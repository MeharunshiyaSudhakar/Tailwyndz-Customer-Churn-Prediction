
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
SPLIT_FILE = ROOT / "reports/phase7/phase7_customer_split.csv"
OUTPUT_DIR = ROOT / "reports/phase8"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
PREDICTION_THRESHOLD = 0.50

TARGET = "CHURNED"

# These columns are identifiers, labels, split/eligibility fields,
# or raw date/diagnostic fields. They are not model predictors.
EXCLUDE_COLUMNS = {
    "CUSTOMER_ID",
    "CHURNED",
    "DATA_SPLIT",
    "SIGNUP_COHORT",
    "INSUFFICIENT_HISTORY_FLAG",
    "NO_OBSERVED_FEATURE_ACTIVITY",
    "SIGNED_UP_AFTER_FEATURE_CUTOFF",
    "SIGNUP_DATE",
    "FIRST_FEATURE_MONTH",
    "LAST_FEATURE_MONTH",
    "OBSERVATION_END_DATE",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Phase 7 split file not found: {SPLIT_FILE}")

    features = pd.read_csv(FEATURE_FILE)
    split = pd.read_csv(SPLIT_FILE)

    required_features = {"CUSTOMER_ID", TARGET}
    required_split = {"CUSTOMER_ID", TARGET, "DATA_SPLIT"}

    if not required_features.issubset(features.columns):
        raise ValueError(
            f"Feature file must contain: {sorted(required_features)}"
        )
    if not required_split.issubset(split.columns):
        raise ValueError(
            f"Split file must contain: {sorted(required_split)}"
        )

    if features["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Feature file contains duplicate CUSTOMER_ID values.")
    if split["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Split file contains duplicate CUSTOMER_ID values.")

    # Use the Phase 7 assignments; do not create a new final test split.
    data = split[["CUSTOMER_ID", TARGET, "DATA_SPLIT"]].merge(
        features,
        on="CUSTOMER_ID",
        how="left",
        suffixes=("_split", ""),
        validate="one_to_one",
    )

    # Verify the labels agree between Phase 5 and Phase 7.
    label_col = f"{TARGET}_split"
    if label_col not in data.columns:
        raise ValueError("Could not verify Phase 7 labels against feature labels.")

    if not (data[label_col] == data[TARGET]).all():
        raise ValueError("CHURNED labels disagree between Phase 5 and Phase 7.")

    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("CHURNED must contain only 0 and 1.")

    # Phase 8 model development uses training customers only.
    train_data = data[data["DATA_SPLIT"] == "train"].copy()
    test_count = int((data["DATA_SPLIT"] == "test").sum())

    if train_data.empty:
        raise ValueError("No training customers found in the Phase 7 split.")

    feature_columns = [
        col for col in features.columns
        if col not in EXCLUDE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No predictor columns remain after exclusions.")

    X = train_data[feature_columns].copy()
    y = train_data[TARGET].copy()

    # Convert numeric-looking data carefully; retain categorical columns.
    for col in X.columns:
        if X[col].dtype == "object":
            # Keep text categories as categories; dates were excluded above.
            X[col] = X[col].replace(r"^\s*$", np.nan, regex=True)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_columns = X_fit.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        col for col in X_fit.columns
        if col not in numeric_columns
    ]

    # Dense output supports both tree boosting and linear models.
    # Every imputer, encoder and scaler is fitted within each model pipeline
    # using X_fit only, never the validation or final test rows.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    models = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    validation_predictions = []

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("model", estimator),
        ])

        pipeline.fit(X_fit, y_fit)
        predicted = pipeline.predict(X_val)

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_val)[:, 1]
        else:
            probabilities = predicted.astype(float)

        # Some metrics can be undefined if a model predicts no positives.
        precision = precision_score(
            y_val, predicted, zero_division=0
        )
        recall = recall_score(
            y_val, predicted, zero_division=0
        )
        f1 = f1_score(y_val, predicted, zero_division=0)

        roc_auc = (
            roc_auc_score(y_val, probabilities)
            if y_val.nunique() == 2 else np.nan
        )
        pr_auc = (
            average_precision_score(y_val, probabilities)
            if y_val.nunique() == 2 else np.nan
        )

        results.append({
            "MODEL": model_name,
            "ACCURACY": accuracy_score(y_val, predicted),
            "PRECISION": precision,
            "RECALL": recall,
            "F1_SCORE": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC_AVERAGE_PRECISION": pr_auc,
            "PREDICTION_THRESHOLD": PREDICTION_THRESHOLD,
        })

        for customer_id, actual, pred, prob in zip(
            train_data.loc[X_val.index, "CUSTOMER_ID"],
            y_val,
            predicted,
            probabilities,
        ):
            validation_predictions.append({
                "CUSTOMER_ID": customer_id,
                "ACTUAL_CHURNED": int(actual),
                "PREDICTED_CHURNED": int(pred),
                "PREDICTED_CHURN_PROBABILITY": float(prob),
                "MODEL": model_name,
            })

    comparison = pd.DataFrame(results).sort_values(
        by="PR_AUC_AVERAGE_PRECISION",
        ascending=False,
    )

    comparison_file = OUTPUT_DIR / "phase8_model_comparison.csv"
    predictions_file = OUTPUT_DIR / "phase8_validation_predictions.csv"
    report_file = OUTPUT_DIR / "phase8_model_comparison_report.txt"

    comparison.to_csv(comparison_file, index=False)
    pd.DataFrame(validation_predictions).to_csv(
        predictions_file, index=False
    )

    lines = [
        "PHASE 8 — BASELINE AND ML MODEL COMPARISON",
        "=" * 52,
        f"Feature dataset: {FEATURE_FILE}",
        f"Phase 7 split: {SPLIT_FILE}",
        f"Eligible customers in Phase 7 split: {len(data)}",
        f"Training customers available for model development: {len(train_data)}",
        f"Final test customers held untouched: {test_count}",
        f"Internal training customers: {len(X_fit)}",
        f"Internal validation customers: {len(X_val)}",
        f"Predictor count: {len(feature_columns)}",
        f"Numeric predictors: {len(numeric_columns)}",
        f"Categorical predictors: {len(categorical_columns)}",
        f"Validation churn rate: {y_val.mean():.4%}",
        f"Prediction threshold for class metrics: {PREDICTION_THRESHOLD}",
        "",
        "VALIDATION RESULTS (sorted by PR-AUC / average precision)",
        "-" * 52,
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "IMPORTANT NOTES",
        "- Model selection used only an internal validation subset of Phase 7 training customers.",
        "- The 446 Phase 7 test customers were not used to fit models or select a winner.",
        "- Preprocessing was fitted within each pipeline on internal training rows only.",
        "- Class metrics use a fixed 0.50 threshold for this comparison; it is not a tuned threshold.",
        "- PR-AUC is reported as average precision.",
        "- Results are validation results, not final test-set performance.",
        "- This experiment does not convert the existing random split into a true future-time holdout.",
        "- Missing activity and trend values may carry limitations; interpretation must be cautious.",
        "",
        "OUTPUT FILES",
        str(comparison_file),
        str(predictions_file),
        str(report_file),
    ]

    report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nPhase 8 model comparison completed.")


if __name__ == "__main__":
    main()
