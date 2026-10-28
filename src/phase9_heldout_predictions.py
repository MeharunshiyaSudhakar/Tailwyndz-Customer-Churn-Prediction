
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
SPLIT_FILE = ROOT / "reports/phase7/phase7_customer_split.csv"
OUTPUT_DIR = ROOT / "reports/phase9"

TARGET = "CHURNED"
THRESHOLD = 0.50
RANDOM_STATE = 42

# Never allow identifiers, target, split assignments, or diagnostic
# eligibility fields to become model predictors.
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
        raise FileNotFoundError(f"Missing feature dataset: {FEATURE_FILE}")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing Phase 7 split: {SPLIT_FILE}")

    features = pd.read_csv(FEATURE_FILE)
    split = pd.read_csv(SPLIT_FILE)

    required_features = {"CUSTOMER_ID", TARGET}
    required_split = {"CUSTOMER_ID", TARGET, "DATA_SPLIT"}

    if not required_features.issubset(features.columns):
        raise ValueError(f"Feature dataset needs {required_features}")
    if not required_split.issubset(split.columns):
        raise ValueError(f"Phase 7 split needs {required_split}")

    if features["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate CUSTOMER_ID values in feature dataset.")
    if split["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate CUSTOMER_ID values in Phase 7 split.")

    # Reuse the exact Phase 7 assignment; do not generate a new split.
    data = split[["CUSTOMER_ID", TARGET, "DATA_SPLIT"]].merge(
        features,
        on="CUSTOMER_ID",
        how="left",
        suffixes=("_split", ""),
        validate="one_to_one",
    )

    split_label = f"{TARGET}_split"
    if split_label not in data.columns:
        raise ValueError("Unable to verify split labels against feature labels.")

    if data[split_label].isna().any() or data[TARGET].isna().any():
        raise ValueError("Missing churn labels found.")
    if not (data[split_label].astype(int) == data[TARGET].astype(int)).all():
        raise ValueError("Churn labels disagree between the split and feature data.")

    data[TARGET] = data[TARGET].astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("CHURNED must contain only 0 and 1.")

    train = data[data["DATA_SPLIT"] == "train"].copy()
    test = data[data["DATA_SPLIT"] == "test"].copy()

    if len(train) != 1781 or len(test) != 446:
        raise ValueError(
            f"Unexpected split sizes: train={len(train)}, test={len(test)}. "
            "Check the Phase 7 split before continuing."
        )

    feature_columns = [
        col for col in features.columns
        if col not in EXCLUDE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No predictor columns remain after exclusions.")

    X_train = train[feature_columns].copy()
    y_train = train[TARGET].copy()
    X_test = test[feature_columns].copy()

    # Blank strings are treated as missing; categories remain categorical.
    for col in feature_columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].replace(r"^\s*$", np.nan, regex=True)
            X_test[col] = X_test[col].replace(r"^\s*$", np.nan, regex=True)

    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        col for col in X_train.columns
        if col not in numeric_columns
    ]

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

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    # Fit only on the 1,781 training customers.
    model.fit(X_train, y_train)

    # Predict probabilities for the 446 held-out customers.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    if len(probabilities) != len(test):
        raise ValueError("Prediction count does not match test customer count.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite churn probabilities found.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Churn probabilities must be between 0 and 1.")

    output = pd.DataFrame({
        "CUSTOMER_ID": test["CUSTOMER_ID"].to_numpy(),
        "ACTUAL_CHURNED": test[TARGET].to_numpy(dtype=int),
        "PREDICTED_CHURN_PROBABILITY": probabilities,
        "PREDICTED_CHURNED": predictions,
        "MODEL": "Logistic Regression",
    })

    if output["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate IDs in prediction output.")

    output_file = OUTPUT_DIR / "phase9_heldout_churn_predictions.csv"
    report_file = OUTPUT_DIR / "phase9_prediction_report.txt"

    output.to_csv(output_file, index=False, float_format="%.8f")

    report_lines = [
        "CORE TASK 7 — HELD-OUT CHURN PROBABILITIES",
        "=" * 52,
        f"Feature dataset: {FEATURE_FILE}",
        f"Phase 7 split: {SPLIT_FILE}",
        "Selected model: Logistic Regression",
        "Model selection basis: highest PR-AUC in initial internal validation",
        f"Training customers used to fit model: {len(train)}",
        f"Held-out customers predicted: {len(test)}",
        f"Predictor columns: {len(feature_columns)}",
        f"Numeric predictors: {len(numeric_columns)}",
        f"Categorical predictors: {len(categorical_columns)}",
        f"Prediction threshold: {THRESHOLD:.2f}",
        "",
        "PREDICTION SUMMARY",
        "-" * 52,
        f"Predicted churned: {int((predictions == 1).sum())}",
        f"Predicted not churned: {int((predictions == 0).sum())}",
        f"Mean predicted churn probability: {probabilities.mean():.6f}",
        f"Median predicted churn probability: {np.median(probabilities):.6f}",
        f"Minimum predicted churn probability: {probabilities.min():.6f}",
        f"Maximum predicted churn probability: {probabilities.max():.6f}",
        f"Actual churned in held-out set (for later evaluation): {int((test[TARGET] == 1).sum())}",
        f"Actual not churned in held-out set (for later evaluation): {int((test[TARGET] == 0).sum())}",
        "",
        "SAFETY AND LIMITATIONS",
        "- Model fitting used training customers only.",
        "- The held-out test set was not used to fit preprocessing or the model.",
        "- Actual labels are included only for later evaluation.",
        "- The 0.50 threshold is a provisional threshold, not a tuned business decision.",
        "- Probabilities are estimates and are not guaranteed to be calibrated.",
        "- Earlier validation scores were unusually high; audit feature/label relationships before trusting results.",
        "- This uses the existing random holdout, not a true future-time holdout.",
        "",
        "OUTPUT FILES",
        str(output_file),
        str(report_file),
    ]

    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print("\nHeld-out prediction generation completed.")


if __name__ == "__main__":
    main()
