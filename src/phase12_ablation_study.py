
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
SPLIT_FILE = ROOT / "reports/phase7/phase7_customer_split.csv"
OUTPUT_DIR = ROOT / "reports/phase12"

TARGET = "CHURNED"

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


def evaluate_feature_set(train_part, valid_part, feature_columns, name):
    numeric = train_part[feature_columns].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical = [
        col for col in feature_columns if col not in numeric
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])

    X_train = train_part[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    X_valid = valid_part[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    y_train = train_part[TARGET].astype(int)
    y_valid = valid_part[TARGET].astype(int)

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_valid)[:, 1]
    predicted = (probability >= 0.5).astype(int)

    return {
        "FEATURE_SET": name,
        "FEATURE_COUNT": len(feature_columns),
        "VALIDATION_CUSTOMERS": len(valid_part),
        "VALIDATION_CHURN_RATE": y_valid.mean(),
        "ACCURACY": accuracy_score(y_valid, predicted),
        "PRECISION": precision_score(
            y_valid, predicted, zero_division=0
        ),
        "RECALL": recall_score(
            y_valid, predicted, zero_division=0
        ),
        "F1_SCORE": f1_score(
            y_valid, predicted, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_valid, probability),
        "PR_AUC_AVERAGE_PRECISION": average_precision_score(
            y_valid, probability
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURE_FILE)
    split = pd.read_csv(SPLIT_FILE)

    if features["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in feature dataset.")

    if split["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in split dataset.")

    data = features.merge(
        split[["CUSTOMER_ID", "DATA_SPLIT"]],
        on="CUSTOMER_ID",
        how="inner",
        validate="one_to_one",
    )

    data = data[
        (data["INSUFFICIENT_HISTORY_FLAG"] == 0)
        & data[TARGET].isin([0, 1])
    ].copy()

    training_pool = data[data["DATA_SPLIT"] == "train"].copy()
    if training_pool.empty:
        raise ValueError("No training customers found.")

    predictors = [
        col for col in features.columns
        if col not in EXCLUDE_COLUMNS
    ]

    # Keep the Phase 7 test set untouched.
    # Use only Phase 7 training customers for an internal validation split.
    train_part, valid_part = train_test_split(
        training_pool,
        test_size=0.20,
        random_state=42,
        stratify=training_pool[TARGET].astype(int),
    )

    all_features = predictors

    # Remove features that could encode inactivity/label timing.
    recency_features = {
        "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY",
    }

    late_and_trend_features = {
        col for col in predictors
        if col.startswith("LATE_PERIOD_")
        or "_TREND_LATE_MINUS_EARLY" in col
    }

    temporal_risk_features = recency_features | late_and_trend_features

    # Profile-only model: customer attributes rather than activity behaviour.
    profile_keywords = (
        "AGE",
        "GENDER",
        "CITY",
        "MEMBERSHIP_TIER",
        "MARKETING_OPT_IN",
        "PREFERRED_STORE_ID",
        "PREFERRED_CATEGORY",
        "HOME_STORE_PRICE_TIER",
        "TENURE_DAYS_AT_CUTOFF",
    )
    profile_features = [
        col for col in predictors
        if col in profile_keywords
    ]

    # Behaviour model excludes profile/demographic fields.
    behaviour_features = [
        col for col in predictors
        if col not in profile_features
    ]

    feature_sets = [
        ("ALL_FEATURES", all_features),
        (
            "WITHOUT_RECENCY",
            [c for c in all_features if c not in recency_features],
        ),
        (
            "WITHOUT_LATE_PERIOD_AND_TRENDS",
            [
                c for c in all_features
                if c not in late_and_trend_features
            ],
        ),
        (
            "WITHOUT_RECENCY_LATE_PERIOD_AND_TRENDS",
            [
                c for c in all_features
                if c not in temporal_risk_features
            ],
        ),
        ("PROFILE_ONLY", profile_features),
        ("BEHAVIOUR_ONLY", behaviour_features),
    ]

    results = []
    for name, cols in feature_sets:
        if not cols:
            print(f"Skipping {name}: no feature columns.")
            continue

        print(f"Evaluating {name} with {len(cols)} feature columns...")
        results.append(
            evaluate_feature_set(train_part, valid_part, cols, name)
        )

    results_df = pd.DataFrame(results).sort_values(
        "PR_AUC_AVERAGE_PRECISION", ascending=False
    )

    output_file = OUTPUT_DIR / "phase12_ablation_results.csv"
    report_file = OUTPUT_DIR / "phase12_ablation_report.txt"

    results_df.to_csv(output_file, index=False)

    lines = [
        "PHASE 12 — FEATURE ABLATION STUDY",
        "=" * 65,
        f"Eligible Phase 7 training-pool customers: {len(training_pool)}",
        f"Internal training customers: {len(train_part)}",
        f"Internal validation customers: {len(valid_part)}",
        f"Internal validation churn rate: "
        f"{valid_part[TARGET].mean():.4%}",
        f"Original feature count: {len(all_features)}",
        f"Recency features removed: {sorted(recency_features)}",
        f"Late-period/trend features removed: "
        f"{sorted(late_and_trend_features)}",
        "",
        "RESULTS SORTED BY PR-AUC",
        "-" * 65,
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['FEATURE_SET']}: features={int(row['FEATURE_COUNT'])}, "
            f"accuracy={row['ACCURACY']:.4f}, "
            f"precision={row['PRECISION']:.4f}, "
            f"recall={row['RECALL']:.4f}, "
            f"F1={row['F1_SCORE']:.4f}, "
            f"ROC-AUC={row['ROC_AUC']:.4f}, "
            f"PR-AUC={row['PR_AUC_AVERAGE_PRECISION']:.4f}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 65,
        "1. Compare each ablation with ALL_FEATURES on the same internal split.",
        "2. A large performance drop after removal suggests those features",
        "   contribute predictive information in this validation sample.",
        "3. A small drop suggests the removed group may add limited incremental value.",
        "4. This does not prove that a feature is leakage-free or causal.",
        "5. Very strong results still require checking label-generation rules.",
        "6. The Phase 7 test customers were not used for this ablation comparison.",
        "7. Threshold is fixed at 0.5; no threshold tuning was performed.",
        "",
        "OUTPUT FILES",
        str(output_file),
        str(report_file),
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nPhase 12 completed.")


if __name__ == "__main__":
    main()
