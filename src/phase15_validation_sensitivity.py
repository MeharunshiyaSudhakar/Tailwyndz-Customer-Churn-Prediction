
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/customer_features_pre_apr2024.csv"
OUT_DIR = ROOT / "reports/phase15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "CHURNED"
RECENCY = "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"
RANDOM_STATE = 42

# Metadata and diagnostic fields excluded from model predictors.
EXCLUDE = {
    "CUSTOMER_ID",
    TARGET,
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

def make_pipeline(X, class_weight):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )),
    ])

def metrics_for(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc_average_precision": average_precision_score(
            y_true, probabilities
        ),
    }

df = pd.read_csv(DATA_PATH)
required = {TARGET, "SIGNUP_DATE", RECENCY, "INSUFFICIENT_HISTORY_FLAG"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0, 1])].copy()
df = df[df["INSUFFICIENT_HISTORY_FLAG"].eq(0)].copy()
df[TARGET] = df[TARGET].astype(int)

# Keep a single fixed stratified split for fair model comparisons.
train_idx, valid_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[TARGET],
)

base_features = [
    c for c in df.columns
    if c not in EXCLUDE and c != TARGET
]
if RECENCY not in base_features:
    raise ValueError(f"{RECENCY} is not in the predictor set.")

experiments = [
    ("all_features_unweighted", base_features, None),
    ("all_features_balanced", base_features, "balanced"),
    (
        "without_recency_unweighted",
        [c for c in base_features if c != RECENCY],
        None,
    ),
    (
        "without_recency_balanced",
        [c for c in base_features if c != RECENCY],
        "balanced",
    ),
]

rows = []
for name, features, weight in experiments:
    X_train = df.loc[train_idx, features].copy()
    X_valid = df.loc[valid_idx, features].copy()
    y_train = df.loc[train_idx, TARGET]
    y_valid = df.loc[valid_idx, TARGET]

    # Convert date-like feature values to numeric day counts, if any.
    for col in X_train.columns:
        if pd.api.types.is_datetime64_any_dtype(X_train[col]):
            X_train[col] = X_train[col].astype("int64") / 86400000000000
            X_valid[col] = X_valid[col].astype("int64") / 86400000000000

    model = make_pipeline(X_train, weight)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)[:, 1]

    row = {
        "experiment": name,
        "class_weight": str(weight),
        "recency_feature_included": RECENCY in features,
        "n_features_before_encoding": len(features),
        "train_customers": len(X_train),
        "validation_customers": len(X_valid),
        "validation_churn_rate": float(y_valid.mean()),
    }
    row.update(metrics_for(y_valid, probabilities))
    rows.append(row)

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT_DIR / "model_sensitivity_comparison.csv", index=False)

# Signup-cohort robustness test. Signup cohort is NOT the label observation
# period, so these metrics must not be described as true temporal validation.
df["SIGNUP_DATE"] = pd.to_datetime(df["SIGNUP_DATE"], errors="coerce")
df["SIGNUP_QUARTER"] = df["SIGNUP_DATE"].dt.to_period("Q").astype("string")

cohort_counts = (
    df.groupby("SIGNUP_QUARTER", dropna=False)
      .agg(CUSTOMERS=(TARGET, "size"), CHURNED=(TARGET, "sum"))
      .reset_index()
)
cohort_counts["CHURN_RATE"] = (
    cohort_counts["CHURNED"] / cohort_counts["CUSTOMERS"]
)
cohort_counts.to_csv(OUT_DIR / "signup_cohort_counts.csv", index=False)

# Use cohorts through 2022Q4 for training and 2023Q1–2023Q2 for testing.
# Require both classes in each side; skip with an explicit reason if not met.
train_quarters = {
    "2021Q1", "2021Q2", "2021Q3", "2021Q4",
    "2022Q1", "2022Q2", "2022Q3", "2022Q4",
}
test_quarters = {"2023Q1", "2023Q2"}

cohort_train = df[df["SIGNUP_QUARTER"].isin(train_quarters)].copy()
cohort_test = df[df["SIGNUP_QUARTER"].isin(test_quarters)].copy()

cohort_result = {
    "validation_type": "signup_cohort_robustness_not_true_temporal_validation",
    "train_cohorts": ",".join(sorted(train_quarters)),
    "test_cohorts": ",".join(sorted(test_quarters)),
    "train_customers": len(cohort_train),
    "test_customers": len(cohort_test),
    "train_churn_rate": (
        float(cohort_train[TARGET].mean()) if len(cohort_train) else np.nan
    ),
    "test_churn_rate": (
        float(cohort_test[TARGET].mean()) if len(cohort_test) else np.nan
    ),
    "status": "",
}

if (
    len(cohort_train) > 0
    and len(cohort_test) > 0
    and cohort_train[TARGET].nunique() == 2
    and cohort_test[TARGET].nunique() == 2
):
    features = base_features
    X_train = cohort_train[features].copy()
    X_test = cohort_test[features].copy()

    for col in X_train.columns:
        if pd.api.types.is_datetime64_any_dtype(X_train[col]):
            X_train[col] = X_train[col].astype("int64") / 86400000000000
            X_test[col] = X_test[col].astype("int64") / 86400000000000

    model = make_pipeline(X_train, "balanced")
    model.fit(X_train, cohort_train[TARGET])
    probabilities = model.predict_proba(X_test)[:, 1]

    cohort_result.update(metrics_for(cohort_test[TARGET], probabilities))
    cohort_result["status"] = "completed_exploratory_cohort_check"
else:
    cohort_result["status"] = (
        "skipped_class_counts_or_sample_size_insufficient"
    )

pd.DataFrame([cohort_result]).to_csv(
    OUT_DIR / "signup_cohort_robustness.csv", index=False
)

summary = f"""# Phase 15 — Validation and Sensitivity Checks

Status: exploratory analysis; not proof of leakage-free or deployment-ready performance.

## Data and split
- Eligible customers: {len(df)}
- Training customers in fixed random split: {len(train_idx)}
- Validation customers in fixed random split: {len(valid_idx)}
- Feature cutoff remains 2024-03-31 as established in the existing pipeline.
- Classification threshold: 0.50.
- Missing-value imputation and scaling/encoding are fitted inside each training pipeline.

## Recency sensitivity
Compare `all_features_*` against `without_recency_*` in
`model_sensitivity_comparison.csv`. A performance drop after removing
`{RECENCY}` indicates predictive dependence on this feature; it does not,
by itself, prove leakage. Investigate how the feature is constructed and
whether it is available at the intended prediction time.

## Class imbalance
Compare weighted and unweighted Logistic Regression using the same random
split. Review recall, precision, F1, PR-AUC, balanced accuracy and ROC-AUC;
there is no single universally best metric.

## Signup-cohort robustness
See `signup_cohort_robustness.csv` and `signup_cohort_counts.csv`.
This is not true temporal validation: customers share the same April–June
2024 churn outcome window, and signup cohorts differ in tenure and exposure.
The check is skipped if either partition lacks both target classes.

## Limitations
- Strong performance may reflect behavioral proxies closely tied to the
  churn definition.
- Cohort differences can reflect customer tenure, exposure and class mix.
- This script does not independently verify the source label-generation
  process or establish causal effects.
"""
(OUT_DIR / "phase15_summary.md").write_text(summary, encoding="utf-8")

print("Phase 15 checks completed.")
print(f"Reports saved to: {OUT_DIR}")
print("\nModel sensitivity comparison:")
print(comparison.round(4).to_string(index=False))
print("\nSignup-cohort robustness:")
print(pd.DataFrame([cohort_result]).round(4).to_string(index=False))
