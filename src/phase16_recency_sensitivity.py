
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
FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
ACTIVITY_FILE = ROOT / "data/processed/monthly_activity_clean.csv"
OUT_DIR = ROOT / "reports/phase16"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "CHURNED"
OLD_RECENCY = "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"
NEW_RECENCY = "MONTHS_SINCE_LAST_POSITIVE_TXN_IN_WINDOW"
CUTOFF = pd.Timestamp("2024-03-31")
START = pd.Timestamp("2023-01-01")
RANDOM_STATE = 42

EXCLUDE = {
    "CUSTOMER_ID", TARGET, "DATA_SPLIT", "SIGNUP_COHORT",
    "INSUFFICIENT_HISTORY_FLAG", "NO_OBSERVED_FEATURE_ACTIVITY",
    "SIGNED_UP_AFTER_FEATURE_CUTOFF", "SIGNUP_DATE",
    "FIRST_FEATURE_MONTH", "LAST_FEATURE_MONTH",
    "OBSERVATION_END_DATE",
}

features_df = pd.read_csv(FEATURE_FILE)
activity = pd.read_csv(ACTIVITY_FILE)

required_features = {
    "CUSTOMER_ID", TARGET, "INSUFFICIENT_HISTORY_FLAG", OLD_RECENCY
}
required_activity = {"CUSTOMER_ID", "MONTH", "TRANSACTIONS"}

if required_features - set(features_df.columns):
    raise ValueError(
        f"Missing feature columns: {required_features - set(features_df.columns)}"
    )
if required_activity - set(activity.columns):
    raise ValueError(
        f"Missing activity columns: {required_activity - set(activity.columns)}"
    )

features_df[TARGET] = pd.to_numeric(features_df[TARGET], errors="coerce")
features_df = features_df[
    features_df[TARGET].isin([0, 1])
    & features_df["INSUFFICIENT_HISTORY_FLAG"].eq(0)
].copy()
features_df[TARGET] = features_df[TARGET].astype(int)

activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")
activity["TRANSACTIONS"] = pd.to_numeric(
    activity["TRANSACTIONS"], errors="coerce"
)
activity = activity[
    activity["MONTH"].between(START, CUTOFF)
].copy()

# Match the existing conservative duplicate-month policy.
duplicate_mask = activity.duplicated(
    ["CUSTOMER_ID", "MONTH"], keep=False
)
activity = activity.loc[~duplicate_mask].copy()

activity = activity.dropna(subset=["TRANSACTIONS"])

positive_activity = activity.loc[activity["TRANSACTIONS"] > 0].copy()
last_positive = (
    positive_activity.groupby("CUSTOMER_ID")["MONTH"]
    .max()
    .rename("LAST_POSITIVE_TXN_MONTH")
)

# Align by customer ID, retaining all eligible customers.
features_df = features_df.merge(
    last_positive, on="CUSTOMER_ID", how="left", validate="one_to_one"
)

features_df[NEW_RECENCY] = (
    (CUTOFF.year - features_df["LAST_POSITIVE_TXN_MONTH"].dt.year) * 12
    + CUTOFF.month - features_df["LAST_POSITIVE_TXN_MONTH"].dt.month
)
features_df["NO_POSITIVE_TXN_IN_FEATURE_WINDOW"] = (
    features_df["LAST_POSITIVE_TXN_MONTH"].isna().astype(int)
)

# Keep the explicit missing recency for customers without positive
# transactions; the model's training-fitted median imputer handles it.
feature_sets = {
    "existing_observed_activity_recency": None,
    "positive_transaction_recency_in_window": None,
    "no_recency_feature": None,
}

base_features = [
    c for c in features_df.columns
    if c not in EXCLUDE
    and c not in {
        OLD_RECENCY, NEW_RECENCY, "LAST_POSITIVE_TXN_MONTH",
        "NO_POSITIVE_TXN_IN_FEATURE_WINDOW",
    }
]

feature_sets["existing_observed_activity_recency"] = (
    base_features + [OLD_RECENCY]
)
feature_sets["positive_transaction_recency_in_window"] = (
    base_features + [NEW_RECENCY, "NO_POSITIVE_TXN_IN_FEATURE_WINDOW"]
)
feature_sets["no_recency_feature"] = base_features

train_idx, valid_idx = train_test_split(
    features_df.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=features_df[TARGET],
)

def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    return Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

def score(y, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc_average_precision": average_precision_score(y, probabilities),
    }

results = []
for experiment, columns in feature_sets.items():
    X_train = features_df.loc[train_idx, columns].copy()
    X_valid = features_df.loc[valid_idx, columns].copy()
    y_train = features_df.loc[train_idx, TARGET]
    y_valid = features_df.loc[valid_idx, TARGET]

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)[:, 1]

    row = {
        "experiment": experiment,
        "train_customers": len(X_train),
        "validation_customers": len(X_valid),
        "validation_churn_rate": float(y_valid.mean()),
        "features_before_encoding": len(columns),
        "no_positive_txn_in_window_train": int(
            features_df.loc[train_idx, "NO_POSITIVE_TXN_IN_FEATURE_WINDOW"].sum()
        ),
        "no_positive_txn_in_window_validation": int(
            features_df.loc[valid_idx, "NO_POSITIVE_TXN_IN_FEATURE_WINDOW"].sum()
        ),
    }
    row.update(score(y_valid, probabilities))
    results.append(row)

comparison = pd.DataFrame(results)
comparison.to_csv(
    OUT_DIR / "recency_model_comparison.csv", index=False
)

customer_audit_cols = [
    "CUSTOMER_ID", TARGET, OLD_RECENCY, NEW_RECENCY,
    "NO_POSITIVE_TXN_IN_FEATURE_WINDOW",
]
features_df[customer_audit_cols].to_csv(
    OUT_DIR / "customer_recency_comparison.csv", index=False
)

summary = """# Phase 16 — Recency Sensitivity Analysis

Status: exploratory sensitivity analysis.

Three balanced Logistic Regression pipelines were evaluated on the same
stratified random train/validation split:
1. Existing recency based on the last observed activity record.
2. Recency based on the last positive-transaction month in the available
   January 2023–March 2024 feature window, plus a no-positive-transaction flag.
3. A baseline excluding recency features.

All preprocessing is fitted within the training pipeline. The positive-
transaction recency is not lifetime recency because the available activity
extract begins in January 2023. Customers without positive transactions in
the feature window retain missing recency and are flagged explicitly.

This is a sensitivity check, not proof of leakage absence, causal impact,
or deployment readiness. Interpret all metrics alongside the label definition
and the limitations of the available activity history.
"""
(OUT_DIR / "phase16_summary.md").write_text(summary, encoding="utf-8")

print("Phase 16 completed.")
print("Reports saved to:", OUT_DIR)
print("\nRecency comparison:")
print(comparison.round(4).to_string(index=False))
print("\nNo-positive-transaction customers in eligible cohort:",
      int(features_df["NO_POSITIVE_TXN_IN_FEATURE_WINDOW"].sum()))
