
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
OUT_DIR = ROOT / "reports/phase17"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "CHURNED"
OLD_RECENCY = "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"
NEW_RECENCY = "MONTHS_SINCE_LAST_POSITIVE_TXN_IN_WINDOW"
CUTOFF = pd.Timestamp("2024-03-31")
START = pd.Timestamp("2023-01-01")
N_SPLITS = 5
TEST_SIZE = 0.20
BASE_SEED = 42

EXCLUDE = {
    "CUSTOMER_ID", TARGET, "DATA_SPLIT", "SIGNUP_COHORT",
    "INSUFFICIENT_HISTORY_FLAG", "NO_OBSERVED_FEATURE_ACTIVITY",
    "SIGNED_UP_AFTER_FEATURE_CUTOFF", "SIGNUP_DATE",
    "FIRST_FEATURE_MONTH", "LAST_FEATURE_MONTH",
    "OBSERVATION_END_DATE",
}

# ---------- Load and validate ----------
df = pd.read_csv(FEATURE_FILE)
activity = pd.read_csv(ACTIVITY_FILE)

required_features = {
    "CUSTOMER_ID", TARGET, "INSUFFICIENT_HISTORY_FLAG", OLD_RECENCY
}
required_activity = {"CUSTOMER_ID", "MONTH", "TRANSACTIONS"}

if required_features - set(df.columns):
    raise ValueError(
        f"Missing feature columns: {required_features - set(df.columns)}"
    )
if required_activity - set(activity.columns):
    raise ValueError(
        f"Missing activity columns: {required_activity - set(activity.columns)}"
    )

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[
    df[TARGET].isin([0, 1])
    & df["INSUFFICIENT_HISTORY_FLAG"].eq(0)
].copy()
df[TARGET] = df[TARGET].astype(int)

activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")
activity["TRANSACTIONS"] = pd.to_numeric(
    activity["TRANSACTIONS"], errors="coerce"
)
activity = activity[
    activity["MONTH"].between(START, CUTOFF)
].copy()
activity = activity.dropna(subset=["MONTH", "TRANSACTIONS"])

# Same conservative duplicate customer-month policy as Phase 16.
duplicate_mask = activity.duplicated(
    ["CUSTOMER_ID", "MONTH"], keep=False
)
activity = activity.loc[~duplicate_mask].copy()

positive = activity.loc[activity["TRANSACTIONS"] > 0]
last_positive = (
    positive.groupby("CUSTOMER_ID")["MONTH"]
    .max()
    .rename("LAST_POSITIVE_TXN_MONTH")
)

df = df.merge(
    last_positive,
    on="CUSTOMER_ID",
    how="left",
    validate="one_to_one",
)

df[NEW_RECENCY] = (
    (CUTOFF.year - df["LAST_POSITIVE_TXN_MONTH"].dt.year) * 12
    + CUTOFF.month - df["LAST_POSITIVE_TXN_MONTH"].dt.month
)
df["NO_POSITIVE_TXN_IN_FEATURE_WINDOW"] = (
    df["LAST_POSITIVE_TXN_MONTH"].isna().astype(int)
)

base_features = [
    c for c in df.columns
    if c not in EXCLUDE
    and c not in {
        OLD_RECENCY,
        NEW_RECENCY,
        "LAST_POSITIVE_TXN_MONTH",
        "NO_POSITIVE_TXN_IN_FEATURE_WINDOW",
    }
]

feature_sets = {
    "existing_observed_activity_recency": base_features + [OLD_RECENCY],
    "positive_transaction_recency_in_window": (
        base_features + [NEW_RECENCY, "NO_POSITIVE_TXN_IN_FEATURE_WINDOW"]
    ),
    "no_recency_feature": base_features,
}

# ---------- Pipeline ----------
def build_pipeline(X):
    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

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
            random_state=BASE_SEED,
        )),
    ])

def evaluate(y_true, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc_average_precision": average_precision_score(
            y_true, probabilities
        ),
    }

# ---------- Repeated validation ----------
rows = []
y = df[TARGET]

print(f"Eligible customers: {len(df)}")
print(f"Churn rate: {y.mean():.4f}")
print(f"Running {N_SPLITS} stratified splits per model...\n")

for split_no in range(N_SPLITS):
    seed = BASE_SEED + split_no

    train_idx, valid_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    y_train = df.loc[train_idx, TARGET]
    y_valid = df.loc[valid_idx, TARGET]

    for experiment, columns in feature_sets.items():
        X_train = df.loc[train_idx, columns].copy()
        X_valid = df.loc[valid_idx, columns].copy()

        model = build_pipeline(X_train)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_valid)[:, 1]

        row = {
            "split": split_no + 1,
            "random_seed": seed,
            "experiment": experiment,
            "train_customers": len(train_idx),
            "validation_customers": len(valid_idx),
            "validation_churn_rate": float(y_valid.mean()),
        }
        row.update(evaluate(y_valid, probabilities))
        rows.append(row)

detail = pd.DataFrame(rows)
detail.to_csv(
    OUT_DIR / "repeated_validation_split_results.csv",
    index=False,
)

metric_cols = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc_average_precision",
]

summary = (
    detail.groupby("experiment")[metric_cols]
    .agg(["mean", "std"])
    .round(4)
)
summary.columns = [
    f"{metric}_{stat}" for metric, stat in summary.columns
]
summary = summary.reset_index()
summary.to_csv(
    OUT_DIR / "repeated_validation_summary.csv",
    index=False,
)

# Pairwise split-level differences: positive means the first model
# outperformed the second on that metric and split.
wide_f1 = detail.pivot(
    index="split", columns="experiment", values="f1"
)
wide_auc = detail.pivot(
    index="split", columns="experiment", values="roc_auc"
)
wide_pr = detail.pivot(
    index="split", columns="experiment",
    values="pr_auc_average_precision",
)

paired_rows = []
comparisons = [
    ("positive_transaction_recency_in_window",
     "existing_observed_activity_recency"),
    ("positive_transaction_recency_in_window",
     "no_recency_feature"),
]
for first, second in comparisons:
    for metric, wide in [
        ("f1", wide_f1),
        ("roc_auc", wide_auc),
        ("pr_auc_average_precision", wide_pr),
    ]:
        differences = wide[first] - wide[second]
        paired_rows.append({
            "model_1": first,
            "model_2": second,
            "metric": metric,
            "mean_split_difference": differences.mean(),
            "std_split_difference": differences.std(ddof=1),
            "splits_model_1_won": int((differences > 0).sum()),
            "splits_tied": int((differences == 0).sum()),
            "splits_model_2_won": int((differences < 0).sum()),
        })

paired = pd.DataFrame(paired_rows).round(4)
paired.to_csv(
    OUT_DIR / "paired_model_differences.csv",
    index=False,
)

summary_text = f"""# Phase 17 — Repeated Stratified Validation

Status: exploratory robustness check, not an independent final test.

- Eligible customers: {len(df)}
- Target prevalence: {y.mean():.4f}
- Repeats: {N_SPLITS}
- Validation fraction per repeat: {TEST_SIZE:.0%}
- Random seeds: {BASE_SEED} through {BASE_SEED + N_SPLITS - 1}
- Model: balanced Logistic Regression
- Classification threshold: 0.50

Three feature strategies were compared using the same split for each repeat:
1. Existing observed-activity recency.
2. Positive-transaction recency within the available feature window plus
   an explicit no-positive-transaction flag.
3. No recency feature.

Imputation, scaling and categorical encoding are fitted inside each
training pipeline. The repeated splits overlap and therefore are not
independent replications. Means and standard deviations describe
sensitivity to these split choices; they are not confidence intervals.

The activity history starts in January 2023, so positive-transaction
recency is limited to that window and is not lifetime recency. Results
do not prove absence of target-proxy effects, leakage, or deployment risk.
A separate untouched final test set or genuinely later outcome window
would be needed for stronger generalization claims.
"""
(OUT_DIR / "phase17_summary.md").write_text(
    summary_text, encoding="utf-8"
)

print("Phase 17 completed.")
print("Reports saved to:", OUT_DIR)
print("\nMean and standard deviation across splits:")
print(summary.to_string(index=False))
print("\nPaired split-level differences:")
print(paired.to_string(index=False))
