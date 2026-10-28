
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "phase3"
REPORTS.mkdir(parents=True, exist_ok=True)

# Load cleaned datasets
profile = pd.read_csv(DATA / "customer_profile_clean.csv")
churn = pd.read_csv(DATA / "churn_label_clean.csv")

# Standardize column names and category casing
profile.columns = profile.columns.str.strip().str.upper()
churn.columns = churn.columns.str.strip().str.upper()

# Check required columns
required_profile = {"CUSTOMER_ID", "SIGNUP_DATE"}
required_churn = {
    "CUSTOMER_ID",
    "CHURNED",
    "INSUFFICIENT_HISTORY_FLAG",
}

missing_profile = required_profile - set(profile.columns)
missing_churn = required_churn - set(churn.columns)

if missing_profile:
    raise ValueError(
        f"Missing profile columns: {missing_profile}. "
        f"Available columns: {list(profile.columns)}"
    )

if missing_churn:
    raise ValueError(
        f"Missing churn columns: {missing_churn}. "
        f"Available columns: {list(churn.columns)}"
    )

# Parse signup dates
profile["SIGNUP_DATE"] = pd.to_datetime(
    profile["SIGNUP_DATE"], errors="coerce"
)

# Merge customer profile with churn labels
df = profile.merge(
    churn[
        ["CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"]
    ],
    on="CUSTOMER_ID",
    how="inner",
    validate="one_to_one",
)

# Keep customers with sufficient history and valid labels
df["CHURNED"] = pd.to_numeric(df["CHURNED"], errors="coerce")
df["INSUFFICIENT_HISTORY_FLAG"] = pd.to_numeric(
    df["INSUFFICIENT_HISTORY_FLAG"], errors="coerce"
)

eligible = df[
    (df["INSUFFICIENT_HISTORY_FLAG"] == 0)
    & (df["CHURNED"].isin([0, 1]))
    & df["SIGNUP_DATE"].notna()
].copy()

# Group signup dates into calendar quarters
eligible["SIGNUP_COHORT"] = (
    eligible["SIGNUP_DATE"].dt.to_period("Q").astype(str)
)

# Summarize cohort size and churn rate
summary = (
    eligible.groupby("SIGNUP_COHORT")
    .agg(
        CUSTOMERS=("CUSTOMER_ID", "nunique"),
        CHURNED_CUSTOMERS=("CHURNED", "sum"),
        CHURN_RATE=("CHURNED", "mean"),
    )
    .reset_index()
)

summary["CHURN_RATE_PCT"] = (summary["CHURN_RATE"] * 100).round(2)
summary = summary.sort_values("SIGNUP_COHORT")

# Save results
output_path = REPORTS / "churn_by_signup_cohort.csv"
summary.to_csv(output_path, index=False)

print("=== CHURN BY SIGNUP COHORT ===")
print(summary[
    ["SIGNUP_COHORT", "CUSTOMERS", "CHURNED_CUSTOMERS", "CHURN_RATE_PCT"]
].to_string(index=False))

print(f"\nEligible customers analysed: {len(eligible)}")
print(f"Customers excluded for insufficient history/invalid label: "
      f"{len(df) - len(df[df['INSUFFICIENT_HISTORY_FLAG'].eq(0) & df['CHURNED'].isin([0, 1])])}")
print(f"\nReport saved to: {output_path}")
