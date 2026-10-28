
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "phase1"

profile = pd.read_csv(DATA / "customer_profile_deduplicated.csv")
activity = pd.read_csv(DATA / "monthly_activity_deduplicated.csv")
churn = pd.read_csv(DATA / "churn_label_deduplicated.csv")

print("=== PROFILE MISSING VALUES ===")
print(profile.isna().sum()[lambda s: s > 0])

print("\n=== CHURN LABEL MISSING VALUES ===")
print(churn.isna().sum()[lambda s: s > 0])

print("\n=== CHURN LABELS AND HISTORY FLAGS ===")
print(pd.crosstab(
    churn["INSUFFICIENT_HISTORY_FLAG"],
    churn["CHURNED"],
    dropna=False
))

# Inspect repeated customer-month groups
activity["MONTH"] = pd.to_datetime(
    activity["MONTH"], errors="coerce"
)

repeated_mask = activity.duplicated(
    ["CUSTOMER_ID", "MONTH"], keep=False
)
repeated = activity[repeated_mask].sort_values(
    ["CUSTOMER_ID", "MONTH"]
)

print("\n=== REPEATED CUSTOMER-MONTH RECORDS ===")
if repeated.empty:
    print("No repeated customer-month records.")
else:
    print(repeated.to_string(index=False))

REPORTS.mkdir(parents=True, exist_ok=True)
repeated.to_csv(
    REPORTS / "remaining_repeated_customer_months.csv",
    index=False
)

print("\nInvestigation report saved.")
