
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_FILE = ROOT / "data/processed/monthly_activity_clean.csv"
OUTPUT_DIR = ROOT / "reports/phase15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Locate the cleaned activity file without guessing if the name differs.
if not ACTIVITY_FILE.exists():
    candidates = list((ROOT / "data/processed").glob("*activity*.csv"))
    candidates = [
        p for p in candidates
        if "customer_features" not in p.name.lower()
    ]
    if len(candidates) != 1:
        print("Could not identify one cleaned monthly activity file.")
        print("CSV files in data/processed:")
        for p in sorted((ROOT / "data/processed").glob("*.csv")):
            print(" -", p.name)
        raise SystemExit(1)
    ACTIVITY_FILE = candidates[0]

activity = pd.read_csv(ACTIVITY_FILE)
required = {"CUSTOMER_ID", "MONTH", "TRANSACTIONS"}
missing = required - set(activity.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")
activity["TRANSACTIONS"] = pd.to_numeric(
    activity["TRANSACTIONS"], errors="coerce"
)

cutoff = pd.Timestamp("2024-03-31")
start = pd.Timestamp("2023-01-01")

pre = activity.loc[
    activity["MONTH"].between(start, cutoff)
].copy()

# Exclude duplicate customer-month groups using the same conservative policy.
duplicate_mask = pre.duplicated(
    ["CUSTOMER_ID", "MONTH"], keep=False
)
pre = pre.loc[~duplicate_mask].copy()

valid = pre.dropna(subset=["TRANSACTIONS"]).copy()

print("Activity file:", ACTIVITY_FILE.name)
print("Rows in feature window after duplicate exclusion:", len(pre))
print("Rows with valid transaction counts:", len(valid))
print("Customers represented:", valid["CUSTOMER_ID"].nunique())
print("Missing transaction values:", int(pre["TRANSACTIONS"].isna().sum()))
print("Negative transaction counts:", int((valid["TRANSACTIONS"] < 0).sum()))

print("\nTransaction count distribution:")
print(valid["TRANSACTIONS"].describe().to_string())

print("\nMonthly-record categories:")
print(
    pd.Series({
        "zero_transactions": int((valid["TRANSACTIONS"] == 0).sum()),
        "positive_transactions": int((valid["TRANSACTIONS"] > 0).sum()),
        "negative_transactions": int((valid["TRANSACTIONS"] < 0).sum()),
    }).to_string()
)

# Compare the latest observed record month with the latest positive-transaction
# month per customer. This is diagnostic only; it does not change model data.
observed = (
    valid.groupby("CUSTOMER_ID")["MONTH"]
    .max()
    .rename("LAST_OBSERVED_MONTH")
)
positive = (
    valid.loc[valid["TRANSACTIONS"] > 0]
    .groupby("CUSTOMER_ID")["MONTH"]
    .max()
    .rename("LAST_POSITIVE_TRANSACTION_MONTH")
)

comparison = pd.concat([observed, positive], axis=1).reset_index()
comparison["LAST_POSITIVE_TRANSACTION_MONTH"] = pd.to_datetime(
    comparison["LAST_POSITIVE_TRANSACTION_MONTH"]
)
comparison["LAST_OBSERVED_MONTH"] = pd.to_datetime(
    comparison["LAST_OBSERVED_MONTH"]
)

comparison["HAS_POSITIVE_TRANSACTION"] = (
    comparison["LAST_POSITIVE_TRANSACTION_MONTH"].notna()
)
comparison["OBSERVED_MONTHS_AFTER_LAST_POSITIVE"] = (
    (comparison["LAST_OBSERVED_MONTH"].dt.year
     - comparison["LAST_POSITIVE_TRANSACTION_MONTH"].dt.year) * 12
    + comparison["LAST_OBSERVED_MONTH"].dt.month
    - comparison["LAST_POSITIVE_TRANSACTION_MONTH"].dt.month
)

comparison.to_csv(
    OUTPUT_DIR / "recency_customer_audit.csv", index=False
)

print("\nCustomer-level recency diagnostic:")
print("Customers audited:", len(comparison))
print(
    "Customers with no positive-transaction month in this window:",
    int((~comparison["HAS_POSITIVE_TRANSACTION"]).sum())
)
print(
    "Customers whose last observed month is later than their last "
    "positive-transaction month:",
    int((comparison["OBSERVED_MONTHS_AFTER_LAST_POSITIVE"] > 0).sum())
)
print(
    "Customers with at least one positive-transaction month:",
    int(comparison["HAS_POSITIVE_TRANSACTION"].sum())
)

print("\nSaved:", OUTPUT_DIR / "recency_customer_audit.csv")
