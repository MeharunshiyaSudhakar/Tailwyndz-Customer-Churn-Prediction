
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "monthly_activity_clean.csv"
OUT = ROOT / "reports" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")
df["TRANSACTIONS"] = pd.to_numeric(
    df["TRANSACTIONS"], errors="coerce"
)
df["TOTAL_SPEND"] = pd.to_numeric(
    df["TOTAL_SPEND"], errors="coerce"
)

df = df.dropna(subset=["MONTH"]).copy()
df["MONTH"] = df["MONTH"].dt.to_period("M").dt.to_timestamp()

# Monthly coverage and activity statistics.
monthly = (
    df.groupby("MONTH")
    .agg(
        activity_rows=("CUSTOMER_ID", "size"),
        unique_customers=("CUSTOMER_ID", "nunique"),
        zero_transaction_rows=(
            "TRANSACTIONS", lambda s: int((s == 0).sum())
        ),
        positive_transaction_rows=(
            "TRANSACTIONS", lambda s: int((s > 0).sum())
        ),
        customers_with_positive_transactions=(
            "CUSTOMER_ID",
            lambda s: s[
                df.loc[s.index, "TRANSACTIONS"] > 0
            ].nunique()
        ),
        total_recorded_spend=("TOTAL_SPEND", "sum"),
        negative_spend_rows=(
            "TOTAL_SPEND", lambda s: int((s < 0).sum())
        )
    )
    .reset_index()
    .sort_values("MONTH")
)

monthly["MONTH"] = monthly["MONTH"].dt.strftime("%Y-%m")
monthly.to_csv(OUT / "monthly_coverage_investigation.csv", index=False)

print("\nMONTHLY COVERAGE INVESTIGATION")
print("=" * 100)
print(monthly.to_string(index=False))

# Compare May and June 2024.
compare = monthly[monthly["MONTH"].isin(["2024-05", "2024-06"])].copy()

if len(compare) == 2:
    compare = compare.set_index("MONTH")
    print("\nMAY vs JUNE 2024")
    for col in [
        "activity_rows",
        "unique_customers",
        "zero_transaction_rows",
        "customers_with_positive_transactions",
        "total_recorded_spend"
    ]:
        may = compare.loc["2024-05", col]
        june = compare.loc["2024-06", col]
        print(f"{col}: May={may}, June={june}, change={june - may}")

# Check whether the customer-month key repeats.
repeated_mask = df.duplicated(
    subset=["CUSTOMER_ID", "MONTH"], keep=False
)
repeated = df[repeated_mask].copy()
repeated.to_csv(
    OUT / "repeated_customer_months_in_coverage.csv", index=False
)

print("\nRows in repeated customer-month groups:", len(repeated))
print("Saved monthly summary to:", OUT / "monthly_coverage_investigation.csv")
