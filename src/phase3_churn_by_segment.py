
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

profile = pd.read_csv(DATA / "customer_profile_clean.csv")
churn = pd.read_csv(DATA / "churn_label_clean.csv")

# Normalize text casing and whitespace.
for col in profile.select_dtypes(include="object").columns:
    profile[col] = profile[col].str.strip()

for col in ["MEMBERSHIP_TIER", "CITY", "HOME_STORE_PRICE_TIER", "GENDER"]:
    if col in profile.columns:
        profile[col] = profile[col].str.title()

df = profile.merge(
    churn[["CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"]],
    on="CUSTOMER_ID",
    how="left",
    validate="one_to_one"
)

df["CHURNED"] = pd.to_numeric(df["CHURNED"], errors="coerce")
df["INSUFFICIENT_HISTORY_FLAG"] = pd.to_numeric(
    df["INSUFFICIENT_HISTORY_FLAG"], errors="coerce"
)

# Use records with valid labels and sufficient history for the primary
# descriptive comparison. Keep all labels in the source data unchanged.
eligible = df[
    df["CHURNED"].isin([0, 1]) &
    (df["INSUFFICIENT_HISTORY_FLAG"] == 0)
].copy()

def segment_summary(data, column):
    summary = (
        data.groupby(column, dropna=False)
        .agg(
            customers=("CUSTOMER_ID", "nunique"),
            churned_customers=("CHURNED", "sum"),
            churn_rate=("CHURNED", "mean")
        )
        .reset_index()
    )
    summary["churn_rate_pct"] = (summary["churn_rate"] * 100).round(2)
    summary["churned_customers"] = summary["churned_customers"].astype(int)
    return summary.sort_values("churn_rate", ascending=False)

for column in ["MEMBERSHIP_TIER", "CITY", "HOME_STORE_PRICE_TIER"]:
    if column not in eligible.columns:
        print(f"Skipping missing column: {column}")
        continue

    result = segment_summary(eligible, column)
    result.to_csv(OUT / f"churn_by_{column.lower()}.csv", index=False)

    print(f"\nCHURN BY {column}")
    print("=" * 65)
    print(
        result[
            [column, "customers", "churned_customers", "churn_rate_pct"]
        ].to_string(index=False)
    )

print("\nEligible customers used:", len(eligible))
print("Excluded for insufficient history or invalid label:",
      len(df) - len(eligible))
print("\nReports saved to:", OUT)
