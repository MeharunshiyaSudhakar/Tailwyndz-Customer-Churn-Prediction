
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "monthly_activity_clean.csv"
OUT = ROOT / "reports" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")
df["TOTAL_SPEND"] = pd.to_numeric(
    df["TOTAL_SPEND"], errors="coerce"
)

# Summary of negative, zero and positive spend values.
spend_summary = {
    "total_rows": int(len(df)),
    "missing_spend": int(df["TOTAL_SPEND"].isna().sum()),
    "negative_spend_rows": int((df["TOTAL_SPEND"] < 0).sum()),
    "zero_spend_rows": int((df["TOTAL_SPEND"] == 0).sum()),
    "positive_spend_rows": int((df["TOTAL_SPEND"] > 0).sum()),
    "negative_spend_transactions_positive": int(
        (
            (df["TOTAL_SPEND"] < 0)
            & (pd.to_numeric(df["TRANSACTIONS"], errors="coerce") > 0)
        ).sum()
    ),
    "negative_spend_avg_basket_positive": int(
        (
            (df["TOTAL_SPEND"] < 0)
            & (pd.to_numeric(df["AVG_BASKET_VALUE"], errors="coerce") > 0)
        ).sum()
    ),
}

# Identify repeated customer-month keys.
keys = ["CUSTOMER_ID", "MONTH"]
repeated = df[df.duplicated(keys, keep=False)].copy()

conflict_keys = []
for key, group in repeated.groupby(keys, dropna=False):
    other_cols = [c for c in df.columns if c not in keys]
    if any(group[c].nunique(dropna=False) > 1 for c in other_cols):
        conflict_keys.append(key)

spend_summary["repeated_customer_month_rows"] = int(len(repeated))
spend_summary["conflicting_customer_month_groups"] = int(
    len(conflict_keys)
)

with open(OUT / "spend_diagnostics_summary.json", "w") as f:
    json.dump(spend_summary, f, indent=4)

# Export suspicious records for traceability.
df[df["TOTAL_SPEND"] < 0].to_csv(
    OUT / "negative_spend_review.csv", index=False
)

print(json.dumps(spend_summary, indent=4))
print("\nSaved diagnostics to:", OUT)
