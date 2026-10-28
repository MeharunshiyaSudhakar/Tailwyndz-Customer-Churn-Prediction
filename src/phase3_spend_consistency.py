
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "monthly_activity_clean.csv"
OUT = ROOT / "reports" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

for col in ["TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Compare recorded spend with transactions × average basket value.
df["EXPECTED_SPEND"] = (
    df["TRANSACTIONS"] * df["AVG_BASKET_VALUE"]
)

df["SPEND_DIFFERENCE"] = (
    df["TOTAL_SPEND"] - df["EXPECTED_SPEND"]
)

# Flag records with large differences.
# This is an investigation threshold, not an automatic error definition.
df["LARGE_SPEND_DIFFERENCE"] = (
    df["SPEND_DIFFERENCE"].abs() > 10
)

review = df[df["LARGE_SPEND_DIFFERENCE"]].copy()
review = review.sort_values(
    "SPEND_DIFFERENCE",
    key=lambda s: s.abs(),
    ascending=False
)

review.to_csv(
    OUT / "spend_consistency_review.csv", index=False
)

print("TOTAL RECORDS:", len(df))
print("RECORDS WITH SPEND DIFFERENCE > 10:", len(review))
print("\nLARGEST DISCREPANCIES:")
print(
    review[
        [
            "CUSTOMER_ID", "MONTH", "TRANSACTIONS",
            "TOTAL_SPEND", "AVG_BASKET_VALUE",
            "EXPECTED_SPEND", "SPEND_DIFFERENCE"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved review to:", OUT / "spend_consistency_review.csv")
