
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
path = root / "data" / "processed" / "monthly_activity_clean.csv"
out = root / "reports" / "phase2"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path)

df["TOTAL_SPEND"] = pd.to_numeric(df["TOTAL_SPEND"], errors="coerce")
df["TRANSACTIONS"] = pd.to_numeric(df["TRANSACTIONS"], errors="coerce")

zero_spend_positive_txns = df[
    (df["TOTAL_SPEND"] == 0) & (df["TRANSACTIONS"] > 0)
]
zero_txns_positive_spend = df[
    (df["TRANSACTIONS"] == 0) & (df["TOTAL_SPEND"] > 0)
]
negative_txns = df[df["TRANSACTIONS"] < 0]

zero_spend_positive_txns.to_csv(
    out / "zero_spend_positive_transactions.csv", index=False
)
zero_txns_positive_spend.to_csv(
    out / "zero_transactions_positive_spend.csv", index=False
)
negative_txns.to_csv(
    out / "negative_transaction_counts.csv", index=False
)

print("Zero spend + positive transactions:", len(zero_spend_positive_txns))
print("Zero transactions + positive spend:", len(zero_txns_positive_spend))
print("Negative transaction counts:", len(negative_txns))
print("Reports saved to:", out)
