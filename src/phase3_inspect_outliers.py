
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "monthly_activity_clean.csv"
OUT = ROOT / "reports" / "phase3"

df = pd.read_csv(DATA)
df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")

# Show the 20 highest spending records.
highest = df.nlargest(20, "TOTAL_SPEND").copy()

columns = [
    "CUSTOMER_ID", "MONTH", "TRANSACTIONS", "TOTAL_SPEND",
    "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES",
    "PROMO_TXN_PCT", "APP_SESSIONS", "SUPPORT_TICKETS"
]
columns = [c for c in columns if c in highest.columns]

highest[columns].to_csv(
    OUT / "top_20_spend_records.csv", index=False
)

# Show the 20 most negative spending records.
lowest = df.nsmallest(20, "TOTAL_SPEND").copy()
lowest[columns].to_csv(
    OUT / "bottom_20_spend_records.csv", index=False
)

print("TOP 20 SPENDING RECORDS")
print(highest[columns].to_string(index=False))

print("\nBOTTOM 20 SPENDING RECORDS")
print(lowest[columns].to_string(index=False))

print("\nSaved both reports to:", OUT)
