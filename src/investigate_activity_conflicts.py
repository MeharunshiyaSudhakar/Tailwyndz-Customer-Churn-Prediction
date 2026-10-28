
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reports" / "eda" / "activity_conflicting_duplicate_rows.csv"
OUT = ROOT / "reports" / "eda" / "activity_conflict_investigation.csv"

df = pd.read_csv(SOURCE, parse_dates=["MONTH"])

if df.empty:
    print("No conflicting duplicate rows to investigate.")
    raise SystemExit(0)

keys = ["CUSTOMER_ID", "MONTH"]
results = []

for (customer_id, month), group in df.groupby(keys):
    spend_values = sorted(group["TOTAL_SPEND"].tolist())

    results.append({
        "CUSTOMER_ID": customer_id,
        "MONTH": month,
        "ROWS_IN_GROUP": len(group),
        "DISTINCT_SPEND_VALUES": len(set(spend_values)),
        "TOTAL_SPEND_VALUES": ", ".join(map(str, spend_values)),
        "SAME_OTHER_COLUMNS": (
            group.drop(columns=["TOTAL_SPEND"]).drop_duplicates().shape[0] == 1
        ),
        "STATUS": "REVIEW_REQUIRED",
        "ACTION": "Preserve source records; do not auto-correct",
    })

result = pd.DataFrame(results)
result.to_csv(OUT, index=False)

print(result.to_string(index=False))
print(f"\nInvestigation saved to: {OUT}")
print("Source workbook was not modified.")
