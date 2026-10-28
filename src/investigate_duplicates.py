
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"

# Read the monthly activity sheet
df = pd.read_excel(FILE, sheet_name="fb Monthly Activity", header=0)

# Find the actual header row if the worksheet has a title row
if "CUSTOMER_ID" not in df.columns:
    header_row = df.index[
        df.astype(str).eq("CUSTOMER_ID").any(axis=1)
    ][0]
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)

# Remove any fully empty rows and normalize column labels
df = df.dropna(how="all")
df.columns = df.columns.astype(str).str.strip()

# Convert relevant columns to numeric
for col in ["TOTAL_SPEND", "TRANSACTIONS"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Identify exact duplicate rows
exact_duplicate_mask = df.duplicated(keep="first")
exact_duplicates = df[exact_duplicate_mask].copy()

# Inspect customer-month groups with more than one row
group_sizes = df.groupby(
    ["CUSTOMER_ID", "MONTH"], dropna=False
).size()

repeated_keys = group_sizes[group_sizes > 1].index
repeated = df.set_index(
    ["CUSTOMER_ID", "MONTH"]
).loc[repeated_keys].reset_index()

# Count groups where rows differ in any column
group_columns = ["CUSTOMER_ID", "MONTH"]
comparison_columns = [c for c in df.columns if c not in group_columns]

group_summary = (
    repeated.groupby(group_columns, dropna=False)[comparison_columns]
    .nunique(dropna=False)
)

conflicting_keys = group_summary.index[
    group_summary.gt(1).any(axis=1)
]

conflicts = repeated.set_index(group_columns).loc[
    conflicting_keys
].reset_index()

OUT = ROOT / "reports" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

exact_duplicates.to_csv(
    OUT / "exact_duplicate_rows.csv", index=False
)
conflicts.to_csv(
    OUT / "conflicting_customer_months.csv", index=False
)

print("TOTAL MONTHLY ACTIVITY ROWS:", len(df))
print("EXACT DUPLICATE ROWS:", int(exact_duplicate_mask.sum()))
print("REPEATED CUSTOMER-MONTH GROUPS:", len(repeated_keys))
print("CUSTOMER-MONTH GROUPS WITH CONFLICTS:", len(conflicting_keys))
print("\nSaved investigation reports to:", OUT)
print("\nConflicting examples:")
print(conflicts.head(10).to_string(index=False))
