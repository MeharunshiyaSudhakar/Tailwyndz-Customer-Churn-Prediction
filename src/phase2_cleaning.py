
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "phase1"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def read_sheet(sheet_name):
    df = pd.read_excel(FILE, sheet_name=sheet_name, header=0)

    # Handle worksheets whose actual headers appear below a title row.
    if "CUSTOMER_ID" not in df.columns:
        matches = df.index[
            df.astype(str).eq("CUSTOMER_ID").any(axis=1)
        ]
        if len(matches) == 0:
            raise ValueError(f"Could not locate CUSTOMER_ID in {sheet_name}")

        header_row = matches[0]
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)

    df.columns = df.columns.astype(str).str.strip()
    return df.dropna(how="all").copy()


# Load all three data tables
profile = read_sheet("fb Customer Profile")
activity = read_sheet("fb Monthly Activity")
churn = read_sheet("fb Churn Label")

# Standardize customer IDs and dates for reliable comparisons
for df in [profile, activity, churn]:
    df["CUSTOMER_ID"] = pd.to_numeric(
        df["CUSTOMER_ID"], errors="raise"
    ).astype("int64")

activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")

# Remove only rows that are identical across every column.
profile_clean = profile.drop_duplicates().copy()
activity_clean = activity.drop_duplicates().copy()
churn_clean = churn.drop_duplicates().copy()

# Keep all conflicting customer-month records for now.
duplicate_keys = activity_clean.duplicated(
    ["CUSTOMER_ID", "MONTH"], keep=False
)
repeated = activity_clean[duplicate_keys].copy()

comparison_cols = [
    c for c in activity_clean.columns
    if c not in ["CUSTOMER_ID", "MONTH"]
]

conflicting_keys = (
    repeated.groupby(["CUSTOMER_ID", "MONTH"])[comparison_cols]
    .nunique(dropna=False)
    .gt(1)
    .any(axis=1)
)
conflict_index = conflicting_keys[conflicting_keys].index

conflicts = (
    repeated.set_index(["CUSTOMER_ID", "MONTH"])
    .loc[conflict_index]
    .reset_index()
)

conflicts.to_csv(
    REPORTS / "conflicting_customer_months_review.csv",
    index=False
)

# Save processed tables; do not overwrite the source workbook.
profile_clean.to_csv(OUT / "customer_profile_deduplicated.csv", index=False)
activity_clean.to_csv(OUT / "monthly_activity_deduplicated.csv", index=False)
churn_clean.to_csv(OUT / "churn_label_deduplicated.csv", index=False)

print("PHASE 2 - INITIAL DUPLICATE HANDLING")
print(f"Profile: {len(profile)} -> {len(profile_clean)} rows")
print(f"Activity: {len(activity)} -> {len(activity_clean)} rows")
print(f"Churn labels: {len(churn)} -> {len(churn_clean)} rows")
print(f"Conflicting customer-month groups flagged: {len(conflict_index)}")
print("Negative spending values retained for investigation.")
print("Original Excel workbook was not modified.")
print(f"Processed data saved in: {OUT}")
