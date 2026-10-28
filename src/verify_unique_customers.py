
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def load_table(sheet_name, required_header):
    raw = pd.read_excel(WORKBOOK, sheet_name=sheet_name, header=None)

    for i in range(len(raw)):
        values = raw.iloc[i].astype(str).str.strip().str.upper()
        if required_header.upper() in values.tolist():
            df = raw.iloc[i + 1:].copy()
            df.columns = raw.iloc[i].astype(str).str.strip()
            df = df.dropna(how="all").reset_index(drop=True)
            df = df.loc[:, ~df.columns.str.lower().eq("nan")]
            return df

    raise ValueError(f"Header {required_header} not found in {sheet_name}")


profile = load_table("fb Customer Profile", "CUSTOMER_ID")
activity = load_table("fb Monthly Activity", "CUSTOMER_ID")
churn = load_table("fb Churn Label", "CUSTOMER_ID")

# Profile: distinguish duplicate rows from unique customers.
profile_unique = profile.drop_duplicates()
profile_conflicts = (
    profile_unique.groupby("CUSTOMER_ID")
    .size()
    .loc[lambda counts: counts > 1]
)

# Churn: remove exact duplicate rows for this audit only.
churn_unique_rows = churn.drop_duplicates()
label_counts = churn_unique_rows.groupby("CUSTOMER_ID")["CHURNED"].nunique()
conflicting_labels = label_counts[label_counts > 1]

# Keep one label per customer only when labels agree.
valid_ids = label_counts[label_counts == 1].index
unique_churn = churn_unique_rows[
    churn_unique_rows["CUSTOMER_ID"].isin(valid_ids)
].drop_duplicates(subset=["CUSTOMER_ID"])

summary = pd.DataFrame([
    {"CHECK": "Profile source rows", "VALUE": len(profile)},
    {"CHECK": "Profile unique customer IDs", "VALUE": profile["CUSTOMER_ID"].nunique()},
    {"CHECK": "Profile exact duplicate rows", "VALUE": profile.duplicated().sum()},
    {"CHECK": "Profile IDs with conflicting profile rows", "VALUE": len(profile_conflicts)},
    {"CHECK": "Churn source rows", "VALUE": len(churn)},
    {"CHECK": "Churn rows after exact deduplication", "VALUE": len(churn_unique_rows)},
    {"CHECK": "Unique churn customer IDs", "VALUE": churn_unique_rows["CUSTOMER_ID"].nunique()},
    {"CHECK": "Customers with conflicting churn labels", "VALUE": len(conflicting_labels)},
    {"CHECK": "Customers with one consistent churn label", "VALUE": len(unique_churn)},
])

summary.to_csv(OUT / "unique_customer_verification.csv", index=False)

print("\nUNIQUE-CUSTOMER VERIFICATION")
print(summary.to_string(index=False))

print("\nCHURN DISTRIBUTION AFTER EXACT DEDUPLICATION")
print(unique_churn["CHURNED"].value_counts(dropna=False).sort_index())

print("\nCHURN PROPORTIONS")
print(unique_churn["CHURNED"].value_counts(normalize=True).sort_index().round(4))

if len(conflicting_labels):
    print("\nWARNING: Conflicting labels found for these customers:")
    print(conflicting_labels.to_string())

print(f"\nSaved: {OUT / 'unique_customer_verification.csv'}")
print("Original workbook was not modified.")
