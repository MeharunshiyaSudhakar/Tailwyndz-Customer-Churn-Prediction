
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def load_table(sheet_name):
    raw = pd.read_excel(WORKBOOK, sheet_name=sheet_name, header=None)

    for i in range(len(raw)):
        headers = (
            raw.iloc[i].fillna("").astype(str).str.strip().str.upper()
        )
        if "CUSTOMER_ID" in headers.tolist():
            df = raw.iloc[i + 1:].copy()
            df.columns = raw.iloc[i].fillna("").astype(str).str.strip()
            df = df.dropna(how="all").reset_index(drop=True)
            df = df.loc[:, [
                str(c).strip().lower() not in ("", "nan", "none")
                for c in df.columns
            ]]
            df["CUSTOMER_ID"] = pd.to_numeric(
                df["CUSTOMER_ID"], errors="coerce"
            ).astype("Int64")
            return df

    raise ValueError(f"CUSTOMER_ID header not found: {sheet_name}")


profile = load_table("fb Customer Profile")
activity = load_table("fb Monthly Activity")
churn = load_table("fb Churn Label")

profile_ids = set(profile["CUSTOMER_ID"].dropna().unique())
activity_ids = set(activity["CUSTOMER_ID"].dropna().unique())
churn_ids = set(churn["CUSTOMER_ID"].dropna().unique())

checks = {
    "Profile IDs missing from activity": sorted(profile_ids - activity_ids),
    "Activity IDs missing from profile": sorted(activity_ids - profile_ids),
    "Profile IDs missing from churn labels": sorted(profile_ids - churn_ids),
    "Churn IDs missing from profile": sorted(churn_ids - profile_ids),
    "Churn IDs missing from activity": sorted(churn_ids - activity_ids),
}

rows = []
for name, ids in checks.items():
    print(f"{name}: {len(ids)}")
    rows.append({
        "CHECK": name,
        "CUSTOMER_COUNT": len(ids),
        "CUSTOMER_IDS": ", ".join(map(str, ids)),
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "customer_coverage_audit.csv", index=False)

print(f"\nSaved: {OUT / 'customer_coverage_audit.csv'}")
print("Source workbook was not modified.")
