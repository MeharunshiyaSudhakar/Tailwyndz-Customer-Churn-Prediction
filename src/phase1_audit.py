
from pathlib import Path
import json
import pandas as pd
import numpy as np

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "raw"
    / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
)

REPORT_DIR = BASE_DIR / "reports" / "phase1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

print("PHASE 1: DATASET AUDIT")
print("Dataset:", DATA_FILE)
print("Reports:", REPORT_DIR)


# ==================================================
# 2. LOAD THE CORRECT HEADERS
# ==================================================

excel = pd.ExcelFile(DATA_FILE)

print("\nWorksheets:", excel.sheet_names)

profile = pd.read_excel(
    excel, sheet_name="fb Customer Profile", header=0
)
activity = pd.read_excel(
    excel, sheet_name="fb Monthly Activity", header=0
)
churn = pd.read_excel(
    excel, sheet_name="fb Churn Label", header=0
)

tables = {
    "customer_profile": profile,
    "monthly_activity": activity,
    "churn_label": churn,
}

# If column names are still Unnamed, try the second row as header.
for name, df in tables.items():
    if "CUSTOMER_ID" not in df.columns:
        if len(df) > 0 and "CUSTOMER_ID" in df.iloc[0].astype(str).values:
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df.iloc[1:].reset_index(drop=True)
            tables[name] = df
        else:
            raise ValueError(
                f"{name}: CUSTOMER_ID header not found. "
                "Inspect the Excel sheet layout."
            )

profile = tables["customer_profile"].copy()
activity = tables["monthly_activity"].copy()
churn = tables["churn_label"].copy()

# Convert columns to expected data types.
for df in [profile, activity, churn]:
    df["CUSTOMER_ID"] = pd.to_numeric(
        df["CUSTOMER_ID"], errors="coerce"
    )

profile["SIGNUP_DATE"] = pd.to_datetime(
    profile["SIGNUP_DATE"], errors="coerce"
)

activity["MONTH"] = pd.to_datetime(
    activity["MONTH"], errors="coerce"
)

for col in ["OBSERVATION_END_DATE", "LAST_PURCHASE_DATE"]:
    churn[col] = pd.to_datetime(churn[col], errors="coerce")

for col in [
    "AGE", "MARKETING_OPT_IN", "PREFERRED_STORE_ID"
]:
    profile[col] = pd.to_numeric(profile[col], errors="coerce")

numeric_activity = [
    "TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE",
    "DISTINCT_CATEGORIES", "PROMO_TXN_PCT",
    "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED",
    "SUPPORT_TICKETS", "COMPLAINT_FLAG"
]

for col in numeric_activity:
    activity[col] = pd.to_numeric(activity[col], errors="coerce")

for col in [
    "MONTHS_OBSERVED", "INSUFFICIENT_HISTORY_FLAG", "CHURNED"
]:
    churn[col] = pd.to_numeric(churn[col], errors="coerce")


# ==================================================
# 3. STRUCTURE AND MISSING-VALUE REPORTS
# ==================================================

structure_rows = []
missing_rows = []
duplicate_rows = []

for name, df in tables.items():
    structure_rows.append({
        "table": name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": " | ".join(df.columns.astype(str)),
    })

    for col in df.columns:
        missing_rows.append({
            "table": name,
            "column": col,
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(
                df[col].isna().mean() * 100, 2
            ),
        })

    duplicate_rows.append({
        "table": name,
        "exact_duplicate_rows": int(df.duplicated().sum()),
    })

pd.DataFrame(structure_rows).to_csv(
    REPORT_DIR / "table_structure.csv", index=False
)
pd.DataFrame(missing_rows).to_csv(
    REPORT_DIR / "missing_values.csv", index=False
)
pd.DataFrame(duplicate_rows).to_csv(
    REPORT_DIR / "exact_duplicates.csv", index=False
)

print("\nTABLE STRUCTURE")
print(pd.DataFrame(structure_rows).to_string(index=False))

print("\nMISSING VALUES")
print(
    pd.DataFrame(missing_rows)
    .query("missing_count > 0")
    .to_string(index=False)
)

print("\nEXACT DUPLICATES")
print(pd.DataFrame(duplicate_rows).to_string(index=False))


# ==================================================
# 4. CUSTOMER ID VALIDATION
# ==================================================

id_rows = []

for name, df in tables.items():
    id_rows.append({
        "table": name,
        "rows": len(df),
        "unique_customer_ids": df["CUSTOMER_ID"].nunique(),
        "missing_customer_ids": int(
            df["CUSTOMER_ID"].isna().sum()
        ),
        "repeated_id_rows": int(
            df["CUSTOMER_ID"].notna().sum()
            - df["CUSTOMER_ID"].nunique()
        ),
    })

id_report = pd.DataFrame(id_rows)
id_report.to_csv(REPORT_DIR / "customer_id_audit.csv", index=False)

print("\nCUSTOMER ID AUDIT")
print(id_report.to_string(index=False))

profile_ids = set(profile["CUSTOMER_ID"].dropna())
activity_ids = set(activity["CUSTOMER_ID"].dropna())
churn_ids = set(churn["CUSTOMER_ID"].dropna())

print("\nID OVERLAP")
print("Profile IDs missing from churn:", len(profile_ids - churn_ids))
print("Churn IDs missing from profile:", len(churn_ids - profile_ids))
print("Activity IDs missing from profile:", len(activity_ids - profile_ids))
print("Profile IDs without activity:", len(profile_ids - activity_ids))


# ==================================================
# 5. DATE AND DUPLICATE CUSTOMER-MONTH AUDIT
# ==================================================

print("\nACTIVITY DATE RANGE")
print("Earliest month:", activity["MONTH"].min())
print("Latest month:", activity["MONTH"].max())
print("Invalid dates:", int(activity["MONTH"].isna().sum()))

monthly_counts = (
    activity.dropna(subset=["MONTH"])
    .groupby("MONTH")
    .size()
    .rename("activity_rows")
    .reset_index()
)

monthly_counts.to_csv(
    REPORT_DIR / "monthly_row_counts.csv", index=False
)

duplicate_pairs = activity[
    activity.duplicated(
        ["CUSTOMER_ID", "MONTH"], keep=False
    )
].sort_values(["CUSTOMER_ID", "MONTH"])

duplicate_pairs.to_csv(
    REPORT_DIR / "duplicate_customer_months.csv", index=False
)

print(
    "Rows involved in repeated customer-month pairs:",
    len(duplicate_pairs)
)

# Customer month coverage
coverage = (
    activity.dropna(subset=["CUSTOMER_ID", "MONTH"])
    .groupby("CUSTOMER_ID")["MONTH"]
    .nunique()
    .rename("months_present")
    .reset_index()
)

coverage["months_missing_from_18"] = (
    18 - coverage["months_present"]
).clip(lower=0)

coverage.to_csv(
    REPORT_DIR / "customer_month_coverage.csv", index=False
)


# ==================================================
# 6. INVALID VALUES AND CATEGORICAL CONSISTENCY
# ==================================================

invalid_rows = []

def record_invalid(table_name, column, mask, issue):
    invalid_rows.append({
        "table": table_name,
        "column": column,
        "issue": issue,
        "affected_rows": int(mask.fillna(False).sum()),
    })

record_invalid(
    "customer_profile", "AGE",
    (profile["AGE"] < 0) | (profile["AGE"] > 120),
    "Age outside 0 to 120"
)

record_invalid(
    "monthly_activity", "TOTAL_SPEND",
    activity["TOTAL_SPEND"] < 0,
    "Negative total spend"
)

record_invalid(
    "monthly_activity", "TRANSACTIONS",
    activity["TRANSACTIONS"] < 0,
    "Negative transaction count"
)

record_invalid(
    "monthly_activity", "PROMO_TXN_PCT",
    ~activity["PROMO_TXN_PCT"].between(0, 1),
    "Promotion proportion outside 0 to 1"
)

record_invalid(
    "churn_label", "CHURNED",
    ~churn["CHURNED"].isin([0, 1]),
    "Churn label outside 0 or 1"
)

pd.DataFrame(invalid_rows).to_csv(
    REPORT_DIR / "invalid_value_checks.csv", index=False
)

print("\nINVALID VALUE CHECKS")
print(pd.DataFrame(invalid_rows).to_string(index=False))

category_rows = []

for col in [
    "GENDER", "CITY", "MEMBERSHIP_TIER",
    "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"
]:
    if col in profile.columns:
        vals = profile[col].dropna().astype(str).str.strip()
        category_rows.append({
            "column": col,
            "missing_values": int(profile[col].isna().sum()),
            "distinct_values": vals.nunique(),
            "values": " | ".join(sorted(vals.unique())),
        })

pd.DataFrame(category_rows).to_csv(
    REPORT_DIR / "categorical_value_audit.csv", index=False
)


# ==================================================
# 7. CHURN LABEL AUDIT
# ==================================================

print("\nCHURN DISTRIBUTION")
print(churn["CHURNED"].value_counts(dropna=False))

print("\nINSUFFICIENT HISTORY")
print(churn["INSUFFICIENT_HISTORY_FLAG"].value_counts(dropna=False))

print(
    "\nMissing last purchase dates:",
    int(churn["LAST_PURCHASE_DATE"].isna().sum())
)

target_report = (
    churn["CHURNED"].value_counts(dropna=False)
    .rename_axis("CHURNED")
    .reset_index(name="customer_count")
)
target_report.to_csv(
    REPORT_DIR / "churn_target_distribution.csv", index=False
)


# ==================================================
# 8. SAFE SEGMENT ANALYSIS
# ==================================================

# Temporary copies only; do not alter the source tables.
profile_unique = profile.drop_duplicates().copy()
churn_unique = churn.drop_duplicates().copy()

if (
    profile_unique["CUSTOMER_ID"].is_unique
    and churn_unique["CUSTOMER_ID"].is_unique
):
    segment_data = profile_unique.merge(
        churn_unique[["CUSTOMER_ID", "CHURNED"]],
        on="CUSTOMER_ID",
        how="left",
        validate="one_to_one"
    )

    for col in ["MEMBERSHIP_TIER", "CITY"]:
        result = (
            segment_data.dropna(subset=["CHURNED"])
            .groupby(col, dropna=False)
            .agg(
                customers=("CUSTOMER_ID", "nunique"),
                churn_rate=("CHURNED", "mean")
            )
            .reset_index()
        )

        result["churn_rate_pct"] = (
            result["churn_rate"] * 100
        ).round(2)

        result.to_csv(
            REPORT_DIR / f"churn_by_{col.lower()}.csv",
            index=False
        )

        print(f"\nChurn by {col}")
        print(result.to_string(index=False))
else:
    print(
        "\nSegment analysis skipped: customer IDs are not unique "
        "after exact duplicate removal."
    )


# ==================================================
# 9. FINAL SUMMARY
# ==================================================

summary = {
    "workbook": DATA_FILE.name,
    "sheets": excel.sheet_names,
    "table_shapes": {
        name: {
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        }
        for name, df in tables.items()
    },
    "profile_unique_ids": int(profile["CUSTOMER_ID"].nunique()),
    "activity_unique_ids": int(activity["CUSTOMER_ID"].nunique()),
    "churn_unique_ids": int(churn["CUSTOMER_ID"].nunique()),
    "activity_start": str(activity["MONTH"].min()),
    "activity_end": str(activity["MONTH"].max()),
    "activity_exact_duplicates": int(activity.duplicated().sum()),
    "duplicate_customer_month_rows": int(len(duplicate_pairs)),
    "missing_last_purchase_dates": int(
        churn["LAST_PURCHASE_DATE"].isna().sum()
    ),
}

with open(
    REPORT_DIR / "phase1_summary.json", "w", encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nPHASE 1 AUDIT COMPLETED")
print("All reports saved to:", REPORT_DIR)
print(json.dumps(summary, indent=4))
