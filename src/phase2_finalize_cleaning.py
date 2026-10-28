
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "phase2"

REPORTS.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# Load the files created by the earlier Phase 2 cleaning script.
profile = pd.read_csv(PROCESSED / "customer_profile_deduplicated.csv")
activity = pd.read_csv(PROCESSED / "monthly_activity_deduplicated.csv")
churn = pd.read_csv(PROCESSED / "churn_label_deduplicated.csv")

# Standardize column names without changing the original workbook.
for df in (profile, activity, churn):
    df.columns = df.columns.str.strip()

# Standardize text casing and whitespace in categorical columns.
categorical_columns = [
    (profile, ["GENDER", "HOME_STORE_PRICE_TIER", "CITY",
               "MEMBERSHIP_TIER", "SIGNUP_CHANNEL"]),
    (activity, []),
    (churn, [])
]

for df, columns in categorical_columns:
    for column in columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.title()
            )

# Parse dates where available.
if "MONTH" in activity.columns:
    activity["MONTH"] = pd.to_datetime(
        activity["MONTH"], errors="coerce"
    )

if "LAST_PURCHASE_DATE" in churn.columns:
    churn["LAST_PURCHASE_DATE"] = pd.to_datetime(
        churn["LAST_PURCHASE_DATE"], errors="coerce"
    )

# Report missing values before any imputation.
missing_reports = []
for name, df in [
    ("customer_profile", profile),
    ("monthly_activity", activity),
    ("churn_label", churn)
]:
    for column in df.columns:
        count = int(df[column].isna().sum())
        if count:
            missing_reports.append({
                "table": name,
                "column": column,
                "missing_count": count,
                "missing_percentage": round(
                    count / len(df) * 100, 2
                )
            })

pd.DataFrame(missing_reports).to_csv(
    REPORTS / "missing_values_review.csv", index=False
)

# Flag negative spend values. Do not change their signs or delete them.
if "TOTAL_SPEND" in activity.columns:
    activity["TOTAL_SPEND"] = pd.to_numeric(
        activity["TOTAL_SPEND"], errors="coerce"
    )

    negative_spend = activity[
        activity["TOTAL_SPEND"] < 0
    ].copy()

    negative_spend.to_csv(
        REPORTS / "negative_spend_review.csv", index=False
    )
else:
    negative_spend = pd.DataFrame()

# Identify repeated customer-month groups and conflicting records.
key_columns = ["CUSTOMER_ID", "MONTH"]
conflict_keys = []

if all(c in activity.columns for c in key_columns):
    repeated_mask = activity.duplicated(
        subset=key_columns, keep=False
    )
    repeated_rows = activity[repeated_mask].copy()

    if not repeated_rows.empty:
        non_key_columns = [
            c for c in activity.columns if c not in key_columns
        ]

        for keys, group in repeated_rows.groupby(
            key_columns, dropna=False
        ):
            has_conflict = any(
                group[c].nunique(dropna=False) > 1
                for c in non_key_columns
            )
            if has_conflict:
                conflict_keys.append(keys)

        repeated_rows.to_csv(
            REPORTS / "repeated_customer_months_review.csv",
            index=False
        )

        if conflict_keys:
            conflict_index = pd.MultiIndex.from_tuples(
                conflict_keys, names=key_columns
            )
            indexed = repeated_rows.set_index(key_columns)
            conflict_rows = indexed.loc[
                indexed.index.isin(conflict_index)
            ].reset_index()

            conflict_rows.to_csv(
                REPORTS / "conflicting_customer_months_review.csv",
                index=False
            )
        else:
            pd.DataFrame().to_csv(
                REPORTS / "conflicting_customer_months_review.csv",
                index=False
            )
    else:
        repeated_rows = pd.DataFrame()

# Keep missing demographic/date values visible for now.
# Imputation and feature-specific handling will be decided later.
# Keep negative spend and conflicting activity rows in the cleaned
# source copies, but document them for controlled handling in Phase 5.

profile.to_csv(PROCESSED / "customer_profile_clean.csv", index=False)
activity.to_csv(PROCESSED / "monthly_activity_clean.csv", index=False)
churn.to_csv(PROCESSED / "churn_label_clean.csv", index=False)

summary = {
    "customer_profile_rows": int(len(profile)),
    "monthly_activity_rows": int(len(activity)),
    "churn_label_rows": int(len(churn)),
    "profile_unique_customer_ids": int(
        profile["CUSTOMER_ID"].nunique()
    ),
    "activity_unique_customer_ids": int(
        activity["CUSTOMER_ID"].nunique()
    ),
    "churn_unique_customer_ids": int(
        churn["CUSTOMER_ID"].nunique()
    ),
    "profile_exact_duplicate_rows": int(
        profile.duplicated().sum()
    ),
    "activity_exact_duplicate_rows": int(
        activity.duplicated().sum()
    ),
    "churn_exact_duplicate_rows": int(
        churn.duplicated().sum()
    ),
    "negative_spend_rows": int(len(negative_spend)),
    "repeated_customer_month_rows": int(
        len(repeated_rows) if "repeated_rows" in locals() else 0
    ),
    "conflicting_customer_month_groups": int(len(conflict_keys)),
    "missing_profile_cells": int(profile.isna().sum().sum()),
    "missing_activity_cells": int(activity.isna().sum().sum()),
    "missing_churn_cells": int(churn.isna().sum().sum()),
    "cleaning_policy": {
        "exact_duplicates": "Removed in the previous Phase 2 step",
        "negative_spend": "Flagged and retained; not converted to positive",
        "conflicting_customer_months": (
            "Flagged and retained; do not choose a value without evidence"
        ),
        "missing_values": (
            "Reported and retained; handling depends on field and use"
        ),
        "last_purchase_date": (
            "Not used as a modelling feature because it may leak the label"
        )
    }
}

with open(REPORTS / "phase2_cleaning_summary.json", "w") as f:
    json.dump(summary, f, indent=4, default=str)

print("\nPHASE 2 CLEANING VALIDATION")
print("-" * 40)
print(json.dumps(summary, indent=4, default=str))
print("\nCleaned source files saved in:", PROCESSED)
print("Review reports saved in:", REPORTS)
