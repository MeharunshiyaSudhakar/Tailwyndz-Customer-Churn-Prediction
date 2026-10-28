
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

if not WORKBOOK.exists():
    raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")


# ---------------------------------------------------------
# 2. LOAD TABLES USING THE REAL HEADER ROW
# ---------------------------------------------------------
def load_table(sheet_name, required_header):
    raw = pd.read_excel(WORKBOOK, sheet_name=sheet_name, header=None)

    for i in range(len(raw)):
        values = (
            raw.iloc[i]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        if required_header.upper() in values:
            headers = raw.iloc[i].fillna("").astype(str).str.strip()
            df = raw.iloc[i + 1:].copy()
            df.columns = headers
            df = df.reset_index(drop=True)

            # Remove empty header columns and completely empty rows.
            keep_columns = [
                str(c).strip().lower() not in ("", "nan", "none")
                for c in df.columns
            ]
            df = df.loc[:, keep_columns]
            df = df.dropna(how="all").reset_index(drop=True)

            # Remove whitespace from column names.
            df.columns = [str(c).strip() for c in df.columns]

            # Normalize numeric-looking customer IDs.
            if "CUSTOMER_ID" in df.columns:
                df["CUSTOMER_ID"] = pd.to_numeric(
                    df["CUSTOMER_ID"], errors="coerce"
                ).astype("Int64")

            return df

    raise ValueError(
        f"Could not find '{required_header}' in sheet '{sheet_name}'"
    )


profile = load_table("fb Customer Profile", "CUSTOMER_ID")
activity = load_table("fb Monthly Activity", "CUSTOMER_ID")
churn = load_table("fb Churn Label", "CUSTOMER_ID")

tables = {
    "profile": (profile, ["CUSTOMER_ID"]),
    "activity": (activity, ["CUSTOMER_ID", "MONTH"]),
    "churn": (churn, ["CUSTOMER_ID"]),
}

print("=" * 72)
print("FRESHBASKET CHURN — COMPLETE EDA VERIFICATION")
print("=" * 72)
print(f"Workbook: {WORKBOOK}")
print(f"Output folder: {OUT}")


# ---------------------------------------------------------
# 3. REQUIRED COLUMNS
# ---------------------------------------------------------
required = {
    "profile": [
        "CUSTOMER_ID", "SIGNUP_DATE", "AGE", "GENDER",
        "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN",
        "PREFERRED_STORE_ID", "PREFERRED_CATEGORY",
        "HOME_STORE_PRICE_TIER",
    ],
    "activity": [
        "CUSTOMER_ID", "MONTH", "TRANSACTIONS", "TOTAL_SPEND",
        "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT",
        "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED",
        "SUPPORT_TICKETS", "COMPLAINT_FLAG",
    ],
    "churn": [
        "CUSTOMER_ID", "OBSERVATION_END_DATE", "LAST_PURCHASE_DATE",
        "MONTHS_OBSERVED", "INSUFFICIENT_HISTORY_FLAG", "CHURNED",
    ],
}

quality_rows = []

for name, (df, keys) in tables.items():
    missing_columns = sorted(set(required[name]) - set(df.columns))
    status = "PASS" if not missing_columns else "FAIL"

    quality_rows.append({
        "CHECK": f"{name}: required columns",
        "VALUE": ", ".join(missing_columns) if missing_columns else "All present",
        "STATUS": status,
    })

    print(f"\n{name.upper()}: {df.shape[0]} rows x {df.shape[1]} columns")
    if missing_columns:
        raise ValueError(f"{name}: missing required columns {missing_columns}")


# ---------------------------------------------------------
# 4. MISSING-VALUE AUDIT
# ---------------------------------------------------------
missing_rows = []

for name, (df, _) in tables.items():
    for column in df.columns:
        missing_rows.append({
            "TABLE": name,
            "COLUMN": column,
            "ROW_COUNT": len(df),
            "MISSING_COUNT": int(df[column].isna().sum()),
            "MISSING_PERCENT": round(
                df[column].isna().mean() * 100, 3
            ) if len(df) else 0,
        })

missing_df = pd.DataFrame(missing_rows)
missing_df.to_csv(OUT / "missing_values.csv", index=False)


# ---------------------------------------------------------
# 5. DUPLICATE AND KEY-CONFLICT AUDIT
# ---------------------------------------------------------
duplicate_rows = []
conflict_files = {}

for name, (df, keys) in tables.items():
    exact_duplicates = int(df.duplicated().sum())
    duplicate_mask = df.duplicated(subset=keys, keep=False)
    duplicate_key_rows = df.loc[duplicate_mask].copy()

    conflict_groups = []
    duplicate_group_count = 0

    for _, group in duplicate_key_rows.groupby(
        keys, dropna=False, sort=False
    ):
        if len(group) <= 1:
            continue

        duplicate_group_count += 1
        non_key = [c for c in df.columns if c not in keys]

        if non_key and len(group[non_key].drop_duplicates()) > 1:
            conflict_groups.append(group)

    conflicts = (
        pd.concat(conflict_groups, ignore_index=True)
        if conflict_groups
        else pd.DataFrame(columns=df.columns)
    )

    duplicate_rows.append({
        "TABLE": name,
        "SOURCE_ROWS": len(df),
        "EXACT_DUPLICATE_ROWS": exact_duplicates,
        "ROWS_SHARING_DUPLICATE_KEYS": len(duplicate_key_rows),
        "DUPLICATE_KEY_GROUPS": duplicate_group_count,
        "CONFLICTING_KEY_GROUPS": len(conflict_groups),
    })

    duplicate_key_rows.to_csv(
        OUT / f"{name}_duplicate_key_rows.csv", index=False
    )
    conflicts.to_csv(
        OUT / f"{name}_conflicting_duplicate_rows.csv", index=False
    )

    conflict_files[name] = conflicts

duplicate_audit = pd.DataFrame(duplicate_rows)
duplicate_audit.to_csv(OUT / "duplicate_key_audit.csv", index=False)

print("\nDUPLICATE AUDIT")
print(duplicate_audit.to_string(index=False))


# ---------------------------------------------------------
# 6. NUMERIC SUMMARIES
# ---------------------------------------------------------
numeric_columns = {
    "profile": [
        "AGE", "MARKETING_OPT_IN", "PREFERRED_STORE_ID",
    ],
    "activity": [
        "TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE",
        "DISTINCT_CATEGORIES", "PROMO_TXN_PCT", "APP_SESSIONS",
        "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS",
        "COMPLAINT_FLAG",
    ],
    "churn": [
        "MONTHS_OBSERVED", "INSUFFICIENT_HISTORY_FLAG", "CHURNED",
    ],
}

summary_rows = []

for name, (df, _) in tables.items():
    for column in numeric_columns[name]:
        values = pd.to_numeric(df[column], errors="coerce")
        summary_rows.append({
            "TABLE": name,
            "COLUMN": column,
            "COUNT": int(values.count()),
            "MISSING_AFTER_CONVERSION": int(values.isna().sum()),
            "MEAN": values.mean(),
            "STD": values.std(),
            "MIN": values.min(),
            "MEDIAN": values.median(),
            "MAX": values.max(),
        })

numeric_summary = pd.DataFrame(summary_rows)
numeric_summary.to_csv(OUT / "numeric_summary.csv", index=False)


# ---------------------------------------------------------
# 7. DATA-QUALITY CHECKS
# ---------------------------------------------------------
def add_check(name, value, passed, note=""):
    quality_rows.append({
        "CHECK": name,
        "VALUE": value,
        "STATUS": "PASS" if passed else "REVIEW",
        "NOTE": note,
    })


# Missing IDs
for name, (df, _) in tables.items():
    add_check(
        f"{name}: missing customer IDs",
        int(df["CUSTOMER_ID"].isna().sum()),
        df["CUSTOMER_ID"].notna().all(),
    )


# Date checks
date_columns = {
    "profile": (profile, ["SIGNUP_DATE"]),
    "activity": (activity, ["MONTH"]),
    "churn": (churn, ["OBSERVATION_END_DATE", "LAST_PURCHASE_DATE"]),
}

for name, (df, columns) in date_columns.items():
    for column in columns:
        original = df[column]
        parsed = pd.to_datetime(original, errors="coerce")
        invalid = int((original.notna() & parsed.isna()).sum())

        add_check(
            f"{name}: {column} invalid dates",
            invalid,
            invalid == 0,
            f"Missing source dates: {int(original.isna().sum())}",
        )


# Activity range checks
activity_numeric = activity.copy()
for column in numeric_columns["activity"]:
    activity_numeric[column] = pd.to_numeric(
        activity_numeric[column], errors="coerce"
    )

range_checks = [
    ("Negative transactions", (activity_numeric["TRANSACTIONS"] < 0).sum()),
    ("Negative average basket", (activity_numeric["AVG_BASKET_VALUE"] < 0).sum()),
    ("Promo percentage outside [0,1]", (
        (activity_numeric["PROMO_TXN_PCT"] < 0)
        | (activity_numeric["PROMO_TXN_PCT"] > 1)
    ).sum()),
    ("Complaint flag outside {0,1}", (
        ~activity_numeric["COMPLAINT_FLAG"].isin([0, 1])
    ).sum()),
    ("Negative total spend", (activity_numeric["TOTAL_SPEND"] < 0).sum()),
]

for label, count in range_checks:
    count = int(count)
    add_check(
        f"Activity: {label}",
        count,
        count == 0,
        "Negative spend may represent refunds/adjustments; investigate, do not auto-delete."
        if label == "Negative total spend" and count > 0 else "",
    )


# Churn domain checks
churn_numeric = churn.copy()
for column in ["CHURNED", "INSUFFICIENT_HISTORY_FLAG", "MONTHS_OBSERVED"]:
    churn_numeric[column] = pd.to_numeric(
        churn_numeric[column], errors="coerce"
    )

invalid_labels = int(
    (~churn_numeric["CHURNED"].isin([0, 1])).sum()
)
invalid_history_flags = int(
    (~churn_numeric["INSUFFICIENT_HISTORY_FLAG"].isin([0, 1])).sum()
)
invalid_months = int(
    (
        (churn_numeric["MONTHS_OBSERVED"] < 1)
        | (churn_numeric["MONTHS_OBSERVED"] > 18)
        | churn_numeric["MONTHS_OBSERVED"].isna()
    ).sum()
)

add_check("Churn: invalid CHURNED values", invalid_labels, invalid_labels == 0)
add_check(
    "Churn: invalid history flags",
    invalid_history_flags,
    invalid_history_flags == 0,
)
add_check(
    "Churn: MONTHS_OBSERVED outside 1–18",
    invalid_months,
    invalid_months == 0,
)


# ---------------------------------------------------------
# 8. UNIQUE-CUSTOMER AND LABEL CONSISTENCY AUDIT
# ---------------------------------------------------------
# Exact duplicates are removed only for this audit calculation.
# The original source DataFrames remain untouched.
churn_unique_rows = churn.drop_duplicates()

labels_per_customer = churn_unique_rows.groupby(
    "CUSTOMER_ID", dropna=False
)["CHURNED"].nunique()

conflicting_label_ids = labels_per_customer[
    labels_per_customer > 1
].index

consistent_ids = labels_per_customer[
    labels_per_customer == 1
].index

unique_churn = churn_unique_rows[
    churn_unique_rows["CUSTOMER_ID"].isin(consistent_ids)
].drop_duplicates(subset=["CUSTOMER_ID"])

unique_summary = pd.DataFrame([
    {"CHECK": "Profile source rows", "VALUE": len(profile)},
    {"CHECK": "Profile unique customer IDs", "VALUE": profile["CUSTOMER_ID"].nunique()},
    {"CHECK": "Profile exact duplicate rows", "VALUE": int(profile.duplicated().sum())},
    {"CHECK": "Activity source rows", "VALUE": len(activity)},
    {"CHECK": "Activity unique customer IDs", "VALUE": activity["CUSTOMER_ID"].nunique()},
    {"CHECK": "Churn source rows", "VALUE": len(churn)},
    {"CHECK": "Churn exact duplicate rows", "VALUE": int(churn.duplicated().sum())},
    {"CHECK": "Churn unique customer IDs", "VALUE": churn["CUSTOMER_ID"].nunique()},
    {"CHECK": "Customers with conflicting labels", "VALUE": len(conflicting_label_ids)},
    {"CHECK": "Customers with consistent labels", "VALUE": len(unique_churn)},
])

unique_summary.to_csv(OUT / "unique_customer_verification.csv", index=False)

churn_distribution = (
    unique_churn["CHURNED"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename_axis("CHURNED")
    .reset_index(name="CUSTOMER_COUNT")
)
churn_distribution["PROPORTION"] = (
    churn_distribution["CUSTOMER_COUNT"] / max(len(unique_churn), 1)
)
churn_distribution.to_csv(OUT / "churn_distribution.csv", index=False)

# Save a plot based on one consistent label per customer.
plt.figure(figsize=(7, 5))
bars = plt.bar(
    churn_distribution["CHURNED"].astype(str),
    churn_distribution["CUSTOMER_COUNT"],
)
plt.title("Unique-Customer Churn Distribution")
plt.xlabel("Churn label (0 = retained, 1 = churned)")
plt.ylabel("Number of customers")
plt.bar_label(bars)
plt.tight_layout()
plt.savefig(OUT / "churn_distribution.png", dpi=150)
plt.close()


# ---------------------------------------------------------
# 9. SAVE DATA-QUALITY REPORT AND PRINT SUMMARY
# ---------------------------------------------------------
quality_df = pd.DataFrame(quality_rows)
quality_df.to_csv(OUT / "data_quality_checks.csv", index=False)

print("\nUNIQUE-CUSTOMER VERIFICATION")
print(unique_summary.to_string(index=False))

print("\nCHURN DISTRIBUTION — EXACT DUPLICATES REMOVED FOR AUDIT")
print(churn_distribution.to_string(index=False))

print("\nDATA-QUALITY CHECKS")
print(quality_df.to_string(index=False))

print("\nOUTPUT FILES")
for path in sorted(OUT.glob("*")):
    if path.is_file():
        print(f" - {path.name}")

print("\nEDA VERIFICATION COMPLETED")
print("No source workbook data was modified.")
print("Review flagged issues before approving data cleaning.")
