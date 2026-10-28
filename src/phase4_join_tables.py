
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 4: JOIN TABLES AND VALIDATE THE CUSTOMER-LEVEL DATASET
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "phase4"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "customer_level_joined.csv"
VALIDATION_FILE = REPORT_DIR / "phase4_validation_report.txt"

# Feature observation window: January 2023 to March 2024
FEATURE_START = pd.Timestamp("2023-01-01")
FEATURE_END = pd.Timestamp("2024-03-31")

print("=" * 65)
print("PHASE 4: CUSTOMER TABLE JOIN")
print("=" * 65)

# ------------------------------------------------------------
# 1. Load cleaned datasets
# ------------------------------------------------------------
profile = pd.read_csv(DATA_DIR / "customer_profile_clean.csv")
activity = pd.read_csv(DATA_DIR / "monthly_activity_clean.csv")
churn = pd.read_csv(DATA_DIR / "churn_label_clean.csv")

for name, df in [
    ("Customer Profile", profile),
    ("Monthly Activity", activity),
    ("Churn Label", churn),
]:
    df.columns = df.columns.str.strip().str.upper()

    if "CUSTOMER_ID" not in df.columns:
        raise ValueError(f"{name} has no CUSTOMER_ID column.")

    df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype("string").str.strip()

    if df["CUSTOMER_ID"].isna().any() or df["CUSTOMER_ID"].eq("").any():
        raise ValueError(f"{name} contains missing or empty CUSTOMER_ID values.")

print("\n[1] INPUT ROW COUNTS")
print(f"Customer profile : {len(profile):,}")
print(f"Monthly activity : {len(activity):,}")
print(f"Churn labels     : {len(churn):,}")

# ------------------------------------------------------------
# 2. Validate customer-level table uniqueness
# ------------------------------------------------------------
for name, df in [
    ("Customer Profile", profile),
    ("Churn Label", churn),
]:
    duplicate_count = int(df["CUSTOMER_ID"].duplicated().sum())

    if duplicate_count:
        raise ValueError(
            f"{name} has {duplicate_count} duplicate CUSTOMER_ID rows. "
            "Resolve these before joining."
        )

print("\n[2] CUSTOMER ID VALIDATION")
print("Profile IDs are unique.")
print("Churn-label IDs are unique.")

# ------------------------------------------------------------
# 3. Parse monthly activity dates
# ------------------------------------------------------------
activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")

if activity["MONTH"].isna().any():
    bad_rows = int(activity["MONTH"].isna().sum())
    raise ValueError(f"Monthly activity has {bad_rows} invalid MONTH values.")

# Normalise dates to month start, for consistent filtering
activity["MONTH"] = activity["MONTH"].dt.to_period("M").dt.to_timestamp()

# Check repeated customer-month combinations
duplicate_customer_months = activity.duplicated(
    subset=["CUSTOMER_ID", "MONTH"], keep=False
)
duplicate_customer_month_rows = int(duplicate_customer_months.sum())

print("\n[3] MONTHLY ACTIVITY VALIDATION")
print(f"Activity rows with repeated customer-month keys: "
      f"{duplicate_customer_month_rows:,}")

# Keep only the feature observation window.
# April-June 2024 is reserved for the churn label.
feature_activity = activity[
    activity["MONTH"].between(FEATURE_START, FEATURE_END)
].copy()

# Identify all duplicated customer-month groups
duplicate_mask = feature_activity.duplicated(
    subset=["CUSTOMER_ID", "MONTH"], keep=False
)

duplicate_groups = (
    feature_activity.loc[duplicate_mask, ["CUSTOMER_ID", "MONTH"]]
    .drop_duplicates()
)

print("\nConflicting/repeated customer-month groups:")
print(duplicate_groups.to_string(index=False))

# Exclude every row belonging to a repeated customer-month group.
# We cannot determine the correct row reliably from the source alone.
feature_activity = feature_activity.merge(
    duplicate_groups.assign(_DUPLICATE_MONTH=True),
    on=["CUSTOMER_ID", "MONTH"],
    how="left",
)

feature_activity = feature_activity[
    feature_activity["_DUPLICATE_MONTH"].isna()
].drop(columns="_DUPLICATE_MONTH")

print(
    "\nRows excluded from feature aggregation due to repeated "
    f"customer-month keys: {int(duplicate_mask.sum())}"
)

print(f"Feature-window rows (Jan 2023-Mar 2024): "
      f"{len(feature_activity):,}")
print(f"Feature-window dates: {feature_activity['MONTH'].min().date()} "
      f"to {feature_activity['MONTH'].max().date()}")

# ------------------------------------------------------------
# 4. Aggregate monthly activity to one row per customer
# ------------------------------------------------------------
# Additive measures: sum across months
sum_columns = [
    "TRANSACTIONS",
    "TOTAL_SPEND",
    "APP_SESSIONS",
    "EMAILS_OPENED",
    "COUPONS_REDEEMED",
    "SUPPORT_TICKETS",
]

# Other measures need a different aggregation strategy.
# AVG_BASKET_VALUE and PROMO_TXN_PCT use transaction-weighted averages.
# DISTINCT_CATEGORIES is averaged across observed months.
# COMPLAINT_FLAG is the maximum (any complaint during the window).
aggregation = {
    **{column: "sum" for column in sum_columns},
    "DISTINCT_CATEGORIES": "mean",
    "COMPLAINT_FLAG": "max",
}

for column in ["AVG_BASKET_VALUE", "PROMO_TXN_PCT"]:
    if column in feature_activity.columns:
        feature_activity[column] = pd.to_numeric(
            feature_activity[column], errors="coerce"
        )

# Create transaction-weighted monthly measure inputs.
# Negative/zero transaction rows remain in the source; they are not
# silently removed. Weighted averages use positive transaction counts.
transactions = pd.to_numeric(
    feature_activity["TRANSACTIONS"], errors="coerce"
)
for column in ["AVG_BASKET_VALUE", "PROMO_TXN_PCT"]:
    if column in feature_activity.columns:
        feature_activity[f"_{column}_WEIGHTED"] = (
            feature_activity[column] * transactions.where(transactions > 0, 0)
        )

# Aggregate basic measures
agg_dict = {
    column: operation
    for column, operation in aggregation.items()
    if column in feature_activity.columns
}

customer_activity = feature_activity.groupby("CUSTOMER_ID").agg(agg_dict)

# Add weighted averages, guarding against zero denominators
positive_txn_weights = transactions.where(transactions > 0, 0)
weight_totals = feature_activity.assign(
    _POSITIVE_TXN_WEIGHT=positive_txn_weights
).groupby("CUSTOMER_ID")["_POSITIVE_TXN_WEIGHT"].sum()

for column in ["AVG_BASKET_VALUE", "PROMO_TXN_PCT"]:
    weighted_column = f"_{column}_WEIGHTED"
    if weighted_column in feature_activity.columns:
        weighted_sum = (
            feature_activity.groupby("CUSTOMER_ID")[weighted_column].sum(
                min_count=1
            )
        )
        customer_activity[column] = (
            weighted_sum / weight_totals.replace(0, pd.NA)
        )

# Number of observed months is useful for later feature engineering
month_counts = feature_activity.groupby("CUSTOMER_ID")["MONTH"].nunique()
customer_activity["OBSERVED_ACTIVITY_MONTHS"] = month_counts

customer_activity = customer_activity.reset_index()

if customer_activity["CUSTOMER_ID"].duplicated().any():
    raise ValueError("Customer activity aggregation did not produce unique IDs.")

print("\n[4] AGGREGATION VALIDATION")
print(f"Customers with activity in feature window: "
      f"{customer_activity['CUSTOMER_ID'].nunique():,}")
print("Aggregated activity has one row per customer.")

# ------------------------------------------------------------
# 5. Join profile, aggregated activity, and churn labels
# ------------------------------------------------------------
# Left join preserves all profile customers.
joined = profile.merge(
    customer_activity,
    on="CUSTOMER_ID",
    how="left",
    validate="one_to_one",
    indicator="_ACTIVITY_MERGE",
)

activity_match_count = int((joined["_ACTIVITY_MERGE"] == "both").sum())
no_feature_activity_count = int(
    (joined["_ACTIVITY_MERGE"] == "left_only").sum()
)
joined = joined.drop(columns="_ACTIVITY_MERGE")

joined = joined.merge(
    churn,
    on="CUSTOMER_ID",
    how="left",
    validate="one_to_one",
    indicator="_CHURN_MERGE",
)

churn_match_count = int((joined["_CHURN_MERGE"] == "both").sum())
missing_churn_count = int((joined["_CHURN_MERGE"] == "left_only").sum())
joined = joined.drop(columns="_CHURN_MERGE")

# Preserve missing activity as missing at this stage.
# No activity record does not automatically prove zero purchases.
if joined["CUSTOMER_ID"].duplicated().any():
    raise ValueError("Final joined dataset contains duplicate customer IDs.")

if len(joined) != len(profile):
    raise ValueError(
        "Final row count differs from the profile row count; inspect the joins."
    )

# ------------------------------------------------------------
# 6. Leakage and label checks
# ------------------------------------------------------------
joined["OBSERVATION_END_DATE"] = pd.to_datetime(
    joined["OBSERVATION_END_DATE"], errors="coerce"
)
joined["LAST_PURCHASE_DATE"] = pd.to_datetime(
    joined["LAST_PURCHASE_DATE"], errors="coerce"
)

label_counts = joined["CHURNED"].value_counts(dropna=False).to_dict()
eligible_count = int(
    (
        joined["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & joined["CHURNED"].isin([0, 1])
    ).sum()
)

# Ensure aggregated features were made only from dates through March 2024
assert feature_activity["MONTH"].max() <= pd.Timestamp("2024-03-01")

# ------------------------------------------------------------
# 7. Save final dataset and validation report
# ------------------------------------------------------------
joined.to_csv(OUTPUT_FILE, index=False)

report_lines = [
    "PHASE 4: JOIN TABLES AND VALIDATION REPORT",
    "=" * 50,
    f"Customer profile rows: {len(profile)}",
    f"Monthly activity source rows: {len(activity)}",
    f"Churn label rows: {len(churn)}",
    f"Feature-window activity rows: {len(feature_activity)}",
    f"Customers with aggregated feature-window activity: "
    f"{customer_activity['CUSTOMER_ID'].nunique()}",
    f"Final joined rows: {len(joined)}",
    f"Final unique customer IDs: {joined['CUSTOMER_ID'].nunique()}",
    f"Profile customers matched to feature activity: {activity_match_count}",
    f"Profile customers without feature-window activity: "
    f"{no_feature_activity_count}",
    f"Profile customers matched to churn labels: {churn_match_count}",
    f"Profile customers without churn labels: {missing_churn_count}",
    f"Repeated customer-month rows in source activity: "
    f"{duplicate_customer_month_rows}",
    f"Eligible customers (history sufficient and valid label): "
    f"{eligible_count}",
    f"CHURNED label counts: {label_counts}",
    "",
    "Leakage control: aggregated activity features use Jan 2023-Mar 2024 only.",
    "April-June 2024 activity was excluded from feature aggregation.",
    "Repeated customer-month rows in source activity: 4 rows across 2 groups.",
    "April 2023 conflict for customer 502065: both rows excluded from "
    "feature aggregation; source values were not altered.",
    "May 2024 conflict for customer 501780: outside the feature window; "
    "not used for predictive features.",
    "Missing activity is preserved as missing, not automatically treated as zero.",
    "Excluded duplicate-month groups must be documented as a data limitation.",
]

VALIDATION_FILE.write_text("\n".join(report_lines), encoding="utf-8")

print("\n[5] JOIN RESULTS")
print(f"Final dataset rows              : {len(joined):,}")
print(f"Unique customer IDs             : {joined['CUSTOMER_ID'].nunique():,}")
print(f"Matched to feature activity      : {activity_match_count:,}")
print(f"No feature-window activity       : {no_feature_activity_count:,}")
print(f"Matched to churn labels          : {churn_match_count:,}")
print(f"Missing churn labels             : {missing_churn_count:,}")
print(f"Eligible customers               : {eligible_count:,}")

print("\n[6] OUTPUTS")
print(f"Customer-level dataset: {OUTPUT_FILE}")
print(f"Validation report     : {VALIDATION_FILE}")
print("\nPhase 4 script completed.")
