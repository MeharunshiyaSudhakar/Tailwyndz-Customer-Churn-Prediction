
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PROFILE_FILE = ROOT / "data/processed/customer_profile_clean.csv"
ACTIVITY_FILE = ROOT / "data/processed/monthly_activity_clean.csv"
LABEL_FILE = ROOT / "data/processed/churn_label_clean.csv"

OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/phase5"

FEATURES_OUT = OUTPUT_DIR / "customer_features_pre_apr2024.csv"
REPORT_OUT = REPORT_DIR / "phase5_feature_validation_report.txt"

FEATURE_START = pd.Timestamp("2023-01-01")
FEATURE_END = pd.Timestamp("2024-03-31")


def require_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def safe_divide(numerator, denominator):
    return numerator.div(denominator.where(denominator.ne(0)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profile = pd.read_csv(PROFILE_FILE)
    activity = pd.read_csv(ACTIVITY_FILE)
    labels = pd.read_csv(LABEL_FILE)

    require_columns(
        profile,
        [
            "CUSTOMER_ID", "SIGNUP_DATE", "AGE", "GENDER", "CITY",
            "MEMBERSHIP_TIER", "MARKETING_OPT_IN", "PREFERRED_STORE_ID",
            "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER",
        ],
        "Customer profile",
    )

    require_columns(
        activity,
        [
            "CUSTOMER_ID", "MONTH", "TRANSACTIONS", "TOTAL_SPEND",
            "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT",
            "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED",
            "SUPPORT_TICKETS", "COMPLAINT_FLAG",
        ],
        "Monthly activity",
    )

    require_columns(
        labels,
        ["CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"],
        "Churn labels",
    )

    # Validate one row per customer in profile and labels.
    for df, name in [(profile, "profile"), (labels, "labels")]:
        if df["CUSTOMER_ID"].isna().any():
            raise ValueError(f"{name} contains missing CUSTOMER_ID values.")
        if df["CUSTOMER_ID"].duplicated().any():
            raise ValueError(f"{name} has duplicate CUSTOMER_ID values.")

    profile["SIGNUP_DATE"] = pd.to_datetime(
        profile["SIGNUP_DATE"], errors="coerce"
    )
    activity["MONTH"] = pd.to_datetime(
        activity["MONTH"], errors="coerce"
    )

    if activity["MONTH"].isna().any():
        raise ValueError("Monthly activity contains invalid MONTH values.")

    
    # Apply the documented conservative policy: exclude every row in a
    # duplicated customer-month group. Do not arbitrarily select one value.
    duplicate_mask = activity.duplicated(
        ["CUSTOMER_ID", "MONTH"], keep=False
    )

    duplicate_groups = (
        activity.loc[duplicate_mask, ["CUSTOMER_ID", "MONTH"]]
        .drop_duplicates()
        .sort_values(["CUSTOMER_ID", "MONTH"])
    )

    excluded_duplicate_rows = int(duplicate_mask.sum())

    if excluded_duplicate_rows:
        print("Duplicate customer-month groups excluded:")
        print(duplicate_groups.to_string(index=False))
        print(f"Total rows excluded: {excluded_duplicate_rows}")

        activity = activity.loc[~duplicate_mask].copy()

    if activity.duplicated(["CUSTOMER_ID", "MONTH"]).any():
        raise ValueError("Duplicate customer-month rows remain after exclusion.")


    # Strict cutoff: April-June 2024 are never used as predictors.
    feature_activity = activity.loc[
        activity["MONTH"].between(FEATURE_START, FEATURE_END)
    ].copy()

    if feature_activity.empty:
        raise ValueError("No activity found inside the feature window.")

    # Explicitly reject any post-cutoff rows in the feature input.
    assert feature_activity["MONTH"].max() <= FEATURE_END

    numeric_columns = [
        "TRANSACTIONS", "TOTAL_SPEND", "APP_SESSIONS",
        "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS",
        "DISTINCT_CATEGORIES", "COMPLAINT_FLAG",
        "AVG_BASKET_VALUE", "PROMO_TXN_PCT",
    ]

    for col in numeric_columns:
        feature_activity[col] = pd.to_numeric(
            feature_activity[col], errors="coerce"
        )

    # Preserve absent monthly records as absent observations; do not fabricate
    # zero-activity months. Sum observed values and report coverage separately.
    grouped = feature_activity.groupby("CUSTOMER_ID", sort=False)

    features = grouped.agg(
        OBSERVED_FEATURE_MONTHS=("MONTH", "nunique"),
        FIRST_FEATURE_MONTH=("MONTH", "min"),
        LAST_FEATURE_MONTH=("MONTH", "max"),
        TOTAL_TRANSACTIONS_PRE_APR2024=("TRANSACTIONS", "sum"),
        TOTAL_SPEND_PRE_APR2024=("TOTAL_SPEND", "sum"),
        MEAN_MONTHLY_SPEND_OBSERVED=("TOTAL_SPEND", "mean"),
        MEDIAN_MONTHLY_SPEND_OBSERVED=("TOTAL_SPEND", "median"),
        TOTAL_APP_SESSIONS_PRE_APR2024=("APP_SESSIONS", "sum"),
        TOTAL_EMAILS_OPENED_PRE_APR2024=("EMAILS_OPENED", "sum"),
        TOTAL_COUPONS_REDEEMED_PRE_APR2024=("COUPONS_REDEEMED", "sum"),
        TOTAL_SUPPORT_TICKETS_PRE_APR2024=("SUPPORT_TICKETS", "sum"),
        MEAN_DISTINCT_CATEGORIES=("DISTINCT_CATEGORIES", "mean"),
        MEAN_PROMO_TXN_PCT=("PROMO_TXN_PCT", "mean"),
        TOTAL_COMPLAINT_FLAGS=("COMPLAINT_FLAG", "sum"),
        ANY_COMPLAINT=("COMPLAINT_FLAG", "max"),
        MEAN_BASKET_VALUE_OBSERVED=("AVG_BASKET_VALUE", "mean"),
    ).reset_index()

    # Activity recency is based only on the last observed pre-April activity.
    features["MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"] = (
        (FEATURE_END.year - features["LAST_FEATURE_MONTH"].dt.year) * 12
        + FEATURE_END.month - features["LAST_FEATURE_MONTH"].dt.month
    )

    # First and last thirds of the available feature window.
    early_end = pd.Timestamp("2023-05-31")
    late_start = pd.Timestamp("2023-11-01")

    early = feature_activity.loc[
        feature_activity["MONTH"] <= early_end
    ].groupby("CUSTOMER_ID").agg(
        EARLY_PERIOD_SPEND=("TOTAL_SPEND", "sum"),
        EARLY_PERIOD_TRANSACTIONS=("TRANSACTIONS", "sum"),
        EARLY_PERIOD_APP_SESSIONS=("APP_SESSIONS", "sum"),
        EARLY_PERIOD_SUPPORT_TICKETS=("SUPPORT_TICKETS", "sum"),
    )

    late = feature_activity.loc[
        feature_activity["MONTH"] >= late_start
    ].groupby("CUSTOMER_ID").agg(
        LATE_PERIOD_SPEND=("TOTAL_SPEND", "sum"),
        LATE_PERIOD_TRANSACTIONS=("TRANSACTIONS", "sum"),
        LATE_PERIOD_APP_SESSIONS=("APP_SESSIONS", "sum"),
        LATE_PERIOD_SUPPORT_TICKETS=("SUPPORT_TICKETS", "sum"),
    )

    features = features.merge(early, on="CUSTOMER_ID", how="left")
    features = features.merge(late, on="CUSTOMER_ID", how="left")

    # Differences are calculated only where both period values are observed.
    features["SPEND_TREND_LATE_MINUS_EARLY"] = (
        features["LATE_PERIOD_SPEND"] - features["EARLY_PERIOD_SPEND"]
    )
    features["TRANSACTION_TREND_LATE_MINUS_EARLY"] = (
        features["LATE_PERIOD_TRANSACTIONS"]
        - features["EARLY_PERIOD_TRANSACTIONS"]
    )
    features["APP_TREND_LATE_MINUS_EARLY"] = (
        features["LATE_PERIOD_APP_SESSIONS"]
        - features["EARLY_PERIOD_APP_SESSIONS"]
    )
    features["SUPPORT_TREND_LATE_MINUS_EARLY"] = (
        features["LATE_PERIOD_SUPPORT_TICKETS"]
        - features["EARLY_PERIOD_SUPPORT_TICKETS"]
    )

    # Profile-derived features; signup dates after the cutoff are flagged.
    profile["TENURE_DAYS_AT_CUTOFF"] = (
        FEATURE_END - profile["SIGNUP_DATE"]
    ).dt.days
    profile["SIGNED_UP_AFTER_FEATURE_CUTOFF"] = (
        profile["SIGNUP_DATE"] > FEATURE_END
    ).astype("int8")

    # Churn label is attached for analysis/target use, not included in the
    # predictor feature list.
    result = profile.merge(features, on="CUSTOMER_ID", how="left", validate="one_to_one")
    result = result.merge(
        labels[["CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"]],
        on="CUSTOMER_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_LABEL"),
    )

    if len(result) != len(profile):
        raise ValueError("Final feature dataset row count differs from profile.")

    if result["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Final feature dataset contains duplicate customers.")

    # Do not treat no observed pre-cutoff activity as zero purchases.
    no_activity = result["OBSERVED_FEATURE_MONTHS"].isna()
    result["NO_OBSERVED_FEATURE_ACTIVITY"] = no_activity.astype("int8")

    result.to_csv(FEATURES_OUT, index=False)

    eligible = result.loc[
        result["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & result["CHURNED"].isin([0, 1])
    ]

    report = [
        "PHASE 5 — LEAKAGE-SAFE FEATURE ENGINEERING VALIDATION",
        "=" * 58,
        f"Feature window start: {FEATURE_START.date()}",
        f"Feature window end: {FEATURE_END.date()}",
        "Post-cutoff activity used: NO",
        "Post-cutoff window excluded: 2024-04-01 through 2024-06-30",
        "",
        f"Profile rows: {len(profile)}",
        f"Monthly source rows: {len(activity)}",
        f"Rows in feature window: {len(feature_activity)}",
        f"Customers with observed feature-window activity: {features['CUSTOMER_ID'].nunique()}",
        f"Customers with no observed feature-window activity: {int(no_activity.sum())}",
        f"Final feature rows: {len(result)}",
        f"Final unique customer IDs: {result['CUSTOMER_ID'].nunique()}",
        f"Eligible labeled customers: {len(eligible)}",
        f"Missing churn labels: {int(result['CHURNED'].isna().sum())}",
        f"Missing history flags: {int(result['INSUFFICIENT_HISTORY_FLAG'].isna().sum())}",
        f"Maximum feature activity month: {feature_activity['MONTH'].max().date()}",
        "",
        "IMPORTANT INTERPRETATION",
        "- Missing monthly records were not filled with zero purchases.",
        "- TOTAL_SPEND is an observed-record aggregate; data-quality issues from earlier phases remain documented.",
        "- Mean PROMO_TXN_PCT is an unweighted mean across observed months.",
        "- Trend values compare observed early and late period totals; they are not normalized for missing months.",
        "- CHURNED and INSUFFICIENT_HISTORY_FLAG are labels/eligibility fields, not predictors.",
        "",
        "OUTPUT",
        str(FEATURES_OUT),
    ]

    REPORT_OUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("\nPhase 5 feature dataset and validation report created.")


if __name__ == "__main__":
    main()
