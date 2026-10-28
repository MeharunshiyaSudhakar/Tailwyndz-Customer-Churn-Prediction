
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
REPORT_DIR = ROOT / "reports/phase6"
REPORT_FILE = REPORT_DIR / "phase6_engagement_churn_findings.txt"
SUMMARY_FILE = REPORT_DIR / "phase6_engagement_churn_summary.csv"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG",
        "TOTAL_TRANSACTIONS_PRE_APR2024", "TOTAL_SPEND_PRE_APR2024",
        "TOTAL_APP_SESSIONS_PRE_APR2024",
        "TOTAL_EMAILS_OPENED_PRE_APR2024",
        "TOTAL_COUPONS_REDEEMED_PRE_APR2024",
        "TOTAL_SUPPORT_TICKETS_PRE_APR2024",
        "MEAN_PROMO_TXN_PCT", "OBSERVED_FEATURE_MONTHS",
        "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY",
        "SPEND_TREND_LATE_MINUS_EARLY",
        "TRANSACTION_TREND_LATE_MINUS_EARLY",
        "APP_TREND_LATE_MINUS_EARLY",
        "SUPPORT_TREND_LATE_MINUS_EARLY",
        "ANY_COMPLAINT", "MEMBERSHIP_TIER", "CITY",
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate CUSTOMER_ID values found.")

    # Analyze only customers whose labels are considered sufficiently observed.
    eligible = df.loc[
        df["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & df["CHURNED"].isin([0, 1])
    ].copy()

    if eligible.empty:
        raise ValueError("No eligible labelled customers found.")

    numeric_features = [
        "TOTAL_TRANSACTIONS_PRE_APR2024",
        "TOTAL_SPEND_PRE_APR2024",
        "TOTAL_APP_SESSIONS_PRE_APR2024",
        "TOTAL_EMAILS_OPENED_PRE_APR2024",
        "TOTAL_COUPONS_REDEEMED_PRE_APR2024",
        "TOTAL_SUPPORT_TICKETS_PRE_APR2024",
        "MEAN_PROMO_TXN_PCT",
        "OBSERVED_FEATURE_MONTHS",
        "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY",
        "SPEND_TREND_LATE_MINUS_EARLY",
        "TRANSACTION_TREND_LATE_MINUS_EARLY",
        "APP_TREND_LATE_MINUS_EARLY",
        "SUPPORT_TREND_LATE_MINUS_EARLY",
        "ANY_COMPLAINT",
    ]

    for col in numeric_features:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")

    summary_rows = []
    for target, group in eligible.groupby("CHURNED"):
        for col in numeric_features:
            values = group[col].dropna()
            summary_rows.append({
                "CHURNED": int(target),
                "GROUP": "Churned" if target == 1 else "Not churned",
                "FEATURE": col,
                "CUSTOMERS_IN_GROUP": len(group),
                "NON_MISSING_VALUES": len(values),
                "MEAN": values.mean() if len(values) else np.nan,
                "MEDIAN": values.median() if len(values) else np.nan,
                "STD": values.std() if len(values) > 1 else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_FILE, index=False)

    churn_rate = eligible["CHURNED"].mean() * 100

    lines = [
        "PHASE 6 — ENGAGEMENT AND SUPPORT VS. CHURN",
        "=" * 55,
        f"Input: {INPUT_FILE}",
        "Feature cutoff: 2024-03-31",
        f"Eligible customers: {len(eligible)}",
        f"Non-churned customers: {(eligible['CHURNED'] == 0).sum()}",
        f"Churned customers: {(eligible['CHURNED'] == 1).sum()}",
        f"Eligible-cohort churn rate: {churn_rate:.2f}%",
        "",
        "NUMERIC FEATURE COMPARISON",
        "Means and medians are descriptive, not causal evidence.",
        "",
    ]

    for feature in numeric_features:
        lines.append(f"{feature}:")
        feature_rows = summary.loc[summary["FEATURE"] == feature]
        for _, row in feature_rows.iterrows():
            lines.append(
                f"  {row['GROUP']}: n={int(row['CUSTOMERS_IN_GROUP'])}, "
                f"non-missing={int(row['NON_MISSING_VALUES'])}, "
                f"mean={row['MEAN']:.3f}, median={row['MEDIAN']:.3f}"
            )
        lines.append("")

    lines.extend([
        "SEGMENT CHURN RATES",
        "Descriptive comparisons only; segment sizes are shown.",
        "",
    ])

    for segment in ["MEMBERSHIP_TIER", "CITY"]:
        lines.append(f"{segment}:")
        segment_summary = (
            eligible.groupby(segment, dropna=False)
            .agg(
                CUSTOMERS=("CUSTOMER_ID", "nunique"),
                CHURNED=("CHURNED", "sum"),
                CHURN_RATE=("CHURNED", "mean"),
            )
            .reset_index()
            .sort_values("CHURN_RATE", ascending=False)
        )

        for _, row in segment_summary.iterrows():
            label = "(Missing)" if pd.isna(row[segment]) else str(row[segment])
            lines.append(
                f"  {label}: customers={int(row['CUSTOMERS'])}, "
                f"churned={int(row['CHURNED'])}, "
                f"churn_rate={row['CHURN_RATE'] * 100:.2f}%"
            )
        lines.append("")

    lines.extend([
        "INTERPRETATION AND LIMITATIONS",
        "- This is descriptive analysis, not a causal test.",
        "- All predictor features use pre-April 2024 activity.",
        "- Missing monthly records were not treated as zero purchases.",
        "- Feature totals reflect observed records and may be affected by unequal history.",
        "- Support ticket counts and complaint indicators do not establish why a customer churned.",
        "- Segment differences may reflect other customer characteristics.",
        "",
        "OUTPUTS",
        str(SUMMARY_FILE),
        str(REPORT_FILE),
    ])

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nPhase 6 analysis completed.")


if __name__ == "__main__":
    main()
