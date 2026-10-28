
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
SPLIT_FILE = ROOT / "reports/phase7/phase7_customer_split.csv"
OUTPUT_DIR = ROOT / "reports/phase11"

TARGET = "CHURNED"

SUSPICIOUS_FEATURES = [
    "MONTHS_SINCE_LAST_OBSERVED_ACTIVITY",
    "OBSERVED_FEATURE_MONTHS",
    "TOTAL_TRANSACTIONS_PRE_APR2024",
    "TOTAL_SPEND_PRE_APR2024",
    "LATE_PERIOD_TRANSACTIONS",
    "LATE_PERIOD_SPEND",
    "LATE_PERIOD_APP_SESSIONS",
    "LATE_PERIOD_SUPPORT_TICKETS",
    "TRANSACTION_TREND_LATE_MINUS_EARLY",
    "SPEND_TREND_LATE_MINUS_EARLY",
    "APP_TREND_LATE_MINUS_EARLY",
    "SUPPORT_TREND_LATE_MINUS_EARLY",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURE_FILE)
    split = pd.read_csv(SPLIT_FILE)

    if features["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in features.")
    if split["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs in split.")

    data = features.merge(
        split[["CUSTOMER_ID", "DATA_SPLIT"]],
        on="CUSTOMER_ID",
        how="inner",
        validate="one_to_one",
    )

    eligible = data[
        (data["INSUFFICIENT_HISTORY_FLAG"] == 0)
        & data[TARGET].isin([0, 1])
    ].copy()

    train = eligible[eligible["DATA_SPLIT"] == "train"].copy()
    rows = []

    for feature in SUSPICIOUS_FEATURES:
        if feature not in train.columns:
            rows.append({
                "FEATURE": feature,
                "STATUS": "COLUMN_NOT_FOUND",
            })
            continue

        x = pd.to_numeric(train[feature], errors="coerce")
        y = train[TARGET].astype(int)
        valid = x.notna() & np.isfinite(x)

        if valid.sum() < 20 or x[valid].nunique() < 2:
            rows.append({
                "FEATURE": feature,
                "STATUS": "INSUFFICIENT_VALID_VALUES",
                "NONMISSING_ROWS": int(valid.sum()),
            })
            continue

        auc = roc_auc_score(y[valid], x[valid])
        rows.append({
            "FEATURE": feature,
            "STATUS": "SCREENED",
            "NONMISSING_ROWS": int(valid.sum()),
            "MISSING_ROWS": int((~valid).sum()),
            "CHURN_RATE_WHEN_MISSING": (
                y[~valid].mean() if (~valid).any() else np.nan
            ),
            "CHURN_RATE_WHEN_PRESENT": (
                y[valid].mean() if valid.any() else np.nan
            ),
            "UNIVARIATE_ROC_AUC": auc,
            "AUC_DISTANCE_FROM_RANDOM": abs(auc - 0.5),
            "FLAG_FOR_REVIEW": bool(auc >= 0.90 or auc <= 0.10),
        })

    screen = pd.DataFrame(rows)

    # Inspect whether feature dates respect the stated cutoff.
    date_checks = []
    cutoff = pd.Timestamp("2024-03-31")

    for column in ["FIRST_FEATURE_MONTH", "LAST_FEATURE_MONTH"]:
        if column not in features.columns:
            continue

        parsed = pd.to_datetime(features[column], errors="coerce")
        date_checks.append({
            "CHECK": f"{column}_AFTER_CUTOFF",
            "COUNT": int((parsed > cutoff).sum()),
            "NONMISSING_DATES": int(parsed.notna().sum()),
        })

    date_checks.append({
        "CHECK": "EXPECTED_FEATURE_CUTOFF",
        "COUNT": 0,
        "NONMISSING_DATES": len(features),
        "DETAIL": "Expected maximum feature month is March 2024.",
    })

    date_report = pd.DataFrame(date_checks)

    screen_file = OUTPUT_DIR / "phase11_leakage_feature_screen.csv"
    date_file = OUTPUT_DIR / "phase11_leakage_date_checks.csv"
    report_file = OUTPUT_DIR / "phase11_leakage_audit_report.txt"

    screen.to_csv(screen_file, index=False)
    date_report.to_csv(date_file, index=False)

    lines = [
        "PHASE 11 — FEATURE LEAKAGE SCREEN",
        "=" * 65,
        f"Eligible training customers screened: {len(train)}",
        "Only training-split labels were used for univariate screening.",
        "",
        "FEATURE SCREEN",
        "-" * 65,
    ]

    for _, row in screen.iterrows():
        if row.get("STATUS") != "SCREENED":
            lines.append(
                f"{row['FEATURE']}: {row.get('STATUS', 'UNKNOWN')}"
            )
            continue

        lines.append(
            f"{row['FEATURE']}: AUC={row['UNIVARIATE_ROC_AUC']:.4f}, "
            f"nonmissing={int(row['NONMISSING_ROWS'])}, "
            f"missing={int(row['MISSING_ROWS'])}, "
            f"review_flag={row['FLAG_FOR_REVIEW']}"
        )

    lines.extend([
        "",
        "DATE CHECKS",
        "-" * 65,
    ])

    for _, row in date_report.iterrows():
        lines.append(
            f"{row['CHECK']}: count={int(row['COUNT'])}"
        )

    lines.extend([
        "",
        "INTERPRETATION AND LIMITATIONS",
        "-" * 65,
        "1. This is a screening report, not proof that leakage exists or is absent.",
        "2. High univariate AUC can reflect real signal, label construction, or leakage.",
        "3. Confirm the precise churn-label definition and observation window.",
        "4. Check that every predictor was available at the intended prediction date.",
        "5. Missingness itself may encode history length or data-collection processes.",
        "6. Date checks do not establish when the underlying measurements became available.",
        "",
        "OUTPUT FILES",
        str(screen_file),
        str(date_file),
        str(report_file),
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nLeakage screening completed.")


if __name__ == "__main__":
    main()
