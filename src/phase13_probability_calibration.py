
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "reports/phase9/phase9_heldout_churn_predictions.csv"
OUTPUT_DIR = ROOT / "reports/phase13"

ID = "CUSTOMER_ID"
ACTUAL = "ACTUAL_CHURNED"
PROB = "PREDICTED_CHURN_PROBABILITY"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {ID, ACTUAL, PROB}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty or df[ID].duplicated().any():
        raise ValueError("Prediction file is empty or contains duplicate IDs.")

    df[ACTUAL] = pd.to_numeric(df[ACTUAL], errors="raise")
    df[PROB] = pd.to_numeric(df[PROB], errors="raise")

    if df[[ACTUAL, PROB]].isna().any().any():
        raise ValueError("Missing labels or probabilities.")

    if not set(df[ACTUAL].unique()).issubset({0, 1}):
        raise ValueError("Actual labels must be binary.")

    if ((df[PROB] < 0) | (df[PROB] > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    y = df[ACTUAL].astype(int)
    p = df[PROB].astype(float)

    brier = brier_score_loss(y, p)
    clipped_p = np.clip(p, 1e-15, 1 - 1e-15)
    logloss = log_loss(y, clipped_p, labels=[0, 1])

    overall = pd.DataFrame([{
        "CUSTOMERS": len(df),
        "ACTUAL_CHURN_RATE": y.mean(),
        "MEAN_PREDICTED_PROBABILITY": p.mean(),
        "BRIER_SCORE": brier,
        "LOG_LOSS": logloss,
        "ABSOLUTE_MEAN_PROBABILITY_GAP": abs(p.mean() - y.mean()),
        "MIN_PROBABILITY": p.min(),
        "MAX_PROBABILITY": p.max(),
        "PROBABILITY_STD": p.std(),
    }])

    # Equal-count bins, ordered by predicted risk.
    ranked = df[[ID, ACTUAL, PROB]].sort_values(
        PROB, ascending=True
    ).reset_index(drop=True)

    ranked["CALIBRATION_BIN"] = (
        pd.qcut(
            ranked.index + 1,
            q=min(10, len(ranked)),
            labels=False,
            duplicates="drop",
        ) + 1
    )

    calibration = ranked.groupby(
        "CALIBRATION_BIN", as_index=False
    ).agg(
        CUSTOMERS=(ID, "nunique"),
        MEAN_PREDICTED_PROBABILITY=(PROB, "mean"),
        OBSERVED_CHURN_RATE=(ACTUAL, "mean"),
        MIN_PREDICTED_PROBABILITY=(PROB, "min"),
        MAX_PREDICTED_PROBABILITY=(PROB, "max"),
        ACTUAL_CHURNERS=(ACTUAL, "sum"),
    )

    calibration["OBSERVED_MINUS_PREDICTED"] = (
        calibration["OBSERVED_CHURN_RATE"]
        - calibration["MEAN_PREDICTED_PROBABILITY"]
    )

    overall_file = OUTPUT_DIR / "phase13_calibration_metrics.csv"
    bins_file = OUTPUT_DIR / "phase13_calibration_bins.csv"
    report_file = OUTPUT_DIR / "phase13_calibration_report.txt"

    overall.to_csv(overall_file, index=False)
    calibration.to_csv(bins_file, index=False)

    lines = [
        "PHASE 13 — HELD-OUT PROBABILITY CALIBRATION AUDIT",
        "=" * 65,
        f"Customers evaluated: {len(df)}",
        f"Observed churn rate: {y.mean():.4%}",
        f"Mean predicted probability: {p.mean():.4%}",
        f"Absolute gap between mean probability and observed rate: "
        f"{abs(p.mean() - y.mean()):.4%}",
        f"Brier score (lower is better): {brier:.6f}",
        f"Log loss (lower is better): {logloss:.6f}",
        f"Minimum predicted probability: {p.min():.8f}",
        f"Maximum predicted probability: {p.max():.8f}",
        "",
        "EQUAL-COUNT CALIBRATION BINS, LOW TO HIGH RISK",
        "-" * 65,
    ]

    for _, row in calibration.iterrows():
        lines.append(
            f"Bin {int(row['CALIBRATION_BIN'])}: "
            f"n={int(row['CUSTOMERS'])}, "
            f"mean predicted={row['MEAN_PREDICTED_PROBABILITY']:.4%}, "
            f"observed churn={row['OBSERVED_CHURN_RATE']:.4%}, "
            f"observed-minus-predicted="
            f"{row['OBSERVED_MINUS_PREDICTED']:.4%}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 65,
        "1. Brier score and log loss assess probabilistic predictions; lower is better.",
        "2. Calibration bins compare mean predicted risk with observed churn.",
        "3. Bin estimates can be noisy because each bin contains a limited number of customers.",
        "4. Good aggregate calibration does not prove calibration within every segment.",
        "5. This is one random held-out sample, not a future-time validation.",
        "6. Calibration cannot rule out target leakage.",
        "7. Do not claim an intervention will prevent churn without experimental evidence.",
        "",
        "OUTPUT FILES",
        str(overall_file),
        str(bins_file),
        str(report_file),
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nCalibration audit completed.")


if __name__ == "__main__":
    main()
