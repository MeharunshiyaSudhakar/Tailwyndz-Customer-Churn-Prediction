
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    ROOT / "reports/phase9/phase9_heldout_churn_predictions.csv"
)
OUTPUT_DIR = ROOT / "reports/phase13"

ID_COL = "CUSTOMER_ID"
ACTUAL_COL = "ACTUAL_CHURNED"
PROB_COL = "PREDICTED_CHURN_PROBABILITY"

TARGETING_SHARES = [0.10, 0.20, 0.30]
INTERVENTION_EFFECTIVENESS = [0.05, 0.10, 0.20]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Phase 9 predictions not found: {PREDICTION_FILE}"
        )

    df = pd.read_csv(PREDICTION_FILE)

    required = {ID_COL, ACTUAL_COL, PROB_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Prediction file is empty.")

    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    df[ACTUAL_COL] = pd.to_numeric(df[ACTUAL_COL], errors="raise")
    df[PROB_COL] = pd.to_numeric(df[PROB_COL], errors="raise")

    if df[[ACTUAL_COL, PROB_COL]].isna().any().any():
        raise ValueError("Missing actual labels or probabilities.")

    if not set(df[ACTUAL_COL].unique()).issubset({0, 1}):
        raise ValueError("Actual churn labels must be 0 or 1.")

    if (
        (df[PROB_COL] < 0) | (df[PROB_COL] > 1)
    ).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    # Rank by model risk, highest probability first.
    ranked = df.sort_values(
        [PROB_COL, ID_COL],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranked.insert(0, "RISK_RANK", np.arange(1, len(ranked) + 1))
    ranked["RISK_PERCENTILE"] = (
        ranked["RISK_RANK"] / len(ranked) * 100
    )

    total_customers = len(ranked)
    total_actual_churners = int(ranked[ACTUAL_COL].sum())
    overall_churn_rate = ranked[ACTUAL_COL].mean()
    total_expected_churn = ranked[PROB_COL].sum()

    targeting_rows = []

    for share in TARGETING_SHARES:
        n_target = max(1, int(np.ceil(total_customers * share)))
        targeted = ranked.head(n_target)

        expected_churn = targeted[PROB_COL].sum()
        observed_churners = int(targeted[ACTUAL_COL].sum())
        observed_rate = targeted[ACTUAL_COL].mean()
        mean_risk = targeted[PROB_COL].mean()

        targeting_rows.append({
            "TARGETING_SHARE": share,
            "CUSTOMERS_TARGETED": n_target,
            "SHARE_OF_ALL_CUSTOMERS": n_target / total_customers,
            "MEAN_PREDICTED_CHURN_RISK": mean_risk,
            "EXPECTED_CHURNERS_FROM_PROBABILITIES": expected_churn,
            "ACTUAL_CHURNERS_IN_HELDOUT_SAMPLE": observed_churners,
            "OBSERVED_CHURN_RATE_IN_TARGETED_GROUP": observed_rate,
            "SHARE_OF_HELDOUT_ACTUAL_CHURNERS_CAPTURED": (
                observed_churners / total_actual_churners
                if total_actual_churners else np.nan
            ),
            "LIFT_VS_OVERALL_OBSERVED_CHURN_RATE": (
                observed_rate / overall_churn_rate
                if overall_churn_rate else np.nan
            ),
        })

    targeting = pd.DataFrame(targeting_rows)

    # Hypothetical scenarios: effectiveness means the assumed fraction
    # of expected churn among targeted customers prevented by intervention.
    # This is a planning assumption, not a measured causal effect.
    scenario_rows = []

    for _, row in targeting.iterrows():
        expected_churn = row["EXPECTED_CHURNERS_FROM_PROBABILITIES"]

        for effectiveness in INTERVENTION_EFFECTIVENESS:
            scenario_rows.append({
                "TARGETING_SHARE": row["TARGETING_SHARE"],
                "CUSTOMERS_TARGETED": int(row["CUSTOMERS_TARGETED"]),
                "ASSUMED_INTERVENTION_EFFECTIVENESS": effectiveness,
                "EXPECTED_CHURN_WITHOUT_INTERVENTION": expected_churn,
                "ILLUSTRATIVE_CHURN_PREVENTED": (
                    expected_churn * effectiveness
                ),
                "ILLUSTRATIVE_EXPECTED_CHURN_AFTER_INTERVENTION": (
                    expected_churn * (1 - effectiveness)
                ),
                "ASSUMPTION_NOTE": (
                    "Assumed proportional reduction in churn among "
                    "targeted customers; not experimentally validated."
                ),
            })

    scenarios = pd.DataFrame(scenario_rows)

    # Decile-style risk bands provide a compact ranking diagnostic.
    ranked["RISK_BAND"] = pd.qcut(
        ranked["RISK_RANK"],
        q=min(10, len(ranked)),
        labels=False,
        duplicates="drop",
    ) + 1

    bands = ranked.groupby("RISK_BAND", as_index=False).agg(
        CUSTOMERS=(ID_COL, "nunique"),
        MIN_PREDICTED_RISK=(PROB_COL, "min"),
        MAX_PREDICTED_RISK=(PROB_COL, "max"),
        MEAN_PREDICTED_RISK=(PROB_COL, "mean"),
        ACTUAL_CHURNERS=(ACTUAL_COL, "sum"),
        OBSERVED_CHURN_RATE=(ACTUAL_COL, "mean"),
    )

    ranked_file = OUTPUT_DIR / "phase13_ranked_customers.csv"
    targeting_file = OUTPUT_DIR / "phase13_targeting_comparison.csv"
    scenarios_file = OUTPUT_DIR / "phase13_intervention_scenarios.csv"
    bands_file = OUTPUT_DIR / "phase13_risk_bands.csv"
    report_file = OUTPUT_DIR / "phase13_retention_report.txt"

    ranked.to_csv(ranked_file, index=False)
    targeting.to_csv(targeting_file, index=False)
    scenarios.to_csv(scenarios_file, index=False)
    bands.to_csv(bands_file, index=False)

    lines = [
        "PHASE 13 — RETENTION SCENARIO ANALYSIS",
        "=" * 65,
        f"Held-out customers: {total_customers}",
        f"Actual held-out churners: {total_actual_churners}",
        f"Observed overall churn rate: {overall_churn_rate:.4%}",
        f"Sum of predicted churn probabilities: {total_expected_churn:.3f}",
        "",
        "TARGETING SCENARIOS",
        "-" * 65,
        "Customers are ranked by predicted churn probability.",
        "Targeting groups use the highest-risk customers first.",
        "",
    ]

    for _, row in targeting.iterrows():
        lines.extend([
            f"Target top {row['TARGETING_SHARE']:.0%}:",
            f"  Customers targeted: {int(row['CUSTOMERS_TARGETED'])}",
            f"  Mean predicted risk: "
            f"{row['MEAN_PREDICTED_CHURN_RISK']:.4%}",
            f"  Expected churn from probabilities: "
            f"{row['EXPECTED_CHURNERS_FROM_PROBABILITIES']:.2f}",
            f"  Actual churners in held-out sample: "
            f"{int(row['ACTUAL_CHURNERS_IN_HELDOUT_SAMPLE'])}",
            f"  Observed targeted-group churn rate: "
            f"{row['OBSERVED_CHURN_RATE_IN_TARGETED_GROUP']:.4%}",
            f"  Actual churners captured: "
            f"{row['SHARE_OF_HELDOUT_ACTUAL_CHURNERS_CAPTURED']:.2%}",
            f"  Lift vs overall observed churn rate: "
            f"{row['LIFT_VS_OVERALL_OBSERVED_CHURN_RATE']:.2f}x",
            "",
        ])

    lines.extend([
        "ILLUSTRATIVE INTERVENTION EFFECTIVENESS SCENARIOS",
        "-" * 65,
        "These values are assumptions, not measured treatment effects.",
        "They do not account for incentive costs, margin, or customer response.",
        "",
    ])

    for _, row in scenarios.iterrows():
        lines.append(
            f"Target {row['TARGETING_SHARE']:.0%}, "
            f"assumed effectiveness "
            f"{row['ASSUMED_INTERVENTION_EFFECTIVENESS']:.0%}: "
            f"illustrative churn prevented="
            f"{row['ILLUSTRATIVE_CHURN_PREVENTED']:.2f}; "
            f"expected churn after intervention="
            f"{row['ILLUSTRATIVE_EXPECTED_CHURN_AFTER_INTERVENTION']:.2f}"
        )

    lines.extend([
        "",
        "LIMITATIONS",
        "-" * 65,
        "1. Results describe the existing random held-out sample.",
        "2. Targeting performance may differ in a future customer cohort.",
        "3. Predicted probabilities have not been calibrated.",
        "4. Expected churn is the sum of model probabilities, not a count of known outcomes.",
        "5. Intervention effectiveness is assumed and not causally estimated.",
        "6. Cost, margin, offer eligibility, and customer treatment data were not supplied.",
        "7. Use a randomized retention experiment to measure actual incremental impact.",
        "",
        "OUTPUT FILES",
        str(ranked_file),
        str(targeting_file),
        str(scenarios_file),
        str(bands_file),
        str(report_file),
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nPhase 13 completed.")


if __name__ == "__main__":
    main()
