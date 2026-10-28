"""Build the final churn prediction file with top-3 drivers per customer.

Inputs (run these first):
    reports/phase9/phase9_heldout_churn_predictions.csv   (phase9_heldout_predictions.py)
    reports/phase11/phase11_shap_customer_feature_values.csv (phase11_shap_analysis.py)

Output:
    reports/phase11/phase11_heldout_top3_customer_drivers.csv

Columns: CUSTOMER_ID, ACTUAL_CHURNED, PREDICTED_CHURN_PROBABILITY,
PREDICTED_CHURNED, MODEL, TOP_DRIVER_1..3. Drivers are the three features with
the largest absolute SHAP value for that customer (association, not causation).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FILE = ROOT / "reports/phase9/phase9_heldout_churn_predictions.csv"
SHAP_FILE = ROOT / "reports/phase11/phase11_shap_customer_feature_values.csv"
OUTPUT_FILE = ROOT / "reports/phase11/phase11_heldout_top3_customer_drivers.csv"


def main():
    predictions = pd.read_csv(PREDICTION_FILE)
    shap_values = pd.read_csv(SHAP_FILE)

    ranked = shap_values.sort_values(
        ["CUSTOMER_ID", "ABS_SHAP_VALUE", "TRANSFORMED_FEATURE"],
        ascending=[True, False, True],
    )
    ranked["RANK"] = ranked.groupby("CUSTOMER_ID").cumcount() + 1
    top3 = (
        ranked[ranked["RANK"] <= 3]
        .pivot(index="CUSTOMER_ID", columns="RANK", values="TRANSFORMED_FEATURE")
        .rename(columns=lambda r: f"TOP_DRIVER_{r}")
        .reset_index()
    )

    output = predictions.merge(top3, on="CUSTOMER_ID", how="left", validate="1:1")
    if output[["TOP_DRIVER_1", "TOP_DRIVER_2", "TOP_DRIVER_3"]].isna().any().any():
        raise ValueError("Some customers have no SHAP drivers.")

    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(output)} rows to {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
