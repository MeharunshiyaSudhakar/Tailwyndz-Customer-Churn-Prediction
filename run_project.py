
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx",
    "data/processed/customer_features_pre_apr2024.csv",
    "reports/phase9/phase9_heldout_churn_predictions.csv",
    "reports/phase10/phase10_metrics.csv",
    "reports/phase11/phase11_shap_summary.csv",
    "reports/phase11/phase11_heldout_top3_customer_drivers.csv",
    "reports/phase12/phase12_ablation_results.csv",
    "reports/phase13/phase13_intervention_scenarios.csv",
    "reports/phase14/final_report.md",
    "reports/phase14/final_report.pdf",
    "presentation/final_presentation.pptx",
    "reports/phase15/phase15_summary.md",
    "reports/phase16/phase16_summary.md",
    "reports/phase17/phase17_summary.md",
    "reports/phase18/phase18_cohort_validation_report.txt",
    "reports/phase18/phase18_test_predictions.csv",
    "reports/phase18/phase18_model_comparison.csv",
    "reports/phase19/phase19_retention_report.txt",
    "reports/phase19/phase19_strategy_comparison.csv",
    "reports/phase19/phase19_recommended_target_list.csv",
]

def main():
    print("=" * 60)
    print("FRESHBASKET CHURN PROJECT — FINAL CHECK")
    print("=" * 60)

    missing = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        status = "OK" if path.is_file() else "MISSING"
        print(f"[{status}] {relative}")
        if not path.is_file():
            missing.append(relative)

    prediction_file = (
        ROOT / "reports/phase11/phase11_heldout_top3_customer_drivers.csv"
    )

    if prediction_file.is_file():
        df = pd.read_csv(prediction_file)
        required_columns = {
            "CUSTOMER_ID",
            "PREDICTED_CHURN_PROBABILITY",
            "PREDICTED_CHURNED",
            "TOP_DRIVER_1",
            "TOP_DRIVER_2",
            "TOP_DRIVER_3",
        }
        absent = required_columns - set(df.columns)

        print("\nHeld-out driver file:")
        print(f"Rows: {len(df)}")
        print(f"Unique customers: {df['CUSTOMER_ID'].nunique()}")

        if absent:
            print("Missing columns:", sorted(absent))
            missing.append("Required prediction columns")
        else:
            print("Required prediction columns: OK")

        if df["CUSTOMER_ID"].duplicated().any():
            print("ERROR: duplicate customer IDs")
            missing.append("Unique prediction customer IDs")

        driver_cols = ["TOP_DRIVER_1", "TOP_DRIVER_2", "TOP_DRIVER_3"]
        if df[driver_cols].isna().any().any():
            print("ERROR: missing driver values")
            missing.append("Complete driver values")

        probs = pd.to_numeric(
            df["PREDICTED_CHURN_PROBABILITY"], errors="coerce"
        )
        if probs.isna().any() or not probs.between(0, 1).all():
            print("ERROR: invalid probability values")
            missing.append("Valid prediction probabilities")

    print("\n" + "=" * 60)
    if missing:
        print("CHECK FAILED — review the missing/invalid items above.")
        sys.exit(1)

    print("All configured final-file checks passed.")
    print("This is a packaging check, not a model re-training run.")
    print("It does not prove future-period model performance.")

if __name__ == "__main__":
    main()
