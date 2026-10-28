
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
REPORT_DIR = ROOT / "reports/phase7"
SPLIT_FILE = REPORT_DIR / "phase7_customer_split.csv"
REPORT_FILE = REPORT_DIR / "phase7_validation_report.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    required = [
        "CUSTOMER_ID",
        "SIGNUP_DATE",
        "CHURNED",
        "INSUFFICIENT_HISTORY_FLAG",
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["CUSTOMER_ID"].isna().any():
        raise ValueError("Missing CUSTOMER_ID values found.")

    if df["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate CUSTOMER_ID values found.")

    df["SIGNUP_DATE"] = pd.to_datetime(
        df["SIGNUP_DATE"], errors="coerce"
    )

    # Restrict split creation to the defined eligible cohort.
    eligible = df.loc[
        df["INSUFFICIENT_HISTORY_FLAG"].eq(0)
        & df["CHURNED"].isin([0, 1])
    ].copy()

    if eligible.empty:
        raise ValueError("No eligible labelled customers found.")

    class_counts = eligible["CHURNED"].value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Not enough examples of both target classes.")

    # Stratification preserves approximate class proportions.
    train_ids, test_ids = train_test_split(
        eligible["CUSTOMER_ID"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=eligible["CHURNED"],
    )

    split_map = pd.DataFrame({
        "CUSTOMER_ID": list(train_ids) + list(test_ids),
        "DATA_SPLIT": (
            ["train"] * len(train_ids)
            + ["test"] * len(test_ids)
        ),
    })

    output = eligible[
        ["CUSTOMER_ID", "SIGNUP_DATE", "CHURNED"]
    ].merge(
        split_map,
        on="CUSTOMER_ID",
        how="inner",
        validate="one_to_one",
    )

    if len(output) != len(eligible):
        raise ValueError("Some eligible customers are missing from the split.")

    if output["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in split output.")

    output["SIGNUP_COHORT"] = (
        output["SIGNUP_DATE"].dt.to_period("Q").astype("string")
    )

    output.to_csv(SPLIT_FILE, index=False)

    lines = [
        "PHASE 7 — VALIDATION STRATEGY REPORT",
        "=" * 48,
        f"Input: {INPUT_FILE}",
        f"Eligible customers: {len(eligible)}",
        f"Train customers: {(output['DATA_SPLIT'] == 'train').sum()}",
        f"Test customers: {(output['DATA_SPLIT'] == 'test').sum()}",
        f"Random state: {RANDOM_STATE}",
        f"Test size: {TEST_SIZE:.0%}",
        "Split method: stratified random customer-level split",
        "Feature cutoff: 2024-03-31",
        "Label window: 2024-04-01 through 2024-06-30",
        "",
        "CLASS DISTRIBUTION BY SPLIT",
    ]

    for split_name, group in output.groupby("DATA_SPLIT"):
        lines.append(
            f"{split_name}: customers={len(group)}, "
            f"churned={(group['CHURNED'] == 1).sum()}, "
            f"not_churned={(group['CHURNED'] == 0).sum()}, "
            f"churn_rate={group['CHURNED'].mean() * 100:.2f}%"
        )

    lines.extend(["", "SIGNUP COHORT DISTRIBUTION BY SPLIT"])

    cohort_table = (
        output.groupby(["DATA_SPLIT", "SIGNUP_COHORT"], dropna=False)
        .agg(
            CUSTOMERS=("CUSTOMER_ID", "nunique"),
            CHURNED=("CHURNED", "sum"),
            CHURN_RATE=("CHURNED", "mean"),
        )
        .reset_index()
    )

    for _, row in cohort_table.iterrows():
        lines.append(
            f"{row['DATA_SPLIT']} | {row['SIGNUP_COHORT']}: "
            f"customers={int(row['CUSTOMERS'])}, "
            f"churned={int(row['CHURNED'])}, "
            f"churn_rate={row['CHURN_RATE'] * 100:.2f}%"
        )

    lines.extend([
        "",
        "LIMITATIONS",
        "- This is a stratified random customer-level split, not a true future-time holdout.",
        "- Signup cohort is reported to inspect distribution; it is not the split mechanism.",
        "- Use the test set only for final evaluation after model and preprocessing choices are fixed.",
        "- Fit imputers, encoders, scalers, feature selection, and models on training data only.",
        "- CUSTOMER_ID, DATA_SPLIT, SIGNUP_COHORT, CHURNED, and history eligibility flags are not predictors.",
        "- All predictor features must remain limited to information available by 2024-03-31.",
        "",
        "OUTPUT",
        str(SPLIT_FILE),
        str(REPORT_FILE),
    ])

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print("\nPhase 7 split and report created.")


if __name__ == "__main__":
    main()
