
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

files = {
    "Customer profile": "customer_profile_deduplicated.csv",
    "Monthly activity": "monthly_activity_deduplicated.csv",
    "Churn labels": "churn_label_deduplicated.csv",
}

for name, filename in files.items():
    df = pd.read_csv(DATA / filename)

    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Exact duplicate rows:", df.duplicated().sum())
    print("Missing values:", int(df.isna().sum().sum()))

    if name == "Monthly activity":
        df["MONTH"] = pd.to_datetime(df["MONTH"], errors="coerce")

        repeated = df.duplicated(
            ["CUSTOMER_ID", "MONTH"], keep=False
        )
        print("Rows in repeated customer-month groups:", int(repeated.sum()))
        print(
            "Repeated customer-month groups:",
            int(
                df.loc[repeated, ["CUSTOMER_ID", "MONTH"]]
                .drop_duplicates()
                .shape[0]
            ),
        )

print("\nValidation completed.")
