
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

activity = pd.read_csv(DATA / "monthly_activity_clean.csv")
churn = pd.read_csv(DATA / "churn_label_clean.csv")

activity["MONTH"] = pd.to_datetime(
    activity["MONTH"], errors="coerce"
)
activity["TRANSACTIONS"] = pd.to_numeric(
    activity["TRANSACTIONS"], errors="coerce"
)

may = activity[activity["MONTH"].dt.strftime("%Y-%m") == "2024-05"]
june = activity[activity["MONTH"].dt.strftime("%Y-%m") == "2024-06"]

may_ids = set(may["CUSTOMER_ID"])
june_ids = set(june["CUSTOMER_ID"])

# Customers recorded in May but absent from June.
missing_june_ids = may_ids - june_ids

# Compare with the supplied churn labels.
comparison = pd.DataFrame({
    "CUSTOMER_ID": sorted(missing_june_ids)
}).merge(
    churn[["CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"]],
    on="CUSTOMER_ID",
    how="left",
    validate="one_to_one"
)

comparison.to_csv(
    OUT / "may_customers_missing_in_june.csv", index=False
)

print("May unique customers:", len(may_ids))
print("June unique customers:", len(june_ids))
print("May customers absent in June:", len(missing_june_ids))

print("\nCHURN LABEL DISTRIBUTION FOR CUSTOMERS ABSENT IN JUNE")
print(comparison["CHURNED"].value_counts(dropna=False).sort_index())

print("\nINSUFFICIENT HISTORY FLAG DISTRIBUTION")
print(
    comparison["INSUFFICIENT_HISTORY_FLAG"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nSaved customer comparison to:", OUT)


# Compare both directions between May and June.
new_in_june = june_ids - may_ids
present_both = may_ids & june_ids

print("\nFULL MAY-JUNE CUSTOMER OVERLAP")
print("Present in both months:", len(present_both))
print("Present in May but absent in June:", len(may_ids - june_ids))
print("Present in June but absent in May:", len(new_in_june))
print(
    "May customers accounted for:",
    len(present_both) + len(may_ids - june_ids)
)
print(
    "June customers accounted for:",
    len(present_both) + len(new_in_june)
)
