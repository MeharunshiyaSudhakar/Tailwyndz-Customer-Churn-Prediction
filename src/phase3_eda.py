
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Load cleaned tables ----------
profile = pd.read_csv(DATA / "customer_profile_clean.csv")
activity = pd.read_csv(DATA / "monthly_activity_clean.csv")
churn = pd.read_csv(DATA / "churn_label_clean.csv")

activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")

numeric_cols = [
    "TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE",
    "DISTINCT_CATEGORIES", "PROMO_TXN_PCT", "APP_SESSIONS",
    "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS",
    "COMPLAINT_FLAG"
]
for col in numeric_cols:
    if col in activity.columns:
        activity[col] = pd.to_numeric(activity[col], errors="coerce")

# ---------- 1. Dataset overview ----------
overview = {
    "profile_rows": int(len(profile)),
    "profile_unique_customers": int(profile["CUSTOMER_ID"].nunique()),
    "activity_rows": int(len(activity)),
    "activity_unique_customers": int(activity["CUSTOMER_ID"].nunique()),
    "churn_rows": int(len(churn)),
    "churn_unique_customers": int(churn["CUSTOMER_ID"].nunique()),
    "activity_first_month": str(activity["MONTH"].min()),
    "activity_last_month": str(activity["MONTH"].max()),
    "activity_missing_month_values": int(activity["MONTH"].isna().sum()),
    "exact_duplicate_rows_profile": int(profile.duplicated().sum()),
    "exact_duplicate_rows_activity": int(activity.duplicated().sum()),
    "exact_duplicate_rows_churn": int(churn.duplicated().sum())
}

with open(OUT / "dataset_overview.json", "w") as f:
    json.dump(overview, f, indent=4)

# ---------- 2. Missing-value report ----------
missing_records = []
for table_name, df in [
    ("customer_profile", profile),
    ("monthly_activity", activity),
    ("churn_label", churn)
]:
    for col in df.columns:
        count = int(df[col].isna().sum())
        missing_records.append({
            "table": table_name,
            "column": col,
            "missing_count": count,
            "missing_percentage": round(count / len(df) * 100, 2)
                if len(df) else 0
        })

pd.DataFrame(missing_records).to_csv(
    OUT / "missing_values_summary.csv", index=False
)

# ---------- 3. Missing months per customer ----------
expected_months = pd.date_range(
    "2023-01-01", "2024-06-01", freq="MS"
)
observed = activity.dropna(subset=["MONTH"]).copy()
observed["MONTH"] = observed["MONTH"].dt.to_period("M")

monthly_counts = observed.groupby("CUSTOMER_ID")["MONTH"].nunique()
missing_month_rows = []

for customer_id in profile["CUSTOMER_ID"].drop_duplicates():
    customer_months = set(
        observed.loc[
            observed["CUSTOMER_ID"] == customer_id, "MONTH"
        ].astype(str)
    )
    missing = [
        month.strftime("%Y-%m")
        for month in expected_months
        if month.strftime("%Y-%m") not in customer_months
    ]
    missing_month_rows.append({
        "CUSTOMER_ID": customer_id,
        "months_observed": len(customer_months),
        "missing_month_count": len(missing),
        "missing_months": ", ".join(missing)
    })

missing_months = pd.DataFrame(missing_month_rows)
missing_months.to_csv(OUT / "missing_months_by_customer.csv", index=False)

# ---------- 4. Transaction and spending anomalies ----------
anomaly_conditions = {
    "zero_transactions": activity["TRANSACTIONS"] == 0,
    "zero_spend": activity["TOTAL_SPEND"] == 0,
    "negative_spend": activity["TOTAL_SPEND"] < 0,
    "zero_spend_positive_transactions": (
        (activity["TOTAL_SPEND"] == 0) &
        (activity["TRANSACTIONS"] > 0)
    ),
    "positive_spend_zero_transactions": (
        (activity["TOTAL_SPEND"] > 0) &
        (activity["TRANSACTIONS"] == 0)
    ),
    "negative_transactions": activity["TRANSACTIONS"] < 0,
}

anomaly_summary = {}
for name, mask in anomaly_conditions.items():
    anomaly_summary[name] = int(mask.fillna(False).sum())
    activity[mask.fillna(False)].to_csv(
        OUT / f"{name}_records.csv", index=False
    )

with open(OUT / "activity_anomaly_summary.json", "w") as f:
    json.dump(anomaly_summary, f, indent=4)

# ---------- 5. Spend summary and IQR outliers ----------
spend = activity["TOTAL_SPEND"].dropna()
q1 = spend.quantile(0.25)
q3 = spend.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = activity[
    (activity["TOTAL_SPEND"] < lower_bound) |
    (activity["TOTAL_SPEND"] > upper_bound)
].copy()
outliers.to_csv(OUT / "spend_iqr_outliers.csv", index=False)

spend_summary = {
    "count": int(spend.count()),
    "mean": float(spend.mean()),
    "median": float(spend.median()),
    "minimum": float(spend.min()),
    "maximum": float(spend.max()),
    "q1": float(q1),
    "q3": float(q3),
    "iqr_lower_bound": float(lower_bound),
    "iqr_upper_bound": float(upper_bound),
    "iqr_outlier_rows": int(len(outliers))
}
with open(OUT / "spend_summary.json", "w") as f:
    json.dump(spend_summary, f, indent=4)

# ---------- 6. Monthly activity coverage ----------
monthly_coverage = (
    observed.groupby("MONTH")
    .agg(
        activity_rows=("CUSTOMER_ID", "size"),
        unique_customers=("CUSTOMER_ID", "nunique")
    )
    .reset_index()
)
monthly_coverage["MONTH"] = monthly_coverage["MONTH"].astype(str)
monthly_coverage.to_csv(OUT / "monthly_activity_coverage.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(
    monthly_coverage["MONTH"],
    monthly_coverage["unique_customers"],
    marker="o"
)
plt.title("Unique Customers with Monthly Activity")
plt.xlabel("Month")
plt.ylabel("Unique customers")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(OUT / "monthly_customer_coverage.png", dpi=150)
plt.close()

# ---------- 7. Churn distribution ----------
if "CHURNED" in churn.columns:
    churn_counts = churn["CHURNED"].value_counts(dropna=False).sort_index()
    churn_counts.rename_axis("CHURNED").reset_index(
        name="customer_count"
    ).to_csv(OUT / "churn_distribution.csv", index=False)

    plt.figure(figsize=(6, 4))
    churn_counts.plot(kind="bar")
    plt.title("Churn Label Distribution")
    plt.xlabel("CHURNED (0 = not churned, 1 = churned)")
    plt.ylabel("Customer count")
    plt.tight_layout()
    plt.savefig(OUT / "churn_distribution.png", dpi=150)
    plt.close()

# ---------- 8. Spend distribution ----------
plt.figure(figsize=(8, 5))
plt.hist(spend, bins=50)
plt.title("Monthly Total Spend Distribution")
plt.xlabel("TOTAL_SPEND")
plt.ylabel("Row count")
plt.tight_layout()
plt.savefig(OUT / "spend_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot(spend, vert=False)
plt.title("Monthly Total Spend — Boxplot")
plt.xlabel("TOTAL_SPEND")
plt.tight_layout()
plt.savefig(OUT / "spend_boxplot.png", dpi=150)
plt.close()

# ---------- 9. Customer profile distributions ----------
for col in ["MEMBERSHIP_TIER", "CITY", "GENDER", "HOME_STORE_PRICE_TIER"]:
    if col in profile.columns:
        counts = profile[col].fillna("Missing").value_counts()
        counts.rename_axis(col).reset_index(
            name="customer_count"
        ).to_csv(OUT / f"{col.lower()}_distribution.csv", index=False)

        plt.figure(figsize=(9, 5))
        counts.plot(kind="bar")
        plt.title(f"Customer Distribution by {col}")
        plt.xlabel(col)
        plt.ylabel("Customer count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / f"{col.lower()}_distribution.png", dpi=150)
        plt.close()

# ---------- 10. Save a compact run summary ----------
summary = {
    "overview": overview,
    "anomalies": anomaly_summary,
    "spending": spend_summary,
    "customers_with_no_activity": int(
        len(set(profile["CUSTOMER_ID"]) - set(activity["CUSTOMER_ID"]))
    ),
    "customers_with_missing_months": int(
        (missing_months["missing_month_count"] > 0).sum()
    ),
    "customers_with_all_18_months": int(
        (missing_months["missing_month_count"] == 0).sum()
    )
}

with open(OUT / "phase3_eda_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nPHASE 3 EDA COMPLETED")
print("=" * 45)
print(json.dumps(summary, indent=4))
print("\nAll Phase 3 outputs saved to:", OUT)
