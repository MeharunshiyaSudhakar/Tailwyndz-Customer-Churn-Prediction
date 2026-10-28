
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "phase11"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

features_path = ROOT / "data" / "processed" / "customer_features_pre_apr2024.csv"
labels_path = ROOT / "data" / "processed" / "churn_label_clean.csv"
activity_path = ROOT / "data" / "processed" / "monthly_activity_clean.csv"

features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)
activity = pd.read_csv(activity_path)

for df in (features, labels, activity):
    df.columns = df.columns.str.strip().str.upper()

activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")
activity["TRANSACTIONS"] = pd.to_numeric(
    activity["TRANSACTIONS"], errors="coerce"
)

# Reconstruct activity-based evidence without modifying labels.
pre = activity[activity["MONTH"] < "2024-04-01"]
post = activity[activity["MONTH"].between("2024-04-01", "2024-06-30")]

pre_stats = pre.groupby("CUSTOMER_ID").agg(
    PRE_ROWS=("MONTH", "size"),
    PRE_POSITIVE_MONTHS=("TRANSACTIONS", lambda s: (s > 0).sum())
).reset_index()

post_stats = post.groupby("CUSTOMER_ID").agg(
    POST_ROWS=("MONTH", "size"),
    POST_POSITIVE_MONTHS=("TRANSACTIONS", lambda s: (s > 0).sum())
).reset_index()

audit = labels.merge(pre_stats, on="CUSTOMER_ID", how="left")
audit = audit.merge(post_stats, on="CUSTOMER_ID", how="left")

count_cols = [
    "PRE_ROWS", "PRE_POSITIVE_MONTHS",
    "POST_ROWS", "POST_POSITIVE_MONTHS"
]
audit[count_cols] = audit[count_cols].fillna(0)

audit["LABEL_EVIDENCE_NOTE"] = "No automatic label change"
audit.loc[
    (audit["CHURNED"] == 1) & (audit["PRE_POSITIVE_MONTHS"] == 0),
    "LABEL_EVIDENCE_NOTE"
] = "Churn-positive label; no earlier positive transactions in extract"

audit.loc[
    (audit["CHURNED"] == 0) & (audit["POST_POSITIVE_MONTHS"] == 0)
    & (audit["POST_ROWS"] > 0),
    "LABEL_EVIDENCE_NOTE"
] = "Non-churn label; observed post-window rows have no positive transactions"

audit.loc[
    (audit["CHURNED"] == 0) & (audit["POST_ROWS"] == 0),
    "LABEL_EVIDENCE_NOTE"
] = "Non-churn label; no post-window activity rows available"

audit_path = REPORT_DIR / "phase11_label_activity_evidence_audit.csv"
audit.to_csv(audit_path, index=False)

# Check identifiers and known target/eligibility columns in model feature table.
forbidden_exact = {
    "CUSTOMER_ID", "CHURNED", "DATA_SPLIT",
    "INSUFFICIENT_HISTORY_FLAG", "NO_OBSERVED_FEATURE_ACTIVITY",
    "OBSERVATION_END_DATE", "LAST_PURCHASE_DATE"
}
present_forbidden = sorted(set(features.columns) & forbidden_exact)

# Check date boundaries for feature-month columns, if present.
date_results = []
for col in ["FIRST_FEATURE_MONTH", "LAST_FEATURE_MONTH",
            "FEATURE_WINDOW_END", "OBSERVATION_END_DATE"]:
    if col in features.columns:
        values = pd.to_datetime(features[col], errors="coerce")
        date_results.append({
            "COLUMN": col,
            "NONMISSING": int(values.notna().sum()),
            "MAX_DATE": str(values.max()) if values.notna().any() else "None",
            "AFTER_2024_03_31": int((values > pd.Timestamp("2024-03-31")).sum())
        })

pd.DataFrame(date_results).to_csv(
    REPORT_DIR / "phase11_final_date_boundary_audit.csv", index=False
)

report = [
    "PHASE 11 FINAL EXPLAINABILITY AND LEAKAGE AUDIT",
    "=" * 52,
    f"Feature table rows: {len(features)}",
    f"Feature table duplicate customer IDs: "
    f"{int(features['CUSTOMER_ID'].duplicated().sum()) if 'CUSTOMER_ID' in features else 'CUSTOMER_ID column missing'}",
    f"Label rows: {len(labels)}",
    f"Activity rows: {len(activity)}",
    f"Predictor table contains forbidden exact columns: {present_forbidden}",
    f"Label evidence audit saved: {audit_path}",
    "",
    "INTERPRETATION",
    "- Date-boundary checks do not by themselves prove absence of target proxies.",
    "- MONTHS_SINCE_LAST_OBSERVED_ACTIVITY has previously shown a high univariate AUC (~0.92).",
    "- Nine investigated customers showed discrepancies between labels and available activity evidence.",
    "- Original labels are preserved; no automatic relabelling was performed.",
    "- SHAP results from the Phase 11 refit are not identical to explanations for the Phase 9 model.",
    "- Model performance and feature effects remain provisional pending source-level label clarification.",
    "",
    "NEXT ACTION",
    "Verify the precise target construction and investigate recency-related predictors before operational use."
]

report_path = REPORT_DIR / "phase11_final_audit_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("\nSaved:", report_path)
print("Saved:", audit_path)
