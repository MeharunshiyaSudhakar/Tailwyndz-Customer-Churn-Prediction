# Phase 16 — Recency Sensitivity Analysis

Status: exploratory sensitivity analysis.

Three balanced Logistic Regression pipelines were evaluated on the same
stratified random train/validation split:
1. Existing recency based on the last observed activity record.
2. Recency based on the last positive-transaction month in the available
   January 2023–March 2024 feature window, plus a no-positive-transaction flag.
3. A baseline excluding recency features.

All preprocessing is fitted within the training pipeline. The positive-
transaction recency is not lifetime recency because the available activity
extract begins in January 2023. Customers without positive transactions in
the feature window retain missing recency and are flagged explicitly.

This is a sensitivity check, not proof of leakage absence, causal impact,
or deployment readiness. Interpret all metrics alongside the label definition
and the limitations of the available activity history.
