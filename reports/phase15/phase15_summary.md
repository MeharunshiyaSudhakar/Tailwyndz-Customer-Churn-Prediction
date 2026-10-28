# Phase 15 — Validation and Sensitivity Checks

Status: exploratory analysis; not proof of leakage-free or deployment-ready performance.

## Data and split
- Eligible customers: 2227
- Training customers in fixed random split: 1781
- Validation customers in fixed random split: 446
- Feature cutoff remains 2024-03-31 as established in the existing pipeline.
- Classification threshold: 0.50.
- Missing-value imputation and scaling/encoding are fitted inside each training pipeline.

## Recency sensitivity
Compare `all_features_*` against `without_recency_*` in
`model_sensitivity_comparison.csv`. A performance drop after removing
`MONTHS_SINCE_LAST_OBSERVED_ACTIVITY` indicates predictive dependence on this feature; it does not,
by itself, prove leakage. Investigate how the feature is constructed and
whether it is available at the intended prediction time.

## Class imbalance
Compare weighted and unweighted Logistic Regression using the same random
split. Review recall, precision, F1, PR-AUC, balanced accuracy and ROC-AUC;
there is no single universally best metric.

## Signup-cohort robustness
See `signup_cohort_robustness.csv` and `signup_cohort_counts.csv`.
This is not true temporal validation: customers share the same April–June
2024 churn outcome window, and signup cohorts differ in tenure and exposure.
The check is skipped if either partition lacks both target classes.

## Limitations
- Strong performance may reflect behavioral proxies closely tied to the
  churn definition.
- Cohort differences can reflect customer tenure, exposure and class mix.
- This script does not independently verify the source label-generation
  process or establish causal effects.
