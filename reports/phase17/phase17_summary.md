# Phase 17 — Repeated Stratified Validation

Status: exploratory robustness check, not an independent final test.

- Eligible customers: 2227
- Target prevalence: 0.2973
- Repeats: 5
- Validation fraction per repeat: 20%
- Random seeds: 42 through 46
- Model: balanced Logistic Regression
- Classification threshold: 0.50

Three feature strategies were compared using the same split for each repeat:
1. Existing observed-activity recency.
2. Positive-transaction recency within the available feature window plus
   an explicit no-positive-transaction flag.
3. No recency feature.

Imputation, scaling and categorical encoding are fitted inside each
training pipeline. The repeated splits overlap and therefore are not
independent replications. Means and standard deviations describe
sensitivity to these split choices; they are not confidence intervals.

The activity history starts in January 2023, so positive-transaction
recency is limited to that window and is not lifetime recency. Results
do not prove absence of target-proxy effects, leakage, or deployment risk.
A separate untouched final test set or genuinely later outcome window
would be needed for stronger generalization claims.
