\# FreshBasket Customer Churn Prediction

\## Phase 15–17: Additional Validation Findings



\*\*Status:\*\* Exploratory and provisional



\### 1. Purpose



These phases extend the Phase 14 evaluation through model sensitivity analysis, alternative recency definitions and repeated stratified validation. They supplement the original report and do not replace its findings.



\### 2. Phase 15 — Model Sensitivity



The balanced Logistic Regression model with the existing recency feature achieved accuracy of 0.9417, balanced accuracy of 0.9260, F1-score of 0.9008 and ROC-AUC of 0.9669.



When recency was removed from the balanced model, F1-score decreased to 0.7671 and ROC-AUC to 0.9121. This indicates that recency-related information contributes substantially to discrimination in this dataset.



A signup-cohort robustness experiment achieved F1-score of 0.8794 and ROC-AUC of 0.9816. This is a cohort-based robustness check, not a true future-period validation.



\### 3. Phase 16 — Positive-Transaction Recency



An alternative recency feature measured time since the last positive-transaction month within the feature window. A separate indicator represented customers with no positive-transaction month in that window.



On the existing split, this model achieved:

\- Accuracy: 0.9484

\- Balanced accuracy: 0.9373

\- Precision: 0.9167

\- Recall: 0.9098

\- F1-score: 0.9132

\- ROC-AUC: 0.9674

\- PR-AUC / average precision: 0.9503



These results were better than those of the existing observed-activity recency model on the same split. However, the alternative feature is still derived from the historical activity window and requires further scrutiny against the target definition.



\### 4. Phase 17 — Repeated Stratified Validation



Five stratified 80/20 splits were evaluated using seeds 42–46. The same splits were used for each model, and preprocessing was fitted within each training pipeline.



| Metric | Existing observed-activity recency | Positive-transaction recency | No recency |

|---|---:|---:|---:|

| Accuracy, mean ± SD | 0.9345 ± 0.0129 | 0.9422 ± 0.0094 | 0.8601 ± 0.0206 |

| Balanced accuracy, mean ± SD | 0.9244 ± 0.0139 | 0.9328 ± 0.0120 | 0.8601 ± 0.0231 |

| Precision, mean ± SD | 0.8839 ± 0.0290 | 0.8977 ± 0.0157 | 0.7234 ± 0.0312 |

| Recall, mean ± SD | 0.8992 ± 0.0203 | 0.9098 ± 0.0192 | 0.8602 ± 0.0294 |

| F1-score, mean ± SD | 0.8913 ± 0.0205 | 0.9036 ± 0.0158 | 0.7858 ± 0.0305 |

| ROC-AUC, mean ± SD | 0.9747 ± 0.0091 | 0.9772 ± 0.0091 | 0.9364 ± 0.0179 |

| PR-AUC / average precision, mean ± SD | 0.9584 ± 0.0153 | 0.9649 ± 0.0120 | 0.8832 ± 0.0238 |



Positive-transaction recency outperformed the existing recency model on all five evaluated splits for F1-score, ROC-AUC and PR-AUC. Compared with the no-recency model, it also showed substantially stronger average discrimination.



These five splits are not independent replications and do not establish formal statistical significance. The findings support selecting positive-transaction recency as a candidate for further evaluation, rather than establishing it as a deployment-ready model.



\### 5. Interpretation and Remaining Limitations



1\. Recency-related features are strongly associated with the churn label, so possible target-proxy effects still need investigation.

2\. The target describes inactivity during April–June 2024, while predictors use activity through March 2024. A cutoff check alone cannot establish independence from target construction.

3\. Customers without positive-transaction months within the feature window require careful handling; the explicit indicator helps represent this case.

4\. Signup-cohort robustness and repeated random splits do not substitute for a defensible future-period evaluation.

5\. Reported results are exploratory. They do not establish causal effects, intervention impact or operational probability calibration.



\### 6. Recommended Next Steps



\- Confirm label-generation rules and missing-month semantics with the dataset owner.

\- Audit the lineage of recency features and their relationship to the target.

\- Preserve the original labels and document unresolved data-quality issues.

\- Evaluate using a defensible time-based holdout after resolving the label and feature-timing concerns.

\- Reassess calibration and retention interventions before operational use.



\### 7. Reproducibility References



\- Phase 15: `reports/phase15/phase15\_summary.md`

\- Phase 16: `reports/phase16/phase16\_summary.md`

\- Phase 17: `reports/phase17/phase17\_summary.md`

\- Repeated split metrics: `reports/phase17/repeated\_validation\_summary.csv`

\- Paired model differences: `reports/phase17/paired\_model\_differences.csv`



