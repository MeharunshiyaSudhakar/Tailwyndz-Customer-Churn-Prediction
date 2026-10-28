# FreshBasket Customer Churn Prediction
## Phase 14 — Final Project Report

**Program:** Tailwyndz Propel Internship
**Assessment:** Customer Churn Prediction for a Retail Loyalty Program
**Status:** Exploratory and provisional

## 1. Executive Summary

This project analyses FreshBasket customer profiles, monthly activity and churn labels. It builds customer-level features, compares classification models, evaluates held-out predictions, examines feature contributions and segments, and explores hypothetical retention scenarios.

The primary validation is a time-ordered signup-cohort split (train on 2021–2022 signups, select on early-2023 signups, test once on 2023Q3–2024Q1 signups). On that test the chosen model reaches PR-AUC 0.654 (no-skill level 0.113), ROC-AUC 0.862 and F1 0.510 — clearly useful, but well below the 0.94 PR-AUC seen on the earlier random split. The random split is kept only as a secondary comparison.

The results are promising but are not ready for operational deployment. The Phase 11 audit identified unresolved inconsistencies between available activity records and churn labels. The strong recency signal also requires further investigation.

## 2. Business Objective

Identify customers who may stop purchasing so the loyalty team can investigate appropriate retention actions. The project covers data quality, feature engineering, model comparison, evaluation, explainability, segmentation and retention scenarios.

## 3. Dataset and Target

The workbook contains customer profiles, monthly activity, churn labels, a glossary and a data dictionary.

The supplied target definition describes churn as no transactions during April–June 2024 despite earlier purchase history. The intended feature cutoff is March 31, 2024; April–June 2024 activity is excluded from the predictor window.

The cleaned source tables contain 2,600 profiles, 28,015 monthly activity rows and 2,600 churn labels. The customer feature table contains 2,600 unique customer IDs. The eligible modelling cohort contains 2,227 customers; 373 were excluded under the insufficient-history eligibility rule.

## 4. Data Quality

- Duplicate rows and repeated customer-month records were audited.
- Two conflicting customer-month groups were investigated; conflicting values were not arbitrarily selected.
- Missing demographic and store-tier fields remain.
- Negative spend and spend/transaction inconsistencies remain documented concerns.
- Missing activity rows are not automatically interpreted as zero transactions.
- Original churn labels were preserved; no automatic relabelling was performed.

## 5. Feature Engineering and Validation

Features were aggregated from January 2023 through March 2024. They include transaction and spend summaries, engagement and support indicators, early/late-period summaries, trends, observed-month counts and recency-related measures.

Primary validation is cohort-based and time-ordered (Section 5A). An earlier stratified random customer-level split (1,781 training-pool and 446 held-out customers) is retained as a secondary comparison (Section 6). Preprocessing was always fitted on training data only.

## 5A. Primary Validation — Signup-Cohort Split (Phase 18)

Members are split by signup quarter so models are always trained on earlier cohorts and tested on later ones:

| Set | Signup cohorts | Members | Churn rate |
|---|---|---:|---:|
| Train | 2021Q1–2022Q4 | 1,384 | 35.3% |
| Validation (model and threshold choice) | 2023Q1–2023Q2 | 358 | 33.0% |
| Final test (used once) | 2023Q3–2024Q1 | 485 (55 churners) | 11.3% |

Models compared: majority-class baseline, a simple rule (flag members whose last observed activity was one or more months earlier), class-weighted Logistic Regression, class-weighted Random Forest and an unweighted Gradient Boosting model. Class imbalance was handled with class weights where available and with a decision threshold tuned on the validation cohort for every model. Gradient Boosting had the best validation PR-AUC (0.976) and was chosen; its threshold (0.424) was tuned on validation only.

| Test result | Value | 95% bootstrap interval |
|---|---:|---:|
| PR-AUC | 0.654 | 0.530–0.765 |
| ROC-AUC | 0.862 | 0.800–0.914 |
| Precision | 0.404 | 0.309–0.500 |
| Recall | 0.691 | 0.558–0.810 |
| F1-score | 0.510 | 0.409–0.599 |

Confusion matrix: TN=374, FP=56, FN=17, TP=38. A no-skill model scores a PR-AUC of 0.113 on this test set.

Other models on the same test cohort: Random Forest PR-AUC 0.633, Logistic Regression 0.465, simple rule 0.359 (the rule's F1 of 0.55 at its fixed cut-off is competitive with the tuned models, so a simple recency rule is a strong and transparent baseline).

Why performance is lower than on the random split: newer members have shorter histories and a much lower churn rate (cohort 2024Q1: 0.8%), so the test population differs from the training population. The model over-predicts risk for the newest members (mean predicted risk 0.31 for 2024Q1 signups against 0.8% actual), which is a cold-start problem that needs separate handling.

Feature-group ablation on the same cohort split (Gradient Boosting, test PR-AUC): all features 0.654; behavioural features only 0.651; without recency 0.370; without recency, late-period and trend features 0.255; demographics only 0.109 (no better than chance). Engagement, support and behavioural features therefore carry nearly all of the predictive value, and the conclusion holds on the stricter split.

Outputs: reports/phase18/ (model comparison, test predictions with top-3 drivers, ablation, segment risk, bootstrap intervals).

## 6. Secondary Evaluation — Random Held-Out Split

This section reports the earlier stratified random split. It is shown for comparison; the cohort-based results in Section 5A are the primary evidence.

The existing Phase 10 report records 446 test customers: 133 actual churners and 313 non-churners.

| Metric | Reported result |
|---|---:|
| Accuracy | 0.9260 |
| Precision | 0.8731 |
| Recall | 0.8797 |
| F1-score | 0.8764 |
| Specificity | 0.9457 |
| Balanced accuracy | 0.9127 |
| ROC-AUC | 0.9628 |
| PR-AUC / average precision | 0.9383 |
| Matthews correlation coefficient | 0.8236 |

Confusion matrix: TN=296, FP=17, FN=16, TP=117.

These metrics are transcribed from the existing report, not recalculated by this script. Strong scores do not rule out target-related leakage or label-quality problems.

## 7. Explainability and Segments

The Phase 11 SHAP analysis ranks MONTHS_SINCE_LAST_OBSERVED_ACTIVITY as the largest mean absolute SHAP contributor. Late-period transactions, observed feature months, email opens and support-related features also rank highly.

These are associations in the current experiment, not causal effects. City, membership tier, store price tier and signup cohort comparisons are descriptive.

## 8. Ablation Study

Phase 12 reported:
- Behaviour-only PR-AUC: 0.9787
- All features PR-AUC: 0.9773
- Without recency PR-AUC: 0.9071
- Without recency, late-period and trend features PR-AUC: 0.8390
- Profile-only PR-AUC: 0.3436

These results are provisional because the experiment uses a single internal validation split and the relationship between recency features and target construction has not been conclusively resolved.

## 9. Calibration and Retention Scenarios (random-split version)

This section reports the earlier random-split scenario. The main retention analysis, built on the cohort-split test set and including a support-ticket outreach scenario, is Section 9A.

The Phase 13 calibration audit reports a mean predicted probability of 0.3289 versus an observed churn rate of 0.2982, an absolute gap of about 3.07 percentage points. The Brier score is 0.0590 and log loss is 0.2194.

Calibration varies across risk ranges. Predicted probabilities should not yet be treated as reliable operational probabilities.

Retention scenarios assume prevention rates of 5%, 10% and 20%. These are assumptions, not measured intervention effects. A randomized retention test is needed to estimate incremental impact.

## 9A. Retention Scenarios on the Cohort Test Set (Phase 19)

**Question.** Which members should the business contact, and how many churners could be saved? Today every member receives retention offers. Members who have already stopped buying cannot be "retained", so the analysis separates reachable members (still buying within the last month of the feature window) from lapsed ones.

**Support experience and churn.** Among members who are still active, churn rises steadily with the number of support tickets in the late window (November 2023 – March 2024), and the pattern holds in both the training and the test cohort:

| Late-window tickets | Train cohort churn | Test cohort churn |
|---|---:|---:|
| 0 | 4.1% | 3.0% |
| 1 | 12.0% | 13.7% |
| 2 | 19.5% | 17.1% |
| 3 or more | 40.6% | 70.0% (10 members) |

**Strategies compared** on the 485 test-cohort members (observed outcomes are used to count the churners each group contains):

| Strategy | Contacts | Churn rate in group | Share of all churners reached |
|---|---:|---:|---:|
| 1. Blanket: contact everyone (current practice) | 485 | 11.3% | 100% |
| 2. Model top 20%, any member (reference; 16 are already lapsed) | 97 | 39.2% | 69% |
| 3. Model top 20%, reachable members only | 97 | 22.7% | 40% |
| 4. Proactive support outreach (reachable, 1+ ticket) | 168 | 17.9% | 55% |
| 5. Targeted coupon (reachable, model-flagged) | 78 | 28.2% | 40% |
| 6. Support outreach and model-flagged (reachable) | 49 | 36.7% | 33% |

**Impact under assumptions.** No campaign data exists, so effectiveness (the share of would-be churners who are saved) is assumed, not measured. At 20% effectiveness the strategies save about 11.0 (blanket), 6.0 (support outreach), 4.4 (coupon) and 3.6 (support outreach plus model flag) members of the 55 who churned. Churn after intervention for the whole cohort therefore falls from 55 to roughly 49 under support outreach, and to about 51 under the narrow strategy 6.

**Cost view.** Contact cost is expressed as a share r of the value of one retained member, so no currency figures are invented. At 20% effectiveness:

| Strategy | Net value, r = 2% | r = 5% | r = 10% |
|---|---:|---:|---:|
| 1. Blanket | 1.3 | −13.2 | −37.5 |
| 4. Support outreach | 2.6 | −2.4 | −10.8 |
| 5. Targeted coupon | 2.8 | 0.5 | −3.4 |
| 6. Support outreach and model-flagged | 2.6 | 1.1 | −1.3 |

Net value is in units of one retained member. Contacting everyone loses value as soon as a contact costs more than about 2% of a retained member's value. The narrow strategy 6 stays positive up to a cost of about 7%, because it contacts only 49 members of whom 37% churn. Strategy 2 looks best on paper but is flattered by the 16 members who had already lapsed; strategy 3 is the like-for-like version.

**Model what-if (association only).** Re-scoring the test cohort with late-window tickets halved lowers the mean predicted risk of the 174 members who raised tickets from 0.308 to 0.255; removing the tickets entirely lowers it to 0.228, and the number of flagged members falls from 94 to 67. This shows how the model uses support tickets; it is not evidence that resolving tickets would cause that reduction. The model's absolute probabilities are too high on this cohort, so only the relative change is informative.

**Reading the results.**
- Stop blanket retention offers: they reach every churner but cost far more than they return unless offers are nearly free.
- The most efficient practical plan is to contact members who are still active, have raised at least one support ticket, and are flagged by the model (49 members here), with a proactive support call and, if useful, a coupon.
- This focused group reaches only a third of churners. Lapsed members (no purchase for two or more months, 16 here, all churned) need a separate win-back offer rather than a prevention offer.
- The groups are small (for strategy 6 the 95% interval on the group churn rate is 25%–51%), so results are indicative.

Outputs: reports/phase19/ (strategy comparison, effectiveness and cost grid, ticket relationship, model what-if, recommended target list).

## 10. Critical Audit Findings and Limitations

- Five churn-positive labels have no earlier positive-transaction months in the available activity extract.
- A broader check found 125 non-churned customers with zero positive post-window months. This includes customers with no post-window activity rows and customers whose observed rows have no positive transactions.
- These are unresolved evidence gaps, not automatic proof that the labels are wrong. Original labels remain unchanged.
- MONTHS_SINCE_LAST_OBSERVED_ACTIVITY has a strong univariate AUC screen of approximately 0.9200. This does not by itself prove leakage, but the target definition and feature timing require verification.
- A date-boundary check alone does not prove that every predictor is independent of target construction.
- The data has a single outcome window (April–June 2024), so the cohort split tests generalisation to newer members, not to a later calendar period. Performance on newer cohorts was materially lower than on the random split.
- Segment differences are descriptive, not causal.

## 11. Recommendations: Data and Validation

1. Clarify churn-label construction and missing-month semantics with the dataset owner.
2. Reconcile label/activity discrepancies against the source without silently changing labels.
3. Verify that all predictors use only permitted pre-label-window information.
4. Re-run the cohort-based evaluation whenever a later outcome window becomes available, and handle brand-new members (short history) with a separate rule or model.
5. Repeat model comparison, calibration, SHAP and ablation on the revised setup.
6. Run a randomized retention pilot before estimating incremental benefit or deploying risk-based targeting: for example, randomly hold out half of the group of still-active, ticket-raising, model-flagged members (strategy 6) and compare churn between contacted and held-out members.
7. Document package versions, preprocessing, random seeds and reproducibility instructions.

## 11A. Business and Marketing Recommendations

These follow from the Phase 18 and 19 results on the cohort test set. Group sizes are small and effectiveness values are assumptions, so they are starting points to be confirmed by the randomised pilot in Section 11, not proven effects.

1. **Stop blanket retention offers.** Contacting all 485 test members reaches every churner, but only 11.3% of those contacted would have left. At an assumed 20% effectiveness a blanket offer saves about 11 members, and it stops paying for itself once a contact costs more than about 2% of a retained member's value (net value +1.3 at 2%, -13.2 at 5%).
2. **Make support tickets the main trigger for retention outreach.** Among members who are still buying, churn rises with late-window tickets: 3.0% with none, 13.7% with one, 17.1% with two and 70% with three or more (only 10 members; 40.6% in the training cohort). When an active member raises a ticket, follow up proactively (call, resolve the issue, goodwill gesture). Prioritise those the model also flags: 49 members, 36.7% of whom churned (95% interval 24%-51%), the most cost-efficient group tested.
3. **Separate win-back from prevention.** Members with no purchase for two or more months (16 in the test cohort, all churned) cannot be retained, only won back. Send them a cheaper win-back message and keep the retention budget for members who are still buying. The model's top-20% list looks strongest (39% churn) partly because it includes these 16; among still-active members it is 22.7%.
4. **Treat engagement channels as early-warning triggers, not proven levers.** Across the eligible cohort, churners had fewer email opens (mean 6.8 vs 12.1), app sessions (10.9 vs 25.4) and coupon redemptions (3.9 vs 6.6), and total email opens and late-period app sessions rank 4th and 6th in the SHAP analysis. These totals partly reflect shorter observed history among churners (8.4 vs 11.0 months), and nothing in this data shows that sending more emails or coupons lowers churn. Use falling app and email activity to trigger a personalised re-engagement message, and test it against a control group before scaling.
5. **Do not target by tier or city.** A demographics-only model has ROC-AUC 0.45 and PR-AUC 0.11 on the cohort test set, no better than chance (no-skill PR-AUC 0.11). Platinum members looked higher (23.5% vs 10% for Gold and Silver) but that is 12 churners among 51 members, too few to base a tier policy on. City churn ranged from 5.7% to 15.8% across groups of 34 to 58 members. Revisit these segments only with more data.
6. **Start small and measure.** Pilot with the still-active, ticket-raising, model-flagged group (about 10% of members), hold out half at random, and replace the assumed effectiveness with the measured effect before any wider rollout.

## 12. Conclusion

The project establishes an exploratory workflow and produces model comparisons, customer-level predictions, feature attribution, segment summaries, ablation results and illustrative retention scenarios.

On the primary cohort-based test the model ranks churn risk well (ROC-AUC 0.86, PR-AUC 0.65 against a 0.11 no-skill level) but precision is modest, so it is best used to prioritise outreach rather than to make automatic decisions. Behavioural features are promising candidates for further validation, while profile-only features showed limited discrimination in this experiment. However, unresolved label/activity discrepancies and recency-related concerns mean that the current scores and feature rankings remain provisional. The next milestone is resolving data definitions and conducting leakage-aware future-period validation—not operational deployment.

## Appendix: Existing Evidence

- EDA: reports/phase3/PHASE3_EDA_FINDINGS.md
- Join validation: reports/phase4/phase4_validation_report.txt
- Feature validation: reports/phase5/phase5_feature_validation_report.txt
- Model comparison: reports/phase8/phase8_model_comparison_report.txt
- Held-out predictions: reports/phase9/phase9_prediction_report.txt
- Evaluation: reports/phase10/phase10_evaluation_report.txt
- Label/leakage audit: reports/phase11/phase11_final_audit_report.txt
- Ablation: reports/phase12/phase12_ablation_report.txt
- Retention/calibration: reports/phase13/phase13_retention_report.txt and phase13_calibration_report.txt
# FreshBasket Customer Churn Prediction

## Phase 15–17: Additional Validation Findings



**Status:** Exploratory and provisional



### 1. Purpose



These phases extend the Phase 14 evaluation through model sensitivity analysis, alternative recency definitions and repeated stratified validation. They supplement the original report and do not replace its findings.



### 2. Phase 15 — Model Sensitivity



The balanced Logistic Regression model with the existing recency feature achieved accuracy of 0.9417, balanced accuracy of 0.9260, F1-score of 0.9008 and ROC-AUC of 0.9669.



When recency was removed from the balanced model, F1-score decreased to 0.7671 and ROC-AUC to 0.9121. This indicates that recency-related information contributes substantially to discrimination in this dataset.



A signup-cohort robustness experiment achieved F1-score of 0.8794 and ROC-AUC of 0.9816. This is a cohort-based robustness check, not a true future-period validation.



### 3. Phase 16 — Positive-Transaction Recency



An alternative recency feature measured time since the last positive-transaction month within the feature window. A separate indicator represented customers with no positive-transaction month in that window.



On the existing split, this model achieved:

- Accuracy: 0.9484

- Balanced accuracy: 0.9373

- Precision: 0.9167

- Recall: 0.9098

- F1-score: 0.9132

- ROC-AUC: 0.9674

- PR-AUC / average precision: 0.9503



These results were better than those of the existing observed-activity recency model on the same split. However, the alternative feature is still derived from the historical activity window and requires further scrutiny against the target definition.



### 4. Phase 17 — Repeated Stratified Validation



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



### 5. Interpretation and Remaining Limitations



1. Recency-related features are strongly associated with the churn label, so possible target-proxy effects still need investigation.

2. The target describes inactivity during April–June 2024, while predictors use activity through March 2024. A cutoff check alone cannot establish independence from target construction.

3. Customers without positive-transaction months within the feature window require careful handling; the explicit indicator helps represent this case.

4. Signup-cohort robustness and repeated random splits do not substitute for a defensible future-period evaluation.

5. Reported results are exploratory. They do not establish causal effects, intervention impact or operational probability calibration.



### 6. Recommended Next Steps



- Confirm label-generation rules and missing-month semantics with the dataset owner.

- Audit the lineage of recency features and their relationship to the target.

- Preserve the original labels and document unresolved data-quality issues.

- Evaluate using a defensible time-based holdout after resolving the label and feature-timing concerns.

- Reassess calibration and retention interventions before operational use.



### 7. Reproducibility References



- Phase 15: `reports/phase15/phase15_summary.md`

- Phase 16: `reports/phase16/phase16_summary.md`

- Phase 17: `reports/phase17/phase17_summary.md`

- Repeated split metrics: `reports/phase17/repeated_validation_summary.csv`

- Paired model differences: `reports/phase17/paired_model_differences.csv`



