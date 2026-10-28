# FreshBasket Loyalty Programme — Customer Churn Prediction

**Programme:** Tailwyndz Propel Internship — Assessment 4
**Author:** S Meharunshiya S (BE Computer Science and Design, Kongu Engineering College)

Predicts which FreshBasket loyalty members will make **zero purchases in April–June 2024**, explains the drivers, compares risk across segments, and simulates retention scenarios.

---

## 1. Results at a glance

| Item | Value |
|---|---|
| Members in source data | 2,600 (2,227 eligible for modelling; 373 excluded for insufficient history) |
| Churn rate (eligible cohort) | about 29.7% |
| Feature window | Jan 2023 – Mar 2024 (predictor cutoff: 31 Mar 2024) |
| Target window | Apr – Jun 2024 |
| Models compared | Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting |
| **Primary validation** | Time-ordered signup-cohort split: train 2021Q1–2022Q4 (1,384), validate 2023Q1–Q2 (358), test 2023Q3–2024Q1 (485 members, 55 churners) |
| Final model | Gradient Boosting (best validation PR-AUC, 0.976); threshold 0.424 tuned on validation |
| **Test results (cohort split)** | PR-AUC 0.654 (no-skill 0.113) · ROC-AUC 0.862 · F1 0.510 · Precision 0.404 · Recall 0.691 |
| Test confusion matrix | TN 374 · FP 56 · FN 17 · TP 38 |
| Secondary check (random 80/20 split) | Logistic Regression: F1 0.876 · ROC-AUC 0.963 · PR-AUC 0.938 (446 members) |
| **Retention scenario** | Proactive outreach to still-active members with support tickets and a model flag: 49 of 485 contacted, 37% of them churn (base rate 11%); about 3.6 members saved at an assumed 20% effectiveness |
| Strongest drivers (SHAP) | Months since last observed activity, late-period transactions, email opens, support tickets |
| Ablation, cohort split (test PR-AUC) | Demographics only 0.11 → behavioural features 0.65 |

Profile data alone (tier, city, age, etc.) has no useful predictive power; behavioural and engagement signals do almost all the work. Performance is lower on the later cohorts than on the random split, because newer members have shorter histories and a lower churn rate (11% vs 30%). The cohort-based numbers are the ones to rely on.

---

## 2. Project structure

```
├── README.md
├── requirements.txt
├── run_project.py              # checks that all final deliverables exist
├── data/
│   ├── raw/                    # original Excel workbook (never modified)
│   └── processed/              # cleaned tables, joined table, feature table
├── src/                        # one script per phase (see run order below)
├── notebooks/                  # data inspection and EDA notebooks
├── reports/
│   ├── phase1 … phase17/       # outputs of each phase
│   └── phase14/final_report.*  # final report (Markdown and PDF)
└── presentation/final_presentation.pptx
```

All scripts locate the project root from their own file path, so the project runs from **any folder** on any operating system. No paths need editing.

---

## 3. Setup

Requires **Python 3.10 or newer**.

```bash
# from the project root folder
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Tested end-to-end on Python 3.12 with pandas 3.0, numpy 2.4, scikit-learn 1.8, shap 0.54, openpyxl 3.1, python-pptx 1.0 and reportlab 4.4. To record your own versions, run `pip freeze > requirements-frozen.txt`.

The raw workbook must be at `data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx`.

---

## 4. How to run (step by step)

Run from the project root. Each step reads the outputs of the earlier ones, so keep this order. The full pipeline takes about one minute.

| # | Command | What it does | Main output |
|---|---|---|---|
| 1 | `python src/phase1_audit.py` | Audits raw tables: shapes, IDs, duplicates, casing, anomalies | `reports/phase1/` |
| 2 | `python src/phase2_cleaning.py` | Removes exact duplicate rows | `data/processed/*_deduplicated.csv` |
| 3 | `python src/phase2_finalize_cleaning.py` | Standardises text casing, flags negative/zero spend, writes clean tables | `data/processed/*_clean.csv` |
| 4 | `python src/phase3_eda.py` | EDA: distributions, missingness, outliers, coverage | `reports/phase3/` |
| 5 | `python src/phase3_churn_by_segment.py` | Churn rate by tier, city, store price tier | `reports/phase3/` |
| 6 | `python src/phase3_churn_by_signup_cohort.py` | Churn rate by signup cohort | `reports/phase3/` |
| 7 | `python src/phase4_join_tables.py` | Joins profile, activity and label into one customer-level table | `data/processed/customer_level_joined.csv` |
| 8 | `python src/phase5_feature_engineering.py` | Builds leakage-safe features from data up to 31 Mar 2024 | `data/processed/customer_features_pre_apr2024.csv` |
| 9 | `python src/phase6_engagement_churn_analysis.py` | Engagement and support vs churn analysis | `reports/phase6/` |
| 10 | `python src/phase7_validation_strategy.py` | Creates the train / held-out split | `reports/phase7/` |
| 11 | `python src/phase8_model_comparison.py` | Compares baseline and ML models on an internal validation set | `reports/phase8/` |
| 12 | `python src/phase9_heldout_predictions.py` | Trains the final model and predicts the held-out members | `reports/phase9/` |
| 13 | `python src/phase10_evaluation_metrics.py` | Precision, recall, F1, PR-AUC, confusion matrix | `reports/phase10/` |
| 14 | `python src/phase11_leakage_audit.py` | Date-boundary and leakage checks on features | `reports/phase11/` |
| 15 | `python src/phase11_shap_analysis.py` | SHAP values per customer | `reports/phase11/` |
| 16 | `python src/phase11_explainability_segments.py` | Feature importance and risk by tier, city and signup cohort | `reports/phase11/` |
| 17 | `python src/phase11_top3_drivers.py` | **Builds the final prediction file** (ID, probability, label, top-3 drivers) | `reports/phase11/phase11_heldout_top3_customer_drivers.csv` |
| 18 | `python src/phase11_final_audit.py` | Label and activity consistency audit | `reports/phase11/` |
| 19 | `python src/phase12_ablation_study.py` | Ablation: demographics only vs behaviour vs all features | `reports/phase12/` |
| 20 | `python src/phase13_probability_calibration.py` | Calibration of predicted probabilities | `reports/phase13/` |
| 21 | `python src/phase13_retention_scenarios.py` | Retention scenarios and targeting lift | `reports/phase13/` |
| 22 | `python src/phase15_recency_audit.py` | Recency feature audit | `reports/phase15/` |
| 23 | `python src/phase15_validation_sensitivity.py` | Recency sensitivity and signup-cohort validation | `reports/phase15/` |
| 24 | `python src/phase16_recency_sensitivity.py` | Positive-transaction recency alternative | `reports/phase16/` |
| 25 | `python src/phase17_repeated_validation.py` | Repeated stratified validation (5 seeds) | `reports/phase17/` |
| 26 | `python src/phase18_cohort_validation.py` | **Primary validation**: signup-cohort split, model comparison, threshold tuning, test predictions with top-3 drivers, ablation, segment risk, bootstrap intervals | `reports/phase18/` |
| 27 | `python src/phase19_retention_scenarios_cohort.py` | **Retention scenarios** on the cohort test set: support-ticket outreach, targeted coupon, blanket and model-based targeting; effectiveness and cost grid; model what-if; target list | `reports/phase19/` |
| 28 | `python run_project.py` | Confirms all final deliverables exist and the prediction files are valid | console |

The data-inspection notebook is `notebooks/01_Dataset_Inspection.ipynb` and the EDA notebook is `notebooks/02_FreshBasket_EDA.ipynb`. Open them with Jupyter from the project root.

### Rebuilding the report and slides (optional)

The report (`reports/phase14/final_report.md`) is written by hand. To regenerate the PDF from it:

```bash
python src/generate_updated_final_pdf.py
```

`src/generate_updated_final_presentation.py` **appends** the Phase 15–17 slides to the existing deck. Run it only once on a deck that does not already contain them; running it twice duplicates those slides. `src/phase14_final_deliverables.py` is the original generator and is no longer part of the run order, because it would overwrite the updated report with older text.

Several small helper scripts in `src/` (for example `investigate_*.py`, `verify_*.py`, `audit_duplicates.py`, `eda.py`) are one-off investigations whose results are already saved in `reports/eda/` and `reports/phase1–3/`. They are not needed to reproduce the results.

---

## 5. Data and cleaning decisions

The workbook has three tables: Customer Profile, Monthly Activity and Churn Label.

| Issue found | Treatment |
|---|---|
| Exact duplicate rows (8 profile, 58 activity, 5 label) | Removed |
| Inconsistent text casing / whitespace | Standardised |
| 2 customer-month groups with conflicting values (customer 502065, Apr 2023; customer 501780, May 2024) | Not guessed; documented and handled explicitly before aggregation |
| 24 negative-spend rows | Kept and flagged; not made positive without evidence |
| 2,350 zero-spend rows, 20 with positive transactions | Kept and flagged; reviewed in `reports/phase2/` |
| Spend outliers (IQR method) | Reviewed in `reports/phase3/`; not deleted |
| 260 missing profile cells (age, gender, store price tier) | Imputed inside the model pipeline, fitted on training data only |
| 120 missing `LAST_PURCHASE_DATE` values | Left missing; the field is **not used as a feature** because it can reveal the label |
| 6 customers with no activity rows | Kept in the joined table; missing activity is not treated as zero transactions |
| 373 customers with insufficient history | Excluded from modelling |

Original churn labels were never altered.

---

## 6. Assumptions and method

- **Target:** `CHURNED = 1` means zero purchase transactions in April–June 2024 despite earlier purchases.
- **Leakage control:** every predictor is built only from months up to March 2024. April–June 2024 activity and `LAST_PURCHASE_DATE` are never used as predictors. `reports/phase11/phase11_leakage_audit_report.txt` documents the date checks.
- **Features:** recency, spend and transaction trends (early vs late period), rolling engagement totals, support-ticket totals and trend, promo usage, tenure, and demographic/tier attributes.
- **Validation (primary):** a time-ordered, signup-cohort split (`src/phase18_cohort_validation.py`). Models are trained on the earliest cohorts (2021Q1–2022Q4), the model and decision threshold are chosen on the next cohorts (2023Q1–Q2), and the latest cohorts (2023Q3–2024Q1) are used once for the final test. All preprocessing is fitted on the training cohort only. Compared: majority-class baseline, a simple recency rule, Logistic Regression, Random Forest and Gradient Boosting. Test intervals come from 1,000 bootstrap resamples.
- **Validation (secondary):** an earlier stratified random 80/20 split (seed 42, 446 held-out members; phases 7–13) and five repeated random splits (phase 17) are kept for comparison only.
- **Class imbalance:** about 30% positives overall (11% in the cohort-split test set). Logistic Regression and Random Forest use class weights; the chosen Gradient Boosting model is unweighted. The decision threshold for each model is tuned on the validation cohort (0.424 for the chosen model) rather than fixed at 0.50. Evaluation uses precision, recall, F1 and PR-AUC, with accuracy for reference.
- **Retention scenarios (primary, `src/phase19_retention_scenarios_cohort.py`):** run on the cohort test set. Five targeting strategies are compared against today's practice of contacting everyone: model top 20% (with and without already-lapsed members), proactive support outreach to still-active members with at least one support ticket in Nov 2023 – Mar 2024, a targeted coupon for model-flagged still-active members, and the combination of support outreach and model flag. "Still active" means the member bought within the last month of the feature window; already-lapsed members need a win-back offer, not a prevention offer. Impact is the assumed effectiveness (5–30%, the share of would-be churners saved) times the churners in each group. Contact cost is expressed as a share of the value of one retained member (2%, 5%, 10%), so no currency figures are invented. A model what-if (re-scoring with fewer support tickets) is included as an association, not a causal effect.
- **Retention scenarios (secondary):** the earlier random-split version (top 10/20/30% targeting, `reports/phase13/`) is kept for comparison.

---

## 7. Deliverables map

| Required deliverable | File |
|---|---|
| Reproducible Python project | `src/`, `requirements.txt`, this README |
| Data-quality and EDA notebook | `notebooks/`, `reports/phase1–3/` |
| Final customer-level modelling dataset | `data/processed/customer_features_pre_apr2024.csv` (also `customer_level_joined.csv`) |
| Model comparison and validation results | `reports/phase18/` (primary, cohort split); `reports/phase8/`, `reports/phase15/`, `reports/phase17/` (secondary) |
| Churn prediction file | `reports/phase18/phase18_test_predictions.csv` (primary; ID, probability, label, top-3 drivers). Random-split version: `reports/phase11/phase11_heldout_top3_customer_drivers.csv` |
| Driver attribution and feature impact | `reports/phase11/`, `reports/phase12/`, `reports/phase18/phase18_ablation.csv` |
| Retention scenario analysis | `reports/phase19/` (primary, cohort split, includes support-ticket outreach scenario); `reports/phase13/` (secondary) |
| Final report | `reports/phase14/final_report.pdf` (and `.md`) |
| Presentation | `presentation/final_presentation.pptx` |

---

## 8. Limitations

1. **Recency dominates the model.** Months since last observed activity is the strongest predictor (AUC about 0.92 on its own); removing it drops PR-AUC from about 0.94 to about 0.85. It is built only from data through March, so it describes members who were already lapsing, but its relationship to how the label was constructed should be confirmed with the dataset owner.
2. **No later calendar period to test on.** The data has a single outcome window (Apr–Jun 2024). The cohort split tests generalisation to newer members, which is the closest available substitute, but not to a later period. The test cohorts also differ from the training cohorts (shorter history, 11% vs 35% churn), so results carry a distribution shift and wide intervals (PR-AUC 0.53–0.77).
   Brand-new members (2024Q1 signups) are over-predicted (mean predicted risk 0.31 vs 0.8% actual); they need a separate rule or model.
3. **Label/activity gaps.** Five churned members have no earlier positive-transaction month in the extract, and 125 non-churned members have no positive purchases after the window starts. Labels were preserved as supplied.
4. **Probabilities are not well calibrated across cohorts.** Scores shift with the churn rate of the cohort, so use them to rank members, not as exact probabilities. (On the random split, mean predicted risk was 0.329 vs 0.298 observed.)
5. **SHAP and segment differences are associations**, not causal effects.
6. **Scenario results are illustrative.** Effectiveness is assumed (no campaign data exists), the targeted groups are small (for the focused group, the 95% interval on its churn rate is 25%–51%), and impact is counted retrospectively from known outcomes. A randomised pilot is needed before relying on these estimates.

---

## 9. Business recommendations

- Stop sending retention offers to everyone: on the cohort test, contacting all members loses value once a contact costs more than about 2% of a retained member's value.
- Contact members who are still active, have raised at least one support ticket and are flagged by the model (49 of 485 here, 37% churn vs 11% overall) with proactive support outreach, and a coupon if needed. Members who have already stopped buying need a separate win-back offer.
- Trigger outreach when a member's activity starts falling (fewer late-period transactions, longer gaps), rather than after they have gone quiet.
- Watch support tickets and falling email engagement as early-warning signals. Among active members, churn climbs from 3% with no tickets to 14% with one, 17% with two and 70% with three or more.
- Run a small randomised retention pilot to measure the real effect and cost before scaling.
- Treat very new members separately (for example, a welcome-journey rule) because their history is too short to score reliably.
- Retrain and re-validate on a later outcome window once one is available.
