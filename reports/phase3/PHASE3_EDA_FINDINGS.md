# Phase 3 — Exploratory Data Analysis and Data Quality Findings

## 1. Objective

This phase evaluates the quality, coverage, spending patterns, and customer churn distribution of the FreshBasket loyalty dataset before predictive modelling.

## 2. Dataset Overview

| Dataset | Cleaned rows | Unique customers |
|---|---:|---:|
| Customer Profile | 2,600 | 2,600 |
| Monthly Activity | 28,015 | 2,594 |
| Churn Label | 2,600 | 2,600 |

Monthly activity covers January 2023 through June 2024. The modelling feature window is January 2023 through March 2024, while April–June 2024 is reserved for the churn-label window.

## 3. Data Quality Findings

- Exact duplicate rows were removed during cleaning.
- Missing profile values include age, gender, home-store price tier, and last purchase date.
- Six profile customers have no monthly activity records.
- Twenty-four negative monthly spending records were identified.
- There are 2,350 zero-spend rows and 2,330 zero-transaction rows in the cleaned activity dataset.
- Twenty rows have zero spend despite positive transaction counts.
- No negative transaction counts were identified.
- Two conflicting customer-month groups remain in the source: customer 502065 in April 2023 and customer 501780 in May 2024.

The conflicting records were not silently corrected. During Phase 4 aggregation, both rows for customer 502065's April 2023 group were excluded from feature aggregation. The May 2024 conflict falls outside the predictive feature window.

### Churn-label and eligibility counts

The cleaned churn-label table contains 2,600 unique customers: 1,782 labelled `CHURNED = 0` and 818 labelled `CHURNED = 1`. Of these, 2,227 customers have sufficient history (`INSUFFICIENT_HISTORY_FLAG = 0`), while 373 have insufficient history (`INSUFFICIENT_HISTORY_FLAG = 1`). Segment and signup-cohort churn analyses use the 2,227 eligible customers with valid labels and sufficient history.

## 4. Spending Distribution and Outliers

Monthly total spending is right-skewed.

- Mean monthly spend: approximately $134.63.
- Median monthly spend: approximately $116.39.
- Minimum recorded spend: approximately −$249.21.
- Maximum recorded spend: approximately $3,731.42.
- IQR screening identified 667 records outside the calculated spending bounds.

The IQR method identifies potential outliers; it does not establish that every flagged value is incorrect. Spending inconsistencies should be interpreted alongside the data dictionary and transaction counts.

A separate consistency screening identified 57 records where total spend differed from transactions multiplied by average basket value by more than $10. This is a diagnostic flag, not proof of an error.

## 5. Monthly Activity Coverage

The number of customers with recorded monthly activity varies over time. A comparison of May and June 2024 found:

- May: 1,600 unique customers.
- June: 1,554 unique customers.
- Customers present in May but absent in June: 110.
- Customers present in both months: 1,490.
- Customers present only in June: 64.

The absence of a monthly record does not necessarily mean that the customer made zero purchases. Missing activity must not automatically be converted to zero without a documented rule.

## 6. Churn by Customer Segment

The following rates were calculated for 2,227 eligible customers with sufficient history and valid churn labels.

### Membership tier

| Tier | Customers | Churned | Churn rate |
|---|---:|---:|---:|
| Silver | 1,269 | 390 | 30.73% |
| Platinum | 260 | 77 | 29.62% |
| Gold | 698 | 195 | 27.94% |

### City

| City | Customers | Churned | Churn rate |
|---|---:|---:|---:|
| Austin | 222 | 80 | 36.04% |
| Columbus | 223 | 76 | 34.08% |
| Nashville | 207 | 63 | 30.43% |
| Raleigh | 255 | 77 | 30.20% |
| Boise | 209 | 62 | 29.67% |
| Denver | 226 | 65 | 28.76% |
| Tulsa | 253 | 72 | 28.46% |
| Madison | 220 | 61 | 27.73% |
| Portland | 205 | 54 | 26.34% |
| Spokane | 207 | 52 | 25.12% |

### Home-store price tier

| Price tier | Customers | Churned | Churn rate |
|---|---:|---:|---:|
| Value | 777 | 252 | 32.43% |
| Missing | 118 | 35 | 29.66% |
| Mainstream | 961 | 280 | 29.14% |
| Upscale | 371 | 95 | 25.61% |

These are descriptive comparisons. They do not establish that membership tier, city, or price tier causes churn. Statistical significance has not yet been tested.

## 7. Churn by Signup Cohort

Customers were grouped by signup quarter. The analysis included 2,227 eligible customers.

- Highest observed churn rate: 2022 Q2, at 43.37% (72 of 166 customers).
- Lowest observed churn rate: 2024 Q1, at 0.83% (1 of 121 customers).
- 2023 Q3 churn rate: 18.08%.
- 2023 Q4 churn rate: 11.76%.

The churn label is based on April–June 2024 activity. Newer cohorts may differ in history, eligibility, or exposure. These differences are descriptive and should not be interpreted as causal evidence.

## 8. Phase 3 Conclusions

1. The dataset contains useful customer profile, activity, and churn-label information, but several data-quality anomalies require careful treatment.
2. Spending outliers and conflicting customer-month records should be documented and handled through explicit, reproducible rules.
3. Missing activity is not automatically equivalent to zero purchases.
4. Churn rates differ across the observed customer segments and signup cohorts, motivating further analysis of engagement and support patterns.
5. All predictive features must use information available through March 2024 to avoid leakage from the April–June 2024 label window.

## 9. Outputs and Limitations

Supporting outputs are stored under `reports/phase3/`, including the churn-by-segment, signup-cohort, spending-consistency, and monthly-coverage analyses.

The findings are exploratory. No causal conclusions, statistical significance claims, or predictive performance claims are made in this phase.
