# Phase 2 — Data Cleaning and Validation Notes

## 1. Duplicate handling

Exact duplicate rows were removed from all three tables.

| Table | Rows after deduplication |
|---|---:|
| Customer Profile | 2,600 |
| Monthly Activity | 28,015 |
| Churn Label | 2,600 |

Validation confirmed that no exact duplicate rows remain in these cleaned files.

## 2. Negative and zero spending

The monthly activity table contains:
- 24 negative-spend rows.
- 2,350 zero-spend rows.
- 25,641 positive-spend rows.
- No missing TOTAL_SPEND values.

All 24 negative-spend rows have positive transaction counts and positive average basket values. The available diagnostics do not establish whether these values represent refunds or data-generation errors. Original values have therefore been retained and flagged for controlled handling during feature engineering.

Zero-spend rows require additional validation against transaction counts before being classified as anomalies.

## 3. Conflicting customer-month records

Two customer-month groups contain conflicting values:
- Customer 502065, April 2023.
- Customer 501780, May 2024.

These groups contain four rows in total. Neither record has been selected as the correct one. The conflicts must be handled explicitly before monthly aggregation to avoid double counting.

## 4. Missing values

The cleaned profile table contains 260 missing cells:
- AGE: 78.
- GENDER: 52.
- HOME_STORE_PRICE_TIER: 130.

The churn-label table contains 120 missing LAST_PURCHASE_DATE values. These values remain missing; no arbitrary imputation has been applied.

LAST_PURCHASE_DATE will not be used as a model feature because it may reveal information about the churn-label period.

## 5. Customer coverage

The profile and churn-label tables each contain 2,600 unique customers. The monthly activity table contains 2,594 unique customers, leaving six profile customers without activity records. These customers should remain represented in the joined dataset, with missing activity handled explicitly.

## 6. Cleaning policy

- Preserve the original Excel workbook.
- Remove exact duplicate rows.
- Standardize categorical text casing and whitespace.
- Retain and flag negative spending values; do not convert them to positive values without evidence.
- Investigate zero-spend records using transaction counts.
- Do not arbitrarily resolve conflicting customer-month records.
- Keep missing values documented and choose any imputation based on feature purpose and model requirements.
- Prevent information from April–June 2024 from entering pre-label feature calculations.

## 7. Remaining checks

Before closing Phase 2:
1. Validate zero-spend rows against transaction counts.
2. Confirm the handling policy for both conflicting customer-month groups.
3. Verify cleaned-table row counts, unique IDs and duplicate counts.
4. Confirm that all review reports are saved.

## 8. Limitations

The available data does not establish the cause of negative spending, the correct value for conflicting monthly records, or the meaning of every zero-spend record. These uncertainties must be disclosed in the final report.

## 9. Zero-spend and transaction consistency

The validation identified:
- 20 rows with zero TOTAL_SPEND and positive TRANSACTIONS.
- Zero rows with zero TRANSACTIONS and positive TOTAL_SPEND.
- Zero rows with negative TRANSACTIONS.

The 20 zero-spend rows are retained and flagged for review. Their cause is not established by the available data, so they will not be automatically deleted or altered.

## 10. Phase 2 closure policy

The cleaned source tables have no exact duplicate rows. Remaining data-quality issues are recorded in the review reports. Negative spend, zero-spend/positive-transaction records, conflicting customer-month groups, and missing values will be handled using explicit rules during feature engineering and documented in the final report.

The original Excel workbook remains unchanged. All cleaning and diagnostic outputs are reproducible from the Python scripts.
