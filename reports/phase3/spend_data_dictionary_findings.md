# Phase 3 — Spend Data Dictionary Findings

## Official field definitions

- `TOTAL_SPEND`: Total spend in the month, in USD. The Data Dictionary notes that a few negative/zero data-entry errors are present.
- `AVG_BASKET_VALUE`: Average spend per transaction.

## EDA findings

The spending analysis covered 28,015 monthly activity records.

- 24 records have negative total spend.
- 2,350 records have zero total spend.
- 20 records have zero total spend despite positive transaction counts.
- 667 records were flagged as potential outliers using the IQR method.
- 57 records have an absolute difference greater than USD 10 between recorded total spend and transactions multiplied by average basket value.

## Handling decision

The Data Dictionary confirms that some negative and zero spending errors are present. However, it does not identify all affected customer-month records or prescribe corrected values.

Therefore, original spending values will be retained in the cleaned source data. Suspicious records will be documented and handled using an explicit, reproducible rule before modelling. No spending value will be arbitrarily changed or deleted.

The consistency check is a screening method, not proof that every flagged record is erroneous, because the Data Dictionary does not explicitly guarantee that total spend must equal transactions multiplied by average basket value in every record.
