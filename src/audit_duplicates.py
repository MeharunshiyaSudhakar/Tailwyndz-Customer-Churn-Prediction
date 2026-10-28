
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def load_table(sheet_name, required_header):
    """Find the actual header row and load the table safely."""
    raw = pd.read_excel(
        WORKBOOK,
        sheet_name=sheet_name,
        header=None
    )

    for i in range(len(raw)):
        values = raw.iloc[i].astype(str).str.strip().str.upper().tolist()
        if required_header.upper() in values:
            df = raw.iloc[i + 1:].copy()
            df.columns = raw.iloc[i].astype(str).str.strip()
            df = df.reset_index(drop=True)

            # Remove completely empty rows and columns.
            df = df.dropna(how="all").dropna(axis=1, how="all")

            # Convert numeric-looking customer IDs consistently.
            if "CUSTOMER_ID" in df.columns:
                df["CUSTOMER_ID"] = pd.to_numeric(
                    df["CUSTOMER_ID"], errors="coerce"
                ).astype("Int64")

            return df

    raise ValueError(
        f"Could not find header '{required_header}' in {sheet_name}"
    )


sheets = {
    "profile": (
        "fb Customer Profile",
        "CUSTOMER_ID",
        ["CUSTOMER_ID"],
    ),
    "activity": (
        "fb Monthly Activity",
        "CUSTOMER_ID",
        ["CUSTOMER_ID", "MONTH"],
    ),
    "churn": (
        "fb Churn Label",
        "CUSTOMER_ID",
        ["CUSTOMER_ID"],
    ),
}

for name, (sheet_name, header, keys) in sheets.items():
    df = load_table(sheet_name, header)

    # Check that the expected key columns exist.
    missing_keys = [col for col in keys if col not in df.columns]
    if missing_keys:
        raise ValueError(
            f"{sheet_name}: missing key columns {missing_keys}. "
            f"Available columns: {df.columns.tolist()}"
        )

    duplicated = df[df.duplicated(subset=keys, keep=False)].copy()
    duplicated.to_csv(
        OUT / f"{name}_duplicate_key_rows.csv", index=False
    )

    conflict_groups = []
    for _, group in duplicated.groupby(keys, dropna=False):
        non_key_columns = [c for c in df.columns if c not in keys]

        if not group[non_key_columns].drop_duplicates().empty:
            if len(group[non_key_columns].drop_duplicates()) > 1:
                conflict_groups.append(group)

    conflicts = (
        pd.concat(conflict_groups, ignore_index=True)
        if conflict_groups
        else pd.DataFrame(columns=df.columns)
    )

    conflicts.to_csv(
        OUT / f"{name}_conflicting_duplicate_rows.csv",
        index=False,
    )

    print(f"\n{'=' * 55}")
    print(f"{name.upper()} — {sheet_name}")
    print(f"Rows in source: {len(df)}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Rows sharing duplicate keys: {len(duplicated)}")
    print(f"Duplicate key groups: {len(conflict_groups) + sum(
        1 for _, g in duplicated.groupby(keys, dropna=False)
        if len(g) > 1
    ) - len(conflict_groups)}")
    print(f"Conflicting duplicate groups: {len(conflict_groups)}")
    print(f"Audit files saved under: {OUT}")

print("\nAudit completed. Original workbook was not modified.")
