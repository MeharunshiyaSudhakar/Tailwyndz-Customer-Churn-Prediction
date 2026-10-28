
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
SPLIT_FILE = ROOT / "reports/phase7/phase7_customer_split.csv"
PREDICTION_FILE = ROOT / "reports/phase9/phase9_heldout_churn_predictions.csv"
OUTPUT_DIR = ROOT / "reports/phase11"

TARGET = "CHURNED"
ACTUAL = "ACTUAL_CHURNED"
PROBABILITY = "PREDICTED_CHURN_PROBABILITY"

EXCLUDE_COLUMNS = {
    "CUSTOMER_ID",
    "CHURNED",
    "DATA_SPLIT",
    "SIGNUP_COHORT",
    "INSUFFICIENT_HISTORY_FLAG",
    "NO_OBSERVED_FEATURE_ACTIVITY",
    "SIGNED_UP_AFTER_FEATURE_CUTOFF",
    "SIGNUP_DATE",
    "FIRST_FEATURE_MONTH",
    "LAST_FEATURE_MONTH",
    "OBSERVATION_END_DATE",
}

SEGMENT_COLUMNS = [
    "MEMBERSHIP_TIER",
    "CITY",
    "HOME_STORE_PRICE_TIER",
]


def require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURE_FILE)
    split = pd.read_csv(SPLIT_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    require_columns(
        features,
        ["CUSTOMER_ID", TARGET] + SEGMENT_COLUMNS,
        "Feature dataset",
    )
    require_columns(split, ["CUSTOMER_ID", "DATA_SPLIT"], "Split file")
    require_columns(
        predictions,
        ["CUSTOMER_ID", ACTUAL, PROBABILITY],
        "Phase 9 predictions",
    )

    for name, df in [
        ("features", features),
        ("split", split),
        ("predictions", predictions),
    ]:
        if df["CUSTOMER_ID"].duplicated().any():
            raise ValueError(f"Duplicate CUSTOMER_ID values in {name}.")

    # Join split assignment without using the target to construct predictors.
    data = features.merge(
        split[["CUSTOMER_ID", "DATA_SPLIT"]],
        on="CUSTOMER_ID",
        how="inner",
        validate="one_to_one",
    )

    # Same eligibility rule used in the earlier modeling phases.
    eligible = data[
        (data["INSUFFICIENT_HISTORY_FLAG"] == 0)
        & data[TARGET].isin([0, 1])
    ].copy()

    train = eligible[eligible["DATA_SPLIT"] == "train"].copy()
    test_predictions = predictions.merge(
        features[["CUSTOMER_ID"] + SEGMENT_COLUMNS],
        on="CUSTOMER_ID",
        how="left",
        validate="one_to_one",
    )

    if train.empty:
        raise ValueError("No eligible training customers found.")

    if test_predictions.empty:
        raise ValueError("No held-out predictions found.")

    # Exclude identifiers, targets, split metadata, and eligibility metadata.
    predictors = [
        c for c in features.columns
        if c not in EXCLUDE_COLUMNS
        and c not in SEGMENT_COLUMNS
        and c not in {"CHURNED"}
    ]

    # Segmentation columns are excluded from the model here to avoid
    # accidentally including them as predictors in this importance analysis.
    # They remain available for descriptive segment-risk reporting.
    numeric_cols = train[predictors].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_cols = [
        c for c in predictors if c not in numeric_cols
    ]

    # Guard against empty or unusable feature sets.
    if not predictors:
        raise ValueError("No predictor columns remain after exclusions.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight=None,
            random_state=42,
        )),
    ])

    X_train = train[predictors].replace([np.inf, -np.inf], np.nan)
    y_train = train[TARGET].astype(int)

    # Feature importance is fitted only on training customers.
    model.fit(X_train, y_train)

    transformed_names = (
        model.named_steps["preprocessor"].get_feature_names_out()
    )
    coefficients = model.named_steps["classifier"].coef_[0]

    importance = pd.DataFrame({
        "TRANSFORMED_FEATURE": transformed_names,
        "LOGISTIC_COEFFICIENT": coefficients,
        "ABS_COEFFICIENT": np.abs(coefficients),
        "DIRECTION": np.where(
            coefficients > 0,
            "Higher predicted churn odds",
            np.where(
                coefficients < 0,
                "Lower predicted churn odds",
                "No linear direction",
            ),
        ),
    }).sort_values("ABS_COEFFICIENT", ascending=False)

    importance.insert(0, "RANK", range(1, len(importance) + 1))

    # Descriptive held-out segment summaries.
    # Rates are calculated only where actual labels are present.
    test_predictions[ACTUAL] = pd.to_numeric(
        test_predictions[ACTUAL], errors="raise"
    )
    test_predictions[PROBABILITY] = pd.to_numeric(
        test_predictions[PROBABILITY], errors="raise"
    )

    segment_tables = []
    for segment in SEGMENT_COLUMNS:
        temp = test_predictions.copy()
        temp[segment] = temp[segment].fillna("Missing").astype(str)

        summary = temp.groupby(segment, dropna=False).agg(
            CUSTOMERS=("CUSTOMER_ID", "nunique"),
            ACTUAL_CHURNED=(ACTUAL, "sum"),
            ACTUAL_CHURN_RATE=(ACTUAL, "mean"),
            MEAN_PREDICTED_CHURN_PROBABILITY=(PROBABILITY, "mean"),
            MEDIAN_PREDICTED_CHURN_PROBABILITY=(PROBABILITY, "median"),
        ).reset_index()

        summary = summary.rename(columns={segment: "SEGMENT_VALUE"})
        summary.insert(0, "SEGMENT", segment)
        segment_tables.append(summary)

    segment_risk = pd.concat(segment_tables, ignore_index=True)

    # Signup cohort analysis using signup dates, not target labels.
    signup = features[["CUSTOMER_ID", "SIGNUP_DATE"]].copy()
    signup["SIGNUP_DATE"] = pd.to_datetime(
        signup["SIGNUP_DATE"], errors="coerce"
    )
    signup["SIGNUP_COHORT"] = (
        signup["SIGNUP_DATE"].dt.to_period("Q").astype("string")
    )

    cohort_data = test_predictions.merge(
        signup[["CUSTOMER_ID", "SIGNUP_COHORT"]],
        on="CUSTOMER_ID",
        how="left",
        validate="one_to_one",
    )

    cohort_summary = cohort_data.groupby(
        "SIGNUP_COHORT", dropna=False
    ).agg(
        CUSTOMERS=("CUSTOMER_ID", "nunique"),
        ACTUAL_CHURNED=(ACTUAL, "sum"),
        ACTUAL_CHURN_RATE=(ACTUAL, "mean"),
        MEAN_PREDICTED_CHURN_PROBABILITY=(PROBABILITY, "mean"),
    ).reset_index()

    # Basic training-only univariate numeric AUC screen.
    # This is a leakage investigation aid, not proof of leakage.
    auc_rows = []
    for col in numeric_cols:
        values = pd.to_numeric(train[col], errors="coerce")
        valid = values.notna() & train[TARGET].notna()
        y = train.loc[valid, TARGET].astype(int)
        x = values.loc[valid]

        if len(y) < 20 or y.nunique() < 2 or x.nunique() < 2:
            continue

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y, x)
        auc_rows.append({
            "FEATURE": col,
            "NONMISSING_TRAINING_ROWS": len(y),
            "UNIVARIATE_ROC_AUC": auc,
            "DISTANCE_FROM_RANDOM": abs(auc - 0.5),
        })

    auc_screen = pd.DataFrame(auc_rows)
    if not auc_screen.empty:
        auc_screen = auc_screen.sort_values(
            "DISTANCE_FROM_RANDOM", ascending=False
        )

    importance_file = OUTPUT_DIR / "phase11_feature_importance.csv"
    segment_file = OUTPUT_DIR / "phase11_segment_risk.csv"
    cohort_file = OUTPUT_DIR / "phase11_signup_cohort_risk.csv"
    auc_file = OUTPUT_DIR / "phase11_univariate_auc_screen.csv"
    report_file = OUTPUT_DIR / "phase11_explainability_report.txt"

    importance.to_csv(importance_file, index=False)
    segment_risk.to_csv(segment_file, index=False)
    cohort_summary.to_csv(cohort_file, index=False)
    auc_screen.to_csv(auc_file, index=False)

    lines = [
        "PHASE 11 — EXPLAINABILITY AND SEGMENT-RISK ANALYSIS",
        "=" * 65,
        f"Eligible training customers: {len(train)}",
        f"Predictor columns before encoding: {len(predictors)}",
        f"Numeric predictors: {len(numeric_cols)}",
        f"Categorical predictors: {len(categorical_cols)}",
        f"Held-out customers summarized: {len(test_predictions)}",
        "",
        "TOP 20 LOGISTIC REGRESSION COEFFICIENTS",
        "-" * 65,
        "Positive coefficients indicate higher predicted churn odds;",
        "negative coefficients indicate lower predicted churn odds,",
        "conditional on the other encoded model features.",
        "Absolute coefficient magnitude is not a causal effect and is",
        "not directly comparable across every feature type.",
        "",
    ]

    for _, row in importance.head(20).iterrows():
        lines.append(
            f"{row['RANK']:>2}. {row['TRANSFORMED_FEATURE']}: "
            f"coef={row['LOGISTIC_COEFFICIENT']:.6f}; "
            f"abs={row['ABS_COEFFICIENT']:.6f}; "
            f"{row['DIRECTION']}"
        )

    lines.extend([
        "",
        "HELD-OUT SEGMENT RISK",
        "-" * 65,
        "Actual churn rates are descriptive results for this random held-out",
        "sample; they do not establish future-period performance or causation.",
        "",
    ])

    for _, row in segment_risk.iterrows():
        lines.append(
            f"{row['SEGMENT']}={row['SEGMENT_VALUE']}: "
            f"n={int(row['CUSTOMERS'])}, "
            f"actual churn={row['ACTUAL_CHURN_RATE']:.4%}, "
            f"mean predicted risk="
            f"{row['MEAN_PREDICTED_CHURN_PROBABILITY']:.4f}"
        )

    lines.extend([
        "",
        "SIGNUP COHORT RISK",
        "-" * 65,
    ])

    for _, row in cohort_summary.iterrows():
        lines.append(
            f"Cohort {row['SIGNUP_COHORT']}: "
            f"n={int(row['CUSTOMERS'])}, "
            f"actual churn={row['ACTUAL_CHURN_RATE']:.4%}, "
            f"mean predicted risk="
            f"{row['MEAN_PREDICTED_CHURN_PROBABILITY']:.4f}"
        )

    lines.extend([
        "",
        "UNIVARIATE AUC LEAKAGE SCREEN",
        "-" * 65,
        "Features with very high univariate AUC need investigation, but",
        "high AUC alone does not prove target leakage.",
    ])

    if auc_screen.empty:
        lines.append("No eligible numeric feature could be screened.")
    else:
        for _, row in auc_screen.head(15).iterrows():
            lines.append(
                f"{row['FEATURE']}: AUC={row['UNIVARIATE_ROC_AUC']:.4f}, "
                f"rows={int(row['NONMISSING_TRAINING_ROWS'])}"
            )

    lines.extend([
        "",
        "IMPORTANT LIMITATIONS",
        "-" * 65,
        "1. Importance uses a logistic regression fitted to training customers.",
        "2. Segment summaries use the saved held-out predictions and labels.",
        "3. The holdout is a random customer split, not a true future-time holdout.",
        "4. Segment differences are associations, not causal explanations.",
        "5. Check suspicious predictors and label-generation rules before deployment.",
        "6. Coefficients are not SHAP values.",
        "",
        "OUTPUT FILES",
        str(importance_file),
        str(segment_file),
        str(cohort_file),
        str(auc_file),
        str(report_file),
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nPhase 11 completed.")


if __name__ == "__main__":
    main()
