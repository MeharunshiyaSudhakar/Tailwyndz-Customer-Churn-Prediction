
from pathlib import Path
import sys

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
EXCLUDE = {
    "CUSTOMER_ID", "CHURNED", "DATA_SPLIT", "SIGNUP_COHORT",
    "INSUFFICIENT_HISTORY_FLAG", "NO_OBSERVED_FEATURE_ACTIVITY",
    "SIGNED_UP_AFTER_FEATURE_CUTOFF", "SIGNUP_DATE",
    "FIRST_FEATURE_MONTH", "LAST_FEATURE_MONTH", "OBSERVATION_END_DATE",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        print("SHAP is not installed in the current Python environment.")
        print("Install it with: python -m pip install shap")
        print("Then rerun: python src\\phase11_shap_analysis.py")
        sys.exit(2)

    features = pd.read_csv(FEATURE_FILE)
    split = pd.read_csv(SPLIT_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    for name, frame in [
        ("features", features),
        ("split", split),
        ("predictions", predictions),
    ]:
        if frame["CUSTOMER_ID"].duplicated().any():
            raise ValueError(f"Duplicate customer IDs in {name}.")

    data = features.merge(
        split[["CUSTOMER_ID", "DATA_SPLIT"]],
        on="CUSTOMER_ID",
        how="inner",
        validate="one_to_one",
    )

    eligible = data[
        (data["INSUFFICIENT_HISTORY_FLAG"] == 0)
        & data[TARGET].isin([0, 1])
    ].copy()

    train = eligible[eligible["DATA_SPLIT"] == "train"].copy()
    holdout_ids = predictions["CUSTOMER_ID"].tolist()

    predictors = [c for c in features.columns if c not in EXCLUDE]
    numeric = train[predictors].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical = [c for c in predictors if c not in numeric]

    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])

    X_train = train[predictors].replace([np.inf, -np.inf], np.nan)
    model.fit(X_train, train[TARGET].astype(int))

    # Explain a bounded, reproducible sample of the held-out feature rows.
    holdout = eligible[
        eligible["CUSTOMER_ID"].isin(holdout_ids)
    ].copy()

    if len(holdout) != len(holdout_ids):
        raise ValueError(
            "Some held-out prediction IDs did not match eligible feature rows."
        )

    X_holdout = holdout[predictors].replace(
        [np.inf, -np.inf], np.nan
    )

    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_classifier = model.named_steps["classifier"]

    X_train_transformed = fitted_preprocessor.transform(X_train)
    X_holdout_transformed = fitted_preprocessor.transform(X_holdout)

    transformed_names = fitted_preprocessor.get_feature_names_out()

    # LinearExplainer explains the fitted logistic model on transformed data.
    background = X_train_transformed[:min(100, len(train))]
    explainer = shap.LinearExplainer(fitted_classifier, background)
    shap_values = explainer(X_holdout_transformed)

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, -1]

    if values.shape != (len(holdout), len(transformed_names)):
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected "
            f"({len(holdout)}, {len(transformed_names)})."
        )

    long_rows = []
    for row_idx, customer_id in enumerate(holdout["CUSTOMER_ID"]):
        for col_idx, feature_name in enumerate(transformed_names):
            long_rows.append({
                "CUSTOMER_ID": customer_id,
                "TRANSFORMED_FEATURE": feature_name,
                "SHAP_VALUE": float(values[row_idx, col_idx]),
                "ABS_SHAP_VALUE": float(abs(values[row_idx, col_idx])),
            })

    shap_detail = pd.DataFrame(long_rows)
    shap_detail_file = OUTPUT_DIR / "phase11_shap_customer_feature_values.csv"
    shap_detail.to_csv(shap_detail_file, index=False)

    summary = shap_detail.groupby(
        "TRANSFORMED_FEATURE", as_index=False
    ).agg(
        MEAN_ABSOLUTE_SHAP=("ABS_SHAP_VALUE", "mean"),
        MEAN_SHAP=("SHAP_VALUE", "mean"),
        CUSTOMERS_EXPLAINED=("CUSTOMER_ID", "nunique"),
    ).sort_values("MEAN_ABSOLUTE_SHAP", ascending=False)

    summary.insert(0, "RANK", range(1, len(summary) + 1))
    summary_file = OUTPUT_DIR / "phase11_shap_summary.csv"
    summary.to_csv(summary_file, index=False)

    report_file = OUTPUT_DIR / "phase11_shap_report.txt"
    lines = [
        "PHASE 11 — SHAP EXPLANATIONS",
        "=" * 60,
        f"Training customers used to fit model: {len(train)}",
        f"Held-out customers explained: {len(holdout)}",
        f"Raw predictors: {len(predictors)}",
        f"Transformed predictors: {len(transformed_names)}",
        "SHAP values explain the fitted model's output in transformed feature space.",
        "They are not causal effects and do not prove the model is leakage-free.",
        "This is a refitted logistic model on the same training split, not a",
        "serialized copy of the original Phase 9 model.",
        "",
        "TOP 20 FEATURES BY MEAN ABSOLUTE SHAP",
        "-" * 60,
    ]

    for _, row in summary.head(20).iterrows():
        lines.append(
            f"{int(row['RANK'])}. {row['TRANSFORMED_FEATURE']}: "
            f"mean_abs_shap={row['MEAN_ABSOLUTE_SHAP']:.6f}; "
            f"mean_shap={row['MEAN_SHAP']:.6f}"
        )

    lines.extend([
        "",
        "OUTPUT FILES",
        str(shap_detail_file),
        str(summary_file),
        str(report_file),
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nSHAP analysis completed.")


if __name__ == "__main__":
    main()
