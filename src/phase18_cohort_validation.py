"""PRIMARY VALIDATION: time-ordered, signup-cohort-based train / validation / test.

Why: the assessment asks for a time- or cohort-based split rather than a random
one. Here members are split by signup quarter, so the model is always trained on
earlier cohorts and evaluated on later ones.

    Train       : signups 2021Q1 - 2022Q4   (fit models)
    Validation  : signups 2023Q1 - 2023Q2   (choose model and threshold)
    Final test  : signups 2023Q3 - 2024Q1   (used once, for reporting)

Only members with sufficient history are used. Predictors are the same
leakage-safe features (data up to 31 Mar 2024) used in earlier phases.
All imputers / scalers / encoders are fitted on the training cohort only.

Outputs (reports/phase18/):
    phase18_model_comparison.csv      validation + test metrics for every model
    phase18_test_predictions.csv      test cohort: ID, probability, label, top-3 drivers
    phase18_ablation.csv              feature-group ablation on the cohort split
    phase18_segment_risk_test.csv     test risk by tier / city
    phase18_test_bootstrap_ci.csv     95% bootstrap intervals for the chosen model
    phase18_cohort_validation_report.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data/processed/customer_features_pre_apr2024.csv"
OUT = ROOT / "reports/phase18"
SEED = 42
TARGET = "CHURNED"

TRAIN_COHORTS = [f"{y}Q{q}" for y in (2021, 2022) for q in (1, 2, 3, 4)]
VAL_COHORTS = ["2023Q1", "2023Q2"]
TEST_COHORTS = ["2023Q3", "2023Q4", "2024Q1"]

EXCLUDE = {
    "CUSTOMER_ID", "CHURNED", "INSUFFICIENT_HISTORY_FLAG",
    "NO_OBSERVED_FEATURE_ACTIVITY", "SIGNED_UP_AFTER_FEATURE_CUTOFF",
    "SIGNUP_DATE", "FIRST_FEATURE_MONTH", "LAST_FEATURE_MONTH",
    "OBSERVATION_END_DATE", "SIGNUP_COHORT", "DATA_SPLIT",
}
PROFILE = ["AGE", "GENDER", "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN",
           "PREFERRED_STORE_ID", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER",
           "TENURE_DAYS_AT_CUTOFF"]
RECENCY = ["MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"]
LATE_TREND = [
    "LATE_PERIOD_SPEND", "LATE_PERIOD_TRANSACTIONS", "LATE_PERIOD_APP_SESSIONS",
    "LATE_PERIOD_SUPPORT_TICKETS", "SPEND_TREND_LATE_MINUS_EARLY",
    "TRANSACTION_TREND_LATE_MINUS_EARLY", "APP_TREND_LATE_MINUS_EARLY",
    "SUPPORT_TREND_LATE_MINUS_EARLY",
]


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ACCURACY": accuracy_score(y, pred),
        "BALANCED_ACCURACY": balanced_accuracy_score(y, pred),
        "PRECISION": precision_score(y, pred, zero_division=0),
        "RECALL": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, p) if len(set(p)) > 1 else 0.5,
        "PR_AUC": average_precision_score(y, p),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def make_preprocessor(X):
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer(
        [("numeric", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                               ("sc", StandardScaler())]), num),
         ("categorical", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                   ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat)],
        remainder="drop", sparse_threshold=0)


def estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                        max_depth=2, random_state=SEED),
    }


def fit(estimator, X_tr, y_tr):
    pipe = Pipeline([("prep", make_preprocessor(X_tr)), ("model", estimator)])
    pipe.fit(X_tr, y_tr)
    return pipe


def top3_drivers(pipe, X_train, X_test, name):
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    Xt_train, Xt_test = prep.transform(X_train), prep.transform(X_test)
    if name == "Logistic Regression":
        values = shap.LinearExplainer(model, Xt_train[:100])(Xt_test).values
    else:
        values = shap.TreeExplainer(model).shap_values(Xt_test)
        if isinstance(values, list):
            values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    order = np.argsort(-np.abs(values), axis=1)[:, :3]
    return [[names[j] for j in row] for row in order]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_FILE, parse_dates=["SIGNUP_DATE"])
    df = df[df["INSUFFICIENT_HISTORY_FLAG"] == 0].copy()
    df["COHORT"] = df["SIGNUP_DATE"].dt.to_period("Q").astype(str)

    train = df[df.COHORT.isin(TRAIN_COHORTS)]
    val = df[df.COHORT.isin(VAL_COHORTS)]
    test = df[df.COHORT.isin(TEST_COHORTS)]
    assert len(train) + len(val) + len(test) == len(df), "cohort lists do not cover all members"
    assert train.SIGNUP_DATE.max() < val.SIGNUP_DATE.min() <= val.SIGNUP_DATE.max() < test.SIGNUP_DATE.min()

    cols = [c for c in df.columns if c not in EXCLUDE | {"COHORT"}]
    X = {k: v[cols] for k, v in (("train", train), ("val", val), ("test", test))}
    y = {"train": train[TARGET].values, "val": val[TARGET].values, "test": test[TARGET].values}

    rows, fitted, val_prob, test_prob = [], {}, {}, {}

    # Baseline 1: majority class
    dummy = fit(DummyClassifier(strategy="most_frequent"), X["train"], y["train"])
    # Baseline 2: simple rule - flag members whose last observed activity is >= k months ago
    ks = sorted(train["MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"].dropna().unique())
    best_k = max(ks, key=lambda k: f1_score(y["train"], (train["MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"].fillna(0) >= k).astype(int)))
    rule = lambda d: (d["MONTHS_SINCE_LAST_OBSERVED_ACTIVITY"].fillna(0) >= best_k).astype(float).values
    for label, pv, pt in (("Dummy baseline", dummy.predict_proba(X["val"])[:, 1], dummy.predict_proba(X["test"])[:, 1]),
                          (f"Rule: months since last activity >= {int(best_k)}", rule(val), rule(test))):
        val_prob[label], test_prob[label] = pv, pt
    for name, est in estimators().items():
        pipe = fit(est, X["train"], y["train"])
        fitted[name] = pipe
        val_prob[name] = pipe.predict_proba(X["val"])[:, 1]
        test_prob[name] = pipe.predict_proba(X["test"])[:, 1]

    for name in val_prob:
        thr_default = 0.5
        thr_tuned = best_f1_threshold(y["val"], val_prob[name]) if name in fitted else 0.5
        for split, ys, ps in (("validation", y["val"], val_prob[name]), ("test", y["test"], test_prob[name])):
            for tname, thr in (("0.50", thr_default), ("val-tuned", thr_tuned)):
                if tname == "val-tuned" and name not in fitted:
                    continue
                rows.append({"MODEL": name, "SPLIT": split, "THRESHOLD_TYPE": tname,
                             "THRESHOLD": round(thr, 4), **metrics(ys, ps, thr)})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "phase18_model_comparison.csv", index=False)

    # Model choice uses VALIDATION PR-AUC only (test untouched).
    val_pr = {n: average_precision_score(y["val"], val_prob[n]) for n in fitted}
    chosen = max(val_pr, key=val_pr.get)
    thr = best_f1_threshold(y["val"], val_prob[chosen])
    pipe = fitted[chosen]

    drivers = top3_drivers(pipe, X["train"], X["test"], chosen)
    p_test = test_prob[chosen]
    pred = pd.DataFrame({
        "CUSTOMER_ID": test.CUSTOMER_ID.values, "SIGNUP_COHORT": test.COHORT.values,
        "ACTUAL_CHURNED": y["test"], "PREDICTED_CHURN_PROBABILITY": np.round(p_test, 6),
        "PREDICTED_CHURNED": (p_test >= thr).astype(int), "MODEL": chosen,
        "TOP_DRIVER_1": [d[0] for d in drivers], "TOP_DRIVER_2": [d[1] for d in drivers],
        "TOP_DRIVER_3": [d[2] for d in drivers],
    })
    pred.to_csv(OUT / "phase18_test_predictions.csv", index=False)

    # Bootstrap CI on the test cohort for the chosen model.
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(p_test), len(p_test))
        if y["test"][idx].sum() == 0:
            continue
        m = metrics(y["test"][idx], p_test[idx], thr)
        boots.append([m["PRECISION"], m["RECALL"], m["F1"], m["ROC_AUC"], m["PR_AUC"]])
    boots = np.array(boots)
    ci = pd.DataFrame({"METRIC": ["PRECISION", "RECALL", "F1", "ROC_AUC", "PR_AUC"],
                       "CI_LOW_2.5": np.percentile(boots, 2.5, axis=0),
                       "CI_HIGH_97.5": np.percentile(boots, 97.5, axis=0)})
    ci.to_csv(OUT / "phase18_test_bootstrap_ci.csv", index=False)

    # Ablation on the same cohort split (chosen model family, validation + test).
    groups = {
        "ALL_FEATURES": cols,
        "BEHAVIOUR_ONLY (no profile)": [c for c in cols if c not in PROFILE],
        "WITHOUT_RECENCY": [c for c in cols if c not in RECENCY],
        "WITHOUT_RECENCY_LATE_AND_TRENDS": [c for c in cols if c not in RECENCY + LATE_TREND],
        "PROFILE_ONLY (demographics)": [c for c in cols if c in PROFILE],
    }
    ab = []
    for gname, gcols in groups.items():
        p = fit(estimators()[chosen], X["train"][gcols], y["train"])
        pv, pt = p.predict_proba(X["val"][gcols])[:, 1], p.predict_proba(X["test"][gcols])[:, 1]
        t = best_f1_threshold(y["val"], pv)
        for split, ys, ps in (("validation", y["val"], pv), ("test", y["test"], pt)):
            ab.append({"FEATURE_SET": gname, "N_FEATURES": len(gcols), "SPLIT": split, **metrics(ys, ps, t if True else 0.5)})
    pd.DataFrame(ab).to_csv(OUT / "phase18_ablation.csv", index=False)

    # Segment risk on the test cohort.
    seg = []
    for col in ("MEMBERSHIP_TIER", "CITY"):
        g = test.assign(P=p_test).groupby(col).agg(MEMBERS=("P", "size"), ACTUAL_CHURN_RATE=(TARGET, "mean"),
                                                    MEAN_PREDICTED_RISK=("P", "mean")).reset_index()
        g.insert(0, "SEGMENT_TYPE", col); g = g.rename(columns={col: "SEGMENT"}); seg.append(g)
    pd.concat(seg).to_csv(OUT / "phase18_segment_risk_test.csv", index=False)

    # Report.
    f = lambda m: (f"F1={m['F1']:.4f} precision={m['PRECISION']:.4f} recall={m['RECALL']:.4f} "
                   f"ROC-AUC={m['ROC_AUC']:.4f} PR-AUC={m['PR_AUC']:.4f}")
    mt = metrics(y["test"], p_test, thr)
    lines = [
        "PHASE 18 - COHORT-BASED (TIME-ORDERED) PRIMARY VALIDATION", "=" * 64,
        f"Train      : signups {TRAIN_COHORTS[0]}-{TRAIN_COHORTS[-1]}  n={len(train)}  churn={y['train'].mean():.1%}",
        f"Validation : signups {VAL_COHORTS[0]}-{VAL_COHORTS[-1]}  n={len(val)}  churn={y['val'].mean():.1%}",
        f"Final test : signups {TEST_COHORTS[0]}-{TEST_COHORTS[-1]}  n={len(test)}  churn={y['test'].mean():.1%} ({int(y['test'].sum())} churners)",
        "", "Later cohorts have shorter tenure and a lower churn rate, so the test set is a",
        "harder, shifted population. PR-AUC should be read against the test prevalence",
        f"({y['test'].mean():.3f}), which is the score of a no-skill model.", "",
        "VALIDATION PR-AUC (model selection)", "-" * 64,
        *[f"{n}: {v:.4f}" for n, v in sorted(val_pr.items(), key=lambda kv: -kv[1])],
        "", f"Chosen model: {chosen}   |   threshold tuned on validation (max F1): {thr:.4f}", "",
        "FINAL TEST RESULTS (chosen model, validation-tuned threshold)", "-" * 64, f(mt),
        f"Confusion matrix: TN={mt['TN']} FP={mt['FP']} FN={mt['FN']} TP={mt['TP']}",
        "95% bootstrap intervals: " + "; ".join(f"{r.METRIC} {r['CI_LOW_2.5']:.3f}-{r['CI_HIGH_97.5']:.3f}" for _, r in ci.iterrows()), "",
        "All models on the test cohort (threshold 0.50 unless val-tuned)", "-" * 64,
    ]
    for _, r in comp[comp.SPLIT == "test"].iterrows():
        lines.append(f"{r.MODEL} [{r.THRESHOLD_TYPE}]: " + f(r))
    lines += ["", "NOTES",
              "- Test cohort used once, after model and threshold were fixed on validation.",
              "- Single outcome window (Apr-Jun 2024): this is a signup-cohort split, which tests",
              "  generalisation to newer members, not to a later calendar period.",
              "- Metrics are associations in this dataset; see README limitations."]
    (OUT / "phase18_cohort_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
