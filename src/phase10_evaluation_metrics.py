
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "reports/phase9/phase9_heldout_churn_predictions.csv"
OUTPUT_DIR = ROOT / "reports/phase10"

TARGET = "ACTUAL_CHURNED"
PREDICTED = "PREDICTED_CHURNED"
PROBABILITY = "PREDICTED_CHURN_PROBABILITY"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Phase 9 prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "CUSTOMER_ID",
        TARGET,
        PREDICTED,
        PROBABILITY,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Prediction file is empty.")

    if df["CUSTOMER_ID"].duplicated().any():
        raise ValueError("Duplicate customer IDs found in predictions.")

    for col in [TARGET, PREDICTED, PROBABILITY]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[[TARGET, PREDICTED, PROBABILITY]].isna().any().any():
        raise ValueError("Missing actual labels or predictions found.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Actual labels must contain only 0 and 1.")

    if not set(df[PREDICTED].unique()).issubset({0, 1}):
        raise ValueError("Predicted labels must contain only 0 and 1.")

    if ((df[PROBABILITY] < 0) | (df[PROBABILITY] > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    y_true = df[TARGET].astype(int)
    y_pred = df[PREDICTED].astype(int)
    y_prob = df[PROBABILITY].astype(float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    roc_auc = (
        roc_auc_score(y_true, y_prob)
        if y_true.nunique() == 2 else np.nan
    )
    pr_auc = (
        average_precision_score(y_true, y_prob)
        if y_true.nunique() == 2 else np.nan
    )

    actual_churn_rate = y_true.mean()
    predicted_churn_rate = y_pred.mean()
    mean_probability = y_prob.mean()

    # Majority-class baseline: predict the most common actual class.
    majority_class = int(y_true.value_counts().idxmax())
    majority_predictions = np.full(len(y_true), majority_class)
    majority_accuracy = accuracy_score(y_true, majority_predictions)
    majority_recall = recall_score(
        y_true, majority_predictions, zero_division=0
    )
    majority_precision = precision_score(
        y_true, majority_predictions, zero_division=0
    )
    majority_f1 = f1_score(
        y_true, majority_predictions, zero_division=0
    )

    metrics = pd.DataFrame([
        {"METRIC": "Customers evaluated", "VALUE": len(df)},
        {"METRIC": "Actual churned", "VALUE": int((y_true == 1).sum())},
        {"METRIC": "Actual not churned", "VALUE": int((y_true == 0).sum())},
        {"METRIC": "Predicted churned", "VALUE": int((y_pred == 1).sum())},
        {"METRIC": "Predicted not churned", "VALUE": int((y_pred == 0).sum())},
        {"METRIC": "Actual churn rate", "VALUE": actual_churn_rate},
        {"METRIC": "Predicted churn rate", "VALUE": predicted_churn_rate},
        {"METRIC": "Mean predicted churn probability", "VALUE": mean_probability},
        {"METRIC": "Accuracy", "VALUE": accuracy},
        {"METRIC": "Precision", "VALUE": precision},
        {"METRIC": "Recall / sensitivity", "VALUE": recall},
        {"METRIC": "Specificity", "VALUE": specificity},
        {"METRIC": "F1-score", "VALUE": f1},
        {"METRIC": "Balanced accuracy", "VALUE": balanced_accuracy},
        {"METRIC": "ROC-AUC", "VALUE": roc_auc},
        {"METRIC": "PR-AUC / average precision", "VALUE": pr_auc},
        {"METRIC": "Matthews correlation coefficient", "VALUE": mcc},
        {"METRIC": "Majority baseline accuracy", "VALUE": majority_accuracy},
        {"METRIC": "Majority baseline precision", "VALUE": majority_precision},
        {"METRIC": "Majority baseline recall", "VALUE": majority_recall},
        {"METRIC": "Majority baseline F1-score", "VALUE": majority_f1},
    ])

    confusion = pd.DataFrame(
        [
            {"ACTUAL": "Not churned (0)", "PREDICTED_NOT_CHURNED": tn,
             "PREDICTED_CHURNED": fp},
            {"ACTUAL": "Churned (1)", "PREDICTED_NOT_CHURNED": fn,
             "PREDICTED_CHURNED": tp},
        ]
    )

    metrics_file = OUTPUT_DIR / "phase10_metrics.csv"
    confusion_file = OUTPUT_DIR / "phase10_confusion_matrix.csv"
    report_file = OUTPUT_DIR / "phase10_evaluation_report.txt"

    metrics.to_csv(metrics_file, index=False)
    confusion.to_csv(confusion_file, index=False)

    lines = [
        "CORE TASK 8 — HELD-OUT EVALUATION AND CLASS IMBALANCE",
        "=" * 60,
        f"Input predictions: {INPUT_FILE}",
        f"Customers evaluated: {len(df)}",
        "Evaluation uses the saved Phase 9 predictions; no model was retrained.",
        "",
        "CONFUSION MATRIX",
        "-" * 60,
        "Rows = actual class; columns = predicted class",
        f"True negatives (TN): {tn}",
        f"False positives (FP): {fp}",
        f"False negatives (FN): {fn}",
        f"True positives (TP): {tp}",
        "",
        "METRICS",
        "-" * 60,
        f"Actual churn rate: {actual_churn_rate:.4%}",
        f"Predicted churn rate: {predicted_churn_rate:.4%}",
        f"Mean predicted probability: {mean_probability:.6f}",
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall / sensitivity: {recall:.4f}",
        f"Specificity: {specificity:.4f}",
        f"F1-score: {f1:.4f}",
        f"Balanced accuracy: {balanced_accuracy:.4f}",
        f"ROC-AUC: {roc_auc:.4f}",
        f"PR-AUC / average precision: {pr_auc:.4f}",
        f"Matthews correlation coefficient: {mcc:.4f}",
        "",
        "MAJORITY-CLASS BASELINE",
        "-" * 60,
        f"Majority class in this evaluation set: {majority_class}",
        f"Baseline accuracy: {majority_accuracy:.4f}",
        f"Baseline precision: {majority_precision:.4f}",
        f"Baseline recall: {majority_recall:.4f}",
        f"Baseline F1-score: {majority_f1:.4f}",
        "",
        "INTERPRETATION NOTES",
        "- Accuracy must be interpreted alongside recall, precision and PR-AUC.",
        "- False negatives are churned customers the model missed.",
        "- False positives are non-churned customers flagged as churned.",
        "- Average precision summarizes precision-recall ranking performance.",
        "- This evaluates one held-out set; it does not prove future-period performance.",
        "- High scores should prompt a leakage audit before business deployment.",
        "- Probabilities have not been checked for calibration in this phase.",
        "",
        "OUTPUT FILES",
        str(metrics_file),
        str(confusion_file),
        str(report_file),
    ]

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nPhase 10 evaluation completed.")


if __name__ == "__main__":
    main()
