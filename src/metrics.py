import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, fbeta_score, average_precision_score,
    confusion_matrix, classification_report
)

def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_proba)),
    }

def threshold_sweep(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        rows.append({
            "threshold": round(float(t), 3),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        })
    return rows

def print_report(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
