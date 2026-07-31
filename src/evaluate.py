import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from utils import PRIMARY_TARGET, CHOSEN_THRESHOLD, reduce_features
from metrics import compute_metrics, threshold_sweep, print_report

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
REPORTS_DIR = Path("../reports"); REPORTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR = Path("../outputs"); OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

def evaluate():
    model = joblib.load(MODELS_DIR / "xgboost.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    test_fe = pd.read_csv('/data/processed/test_fe.csv')
    test_fe = reduce_features(test_fe)

    X_test, y_test = test_fe[feature_cols], test_fe[PRIMARY_TARGET]
    test_proba = model.predict_proba(X_test)[:, 1]

    m = compute_metrics(y_test, test_proba, CHOSEN_THRESHOLD)
    print(f"Recall: {m['recall']:.4f} | Precision: {m['precision']:.4f} | "
          f"F2: {m['f2']:.4f} | AUPRC: {m['auprc']:.4f}")
    print_report(y_test, test_proba, CHOSEN_THRESHOLD)

    # PR curve
    precisions, recalls, _ = precision_recall_curve(y_test, test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color="black", linewidth=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve (AUPRC={m['auprc']:.4f})")
    plt.savefig(REPORTS_DIR / "pr_curve.png", dpi=120)
    plt.close()

    # Confusion matrix
    y_pred = (test_proba >= CHOSEN_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Risk", "Risk"]).plot(cmap="Blues", colorbar=False)
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # Save metrics + operating points
    sweep = threshold_sweep(y_test, test_proba)
    metrics_out = {
        "target_used": PRIMARY_TARGET,
        "threshold": CHOSEN_THRESHOLD,
        "test_metrics": m,
        "operating_points": sweep,
        "note": ("92-95% recall with 80-85% precision not achievable with tabular "
                 "vitals alone at this class imbalance; reporting full PR tradeoff.")
    }
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("Saved metrics.json, pr_curve.png, confusion_matrix.png")
    return metrics_out

if __name__ == "__main__":
    evaluate()
