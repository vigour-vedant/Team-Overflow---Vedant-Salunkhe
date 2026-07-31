import pandas as pd
import joblib
from pathlib import Path
from utils import CHOSEN_THRESHOLD, reduce_features

MODELS_DIR = Path("../models")
OUTPUTS_DIR = Path("../outputs"); OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

def predict(input_csv, output_csv="../outputs/predictions.csv"):
    model = joblib.load(MODELS_DIR / "xgboost.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    df = pd.read_csv(input_csv)
    df = reduce_features(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:5]}...")

    proba = model.predict_proba(df[feature_cols])[:, 1]
    pred = (proba >= CHOSEN_THRESHOLD).astype(int)

    out = df[["patient_id", "HourInStay"]].copy()
    out["sepsis_risk_proba"] = proba
    out["sepsis_alert"] = pred
    out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv} — {pred.sum()} alerts flagged out of {len(pred)} rows.")
    return out

if __name__ == "__main__":
    import sys
    predict(sys.argv[1] if len(sys.argv) > 1 else "../data/processed/test_fe.csv")
