import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go

# ---- Paths ----
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"

CHOSEN_THRESHOLD = 0.16

st.set_page_config(page_title="Early Sepsis Alert", page_icon="🩺", layout="wide")

# ---- Load model ----
@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "xgboost.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return model, feature_cols

try:
    model, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ---- Header ----
st.title("🩺 Early Sepsis Alert")
st.caption("Predicts sepsis risk 12 hours ahead using ICU vital sign trends. "
           "Built for early-warning triage support, not diagnosis.")

if not model_loaded:
    st.error(
        "Model files not found in `models/`. Run `src/train.py` first to "
        "generate `xgboost.pkl` and `feature_cols.pkl`."
    )
    st.stop()

st.warning(
    "⚠️ **Research demo, not a clinical tool.** At this dataset's class "
    "imbalance, the model trades precision for recall — expect a meaningful "
    "false-alarm rate. See the Model Performance tab for honest numbers.",
    icon="⚠️"
)

tab1, tab2, tab3 = st.tabs(["📁 Batch Risk Scoring", "🔢 Manual Check", "📊 Model Performance"])

# ============================================================
# TAB 1 — Batch scoring from an already feature-engineered CSV
# ============================================================
with tab1:
    st.subheader("Score a patient file")
    st.write(
        "Upload a CSV already run through `feature_engineering.py` "
        "(i.e. `test_fe.csv` format) — must include all model feature columns."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in df.columns]

        if missing:
            st.error(f"Missing {len(missing)} required feature columns. "
                      f"First few: {missing[:5]}")
        else:
            proba = model.predict_proba(df[feature_cols])[:, 1]
            pred = (proba >= CHOSEN_THRESHOLD).astype(int)

            result = df.copy()
            result["sepsis_risk_score"] = proba
            result["alert"] = np.where(pred == 1, "🔴 HIGH RISK", "🟢 Low Risk")

            id_cols = [c for c in ["patient_id", "HourInStay"] if c in df.columns]
            display_cols = id_cols + ["sepsis_risk_score", "alert"]

            st.write(f"**{pred.sum()} / {len(pred)}** rows flagged as high risk "
                     f"(threshold = {CHOSEN_THRESHOLD})")

            st.dataframe(
                result[display_cols].sort_values("sepsis_risk_score", ascending=False),
                use_container_width=True, height=400
            )

            csv_out = result[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored results", csv_out,
                                "sepsis_risk_scores.csv", "text/csv")

# ============================================================
# TAB 2 — Manual single-point vitals check (simplified, raw vitals only)
# ============================================================
with tab2:
    st.subheader("Quick manual risk check")
    st.caption(
        "Enter current vitals for a rough single-point risk estimate. "
        "This is a simplified version — the real model additionally uses "
        "trailing 8h/24h trend features, so this manual check is less "
        "accurate than a full patient history upload."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        hr = st.number_input("Heart Rate (bpm)", 30, 220, 85)
        sbp = st.number_input("Systolic BP (mmHg)", 50, 250, 115)
        map_val = st.number_input("MAP (mmHg)", 30, 180, 80)
        resp = st.number_input("Respiratory Rate (/min)", 5, 60, 16)
    with col2:
        temp = st.number_input("Temperature (°C)", 30.0, 42.0, 37.0, step=0.1)
        o2sat = st.number_input("O2 Saturation (%)", 50, 100, 97)
        wbc = st.number_input("WBC (x10^9/L)", 0.0, 50.0, 8.0, step=0.1)
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 1.5, step=0.1)
    with col3:
        creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0, step=0.1)
        bun = st.number_input("BUN (mg/dL)", 0.0, 150.0, 15.0, step=1.0)
        platelets = st.number_input("Platelets (x10^9/L)", 0, 800, 250)

    if st.button("Compute clinical risk indicators", type="primary"):
        shock_index = hr / sbp if sbp > 0 else 0
        sirs = int(temp > 38 or temp < 36) + int(hr > 90) + int(resp > 20) + int(wbc > 12 or wbc < 4)
        qsofa = int(resp >= 22) + int(sbp <= 100)
        hypotension = map_val < 65
        hypoxia = o2sat < 90

        st.markdown("### Clinical Indicator Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shock Index", f"{shock_index:.2f}", "⚠️ High" if shock_index > 0.9 else "Normal")
        c2.metric("SIRS Score", f"{sirs}/4", "⚠️ Positive" if sirs >= 2 else "Negative")
        c3.metric("qSOFA (proxy)", f"{qsofa}/2", "⚠️ High" if qsofa >= 2 else "Low")
        c4.metric("MAP", f"{map_val} mmHg", "⚠️ Hypotensive" if hypotension else "Normal")

        flags = []
        if shock_index > 0.9: flags.append("Elevated shock index")
        if sirs >= 2: flags.append("SIRS criteria met (≥2)")
        if qsofa >= 2: flags.append("qSOFA proxy elevated")
        if hypotension: flags.append("MAP below 65 mmHg (hypotension)")
        if hypoxia: flags.append("O2 saturation below 90%")

        if flags:
            st.error("**Clinical flags raised:**\n" + "\n".join(f"- {f}" for f in flags))
        else:
            st.success("No major clinical flags raised on current single-point vitals.")

        st.info(
            "Note: this tab checks instantaneous clinical heuristics only. "
            "The trained ML model (Tab 1) additionally weighs 12-hour trend "
            "patterns and is the primary prediction engine — a single reading "
            "here should not be treated as the model's actual risk output."
        )

# ============================================================
# TAB 3 — Model performance / honesty panel
# ============================================================
with tab3:
    st.subheader("Model Performance (held-out test set)")

    metrics_path = OUTPUTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        m = metrics.get("test_metrics", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{m.get('recall', 0):.1%}")
        c2.metric("Precision", f"{m.get('precision', 0):.1%}")
        c3.metric("F2 Score", f"{m.get('f2', 0):.3f}")
        c4.metric("AUPRC", f"{m.get('auprc', 0):.3f}")

        st.warning(
            "**Honest limitation:** at this dataset's ~2% positive rate for "
            "12h-ahead sepsis onset, tabular vital signs alone do not carry "
            "enough signal to reach 90%+ recall with 80%+ precision "
            "simultaneously — confirmed across feature reduction, SMOTE, and "
            "hyperparameter tuning. The PR curve below shows the full "
            "achievable tradeoff instead of a single cherry-picked threshold, "
            "consistent with how the original PhysioNet 2019 Sepsis Challenge "
            "evaluated submissions."
        )
    else:
        st.info("Run `src/evaluate.py` to generate `outputs/metrics.json`.")

    st.markdown("### Evaluation Plots")
    plot_cols = st.columns(2)
    plots = [
        ("pr_curve.png", "Precision-Recall Curve"),
        ("roc_curve.png", "ROC Curve"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("feature_importance.png", "Top Feature Importances"),
    ]
    for i, (fname, caption) in enumerate(plots):
        path = REPORTS_DIR / fname
        if path.exists():
            plot_cols[i % 2].image(str(path), caption=caption, use_container_width=True)
        else:
            plot_cols[i % 2].info(f"{fname} not found — run `src/evaluate.py`.")

st.divider()
st.caption("Early Sepsis Alert — hackathon research demo. Not for clinical use.")
