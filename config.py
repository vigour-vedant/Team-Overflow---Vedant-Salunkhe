"""Global configuration for Early Sepsis Alert pipeline."""
from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"

for d in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Column names ----
ID_COL = "patient_id"
TIME_COL = "HourInStay"
RAW_TARGET = "SepsisLabel"

# ---- Prediction horizons ----
HORIZON_SHORT = 6
HORIZON_LONG = 12
PRIMARY_TARGET = "target_12h"   # target actually used for the final model

# ---- Train/test split ----
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---- Feature engineering ----
TREND_VITALS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "WBC", "Lactate", "Platelets", "Creatinine", "BUN",
    "PaCO2", "pH", "Bilirubin_total", "Glucose"
]
ROLLING_WINDOWS = [8, 24]     # hours (reduced from [4,8,16,24] to cut redundancy)
DELTA_WINDOWS = [1, 3, 6]     # hours
SLOPE_WINDOW = 6              # hours
SPARSE_LABS = [
    "EtCO2", "Bilirubin_direct", "TroponinI", "Fibrinogen",
    "AST", "Alkalinephos", "Lactate", "SaO2", "PaCO2"
]

# ---- Model ----
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=8,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=2.0,
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ---- Decision threshold ----
CHOSEN_THRESHOLD = 0.16

# ---- Metric targets (for reference; not fully achievable at this class imbalance) ----
RECALL_TARGET = (0.92, 0.95)
PRECISION_TARGET = (0.80, 0.85)
F2_TARGET = (0.88, 0.92)
