import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from utils import ID_COL, TIME_COL, PRIMARY_TARGET, RANDOM_STATE, get_feature_cols, reduce_features

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

BEST_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    min_child_weight=8, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=2.0, eval_metric="aucpr",
    random_state=RANDOM_STATE, n_jobs=-1
)

def train():
    train_fe = pd.read_csv('/data/processed/train_fe.csv')
    train_fe = reduce_features(train_fe)
    feature_cols = get_feature_cols(train_fe)

    X_train, y_train = train_fe[feature_cols], train_fe[PRIMARY_TARGET]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(**BEST_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    joblib.dump(model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    print(f"Model saved. Trained on {len(feature_cols)} features, {len(train_fe)} rows.")
    return model, feature_cols

if __name__ == "__main__":
    train()
