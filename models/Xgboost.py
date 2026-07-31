# Add to src/train.py, after the XGBoost training block

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_rf_baseline(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    print("Random Forest baseline saved.")
    return rf

def save_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Scaler saved (for any linear/distance-based models).")
    return scaler

# then inside train():
    train_rf_baseline(X_train, y_train)
    save_scaler(X_train)
