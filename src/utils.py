import pandas as pd
from pathlib import Path

ID_COL = "patient_id"
TIME_COL = "HourInStay"
RAW_TARGET = "SepsisLabel"
PRIMARY_TARGET = "target_12h"
CHOSEN_THRESHOLD = 0.16
RANDOM_STATE = 42

def load_csv(path):
    return pd.read_csv(Path(path))

def get_feature_cols(df, exclude=None):
    exclude = exclude or [ID_COL, TIME_COL, RAW_TARGET, "target_6h", "target_12h"]
    return [c for c in df.columns if c not in exclude]

def reduce_features(df):
    drop_cols = [c for c in df.columns if "_roll4h_" in c or "_roll16h_" in c]
    return df.drop(columns=drop_cols)
