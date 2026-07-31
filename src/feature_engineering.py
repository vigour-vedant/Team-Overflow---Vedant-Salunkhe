import pandas as pd
import numpy as np
from pathlib import Path

ID_COL, TIME_COL, TARGET_COL = "patient_id", "HourInStay", "SepsisLabel"
HORIZON_SHORT, HORIZON_LONG = 6, 12
TREND_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                 "WBC", "Lactate", "Platelets", "Creatinine", "BUN",
                 "PaCO2", "pH", "Bilirubin_total", "Glucose"]
ROLLING_WINDOWS = [4, 8, 16, 24]
DELTA_WINDOWS = [1, 3, 6]
SPARSE_LABS = ["EtCO2", "Bilirubin_direct", "TroponinI", "Fibrinogen",
               "AST", "Alkalinephos", "Lactate", "SaO2", "PaCO2"]

def make_future_onset_labels(labels, horizons):
    n = len(labels)
    out = {h: np.zeros(n, dtype=int) for h in horizons}
    for h in horizons:
        for t in range(n):
            start, end = t + 1, min(t + 1 + h, n)
            out[h][t] = 1 if start < n and labels[start:end].max() > 0 else 0
    return out

def build_targets(df):
    frames = []
    for pid, g in df.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL).copy()
        future = make_future_onset_labels(g[TARGET_COL].values, [HORIZON_SHORT, HORIZON_LONG])
        g[f"target_{HORIZON_SHORT}h"] = future[HORIZON_SHORT]
        g[f"target_{HORIZON_LONG}h"] = future[HORIZON_LONG]
        frames.append(g)
    return pd.concat(frames, ignore_index=True)

def add_rolling_features(df, vitals, windows):
    df = df.sort_values([ID_COL, TIME_COL]).copy()
    grouped = df.groupby(ID_COL, sort=False)
    for vital in vitals:
        for w in windows:
            roll = grouped[vital].rolling(window=w, min_periods=1)
            df[f"{vital}_roll{w}h_mean"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{vital}_roll{w}h_std"] = roll.std().reset_index(level=0, drop=True).fillna(0)
            df[f"{vital}_roll{w}h_min"] = roll.min().reset_index(level=0, drop=True)
            df[f"{vital}_roll{w}h_max"] = roll.max().reset_index(level=0, drop=True)
    return df

def add_delta_features(df, vitals, windows):
    df = df.sort_values([ID_COL, TIME_COL]).copy()
    grouped = df.groupby(ID_COL, sort=False)
    for vital in vitals:
        for w in windows:
            df[f"{vital}_delta{w}h"] = df[vital] - grouped[vital].shift(w)
    delta_cols = [c for c in df.columns if "_delta" in c]
    df[delta_cols] = df[delta_cols].fillna(0)
    return df

def rolling_slope(series, window):
    x = np.arange(window)
    def _slope(y):
        if len(y) < 2 or np.all(y == y[0]):
            return 0.0
        try:
            return np.polyfit(x[:len(y)], y, 1)[0]
        except Exception:
            return 0.0
    return series.rolling(window, min_periods=2).apply(_slope, raw=True).fillna(0)

def add_slope_features(df, vitals, window=6):
    df = df.sort_values([ID_COL, TIME_COL]).copy()
    grouped = df.groupby(ID_COL, sort=False)
    for vital in vitals:
        df[f"{vital}_slope{window}h"] = grouped[vital].transform(lambda s: rolling_slope(s, window))
    return df

def add_clinical_scores(df):
    df = df.copy()
    df["shock_index"] = (df["HR"] / df["SBP"].replace(0, np.nan)).fillna(0)
    sirs = (((df["Temp"] > 38) | (df["Temp"] < 36)).astype(int)
            + (df["HR"] > 90).astype(int)
            + ((df["Resp"] > 20) | (df["PaCO2"] < 32)).astype(int)
            + ((df["WBC"] > 12) | (df["WBC"] < 4)).astype(int))
    df["sirs_score"] = sirs
    df["sirs_positive"] = (sirs >= 2).astype(int)
    df["qsofa_proxy_score"] = (df["Resp"] >= 22).astype(int) + (df["SBP"] <= 100).astype(int)
    df["bun_creatinine_ratio"] = (df["BUN"] / df["Creatinine"].replace(0, np.nan)).fillna(0)
    df["hypotension_flag"] = (df["MAP"] < 65).astype(int)
    df["hypoxia_flag"] = (df["O2Sat"] < 90).astype(int)
    return df

def add_measurement_flags(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_measured"] = (df[c] != 0).astype(int)
    return df

def run_feature_engineering(input_path, output_path):
    df = pd.read_csv(input_path)
    df = build_targets(df)
    df = add_rolling_features(df, TREND_VITALS, ROLLING_WINDOWS)
    df = add_delta_features(df, TREND_VITALS, DELTA_WINDOWS)
    df = add_slope_features(df, TREND_VITALS, window=6)
    df = add_clinical_scores(df)
    df = add_measurement_flags(df, SPARSE_LABS)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df_final = df[df[TARGET_COL] == 0].reset_index(drop=True)
    df_final.to_csv(output_path, index=False)
    print(f"Saved {output_path}: {df_final.shape}")
    return df_final

if __name__ == "__main__":
    run_feature_engineering("../data/processed/train.csv", "../data/processed/train_fe.csv")
    run_feature_engineering("../data/processed/test.csv", "../data/processed/test_fe.csv")
