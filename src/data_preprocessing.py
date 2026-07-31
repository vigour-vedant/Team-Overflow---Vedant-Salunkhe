import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def build_patient_id(df, time_col="HourInStay"):
    df = df.rename(columns={"Unnamed: 0": time_col}) if "Unnamed: 0" in df.columns else df
    df[time_col] = df[time_col] if time_col in df.columns else df["Unnamed: 0"]
    df["patient_id"] = (df[time_col] == 1).cumsum()
    return df

def split_train_test(raw_csv_path, out_dir, test_size=0.20, random_state=42):
    df = pd.read_csv('/content/extracted_folder/Early_Sepsis_Alert/Data/Raw Data/vital_dataset.csv')
    df = df.rename(columns={"Unnamed: 0": "HourInStay"})
    df["patient_id"] = (df["HourInStay"] == 1).cumsum()

    cols = ["patient_id", "HourInStay"] + [c for c in df.columns if c not in ("patient_id", "HourInStay")]
    df = df[cols]

    patient_labels = df.groupby("patient_id")["SepsisLabel"].max()
    train_ids, test_ids = train_test_split(
        patient_labels.index, test_size=test_size, random_state=random_state,
        stratify=patient_labels.values
    )

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

if __name__ == "__main__":
    split_train_test("../data/raw/vital_dataset.csv", "../data/processed")
