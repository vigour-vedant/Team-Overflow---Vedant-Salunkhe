# Team-Overflow---Vedant-Salunkhe

# Early Sepsis Alert

Predicts whether an ICU patient will develop sepsis in the next **6–12 hours**
using hourly vital sign time-series data, so clinicians can intervene before
onset rather than after.

## Problem
A missed sepsis signal can be fatal. A false alarm costs a few minutes of
review. This asymmetry means the model is optimized for **recall** first,
using **F2** as the primary metric (weights recall 4x over precision).

## Dataset
PhysioNet-style ICU vitals dataset: hourly readings (HR, SBP, MAP, Resp,
Lactate, WBC, etc.) across 1,533 patients, ~139 of whom develop sepsis.

## Approach
1. **`data_preprocessing.py`** — patient-level 80/20 train/test split
   (stratified on sepsis outcome, no row-level leakage across patients).
2. **`feature_engineering.py`** — builds a forward-shifted target
   (`target_12h`: will sepsis onset in the next 12h?), plus rolling window
   stats, deltas, slopes, and clinical composite scores (SIRS, qSOFA proxy,
   shock index) per vital.
3. **`train.py`** — XGBoost, regularized, `scale_pos_weight` for class
   imbalance.
4. **`evaluate.py`** — reports a full precision-recall curve rather than a
   single hard threshold, following the same evaluation philosophy as the
   original PhysioNet 2019 Sepsis Challenge (plain fixed-threshold
   precision/recall breaks down at this class imbalance).

## Results (test set, target_12h, threshold=0.16)
| Metric | Value |
|---|---|
| Recall | 0.338 |
| Precision | 0.095 |
| F2 | 0.224 |
| AUPRC | 0.085 |

**Honest limitation:** hourly tabular vitals alone do not carry enough signal
to reach 90%+ recall at 80%+ precision simultaneously for a 12h-ahead alert —
confirmed across feature reduction, SMOTE, and hyperparameter tuning. See
`reports/pr_curve.png` for the full tradeoff curve and `outputs/metrics.json`
for operating points at other thresholds.

## Project Structure

early-sepsis-alert/
├── data/ # raw + processed datasets
├── notebooks/ # EDA, feature engineering, model experiments
├── src/ # production pipeline scripts
├── models/ # trained model + feature list (.pkl)
├── reports/ # evaluation plots (PR curve, confusion matrix, etc.)
├── outputs/ # predictions.csv, metrics.json
└── app/ # demo interface
## Usage
```bash
cd src
python data_preprocessing.py
python feature_engineering.py
python train.py
python evaluate.py
python predict.py ../data/processed/test_fe.csv
```

## Requirements
See `requirements.txt`. Python 3.10+.
