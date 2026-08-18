"""
Stage 1 (adapted): Data Ingestion for CICIDS2017
-----------------------------------------
Replaces the synthetic generator with the real Clean_CICIDS2017.csv file.
Everything from Stage 2 onward is unchanged and doesn't care that the data
is now real instead of synthetic — it just needs numeric features + a
binary 'label' column, which is exactly what this produces.

Run this in the SAME folder as Clean_CICIDS2017.csv.

    python3 01_data_ingestion.py
"""

import pandas as pd
import numpy as np
import os

CSV_PATH = r"C:\Users\Menanki Shekhawat\TrustCTI\Clean_CICIDS2017.csv"   # change path if needed
OUTPUT_DIR = "data"
RANDOM_SEED = 42

# The full file has ~2.8 million rows and 78 raw features. We do two things
# to keep this workable and interpretable for a demo pipeline:
#   1. Select a focused, human-readable subset of features (not all 78) so
#      Stage 3's SHAP explanations stay readable ("SYN flag count increased
#      risk" is meaningful; "Fwd IAT Std" is not, to a non-expert analyst).
#   2. Cap rows per class so training/SHAP/adversarial testing stay fast.
SELECTED_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Idle Mean",
]

LABEL_COL = "Label"
MAX_ROWS_PER_CLASS = 25000  # increase/decrease depending on your machine's RAM


def load_and_sample(csv_path: str, max_rows_per_class: int) -> pd.DataFrame:
    print("Reading CSV (the full file may take a minute or two)...")
    usecols = SELECTED_FEATURES + [LABEL_COL]
    df = pd.read_csv(csv_path, usecols=usecols)
    print(f"Loaded {len(df):,} rows")

    # Binary label: BENIGN -> 0, every attack type (DDoS, PortScan, etc.) -> 1
    df["label"] = (df[LABEL_COL] != "BENIGN").astype(int)

    print("\nOriginal label breakdown (attack types collapse into 'malicious'):")
    print(df[LABEL_COL].value_counts())

    # Stratified cap: sample up to max_rows_per_class from EACH class
    # (benign / malicious) so rare attack types aren't drowned out, but the
    # dataset stays a manageable size. Done with explicit per-class
    # filtering rather than groupby().apply(), which has an edge case that
    # can silently drop the grouping column when only one class is present
    # in a given slice of data.
    sampled_parts = []
    for cls_value in sorted(df["label"].unique()):
        subset = df[df["label"] == cls_value]
        n = min(len(subset), max_rows_per_class)
        sampled_parts.append(subset.sample(n, random_state=RANDOM_SEED))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.drop(columns=[LABEL_COL]).reset_index(drop=True)

    print(f"\nSampled dataset shape: {sampled.shape}")
    print("Label distribution after sampling:")
    print(sampled["label"].value_counts(normalize=True))
    return sampled


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != "label"]

    # CICIDS2017 is known to contain +/- infinity in rate columns (e.g.
    # Flow Bytes/s when Flow Duration is 0 -> division by zero). Convert
    # those to NaN so they get handled like any other missing value,
    # instead of silently poisoning the model with inf.
    n_inf = np.isinf(df[feature_cols]).sum().sum()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    print(f"\nReplaced {n_inf} infinite values with NaN")

    n_missing = df[feature_cols].isna().sum().sum()
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    print(f"Imputed {n_missing} missing/converted values with column medians")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_sample(CSV_PATH, MAX_ROWS_PER_CLASS)
    df = clean_and_normalize(df)

    out_path = os.path.join(OUTPUT_DIR, "threat_data_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned dataset to {out_path}")
    print(f"Final shape: {df.shape}")