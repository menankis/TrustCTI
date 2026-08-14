"""
Stage 1: Data Ingestion & Normalization
-----------------------------------------
Simulates pulling threat-related events from multiple fragmented sources
(login systems, network logs, endpoint tools) — exactly the "fragmented,
inconsistent data" problem the CODASPY 2026 paper identifies as the
#1 technical barrier (78.6% of practitioners called it major/significant).

We deliberately inject messiness (missing values, inconsistent scales,
duplicate rows) and then clean it — because a real ingestion layer's job
is to fix that, not assume clean data.

Output: data/threat_data_processed.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 5000
OUTPUT_DIR = "data"

np.random.seed(RANDOM_SEED)


def generate_raw_events(n_samples: int) -> pd.DataFrame:
    """Simulate raw, messy CTI event data from multiple sources."""

    src_ip_reputation = np.random.normal(70, 20, n_samples).clip(0, 100)
    login_attempts_last_hour = np.random.poisson(2, n_samples)
    geo_mismatch = np.random.binomial(1, 0.08, n_samples)
    unusual_time_access = np.random.binomial(1, 0.15, n_samples)
    data_transfer_mb = np.random.exponential(50, n_samples)
    known_malicious_indicators = np.random.poisson(0.3, n_samples)
    days_since_last_patch = np.random.exponential(20, n_samples)
    protocol_risk_score = np.random.normal(30, 15, n_samples).clip(0, 100)
    user_privilege_level = np.random.randint(1, 6, n_samples)
    prior_incident_count = np.random.poisson(0.5, n_samples)

    # ---- Ground-truth label logic (a weighted "risk function" + noise) ----
    risk_score = (
        (100 - src_ip_reputation) * 0.02
        + login_attempts_last_hour * 0.15
        + geo_mismatch * 2.5
        + unusual_time_access * 1.2
        + (data_transfer_mb > 150).astype(int) * 1.5
        + known_malicious_indicators * 2.0
        + (days_since_last_patch > 45).astype(int) * 1.0
        + protocol_risk_score * 0.02
        + prior_incident_count * 1.3
        + np.random.normal(0, 0.8, n_samples)  # noise so it's not trivially separable
    )
    threshold = np.percentile(risk_score, 85)  # ~15% malicious, realistic imbalance
    label = (risk_score > threshold).astype(int)

    df = pd.DataFrame({
        "src_ip_reputation_score": src_ip_reputation,
        "login_attempts_last_hour": login_attempts_last_hour,
        "geo_mismatch": geo_mismatch,
        "unusual_time_access": unusual_time_access,
        "data_transfer_mb": data_transfer_mb,
        "known_malicious_indicator_count": known_malicious_indicators,
        "days_since_last_patch": days_since_last_patch,
        "protocol_risk_score": protocol_risk_score,
        "user_privilege_level": user_privilege_level,
        "prior_incident_count": prior_incident_count,
        "label": label,
    })

    # ---- Inject real-world messiness (this is the "fragmented data" problem) ----
    # 1. Missing values scattered across a few columns
    for col in ["src_ip_reputation_score", "data_transfer_mb", "days_since_last_patch"]:
        missing_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. Duplicate rows (common when multiple tools log the same event)
    dupes = df.sample(frac=0.02, random_state=RANDOM_SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """This is the actual 'ingestion & normalization' logic a pipeline needs."""

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Impute missing numeric values with column median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("label")
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"Imputed {n_missing} missing values in '{col}' with median")

    # Clip obviously invalid values (e.g. negative reputation scores)
    df["src_ip_reputation_score"] = df["src_ip_reputation_score"].clip(0, 100)
    df["protocol_risk_score"] = df["protocol_risk_score"].clip(0, 100)
    df["days_since_last_patch"] = df["days_since_last_patch"].clip(0, None)
    df["data_transfer_mb"] = df["data_transfer_mb"].clip(0, None)

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating raw multi-source threat event data...")
    raw_df = generate_raw_events(N_SAMPLES)
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Missing values:\n{raw_df.isna().sum()[raw_df.isna().sum() > 0]}\n")

    print("Cleaning and normalizing...")
    clean_df = clean_and_normalize(raw_df)
    print(f"\nFinal dataset shape: {clean_df.shape}")
    print(f"Label distribution:\n{clean_df['label'].value_counts(normalize=True)}")

    out_path = os.path.join(OUTPUT_DIR, "threat_data_processed.csv")
    clean_df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned dataset to {out_path}")