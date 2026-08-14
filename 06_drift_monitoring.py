"""
Stage 6: Drift Monitoring
-----------------------------------------
Addresses the paper's "model drift" concern: a model trained on today's
attacker behavior can silently degrade as attackers change tactics.

We simulate this by generating three time periods of data. Periods 2 and 3
introduce a gradual behavioral shift (attackers increasingly avoid the
"obvious" signals like known_malicious_indicator_count and instead rely on
subtler patterns) — then we score the ORIGINAL (Day 1) model against each
period without retraining, to show performance decay over time.

Output: reports/drift_report.json, reports/drift_chart.png
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = "models/threat_model.pkl"
REPORT_PATH = "reports/drift_report.json"
CHART_PATH = "reports/drift_chart.png"
RANDOM_SEED = 42
N_SAMPLES_PER_PERIOD = 1500


def generate_period_data(n_samples, drift_level: float, seed: int) -> pd.DataFrame:
    """
    drift_level: 0.0 = original attacker behavior, 1.0 = fully shifted behavior.
    As drift increases, attackers rely less on 'known_malicious_indicator_count'
    and 'geo_mismatch' (the model's top 2 signals) and more on subtler patterns
    the original model was never trained to weight heavily.
    """
    rng = np.random.RandomState(seed)

    src_ip_reputation = rng.normal(70, 20, n_samples).clip(0, 100)
    login_attempts_last_hour = rng.poisson(2, n_samples)
    geo_mismatch = rng.binomial(1, max(0.08 - 0.06 * drift_level, 0.01), n_samples)
    unusual_time_access = rng.binomial(1, 0.15, n_samples)
    data_transfer_mb = rng.exponential(50, n_samples)
    # Attackers increasingly avoid tripping known-indicator lists as drift increases
    known_malicious_indicators = rng.poisson(max(0.3 - 0.25 * drift_level, 0.02), n_samples)
    days_since_last_patch = rng.exponential(20, n_samples)
    protocol_risk_score = rng.normal(30, 15, n_samples).clip(0, 100)
    user_privilege_level = rng.randint(1, 6, n_samples)
    prior_incident_count = rng.poisson(0.5, n_samples)
    # New subtler signal that emerges with drift (model was never trained on this pattern)
    session_length_anomaly = rng.normal(0.3 * drift_level, 0.1, n_samples).clip(0, 1)

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
        + session_length_anomaly * 4.0 * drift_level  # this signal the model can't see
        + rng.normal(0, 0.8, n_samples)
    )
    threshold = np.percentile(risk_score, 85)
    label = (risk_score > threshold).astype(int)

    return pd.DataFrame({
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


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    model = joblib.load(MODEL_PATH)

    periods = [
        ("Month 0 (deployment)", 0.0),
        ("Month 2", 0.35),
        ("Month 4", 0.65),
        ("Month 6", 1.0),
    ]

    drift_results = []
    for label_name, drift_level in periods:
        df = generate_period_data(N_SAMPLES_PER_PERIOD, drift_level, seed=RANDOM_SEED + int(drift_level * 100))
        X = df.drop(columns=["label"])
        y = df["label"]

        y_pred = model.predict(X)
        metrics = {
            "period": label_name,
            "drift_level": drift_level,
            "accuracy": round(accuracy_score(y, y_pred), 3),
            "recall_malicious": round(recall_score(y, y_pred), 3),
            "precision_malicious": round(precision_score(y, y_pred), 3),
            "f1_malicious": round(f1_score(y, y_pred), 3),
        }
        drift_results.append(metrics)
        print(metrics)

    with open(REPORT_PATH, "w") as f:
        json.dump(drift_results, f, indent=2)
    print(f"\nSaved drift report to {REPORT_PATH}")

    # Retraining trigger check (paper: orgs need clear retraining triggers, not guesswork)
    RETRAIN_RECALL_THRESHOLD = 0.75
    for r in drift_results:
        if r["recall_malicious"] < RETRAIN_RECALL_THRESHOLD:
            print(f"RETRAINING TRIGGER: recall dropped to {r['recall_malicious']} "
                  f"at '{r['period']}' (below {RETRAIN_RECALL_THRESHOLD} threshold)")

    # Chart
    periods_labels = [r["period"] for r in drift_results]
    recall_vals = [r["recall_malicious"] for r in drift_results]
    precision_vals = [r["precision_malicious"] for r in drift_results]
    accuracy_vals = [r["accuracy"] for r in drift_results]

    plt.figure(figsize=(8, 5))
    plt.plot(periods_labels, recall_vals, marker="o", label="Recall (malicious)")
    plt.plot(periods_labels, precision_vals, marker="o", label="Precision (malicious)")
    plt.plot(periods_labels, accuracy_vals, marker="o", label="Accuracy")
    plt.axhline(y=RETRAIN_RECALL_THRESHOLD, color="red", linestyle="--", alpha=0.5, label="Retrain trigger")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model performance decay as attacker behavior shifts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"Saved drift chart to {CHART_PATH}")