"""
Stage 6 (adapted): Drift Monitoring — CICIDS2017 schema
-----------------------------------------
The original version generated fully synthetic "future" data. With a real,
fixed dataset like CICIDS2017 we don't have actual future traffic to test
against, so instead we simulate drift the way the paper describes it:
attackers progressively adjust their behavior to evade the *existing*
detection signals.

IMPORTANT: real CICIDS2017 is known to be extremely separable (models
often hit 98-99%+ accuracy) because it uses MANY features, not just a
handful. A first version of this script perturbed a fixed guessed list of
7 features by a fixed percentage and saw almost no decay — because the
model was still getting plenty of signal from the untouched features.

Fix: instead of guessing which features matter, we ask the model directly
via feature_importances_, target whichever features it actually relies on
most, and INTERPOLATE malicious traffic toward the average benign traffic
values for those features (rather than just scaling them down). At
drift_level=1.0, malicious flows look statistically like average benign
flows on the features the model leans on hardest — which is what real
attacker adaptation is trying to achieve.

Output: reports/drift_report.json, reports/drift_chart.png
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = "models/threat_model.pkl"
DATA_PATH = "data/threat_data_processed.csv"
REPORT_PATH = "reports/drift_report.json"
CHART_PATH = "reports/drift_chart.png"
RANDOM_SEED = 42
RETRAIN_RECALL_THRESHOLD = 0.75

# How many of the model's top features to target for drift simulation.
# Covers the features actually driving predictions instead of a guess.
TOP_N_FEATURES_TO_DRIFT = 10


def get_top_features(model, feature_names, top_n):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n).index.tolist()


def simulate_period(X_test: pd.DataFrame, y_test: pd.Series, drift_level: float,
                     ranked_features: list, benign_reference: pd.Series):
    """
    Staggered ramp: each feature (ranked by importance) starts shifting at
    a different point in the drift timeline and blends smoothly from its
    real value toward the benign average, rather than either (a) blending
    all features at once — which can hit multiple decision thresholds
    simultaneously and cause a sudden cliff — or (b) fully swapping one
    feature at a time — which can also cliff if that one feature alone is
    highly decisive. Staggering avoids both failure modes.

    For feature rank i (0-indexed) of N total ranked features, its blend
    intensity at a given drift_level is:
        clip(drift_level * N - i, 0, 1)
    So feature 0 starts blending immediately, feature 1 starts once
    feature 0 is partway done, and so on — features shift into "evaded"
    state in sequence, each gradually.
    """
    X_period = X_test.astype(float).copy()
    malicious_idx = y_test[y_test == 1].index
    n = len(ranked_features)

    for i, feat in enumerate(ranked_features):
        intensity = min(max(drift_level * n - i, 0.0), 1.0)
        if intensity <= 0:
            continue
        original_vals = X_period.loc[malicious_idx, feat]
        target_val = benign_reference[feat]
        X_period.loc[malicious_idx, feat] = (
            original_vals * (1 - intensity) + target_val * intensity
        )

    return X_period


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    target_features = get_top_features(model, X.columns, TOP_N_FEATURES_TO_DRIFT)
    benign_reference = X_test[y_test == 0][target_features].mean()

    print(f"Targeting the model's top {TOP_N_FEATURES_TO_DRIFT} features for drift simulation:")
    for feat in target_features:
        print(f"  - {feat} (benign avg: {benign_reference[feat]:.2f})")

    periods = [
        ("Month 0", 0.0),
        ("Month 1", 0.15),
        ("Month 2", 0.3),
        ("Month 3", 0.45),
        ("Month 4", 0.6),
        ("Month 5", 0.8),
        ("Month 6", 1.0),
    ]

    drift_results = []
    for label_name, drift_level in periods:
        X_period = simulate_period(X_test, y_test, drift_level, target_features, benign_reference)
        y_pred = model.predict(X_period)

        metrics = {
            "period": label_name,
            "drift_level": drift_level,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "recall_malicious": round(recall_score(y_test, y_pred), 3),
            "precision_malicious": round(precision_score(y_test, y_pred), 3),
            "f1_malicious": round(f1_score(y_test, y_pred), 3),
        }
        drift_results.append(metrics)
        print(metrics)

    with open(REPORT_PATH, "w") as f:
        json.dump({"target_features": target_features, "periods": drift_results}, f, indent=2)
    print(f"\nSaved drift report to {REPORT_PATH}")

    for r in drift_results:
        if r["recall_malicious"] < RETRAIN_RECALL_THRESHOLD:
            print(f"RETRAINING TRIGGER: recall dropped to {r['recall_malicious']} "
                  f"at '{r['period']}' (below {RETRAIN_RECALL_THRESHOLD} threshold)")

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
    plt.title("Model performance decay as attacker behavior shifts (CICIDS2017)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"Saved drift chart to {CHART_PATH}")