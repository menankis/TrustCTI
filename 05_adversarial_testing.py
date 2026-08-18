"""
Stage 5 (adapted): Adversarial Robustness Testing — CICIDS2017 schema
-----------------------------------------
Two attacks: evasion and data poisoning.

IMPORTANT FIX: an earlier version of the evasion test scaled a fixed list
of features down by a fixed 30% — this produced a near-zero (0.5%) evasion
rate on real CICIDS2017 data, which was misleading rather than reassuring.
The problem: real feature values span huge ranges (Flow Bytes/s in the
millions, packet totals in the thousands), so a flat 30% cut barely moves
a flow's fingerprint. This mirrors the exact issue Stage 6's drift test had
before being fixed.

Fix: same approach as the corrected Stage 6 — use the model's actual
feature_importances_ to find what it relies on, then INTERPOLATE malicious
flows toward real benign averages on those features, at several intensity
levels, instead of a single fixed percentage cut.

Output: reports/adversarial_robustness_report.json
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

MODEL_PATH = "models/threat_model.pkl"
DATA_PATH = "data/threat_data_processed.csv"
REPORT_PATH = "reports/adversarial_robustness_report.json"
RANDOM_SEED = 42
TOP_N_FEATURES_TO_ATTACK = 10
INTENSITY_LEVELS = (0.25, 0.5, 0.75, 1.0)


def get_top_features(model, feature_names, top_n):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n).index.tolist()


def evasion_attack_test(model, X_test, y_test, target_features, benign_reference, intensity):
    """
    Blend correctly-caught malicious flows toward real benign averages on
    the model's top features, at the given intensity (0=no change,
    1=fully replaced with benign averages), and check how many flip.
    """
    malicious_mask = (y_test == 1)
    correctly_caught = malicious_mask & (model.predict(X_test) == 1)
    X_malicious = X_test[correctly_caught].astype(float).copy()

    if len(X_malicious) == 0:
        return {"error": "no correctly-caught malicious samples to test"}

    original_preds = model.predict(X_malicious)
    original_scores = model.predict_proba(X_malicious)[:, 1]

    X_perturbed = X_malicious.copy()
    for feat in target_features:
        target_val = benign_reference[feat]
        X_perturbed[feat] = X_perturbed[feat] * (1 - intensity) + target_val * intensity

    perturbed_preds = model.predict(X_perturbed)
    perturbed_scores = model.predict_proba(X_perturbed)[:, 1]

    n_evaded = int(((original_preds == 1) & (perturbed_preds == 0)).sum())
    evasion_rate = n_evaded / len(X_malicious)

    return {
        "intensity": intensity,
        "malicious_samples_tested": int(len(X_malicious)),
        "samples_that_evaded": n_evaded,
        "evasion_rate": round(evasion_rate, 3),
        "avg_threat_score_before": round(float(original_scores.mean()), 3),
        "avg_threat_score_after": round(float(perturbed_scores.mean()), 3),
    }


def poisoning_attack_test(X_train, y_train, X_test, y_test, poison_fractions=(0.0, 0.02, 0.05, 0.10)):
    """Unchanged — this logic doesn't depend on feature scale."""
    results = []
    rng = np.random.RandomState(RANDOM_SEED)

    for frac in poison_fractions:
        y_poisoned = y_train.copy()
        n_poison = int(frac * len(y_train))
        if n_poison > 0:
            poison_idx = rng.choice(y_train.index, size=n_poison, replace=False)
            y_poisoned.loc[poison_idx] = 1 - y_poisoned.loc[poison_idx]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X_train, y_poisoned)
        y_pred = model.predict(X_test)

        results.append({
            "poison_fraction": frac,
            "n_labels_flipped": n_poison,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "recall_malicious": round(recall_score(y_test, y_pred), 3),
            "precision_malicious": round(precision_score(y_test, y_pred), 3),
        })

    return results


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    target_features = get_top_features(model, X.columns, TOP_N_FEATURES_TO_ATTACK)
    benign_reference = X_test[y_test == 0][target_features].mean()

    print(f"Targeting the model's top {TOP_N_FEATURES_TO_ATTACK} features for evasion testing:")
    for feat in target_features:
        print(f"  - {feat} (benign avg: {benign_reference[feat]:.2f})")

    print("\n=== Running evasion attack test at multiple intensities ===")
    evasion_results = []
    for intensity in INTENSITY_LEVELS:
        result = evasion_attack_test(model, X_test, y_test, target_features, benign_reference, intensity)
        evasion_results.append(result)
        print(result)

    print("\n=== Running data poisoning test (retrains the model 4x) ===")
    poisoning_results = poisoning_attack_test(X_train, y_train, X_test, y_test)
    for r in poisoning_results:
        print(r)

    report = {
        "target_features": target_features,
        "evasion_attack_by_intensity": evasion_results,
        "poisoning_attack": poisoning_results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved full adversarial robustness report to {REPORT_PATH}")