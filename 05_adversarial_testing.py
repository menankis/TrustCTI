"""
Stage 5: Adversarial Robustness Testing
-----------------------------------------
Addresses the paper's finding that AI systems are themselves attackable
(35.7% of practitioners had directly encountered adversarial AI threats).
Two attack simulations:

1. Evasion attack: take malicious events the model correctly caught,
   nudge the most influential feature just enough to see if the model
   flips its decision to "benign" — this simulates an attacker who knows
   roughly what the model looks at and tunes their behavior to slip under it.

2. Data poisoning simulation: inject a small number of mislabeled training
   examples and see how much detection performance degrades — this
   simulates an attacker who can influence training/feedback data.

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

# Features an attacker could realistically manipulate (vs. ones they can't,
# like "prior_incident_count" which reflects historical record, not live behavior)
MANIPULABLE_FEATURES = [
    "login_attempts_last_hour",
    "data_transfer_mb",
    "unusual_time_access",
    "protocol_risk_score",
]


def evasion_attack_test(model, X_test, y_test, feature_names, perturb_pct=0.3):
    """
    For correctly-detected malicious events, perturb manipulable features
    by perturb_pct and check how many predictions flip to 'benign'.
    """
    malicious_mask = (y_test == 1)
    correctly_caught = malicious_mask & (model.predict(X_test) == 1)
    X_malicious = X_test[correctly_caught].copy()

    if len(X_malicious) == 0:
        return {"error": "no correctly-caught malicious samples to test"}

    original_preds = model.predict(X_malicious)
    X_perturbed = X_malicious.copy()

    for feat in MANIPULABLE_FEATURES:
        if feat in X_perturbed.columns:
            if feat in ["unusual_time_access"]:
                # binary feature: flip it
                X_perturbed[feat] = 1 - X_perturbed[feat]
            else:
                # reduce numeric "suspicious-looking" features
                X_perturbed[feat] = X_perturbed[feat] * (1 - perturb_pct)

    perturbed_preds = model.predict(X_perturbed)
    perturbed_scores = model.predict_proba(X_perturbed)[:, 1]

    n_evaded = int(((original_preds == 1) & (perturbed_preds == 0)).sum())
    evasion_rate = n_evaded / len(X_malicious)

    return {
        "malicious_samples_tested": int(len(X_malicious)),
        "samples_that_evaded_after_perturbation": n_evaded,
        "evasion_rate": round(evasion_rate, 3),
        "avg_threat_score_before": round(float(model.predict_proba(X_malicious)[:, 1].mean()), 3),
        "avg_threat_score_after": round(float(perturbed_scores.mean()), 3),
        "perturbed_features": MANIPULABLE_FEATURES,
        "perturbation_magnitude": perturb_pct,
    }


def poisoning_attack_test(X_train, y_train, X_test, y_test, poison_fractions=(0.0, 0.02, 0.05, 0.10)):
    """
    Flip labels on an increasing fraction of training data (label poisoning)
    and measure how much detection recall degrades.
    """
    results = []
    rng = np.random.RandomState(RANDOM_SEED)

    for frac in poison_fractions:
        y_poisoned = y_train.copy()
        n_poison = int(frac * len(y_train))
        if n_poison > 0:
            poison_idx = rng.choice(y_train.index, size=n_poison, replace=False)
            y_poisoned.loc[poison_idx] = 1 - y_poisoned.loc[poison_idx]  # flip label

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

    print("=== Running evasion attack test ===")
    evasion_results = evasion_attack_test(model, X_test, y_test, X.columns)
    print(json.dumps(evasion_results, indent=2))

    print("\n=== Running data poisoning test (this retrains the model 4x, may take a moment) ===")
    poisoning_results = poisoning_attack_test(X_train, y_train, X_test, y_test)
    for r in poisoning_results:
        print(r)

    report = {
        "evasion_attack": evasion_results,
        "poisoning_attack": poisoning_results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved full adversarial robustness report to {REPORT_PATH}")