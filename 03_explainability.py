"""
Stage 3: Explainability Layer
-----------------------------------------
Attaches a human-readable "why" to every AI prediction using SHAP
(SHapley Additive exPlanations).

This directly addresses the paper's central human-factor finding:
"Interpretability is almost as important as accuracy" — if an analyst
doesn't understand why an alert fired, they ignore it, and the AI
system becomes operationally useless (every survey respondent rated
trust as at least a moderate challenge).

Output: data/predictions_with_explanations.csv
"""

import pandas as pd
import numpy as np
import joblib
import shap
import json

MODEL_PATH = "models/threat_model.pkl"
DATA_PATH = "data/predictions_with_scores.csv"
OUTPUT_PATH = "data/predictions_with_explanations.csv"
TOP_N_REASONS = 3


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in
                    ("true_label", "predicted_label", "threat_score")]
    X = df[feature_cols]
    return model, df, X, feature_cols


def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classifiers, TreeExplainer can return a list [class0, class1]
    # or a single array — normalize to "contribution toward malicious class"
    if isinstance(shap_values, list):
        malicious_shap = shap_values[1]
    elif shap_values.ndim == 3:
        malicious_shap = shap_values[:, :, 1]
    else:
        malicious_shap = shap_values

    return malicious_shap


def top_reasons_for_row(shap_row, feature_values, feature_names, top_n=3):
    """Turn raw SHAP numbers into a plain-language explanation string."""
    contributions = list(zip(feature_names, shap_row, feature_values))
    # Sort by absolute impact on the prediction, strongest first
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for name, shap_val, feat_val in contributions[:top_n]:
        direction = "increased" if shap_val > 0 else "decreased"
        readable_name = name.replace("_", " ")
        reasons.append(f"{readable_name}={feat_val:.1f} {direction} risk")

    return " | ".join(reasons)


if __name__ == "__main__":
    print("Loading model and predictions...")
    model, df, X, feature_cols = load_model_and_data()

    print(f"Computing SHAP explanations for {len(X)} predictions...")
    shap_values = compute_shap_values(model, X)

    print(f"Generating top-{TOP_N_REASONS} human-readable reasons per row...")
    explanations = []
    for i in range(len(X)):
        reason = top_reasons_for_row(
            shap_values[i], X.iloc[i].values, feature_cols, TOP_N_REASONS
        )
        explanations.append(reason)

    df["explanation"] = explanations

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved explained predictions to {OUTPUT_PATH}")

    print("\n=== Sample explanations (3 highest-scored alerts) ===")
    top_alerts = df.sort_values("threat_score", ascending=False).head(3)
    for _, row in top_alerts.iterrows():
        print(f"\nThreat score: {row['threat_score']:.3f} | "
              f"True label: {'malicious' if row['true_label'] == 1 else 'benign'}")
        print(f"Reason: {row['explanation']}")