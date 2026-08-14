"""
Stage 2: AI Threat Detection & Scoring
-----------------------------------------
Trains a classifier to score/prioritize incoming events by threat likelihood.
Deliberately using a Random Forest (not a black-box deep model) because
Stage 3 (explainability) and Stage 4 (human-in-the-loop routing) depend on
having clear, per-prediction confidence + feature contributions — this
mirrors the paper's finding that interpretability is "almost as important
as accuracy" for analyst trust.

Output: models/threat_model.pkl, data/predictions_with_scores.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = "data/threat_data_processed.csv"
MODEL_DIR = "models"
RANDOM_SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",  # handles the 85/15 imbalance
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # threat_score

    print("=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "malicious"]))

    return y_pred, y_proba


def feature_importance_report(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print("=== Feature Importances (which signals drive the model) ===")
    print(importances.to_string())
    return importances


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    model = train_model(X_train, y_train)
    y_pred, y_proba = evaluate(model, X_test, y_test)
    print()
    feature_importance_report(model, X.columns)

    # Save model for Stage 3 (explainability) and Stage 4 (human-in-the-loop)
    model_path = os.path.join(MODEL_DIR, "threat_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to {model_path}")

    # Save test-set predictions with threat scores — this feeds later stages
    results = X_test.copy()
    results["true_label"] = y_test.values
    results["predicted_label"] = y_pred
    results["threat_score"] = y_proba
    results_path = "data/predictions_with_scores.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved predictions with threat scores to {results_path}")