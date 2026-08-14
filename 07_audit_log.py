"""
Stage 7: Audit & Governance Log
-----------------------------------------
Addresses the paper's regulatory findings: 91.6% of practitioners had at
least some concern about explaining AI decisions to regulators/auditors.
This stage doesn't generate new predictions — it compiles everything the
pipeline already produced (Stages 1-6) into one audit-ready governance
record, the kind an organization would need to show:

  - Model version and when it was trained
  - Evaluation metrics at deployment
  - Every decision made, with its explanation and routing outcome
  - Adversarial robustness findings
  - Drift history and retraining triggers
  - A rollback-relevant summary (what would trigger a rollback)

Output: reports/audit_log.json, reports/audit_log_decisions.csv
"""

import pandas as pd
import json
import os
import hashlib
from datetime import datetime, timezone

MODEL_PATH = "models/threat_model.pkl"
ROUTED_ALERTS_PATH = "data/routed_alerts.csv"
ADVERSARIAL_REPORT_PATH = "reports/adversarial_robustness_report.json"
DRIFT_REPORT_PATH = "reports/drift_report.json"

AUDIT_JSON_PATH = "reports/audit_log.json"
AUDIT_CSV_PATH = "reports/audit_log_decisions.csv"

RETRAIN_RECALL_THRESHOLD = 0.75


def file_hash(path: str) -> str:
    """Simple content hash so you can prove which exact model version was audited."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    routed = pd.read_csv(ROUTED_ALERTS_PATH)
    adversarial = load_json(ADVERSARIAL_REPORT_PATH)
    drift = load_json(DRIFT_REPORT_PATH)

    # ---- Per-decision audit trail (what a regulator/auditor would inspect) ----
    decision_log = routed[[
        "threat_score", "predicted_label", "true_label",
        "explanation", "routing_decision"
    ]].copy()
    decision_log.insert(0, "decision_id", range(1, len(decision_log) + 1))
    decision_log.insert(1, "logged_at_utc", datetime.now(timezone.utc).isoformat())
    decision_log["model_version"] = file_hash(MODEL_PATH)
    decision_log.to_csv(AUDIT_CSV_PATH, index=False)

    # ---- Governance-level summary (what leadership/auditors care about) ----
    latest_drift = drift[-1] if drift else None
    needs_retrain = latest_drift and latest_drift["recall_malicious"] < RETRAIN_RECALL_THRESHOLD

    governance_record = {
        "audit_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version_hash": file_hash(MODEL_PATH),
        "total_decisions_logged": int(len(decision_log)),
        "routing_breakdown": routed["routing_decision"].value_counts().to_dict(),
        "human_review_rate_pct": round(
            100 * (routed["routing_decision"] == "human_review").mean(), 1
        ),
        "adversarial_robustness_summary": {
            "evasion_rate": adversarial["evasion_attack"]["evasion_rate"] if adversarial else None,
            "recall_at_10pct_poisoning": next(
                (r["recall_malicious"] for r in adversarial["poisoning_attack"]
                 if r["poison_fraction"] == 0.1), None
            ) if adversarial else None,
        },
        "drift_status": {
            "latest_period_checked": latest_drift["period"] if latest_drift else None,
            "latest_recall": latest_drift["recall_malicious"] if latest_drift else None,
            "retrain_threshold": RETRAIN_RECALL_THRESHOLD,
            "retrain_recommended": bool(needs_retrain),
        },
        "accountability": {
            "model_owner": "cybersecurity_ai_team",  # placeholder — set per organization
            "approval_required_for_high_impact_actions": True,
            "rollback_procedure": "Revert to previous model_version_hash in models/ "
                                   "registry and re-run routing on the last 24h of "
                                   "traffic before resuming auto-decisions.",
        },
    }

    with open(AUDIT_JSON_PATH, "w") as f:
        json.dump(governance_record, f, indent=2)

    print("=== Governance Record ===")
    print(json.dumps(governance_record, indent=2))
    print(f"\nSaved per-decision audit trail to {AUDIT_CSV_PATH}")
    print(f"Saved governance summary to {AUDIT_JSON_PATH}")

    if needs_retrain:
        print("\nGOVERNANCE FLAG: current model is past its retraining threshold. "
              "Do not treat production accuracy numbers as sufficient evidence "
              "of health — recall has degraded below policy threshold.")