# TrustCTI — Explainable, Human-Verified AI Threat Intelligence Pipeline

Most AI cybersecurity projects optimize for one number: accuracy. This project
argues that's not enough for a financial institution — a model can't be
deployed if nobody can explain it, defend it against attackers, or prove to
an auditor that it's behaving properly.

TrustCTI is a 7-stage pipeline that treats trust, explainability, and
governance as first-class features, not afterthoughts — directly informed by
CODASPY 2026 research on why AI-driven Cyber Threat Intelligence struggles to
move from research prototype to real financial-sector deployment.

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_data_ingestion.py` | Simulates multi-source threat data, cleans and normalizes it |
| 2 | `02_train_model.py` | Trains a Random Forest to score events by threat likelihood |
| 3 | `03_explainability.py` | SHAP-based, human-readable reason for every prediction |
| 4 | `04_human_loop_routing.py` | Routes low-confidence alerts to analyst review instead of auto-deciding |
| 5 | `05_adversarial_testing.py` | Tests evasion attacks and label-poisoning attacks against the model |
| 6 | `06_drift_monitoring.py` | Simulates attacker behavior shifting over time, tracks performance decay |
| 7 | `07_audit_log.py` | Compiles every decision into an audit-ready governance record |

## Setup

```bash
pip install scikit-learn pandas numpy joblib shap matplotlib --break-system-packages
```

## Run everything

```bash
python3 run_pipeline.py
```

Or run any stage individually (each stage reads the previous stage's output
from `data/`, `models/`, or `reports/`).

## Why this design

Each stage maps directly to a barrier or safeguard identified in the source
research:

- **Explainability (Stage 3)** — interpretability matters as much as accuracy
  for analyst trust
- **Human-in-the-loop (Stage 4)** — AI shouldn't take high-impact action
  without human approval
- **Adversarial testing (Stage 5)** — AI systems are themselves attackable
  and need to be tested as such
- **Drift monitoring (Stage 6)** — models silently degrade as attackers adapt;
  continuous monitoring, not one-time testing, is required
- **Audit log (Stage 7)** — regulators require organizations to explain and
  defend AI-driven decisions

## Project structure after running

```
data/       # raw + processed datasets, predictions, explanations, routing decisions
models/     # trained model (threat_model.pkl)
reports/    # adversarial report, drift report + chart, audit log
```
