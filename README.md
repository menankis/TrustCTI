# TrustCTI — Explainable, Human-Verified AI Threat Intelligence Pipeline

Most AI cybersecurity projects optimize for one number: accuracy. This project
argues that's not enough for a financial institution — a model can't be
deployed if nobody can explain it, defend it against attackers, or prove to
an auditor that it's behaving properly.

TrustCTI is a 7-stage pipeline that treats trust, explainability, and
governance as first-class features, not afterthoughts — directly informed by
CODASPY 2026 research on why AI-driven Cyber Threat Intelligence struggles to
move from research prototype to real financial-sector deployment.

Built and validated on **CICIDS2017**, a real, peer-reviewed network
intrusion detection dataset.

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_data_ingestion.py` | Loads and cleans real CICIDS2017 traffic data, handles infinities/duplicates |
| 2 | `02_train_model.py` | Trains a Random Forest to score events by threat likelihood |
| 3 | `03_explainability.py` | SHAP-based, human-readable reason for every prediction |
| 4 | `04_human_loop_routing.py` | Routes low-confidence alerts to analyst review instead of auto-deciding |
| 5 | `05_adversarial_testing.py` | Tests evasion attacks (at increasing intensity) and label-poisoning attacks |
| 6 | `06_drift_monitoring.py` | Simulates attacker behavior adapting over time, tracks performance decay |
| 7 | `07_audit_log.py` | Compiles every decision into an audit-ready governance record |

## Setup

```bash
pip install scikit-learn pandas numpy joblib shap matplotlib --break-system-packages
```

Download `Clean_CICIDS2017.csv` and place it in the project root before running.

## Run everything

```bash
python3 run_pipeline.py
```

Or run any stage individually — each reads the previous stage's output from
`data/`, `models/`, or `reports/`.

## Key results from this run

- **Routing:** 9,322 total alerts processed — 4,239 auto-escalated (100%
  accuracy), 4,665 auto-closed (99.9% accuracy), and only 418 (4.5%) needed
  human review — meaning the vast majority of decisions were handled
  automatically and correctly, with humans reserved for the genuinely
  uncertain cases
- **Drift under attacker adaptation:** recall on real malicious traffic fell
  from **0.99 → 0.82 → 0.30 → 0.0** across simulated adaptation stages, while
  accuracy declined far more gently (**0.99 → 0.52**) — proof that accuracy
  alone can look healthy while the model is quietly missing nearly all real
  threats
- **Adversarial evasion:** evasion rate climbs from ~10% under mild traffic
  disguise up to ~99-100% once an attacker fully mimics benign traffic on the
  model's top-relied-on features — demonstrating this class of model is
  genuinely vulnerable to an informed adversary, not just a naive one
- **Explainability:** every prediction ships with a plain-English, SHAP-based
  reason (e.g. "SYN Flag Count increased risk"), so no alert reaches an
  analyst or an auditor unexplained
- **Governance:** every decision is logged with a model version hash,
  explanation, and routing outcome; automatic retraining triggers fire the
  moment malicious-class recall drops below a 0.75 policy threshold

## Why this design

Each stage maps directly to a barrier or safeguard identified in the source
research:

- **Explainability (Stage 3)** — interpretability matters as much as accuracy
  for analyst trust
- **Human-in-the-loop (Stage 4)** — AI shouldn't take high-impact action
  without human approval
- **Adversarial testing (Stage 5)** — AI systems are themselves attackable
  and need to be tested against an informed adversary, not just a naive one
- **Drift monitoring (Stage 6)** — models silently degrade as attackers adapt;
  continuous, multi-metric monitoring — not a single accuracy check — is
  required to catch it
- **Audit log (Stage 7)** — regulators require organizations to explain and
  defend AI-driven decisions

## Known limitations

- Evaluated on a single, well-studied academic dataset (CICIDS2017), which is
  known to be highly separable — real-world traffic is likely messier
- Drift is *simulated* by reshaping real malicious flows toward the model's
  own top features, not measured against independently-collected future data
- The evasion attack assumes a moderately informed attacker (aware of which
  features the model relies on); it is not a formal gradient-based attack,
  since Random Forests are non-differentiable
- Routing thresholds (0.75 / 0.15) and the retraining trigger (0.75 recall)
  are reasonable starting policies, not tuned against real analyst capacity
  or organizational risk tolerance

## Project structure after running

```
data/       # processed dataset, predictions, explanations, routing decisions
models/     # trained model (threat_model.pkl)
reports/    # adversarial report, drift report + chart, audit log
```
