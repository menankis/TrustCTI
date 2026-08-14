"""
TrustCTI — Full Pipeline Runner
-----------------------------------------
Runs all 7 stages in order. This is the single entry point for demos:

    python3 run_pipeline.py

Stage 1: Data ingestion & normalization
Stage 2: AI threat detection & scoring
Stage 3: Explainability layer (SHAP)
Stage 4: Human-in-the-loop routing
Stage 5: Adversarial robustness testing
Stage 6: Drift monitoring
Stage 7: Audit & governance log
"""

import subprocess
import sys
import time

STAGES = [
    ("Stage 1: Data ingestion & normalization", "01_data_ingestion.py"),
    ("Stage 2: AI threat detection & scoring", "02_train_model.py"),
    ("Stage 3: Explainability layer", "03_explainability.py"),
    ("Stage 4: Human-in-the-loop routing", "04_human_loop_routing.py"),
    ("Stage 5: Adversarial robustness testing", "05_adversarial_testing.py"),
    ("Stage 6: Drift monitoring", "06_drift_monitoring.py"),
    ("Stage 7: Audit & governance log", "07_audit_log.py"),
]


def run_stage(name: str, script: str) -> float:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nFAILED at: {name}")
        sys.exit(1)
    return elapsed


if __name__ == "__main__":
    print("Running TrustCTI pipeline end to end...\n")
    total_start = time.time()

    for name, script in STAGES:
        elapsed = run_stage(name, script)
        print(f"\n[{name} completed in {elapsed:.1f}s]")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print("Outputs: data/, models/, reports/")
    print(f"{'=' * 60}")