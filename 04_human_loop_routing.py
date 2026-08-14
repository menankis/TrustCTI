"""
Stage 4: Human-in-the-Loop Routing
-----------------------------------------
Implements the paper's Safeguard 1: "AI should not automatically enforce
high-impact actions without appropriate human approval." Low-confidence or
borderline predictions get routed to an analyst review queue instead of
being auto-closed as benign or auto-escalated as malicious.

Routing logic (confidence bands, not a single cutoff):
  - threat_score >= HIGH_THRESHOLD        -> auto-escalate (high confidence malicious)
  - threat_score <= LOW_THRESHOLD         -> auto-close (high confidence benign)
  - LOW_THRESHOLD < score < HIGH_THRESHOLD -> route to human analyst review

Output: data/routed_alerts.csv, data/routing_summary.json
"""

import pandas as pd
import json

INPUT_PATH = "data/predictions_with_explanations.csv"
OUTPUT_PATH = "data/routed_alerts.csv"
SUMMARY_PATH = "data/routing_summary.json"

# Confidence band thresholds — tune these based on analyst capacity vs. risk tolerance
HIGH_THRESHOLD = 0.75   # above this: confident enough to auto-escalate
LOW_THRESHOLD = 0.15    # below this: confident enough to auto-close


def route_alert(threat_score: float) -> str:
    if threat_score >= HIGH_THRESHOLD:
        return "auto_escalate"
    elif threat_score <= LOW_THRESHOLD:
        return "auto_close"
    else:
        return "human_review"


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    df["routing_decision"] = df["threat_score"].apply(route_alert)

    # Human-readable review priority for the queue (higher score = review first)
    review_queue = (
        df[df["routing_decision"] == "human_review"]
        .sort_values("threat_score", ascending=False)
        .reset_index(drop=True)
    )

    df.to_csv(OUTPUT_PATH, index=False)

    counts = df["routing_decision"].value_counts()
    summary = {
        "total_alerts": int(len(df)),
        "auto_escalated": int(counts.get("auto_escalate", 0)),
        "auto_closed": int(counts.get("auto_close", 0)),
        "sent_to_human_review": int(counts.get("human_review", 0)),
        "pct_requiring_human_review": round(
            100 * counts.get("human_review", 0) / len(df), 1
        ),
        "thresholds": {"high": HIGH_THRESHOLD, "low": LOW_THRESHOLD},
    }
    # Sanity check: how well did auto-decisions match ground truth?
    auto_escalated = df[df["routing_decision"] == "auto_escalate"]
    auto_closed = df[df["routing_decision"] == "auto_close"]
    if len(auto_escalated) > 0:
        summary["auto_escalate_accuracy"] = round(
            (auto_escalated["true_label"] == 1).mean(), 3
        )
    if len(auto_closed) > 0:
        summary["auto_close_accuracy"] = round(
            (auto_closed["true_label"] == 0).mean(), 3
        )

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Routing Summary ===")
    print(json.dumps(summary, indent=2))

    print(f"\nSaved routed alerts to {OUTPUT_PATH}")
    print(f"Saved routing summary to {SUMMARY_PATH}")

    print("\n=== Sample of human review queue (top 3, highest priority first) ===")
    for _, row in review_queue.head(3).iterrows():
        print(f"\nThreat score: {row['threat_score']:.3f} | "
              f"True label: {'malicious' if row['true_label'] == 1 else 'benign'}")
        print(f"Reason: {row['explanation']}")