"""Read author logs after training, without instrumenting the training process."""
import argparse
import json
import math
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
args = parser.parse_args()
console = (args.run_root / "console.log").read_text()
nonfinite_loss = bool(re.search(
    r"(?:target total loss|classification):\s*[+-]?(?:nan|inf)\b",
    console, re.IGNORECASE,
))
domains = ("art", "clipart", "product", "real_world")
rows = []
for domain in domains:
    logs = [p for p in (args.run_root / "outputs").rglob("train.log") if p.parent.name == domain]
    if len(logs) != 1:
        raise RuntimeError(f"Expected one author log for {domain}, found {len(logs)}")
    matches = re.findall(r"^final_accuracy:\s+(\S+)", logs[0].read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"Missing or ambiguous final accuracy for {domain}")
    score = float(matches[0])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError(f"Invalid final accuracy for {domain}: {score}")
    checkpoint = logs[0].parent / "last.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing checkpoint: {checkpoint}")
    rows.append({"target": domain, "final_accuracy": score,
                 "log": str(logs[0]), "checkpoint": str(checkpoint)})
summary = {
    "complete": True, "completed_tasks": len(rows), "expected_tasks": 4,
    "training_valid": not nonfinite_loss,
    "invalid_reason": "Author console reports nonfinite training loss" if nonfinite_loss else None,
    "protocol": "multi-source single-target; four targets in one process",
    "seed": 1, "steps_per_target": 1000, "backbone": "RN50",
    "author_python_code_modified": False,
    "evaluation": "author shuffled target loader with drop_last=True, batch size 30",
    "inference": "author temporal text-feature average",
    "tasks": rows, "mean_accuracy": sum(r["final_accuracy"] for r in rows) / len(rows),
}
provenance = json.loads((args.run_root / "provenance.json").read_text())
summary["author_python_code_modified"] = provenance["author_python_code_modified"]
summary["control_variant"] = provenance.get("control_variant", "unmodified")
for key in ("ablation", "enhanced_pseudo_label", "OT_clustering", "loss_semantics"):
    if key in provenance:
        summary[key] = provenance[key]
if summary["control_variant"] in ("init_checked", "protocol_checked"):
    checks = [json.loads(line) for line in (args.run_root / "numerical_checks.jsonl").read_text().splitlines()]
    completed_checks = {r["target"]: r["checked_updates"] for r in checks if r["phase"] == "finished"}
    guard_passed = all(completed_checks.get(domain) == 1000 for domain in domains)
    guard_passed = guard_passed and not any(r["phase"] == "failure" for r in checks)
    summary["numerical_checks_passed"] = guard_passed
    summary["checked_updates_per_target"] = completed_checks
    if not guard_passed:
        summary["training_valid"] = False
        summary["invalid_reason"] = "Numerical checks failed or did not cover all optimizer updates"
if summary["control_variant"] == "protocol_checked":
    coverage = {}
    for row in rows:
        checks_path = Path(row["log"]).parent / "data_protocol_checks.jsonl"
        checks = [json.loads(line) for line in checks_path.read_text().splitlines()]
        finishes = [r for r in checks if r["phase"] == "finished"]
        if len(finishes) == 1 and not any(r["phase"] == "failure" for r in checks):
            coverage[row["target"]] = finishes[0]["checked_steps"]
    summary["protocol_valid"] = all(coverage.get(domain) == 1000 for domain in domains)
    summary["protocol_checked_updates_per_target"] = coverage
    if not summary["protocol_valid"]:
        summary["training_valid"] = False
        summary["invalid_reason"] = "Data protocol audit failed or was incomplete"
(args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
