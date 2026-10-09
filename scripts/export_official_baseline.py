"""Expose completed, protocol-checked author SPL metrics for paired plotting."""
import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("author_run", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
summary = json.loads((args.author_run / "summary.json").read_text())
if not (summary["training_valid"] and summary.get("protocol_valid") and summary["OT_clustering"] == 0):
    raise RuntimeError("Official baseline is not a valid SPL comparison")
if args.destination.exists():
    raise FileExistsError(args.destination)
args.destination.mkdir(parents=True)
for task in summary["tasks"]:
    source = Path(task["log"]).parent
    destination = args.destination / task["target"]
    destination.mkdir()
    for name in ("train_metrics.csv", "eval_metrics.csv", "result.json", "data_protocol_checks.jsonl"):
        shutil.copy2(source / name, destination / name)
shutil.copy2(args.author_run / "provenance.json", args.destination / "provenance.json")
shutil.copy2(args.author_run / "summary.json", args.destination / "summary.json")
