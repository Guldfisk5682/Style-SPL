"""Require exact original/replayed S1 states before interpreting trajectories."""
import argparse
import json
import sys
from pathlib import Path

import torch
from verify_resume import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("original", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    keys = ["prompt", "optimizer", "scheduler", "step", "target_feature_count", "valley",
            "running_means", "running_count", "best_instant", "rng", "source_stream", "target_stream"]
    report = {"exact_replay": True, "compared_fields": keys, "targets": {}}
    try:
        for target in ["art", "clipart", "product", "real_world"]:
            a = torch.load(args.original / target / "checkpoints/last.pth", map_location="cpu", weights_only=True)
            b = torch.load(args.replay / target / "checkpoints/last.pth", map_location="cpu", weights_only=True)
            for key in keys:
                compare(a[key], b[key], f"{target}/{key}")
            report["targets"][target] = "exact"
    except Exception as exc:
        report.update(exact_replay=False, error=str(exc))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
