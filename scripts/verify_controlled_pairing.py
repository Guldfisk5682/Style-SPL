"""Verify that architecture comparisons saw identical data and initial classes."""
import argparse
import json
from pathlib import Path

import torch
from verify_resume import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--arms", nargs="+", default=["r1_shared", "r2_pooled", "r3_split", "r4_silu32"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"matched": True, "reference": args.arms[0], "comparisons": {},
              "checked": ["initial class SHA", "frozen bank SHA", "bank buffers", "source centroids/counts", "RNG", "source/target stream state", "scheduler"]}
    for target in ["art", "clipart", "product", "real_world"]:
        reference = args.root / args.arms[0] / target
        a = torch.load(reference / "checkpoints/last.pth", map_location="cpu", weights_only=True)
        for arm in args.arms[1:]:
            b = torch.load(args.root / arm / target / "checkpoints/last.pth", map_location="cpu", weights_only=True)
            assert a["step"] == b["step"] == 1000
            for key in ["initial_class_prompt_sha256", "frozen_bank_input_sha256"]:
                assert a["config"][key] == b["config"][key], (arm, target, key)
            for key in ["running_means", "running_count", "rng", "source_stream", "target_stream", "scheduler"]:
                compare(a[key], b[key], f"{arm}/{target}/{key}")
            for key in a["prompt"]:
                if key.startswith("style_bank."):
                    compare(a["prompt"][key], b["prompt"][key], f"{arm}/{target}/{key}")
            report["comparisons"][f"{arm}/{target}"] = "exact"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
