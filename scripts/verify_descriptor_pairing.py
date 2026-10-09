"""Compare full control runs against R4 without expecting identical bank inputs."""
import argparse
import json
from pathlib import Path

import torch
from verify_resume import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"matched": True, "reference": "r4_real", "tasks": {},
              "checked": ["parameter count", "original bank SHA", "initial class and Pooled tokens",
                          "training RNG", "source and target stream states", "centroids and counts",
                          "scheduler", "unchanged effective descriptors"]}
    mappings = []
    for target in ("art", "clipart", "product", "real_world"):
        reference = args.root / "r4_real" / target
        a = torch.load(reference / "checkpoints/last.pth", map_location="cpu", weights_only=True)
        a0 = torch.load(reference / "mechanism/step0000.pt", map_location="cpu", weights_only=True)["prompt_state"]
        for arm in ("r4_shuffled", "r4_random"):
            task = args.root / arm / target
            b = torch.load(task / "checkpoints/last.pth", map_location="cpu", weights_only=True)
            b0 = torch.load(task / "mechanism/step0000.pt", map_location="cpu", weights_only=True)["prompt_state"]
            assert a["step"] == b["step"] == 1000
            for key in ("trainable_parameters", "initial_class_prompt_sha256", "frozen_bank_input_sha256",
                        "seed", "M1", "M2", "batch_size", "prompt_learning_rate", "w_scale", "t_weight",
                        "independent_pooled", "split_projector", "projector_architecture", "bottleneck_dim"):
                assert a["config"][key] == b["config"][key], (arm, target, key)
            compare(a0["ctx_source_combined"], b0["ctx_source_combined"], "Pooled initialization")
            for key in ("rng", "source_stream", "target_stream", "running_means", "running_count", "scheduler"):
                compare(a[key], b[key], f"{arm}/{target}/{key}")
            for key, value in b0.items():
                if key.startswith("style_bank."):
                    compare(value, b["prompt"][key], "Descriptor unchanged during training")
                elif "projector" in key and (arm == "r4_shuffled" or key.endswith((".0.weight", ".0.bias", "expansion"))):
                    compare(a0[key], value, "Matched R4 initialization")
            if arm == "r4_shuffled":
                mappings.append(b["config"]["descriptor_control"]["domain_to_descriptor"])
            report["tasks"][f"{arm}/{target}"] = "exact matched data/RNG/centroids and frozen condition"
    assert all(value == mappings[0] for value in mappings)
    report["global_shuffle_mapping"] = mappings[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
