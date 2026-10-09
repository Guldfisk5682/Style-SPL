"""CPU full-pool paired predictions from saved temporal text features.

Bootstrap intervals describe image sampling with the three trained models
held fixed. They do NOT account for training seeds or control-draw variance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--image_cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    arms = ("r4_real", "r4_shuffled", "r4_random")
    report = {"prediction_type": "Final saved temporal-average target text; full fixed target pool",
              "limitations": "Image bootstrap conditional on fixed trained models, NOT seed/control variance",
              "targets": {}, "comparisons": {}}
    rng = np.random.default_rng(20261010)
    bootstrap = {arm: np.zeros(10000) for arm in arms[1:]}
    final_differences = {arm: [] for arm in arms[1:]}
    for target in ("art", "clipart", "product", "real_world"):
        cache = torch.load(args.image_cache / f"{target}.pt", map_location="cpu", weights_only=True)
        rows, correct, predictions = {}, {}, {}
        for arm in arms:
            checkpoint = torch.load(args.root / arm / target / "checkpoints/last.pth", map_location="cpu", weights_only=True)
            assert checkpoint["target_feature_count"] > 0
            text = checkpoint["prompt"]["target_features"]
            predictions[arm] = (cache["normalized"] @ text.T).argmax(-1)
            correct[arm] = predictions[arm] == cache["labels"]
            rows[arm] = {"accuracy": correct[arm].float().mean().item(),
                         "temporal_feature_count": checkpoint["target_feature_count"]}
        pairs = {}
        for arm in arms[1:]:
            difference = correct["r4_real"].int() - correct[arm].int()
            n = difference.numel()
            counts = [(difference == value).sum().item() for value in (-1, 0, 1)]
            samples = rng.multinomial(n, np.array(counts) / n, size=10000)
            bootstrap[arm] += (samples[:, 2] - samples[:, 0]) / n / 4
            final_differences[arm].append(difference.float().mean().item())
            pairs[arm] = {"real_only_correct": counts[2], "control_only_correct": counts[0],
                          "real_minus_control_pp": 100 * difference.float().mean().item(),
                          "prediction_disagreement_fraction": (predictions["r4_real"] != predictions[arm]).float().mean().item()}
        report["targets"][target] = {"samples": cache["labels"].numel(), "arms": rows, "paired": pairs}
    for arm in arms[1:]:
        report["comparisons"][arm] = {"mean_real_minus_control_pp": 100 * float(np.mean(final_differences[arm])),
                                     "conditional_image_bootstrap_95pct_pp": (100 * np.quantile(bootstrap[arm], [.025, .975])).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
