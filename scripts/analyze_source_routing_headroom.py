"""Read-only source-teacher diversity diagnostic; no new router or training.

Hard selection oracle is an upper bound only for choosing ONE teacher's
top-1 prediction. It is not an upper bound for class-dependent logit routing
or the Base/Pooled/Weighted combined teacher used by SPL.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.round_analysis_metrics import teacher_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--image_cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["r4_real"])
    args = p.parse_args()
    report = {"arms": {}, "uses_target_labels": "offline accuracy only",
              "limits": ["No style similarities are extracted or fitted here.",
                         "Hard teacher-selection oracle is NOT an upper bound for class-dependent SPL routing.",
                         "Teacher agreement alone does not establish equal calibration or NLL."]}
    for arm in args.arms:
        tasks = {}
        for target in ("art", "clipart", "product", "real_world"):
            root = args.root / arm / target
            snapshot = torch.load(root / "mechanism/step1000.pt", map_location="cpu", weights_only=True)
            config = json.loads((root / "config.json").read_text())
            cache = torch.load(args.image_cache / f"{target}.pt", map_location="cpu", weights_only=True)
            texts, images, labels = snapshot["text_features"], cache["normalized"], cache["labels"]
            source_logits = torch.stack([images @ text.T for text in texts[:3]])
            predictions = source_logits.argmax(-1)
            correct = predictions == labels[None, :]
            consensus = (predictions == predictions[:1]).all(0)
            scores, _, _ = teacher_scores(images, cache["raw"], texts[:3], texts[3], texts[3],
                                          snapshot["running_means"], config["w_scale"])
            per_class = []
            for cls in labels.unique():
                mask = labels == cls
                per_class.append({"class_index": cls.item(), "count": mask.sum().item(),
                                  "source_accuracy": correct[:, mask].float().mean(-1).tolist(),
                                  "all_source_agreement": consensus[mask].float().mean().item(),
                                  "hard_selection_oracle_accuracy": correct[:, mask].any(0).float().mean().item()})
            tasks[target] = {"source_domain_order": config["source_domain_order"],
                "source_accuracy": correct.float().mean(-1).tolist(),
                "all_source_prediction_agreement": consensus.float().mean().item(),
                "hard_selection_oracle_accuracy": correct.any(0).float().mean().item(),
                "uniform_source_logits_accuracy": (source_logits.mean(0).argmax(-1) == labels).float().mean().item(),
                "semantic_weighted_source_accuracy": (scores["weighted_source"].argmax(-1) == labels).float().mean().item(),
                "per_class": per_class}
        report["arms"][arm] = tasks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({arm: {t: {k: v for k, v in row.items() if k != "per_class"} for t, row in tasks.items()}
                      for arm, tasks in report["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
