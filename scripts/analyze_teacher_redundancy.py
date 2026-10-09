"""CPU-only saved-state teacher redundancy and logit-gradient diagnostics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from clip_custom import clip
from scripts.round_analysis_metrics import teacher_scores


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--image_cache", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    classes = sorted(p.name for p in (args.data_root / "art").iterdir() if p.is_dir())
    model, _ = clip.load("RN50", device="cpu")
    model.float().eval().requires_grad_(False)
    tokens = clip.tokenize([f"A photo of a {name}" for name in classes])
    base_text = F.normalize(model.encode_text(tokens).float(), dim=-1)
    scale = model.logit_scale.exp()
    report = {"protocol": "Frozen saved-state CPU diagnostics; no optimizer or training updates",
              "logit_scale": scale.item(), "tasks": {}}
    for arm in ["r1_shared", "r2_pooled", "r3_split", "r4_silu32"]:
        for target in ["art", "clipart", "product", "real_world"]:
            state_path = args.run_root / arm / target / "mechanism/step1000.pt"
            if not state_path.exists():
                continue
            saved = torch.load(state_path, map_location="cpu", weights_only=True)
            images = torch.load(args.image_cache / f"{target}.pt", map_location="cpu", weights_only=True)
            assert images["classes"] == classes
            config = json.loads((args.run_root / arm / target / "config.json").read_text())
            if not (saved["running_count"] > 0).all():
                raise ValueError("Final weighted teacher requires complete centroids")
            texts = saved["text_features"]
            scores, weights, individual = teacher_scores(images["normalized"], images["raw"],
                texts[:3], texts[3], base_text, saved["running_means"], config["w_scale"])
            scores["adapted_without_base"] = (scores["pooled"] + scores["weighted_source"]) / 2
            log_student = (scale * (images["normalized"] @ texts[4].T)).log_softmax(-1)
            student = log_student.exp()
            probabilities = {k: (scale*v).softmax(-1) for k, v in scores.items()}
            stats = {}
            for name, probability in probabilities.items():
                kl = (probability*(probability.clamp_min(1e-38).log() - log_student)).sum(-1)
                stats[name] = {"kl_to_student": kl.mean().item(),
                    "prediction_disagreement_with_student": (scores[name].argmax(-1) != student.argmax(-1)).float().mean().item(),
                    "accuracy_offline_only": (scores[name].argmax(-1) == images["labels"]).float().mean().item()}
            gradient = {k: student - p for k, p in probabilities.items()}
            combined, base, adapted = gradient["combined"], gradient["base"], gradient["adapted_without_base"]
            mean_norm = lambda x: x.double().norm(dim=-1).mean().item()
            entry = {"branches": stats,
                "pooled_weighted_prediction_disagreement": (scores["pooled"].argmax(-1) != scores["weighted_source"].argmax(-1)).float().mean().item(),
                "individual_source_kl_to_student": [(p*(p.clamp_min(1e-38).log()-log_student)).sum(-1).mean().item() for p in (scale*individual).softmax(-1)],
                "logit_gradient": {"semantics": "Soft CE derivative wrt scaled Student logits: q_student - p_teacher; detached Teacher; not a parameter-gradient attribution",
                    "combined_norm": mean_norm(combined), "base_only_norm": mean_norm(base), "adapted_without_base_norm": mean_norm(adapted),
                    "adapted_without_base_to_combined_norm_ratio": mean_norm(adapted)/mean_norm(combined),
                    "combined_vs_base_flattened_cosine": F.cosine_similarity(combined.flatten().double(), base.flatten().double(), dim=0).item()},
                "snapshot_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
            report["tasks"][f"{arm}/{target}"] = entry
            print(json.dumps({"arm": arm, "target": target, "branch_kl": {k: v["kl_to_student"] for k, v in stats.items()},
                              "gradient": entry["logit_gradient"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
