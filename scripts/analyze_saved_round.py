"""Extract diagnostics from one completed round, without any parameter update."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from round_analysis_metrics import prompt_geometry, text_geometry, stage_contributions, teacher_scores

DOMAINS = ["art", "clipart", "product", "real_world"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def evaluate_teachers(cache, texts, base_text, checkpoint, config, scale, device):
    names = ["base", "pooled", "weighted_source", "combined", "student_post_update", "student_temporal"]
    predictions = {name: [] for name in names}
    confidence = {name: [] for name in names}
    source_predictions = [[] for _ in range(3)]
    weight_sum = torch.zeros(3, dtype=torch.double)
    entropy_sum = max_sum = 0.
    weight_count = 0
    centroids = checkpoint["running_means"].to(device)
    if not (checkpoint["running_count"] > 0).all():
        raise RuntimeError("Weighted teacher unavailable: incomplete saved centroids")
    for start in range(0, len(cache["labels"]), 128):
        raw = cache["raw"][start:start+128].to(device)
        normalized = cache["normalized"][start:start+128].to(device)
        scores, weights, source_scores = teacher_scores(normalized, raw, texts[:3], texts[3], base_text, centroids, config["w_scale"])
        scores["student_post_update"] = normalized @ texts[4].T
        # Do NOT normalize averaged text before predicting: retain original inference.
        scores["student_temporal"] = normalized @ checkpoint["prompt"]["target_features"].to(device).T
        for name, score in scores.items():
            predictions[name].append(score.argmax(-1).cpu())
            confidence[name].append((scale * score).softmax(-1).max(-1).values.cpu())
        for i in range(3):
            source_predictions[i].append(source_scores[i].argmax(-1).cpu())
        count = weights.shape[1] * weights.shape[2]
        entropy_sum += (-(weights * weights.clamp_min(1e-38).log()).sum(0)).double().sum().item()
        max_sum += weights.max(0).values.double().sum().item()
        weight_sum += weights.double().sum((1, 2)).cpu()
        weight_count += count
    predictions = {k: torch.cat(v) for k, v in predictions.items()}
    confidence = {k: torch.cat(v) for k, v in confidence.items()}
    labels = cache["labels"]
    accuracy = {k: (v == labels).float().mean().item() for k, v in predictions.items()}
    per_class = []
    for label, name in enumerate(cache["classes"]):
        mask = labels == label
        per_class.append({"class": name, "n": mask.sum().item(),
                          **{k: (v[mask] == label).float().mean().item() for k, v in predictions.items()}})
    base_correct = predictions["base"] == labels
    combined_correct = predictions["combined"] == labels
    disagreement = {}
    for branch in ("pooled", "weighted_source", "combined"):
        correct = predictions[branch] == labels
        disagreement[branch] = {
            "rescued_base_errors": (correct & ~base_correct).sum().item(),
            "spoiled_base_correct": (~correct & base_correct).sum().item(),
            "agreement_with_base": (predictions[branch] == predictions["base"]).float().mean().item()}
    result = {"n": len(labels), "accuracy": accuracy,
              "source_individual_accuracy": [(torch.cat(p) == labels).float().mean().item() for p in source_predictions],
              "mean_confidence": {k: v.mean().item() for k, v in confidence.items()},
              "routing": {"source_order": config["source_domain_order"],
                          "mean_weight": (weight_sum / weight_count).tolist(),
                          "mean_source_weight_entropy": entropy_sum / weight_count,
                          "mean_max_source_weight": max_sum / weight_count},
              "vs_base": disagreement,
              "student_agreement_with_combined": (predictions["student_post_update"] == predictions["combined"]).float().mean().item(),
              "student_correct_teacher_wrong": ((predictions["student_post_update"] == labels) & ~combined_correct).sum().item(),
              "teacher_correct_student_wrong": ((predictions["student_post_update"] != labels) & combined_correct).sum().item()}
    rows = [{"path": path, "label": int(labels[i]),
             **{k: int(p[i]) for k, p in predictions.items()}} for i, path in enumerate(cache["paths"])]
    return result, per_class, rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style_root", type=Path, required=True)
    p.add_argument("--baseline_root", type=Path, required=True)
    p.add_argument("--runtime_root", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier analysis; use a new output directory")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.runtime_root.resolve()))
    from clip_custom import clip
    from model import PromptGenerator, Custom_Clip
    from style import DomainStyleProjector, FixedStyleBank, calibrate_initial_output, preprocess_identity
    from runtime import fix_random_seed
    fix_random_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_model.float().eval().requires_grad_(False)
    encoder = Custom_Clip(clip_model).eval()
    baseline_summary = json.loads((args.baseline_root / "summary.json").read_text())
    baseline_paths = {t["target"]: Path(t["checkpoint"]) for t in baseline_summary["tasks"]}
    protected = [args.style_root / target / "checkpoints/last.pth" for target in DOMAINS] + list(baseline_paths.values())
    before_hashes = {str(path): sha(path) for path in protected}
    report = {"protocol": "Post-training read-only diagnostics; final post-update states; full target data, shuffle=False, drop_last=False.",
              "training_updates": 0, "targets": {},
              "limits": ["Historical teacher accuracy was not logged and cannot be recovered from final states.",
                         "Recorded Instant Accuracy uses pre-update features and shuffled dropped-tail evaluation; replay uses post-update state on full data.",
                         "Across-target comparison contains four separately trained class prompts/projectors; same-model domain comparison isolates conditioning.",
                         "Official B0 checkpoint lacks running centroids; weighted B0 teacher cannot be exactly restored.",
                         "Stage shares decompose raw prompt differences exactly, not nonlinear text-encoder differences."],
              "input_checkpoint_sha256": before_hashes, "preprocess": preprocess_identity(preprocess)}
    exports = {}
    caches = {}
    for target in DOMAINS:
        dataset = ImageFolder(args.data_root / target, transform=preprocess)
        loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)
        raw_features, labels = [], []
        print(f"Extracting frozen image embeddings for offline evaluation: {target}, n={len(dataset)}", flush=True)
        for images, gt in loader:
            raw_features.append(clip_model.visual(images.to(device)).cpu())
            labels.append(gt)
        raw = torch.cat(raw_features)
        caches[target] = {"raw": raw, "normalized": F.normalize(raw, dim=-1), "labels": torch.cat(labels),
                          "classes": dataset.classes,
                          "paths": [str(Path(path).relative_to(args.data_root)) for path, _label in dataset.samples]}
        cache_root = args.output / "image_embeddings"
        cache_root.mkdir(exist_ok=True)
        torch.save(caches[target], cache_root / f"{target}.pt")
    for target in DOMAINS:
        task = args.style_root / target
        config = json.loads((task / "config.json").read_text())
        for name in ("model.py", "style.py", "spl.py", "clip_custom/model.py"):
            if sha(args.runtime_root / name) != config["runtime_sha256"][name]:
                raise RuntimeError(f"Frozen runtime provenance mismatch: {name}")
        classes = caches[target]["classes"]
        assert all(cache["classes"] == classes for cache in caches.values())
        payload = torch.load(task / "style_bank.pt", map_location="cpu", weights_only=True)
        saved = torch.load(task / "checkpoints/last.pth", map_location="cpu", weights_only=True)
        sources = config["source_domain_order"]
        settings = SimpleNamespace(device=device, M1=16, M2=16, style_spl_enabled=1, seed=config["seed"])
        prompt = PromptGenerator(classes, clip_model, sources, target, settings, style_bank=payload)
        prompt.load_state_dict(saved["prompt"], strict=True); prompt.eval().requires_grad_(False)
        bank = FixedStyleBank(payload)
        for name, value in bank.state_dict().items():
            torch.testing.assert_close(saved["prompt"]["style_bank."+name], value, rtol=0, atol=0)
        names = [*sources, "pooled", target]
        tokens = torch.stack([prompt.domain_tokens("source", i)[0] for i in range(3)] +
                             [prompt.domain_tokens("pooled")[0], prompt.domain_tokens("target")[0]])
        full_prompts = [prompt.forward_source(i) for i in range(3)] + list(prompt())
        texts = torch.stack([encoder.forward_txt(t, prompt.tokenized_prompts) for t in full_prompts])
        live_texts_cpu = texts.cpu()
        final_geometry, per_class_cosine = text_geometry(live_texts_cpu)
        torch.manual_seed(config["style_initialization_seed"])
        initial_projector = DomainStyleProjector()
        calibrate_initial_output(initial_projector, bank)
        initial_tokens = torch.stack([initial_projector(bank.entry(i)) for i in range(5)])
        init_geom = prompt_geometry(initial_tokens, torch.ones_like(saved["prompt"]["ctx_cls"]) * .02)
        # Actual initial class RMS was not saved; remove its approximate ratios.
        for key in ("shared_class_prompt_rms", "domain_to_shared_class_rms_ratio", "centered_domain_to_shared_class_rms_ratio"):
            init_geom.pop(key)
        final_geom = prompt_geometry(tokens.cpu(), saved["prompt"]["ctx_cls"])
        stage_tokens = torch.stack([prompt.style_projector.stage_tokens(prompt.style_bank.entry(i)) for i in range(5)])
        stages = stage_contributions(stage_tokens.cpu(), prompt.style_projector.expansion.cpu(), names)
        # .cpu() above only copies a tensor; Module parameters remain on the GPU.
        base_tokens = clip.tokenize([f"A photo of a {name}" for name in classes]).to(device)
        base_text = F.normalize(clip_model.encode_text(base_tokens).float(), dim=-1)
        teacher_result, teacher_classes, prediction_rows = evaluate_teachers(caches[target], texts, base_text, saved, config, encoder.logit_scale.exp(), device)
        b0_saved = torch.load(baseline_paths[target], map_location="cpu", weights_only=True)
        settings.style_spl_enabled = 0
        b0 = PromptGenerator(classes, clip_model, sources, target, settings)
        b0.load_state_dict(b0_saved["prompt"], strict=True); b0.eval().requires_grad_(False)
        b0_tokens = torch.stack([b0.domain_tokens("source", i)[0] for i in range(3)] + [b0.domain_tokens("pooled")[0], b0.domain_tokens("target")[0]])
        b0_texts = torch.stack([encoder.forward_txt(t, b0.tokenized_prompts) for t in ([b0.forward_source(i) for i in range(3)]+list(b0()))])
        b0_geometry, _ = text_geometry(b0_texts.cpu())
        normalized = caches[target]["normalized"].to(device)
        labels = caches[target]["labels"].to(device)
        b0_accuracy = {name: ((normalized @ text.T).argmax(-1) == labels).float().mean().item()
                       for name, text in (("base", base_text), ("pooled", b0_texts[3]), ("student_post_update", b0_texts[4]),
                                          ("student_temporal", b0_saved["prompt"]["target_features"].to(device)))}
        rows = [{"a": names[i], "b": names[j], "class": cls, "cosine": per_class_cosine[i,j,k].item()}
                for i in range(5) for j in range(i+1,5) for k, cls in enumerate(classes)]
        destination = args.output / target; destination.mkdir()
        write_csv(destination / "same_class_cross_domain_cosine.csv", rows)
        write_csv(destination / "teacher_per_class_accuracy.csv", teacher_classes)
        write_csv(destination / "teacher_predictions.csv", prediction_rows)
        for name in ("eval_metrics.csv", "train_metrics.csv", "result.json", "init_diagnostics.json"):
            (destination / name).write_bytes((task / name).read_bytes())
        for name in ("eval_metrics.csv", "train_metrics.csv", "result.json"):
            (destination / ("baseline_"+name)).write_bytes((args.baseline_root / target / name).read_bytes())
        entry = {"domain_order": names, "prompt_initial": init_geom, "prompt_final": final_geom,
                 "b0_prompt_final": prompt_geometry(b0_tokens.cpu(), b0_saved["prompt"]["ctx_cls"]),
                 "text_final": final_geometry, "b0_text_final": b0_geometry,
                 "stage_domain_difference": stages, "teacher_diagnostic": teacher_result,
                 "b0_diagnostic_accuracy": b0_accuracy,
                 "formal_final_accuracy": json.loads((task/"result.json").read_text())["final_accuracy"],
                 "formal_b0_accuracy": json.loads((args.baseline_root/target/"result.json").read_text())["final_accuracy"]}
        report["targets"][target] = entry
        exports[target] = {"domain_order": names, "prompt_tokens": tokens.cpu(), "text_features_post_update": live_texts_cpu,
                           "target_features_temporal": saved["prompt"]["target_features"],
                           "b0_target_post_update": b0_texts[4].cpu(), "b0_target_temporal": b0_saved["prompt"]["target_features"],
                           "classes": classes}
        write_json(destination / "metrics.json", entry)
        print(json.dumps({"target": target, "teachers": teacher_result["accuracy"], "rms_ratio": final_geom["domain_to_shared_class_rms_ratio"],
                          "target_source_text_cosine": [final_geometry["mean"][i][4] for i in range(3)],
                          "stages_signed": stages["source_target_pairs"]["signed_shares"]}), flush=True)
        del prompt, b0, texts, b0_texts, normalized, labels
    cross = {}
    for name, key in (("post_update", "text_features_post_update"), ("temporal", "target_features_temporal"),
                      ("b0_post_update", "b0_target_post_update"), ("b0_temporal", "b0_target_temporal")):
        matrices = [exports[t][key][4] if key == "text_features_post_update" else exports[t][key] for t in DOMAINS]
        stats, values = text_geometry(torch.stack(matrices))
        cross[name] = stats
        rows = [{"target_model_a": DOMAINS[i], "target_model_b": DOMAINS[j], "class": cls, "cosine": values[i,j,k].item()}
                for i in range(4) for j in range(i+1,4) for k, cls in enumerate(exports[DOMAINS[0]]["classes"])]
        write_csv(args.output / f"cross_target_{name}_cosine.csv", rows)
    report["cross_target_models"] = cross
    report["checkpoints_unchanged_after_analysis"] = all(sha(path) == before_hashes[str(path)] for path in protected)
    if not report["checkpoints_unchanged_after_analysis"]:
        raise RuntimeError("Analysis changed a checkpoint")
    torch.save(exports, args.output / "text_feature_exports.pt")
    write_json(args.output / "analysis.json", report)
    print("Read-only diagnostic extraction completed.", flush=True)


if __name__ == "__main__":
    main()
