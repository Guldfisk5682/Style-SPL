"""Recover frozen B0 teachers and source sampling centroids WITHOUT training.

Python RNG alone drives the original RandomDomainSampler. Its constructor
consumes one complete sequence, and each iterator consumes another full
sequence even when the final epoch is only partly used. Replay that sequence
across all four tasks, using cached frozen RN50 features, not target labels.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from clip_custom import clip
from model import PromptGenerator, Custom_Clip
from runtime import fix_random_seed

DOMAINS = ("art", "clipart", "product", "real_world")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--author_run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fix_random_seed(1)
    torch.set_num_threads(4)
    clip_model, _ = clip.load("RN50", device="cuda")
    clip_model.float().eval().requires_grad_(False)
    encoder = Custom_Clip(clip_model).eval().requires_grad_(False)
    data_module = load_module("frozen_b0_dataset", args.author_run / "author_code/dataset.py")
    sampler_module = load_module("frozen_b0_sampler", args.author_run / "author_code/samplers.py")
    author_provenance = json.loads((args.author_run / "provenance.json").read_text())
    for name in ("dataset.py", "samplers.py"):
        assert sha(args.author_run / "author_code" / name) == author_provenance["runtime_sha256"][name]
    author_summary = json.loads((args.runs / "b0_protocol_checked_20261009_seed1/summary.json").read_text())
    checkpoints = {t["target"]: Path(t["checkpoint"]) for t in author_summary["tasks"]}
    old_texts = torch.load(args.runs / "style_reliability_20261010/styles/b0_texts.pt", map_location="cpu", weights_only=True)
    caches = {d: torch.load(args.runs / "analysis_round_20261009/image_embeddings" / f"{d}.pt", map_location="cpu", weights_only=True) for d in DOMAINS}
    classes = caches[DOMAINS[0]]["classes"]
    assert all(c["classes"] == classes for c in caches.values())
    exports, audits = {}, {}
    # CLIP/Prompt initializations do not use Python random. Reset exactly once,
    # and retain source sampler RNG state across the author's four task order.
    random.seed(1)
    for target in DOMAINS:
        sources = [d for d in DOMAINS if d != target]
        dataset = data_module.MultiSourceDataset(str(args.data), sources)
        assert dataset.classnames == classes
        paths = [Path(item.impath).resolve().relative_to(args.data.resolve()).as_posix() for item in dataset.data]
        manifest = hashlib.sha256(json.dumps(paths).encode()).hexdigest()
        audit_path = args.runs / "b0_protocol_checked_20261009_seed1" / target / "data_protocol_checks.jsonl"
        original_audit = json.loads(audit_path.read_text().splitlines()[0])
        assert manifest == original_audit["source_manifest_sha256"]
        maps = {s: {path: i for i, path in enumerate(caches[s]["paths"])} for s in sources}
        features = torch.stack([caches[sources[item.domain]]["raw"][maps[sources[item.domain]][path]]
                                for item, path in zip(dataset.data, paths)]).cuda()
        labels = torch.tensor([item.label for item in dataset.data], device="cuda")
        domains = torch.tensor([item.domain for item in dataset.data], device="cuda")
        # Only SOURCE labels enter the preparation step. No held-out label access.
        for item, path in zip(dataset.data, paths):
            source = sources[item.domain]
            assert classes[item.label] == Path(path).parts[1]
            assert source != target
        sampler = sampler_module.RandomDomainSampler(dataset.data, 30, 3)
        full = torch.zeros(3, len(classes), 1024)
        full_counts = torch.zeros(3, len(classes), dtype=torch.int64)
        for source_i in range(3):
            for cls in range(len(classes)):
                selected = (domains == source_i) & (labels == cls)
                full[source_i, cls] = features[selected].double().mean(0).float().cpu()
                full_counts[source_i, cls] = selected.sum().cpu()
        means = torch.zeros(3, len(classes), 1024, device="cuda")
        counts = torch.zeros(3, len(classes), device="cuda")
        expected = {int(row["step"]): float(row["train/source_centroid_coverage"])
                    for row in csv.DictReader((args.runs / "b0_protocol_checked_20261009_seed1" / target / "train_metrics.csv").open())}
        sequence, cursor, history, coverage_checks = [], 0, [], 0
        for step in range(1, 1001):
            if cursor >= len(sequence):
                sequence = list(iter(sampler))
                cursor = 0
                assert len(sequence) % 30 == 0
            indexes = sequence[cursor:cursor + 30]
            cursor += 30
            assert len(indexes) == 30
            history.append(indexes)
            idx = torch.tensor(indexes, device="cuda")
            y, d, raw = labels[idx], domains[idx], features[idx]
            for source_i in range(3):
                mask = d == source_i
                assert mask.sum() == 10
                for cls in torch.unique(y[mask]):
                    batch = raw[mask][y[mask] == cls]
                    means[source_i, cls] = means[source_i, cls] * counts[source_i, cls] + batch.sum(0)
                    counts[source_i, cls] += len(batch)
                    means[source_i, cls] /= counts[source_i, cls]
            if step in expected:
                observed = (counts > 0).float().mean().item()
                assert observed == expected[step], (target, step, observed, expected[step])
                coverage_checks += 1
        assert (counts > 0).all() and (counts.sum(1) == 10000).all()
        path = checkpoints[target]
        before = sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        prompt = PromptGenerator(classes, clip_model, sources, target,
                                 SimpleNamespace(device="cuda", M1=16, M2=16, style_spl_enabled=0))
        prompt.load_state_dict(saved["prompt"], strict=True)
        prompt.eval().requires_grad_(False)
        source_texts = torch.stack([encoder.forward_txt(prompt.forward_source(i), prompt.tokenized_prompts) for i in range(3)]).cpu()
        torch.testing.assert_close(source_texts, old_texts[target]["source_texts"], atol=0, rtol=0)
        pooled, student = prompt()
        pooled_text = encoder.forward_txt(pooled, prompt.tokenized_prompts).cpu()
        student_text = encoder.forward_txt(student, prompt.tokenized_prompts).cpu()
        assert sha(path) == before
        trace = torch.tensor(history)
        record = {"source_manifest_matches_training": True, "source_manifest_sha256": manifest,
                  "coverage_checks_matched": coverage_checks, "source_samples_per_domain": counts.sum(1).tolist(),
                  "sampled_indices_sha256": hashlib.sha256(trace.numpy().tobytes()).hexdigest(),
                  "source_texts_match_prior_export_exactly": True, "checkpoint_sha256": before,
                  "recovery_limit": "Original checkpoint omitted centroids; sampling order is replayed and all saved coverage checks match, but no original centroid tensor exists for bitwise verification. Cached image encoding batch=64 versus training batch=30 may introduce small numerical differences.",
                  "full_vs_replay_centroid_max_abs": (full - means.cpu()).abs().max().item()}
        exports[target] = {"source_order": sources, "classes": classes, "source_texts": source_texts,
                           "pooled_text": pooled_text, "student_text": student_text,
                           "student_temporal_text": saved["prompt"]["target_features"],
                           "base_text": old_texts[target]["base_text"], "logit_scale": old_texts[target]["logit_scale"],
                           "replay_centroids": means.cpu(), "replay_counts": counts.cpu(),
                           "full_centroids": full, "full_counts": full_counts, "replay_indices": trace,
                           "audit": record}
        audits[target] = record
        print(json.dumps({"target": target, **record}), flush=True)
        del prompt, features
    torch.save(exports, args.output / "b0_static_state.pt")
    provenance = {"training_updates": 0, "target_labels_accessed": False, "tasks": audits,
                  "author_runtime_sha256": {n: sha(args.author_run / "author_code" / n) for n in ("main.py", "samplers.py", "dataset.py")},
                  "preparation_code_sha256": sha(Path(__file__))}
    (args.output / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
