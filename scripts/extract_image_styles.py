"""Read-only, label-free per-image RN50 style extraction for saved teachers."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from clip_custom import clip
from model import PromptGenerator, Custom_Clip
from style import RN50StyleExtractor, UnlabelledImages, preprocess_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--export_b0", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    model, preprocess = clip.load("RN50", device="cuda")
    model.float().eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)
    # Labels in the previous embedding caches are NOT read in this extractor.
    for target in args.targets:
        destination = args.output / f"{target}.pt"
        if destination.exists():
            raise FileExistsError(destination)
        dataset = UnlabelledImages(args.data / target, preprocess)
        reference = torch.load(args.runs / "analysis_round_20261009/image_embeddings" / f"{target}.pt",
                               map_location="cpu", weights_only=True)
        paths = [str(path.relative_to(args.data)) for path in dataset.paths]
        assert paths == reference["paths"]
        bank_path = args.runs / "s1_protocol_checked_20261009_seed1" / target / "style_bank.pt"
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        identity = bank["metadata"]["cache_identity"]
        assert dataset.manifest() == identity["manifests"][target]
        assert preprocess_identity(preprocess) == identity["preprocess_id"]
        bn_before = {k: v.clone() for k, v in model.visual.named_buffers()}
        blocks = [[] for _ in range(4)]
        raw = []
        with RN50StyleExtractor(model.visual) as extractor:
            for images in DataLoader(dataset, batch_size=64, num_workers=4,
                                     shuffle=False, drop_last=False, pin_memory=True):
                statistics, embedding = extractor(images.cuda())
                for stage, (mean, std) in enumerate(statistics):
                    blocks[stage].append(torch.cat((mean, std), -1).cpu())
                raw.append(embedding.cpu())
        blocks = [torch.cat(values) for values in blocks]
        raw = torch.cat(raw)
        raw_error = (raw - reference["raw"]).abs().max().item()
        torch.testing.assert_close(raw, reference["raw"], atol=3e-5, rtol=3e-5)
        bank_errors = []
        for block, (mean, std) in zip(blocks, bank["target"]):
            expected = torch.cat((mean, std))
            observed = block.double().mean(0).float()
            bank_errors.append((observed - expected).abs().max().item())
            torch.testing.assert_close(observed, expected, atol=3e-5, rtol=3e-5)
        for k, v in model.visual.named_buffers():
            torch.testing.assert_close(v, bn_before[k], atol=0, rtol=0)
        metadata = {"target": target, "n": len(paths), "manifest": dataset.manifest(),
                    "preprocess": preprocess_identity(preprocess), "training_updates": 0,
                    "target_labels_accessed": False, "bn_unchanged": True,
                    "raw_embedding_max_abs_error": raw_error, "bank_mean_max_abs_errors": bank_errors,
                    "reference_bank_sha256": sha(bank_path), "extractor_sha256": sha(Path(__file__))}
        torch.save({"blocks": blocks, "paths": paths, "metadata": metadata}, destination)
        destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
    if args.export_b0:
        encoder = Custom_Clip(model).eval()
        summary = json.loads((args.runs / "b0_protocol_checked_20261009_seed1/summary.json").read_text())
        exports = {}
        for task in summary["tasks"]:
            target = task["target"]
            cache = torch.load(args.runs / "analysis_round_20261009/image_embeddings" / f"{target}.pt",
                               map_location="cpu", weights_only=True)
            classes = cache["classes"]
            bank = torch.load(args.runs / "s1_protocol_checked_20261009_seed1" / target / "style_bank.pt",
                              map_location="cpu", weights_only=True)
            sources = bank["metadata"]["source_domain_order"]
            path = Path(task["checkpoint"])
            before = sha(path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            settings = SimpleNamespace(device="cuda", M1=16, M2=16, style_spl_enabled=0)
            prompt = PromptGenerator(classes, model, sources, target, settings)
            prompt.load_state_dict(saved["prompt"], strict=True)
            prompt.eval().requires_grad_(False)
            texts = torch.stack([encoder.forward_txt(prompt.forward_source(i), prompt.tokenized_prompts)
                                 for i in range(3)]).cpu()
            base = model.encode_text(clip.tokenize([f"A photo of a {name}" for name in classes]).cuda()).float()
            base = torch.nn.functional.normalize(base, dim=-1).cpu()
            assert sha(path) == before
            exports[target] = {"source_texts": texts, "base_text": base, "classes": classes,
                               "source_domain_order": sources, "checkpoint_sha256": before,
                               "logit_scale": model.logit_scale.exp().item(),
                               "original_semantic_centroids_available": False}
        torch.save(exports, args.output / "b0_texts.pt")
        print("B0 source texts and frozen CLIP scale exported; no training updates", flush=True)


if __name__ == "__main__":
    main()
