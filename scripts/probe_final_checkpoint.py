"""Counterfactual/logit and gradient diagnostics on preserved existing states."""
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
from controlled_diagnostics import TrajectoryProbe, objective_gradient_probe
from dataloader import load_pseudo_label_data
from dataset import MultiSourceDataset
from samplers import RandomDomainSampler
from model import PromptGenerator, Custom_Clip
from runtime import fix_random_seed, ReplayableLoader, restore_rng
from spl import LossValley, spl_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image_cache", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    fix_random_seed(1)
    clip_model, preprocess = clip.load("RN50", device="cuda")
    clip_model.float().eval().requires_grad_(False)
    encoder = Custom_Clip(clip_model).eval()
    protected = {}
    for target in ["art", "clipart", "product", "real_world"]:
        task = args.run_root / target
        path = task / "checkpoints/last.pth"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(saved["config"])
        config.update(device="cuda", pin_memory=True, diagnostic_image_cache=args.image_cache,
                      counterfactual_step=200, independent_pooled=config.get("independent_pooled", False),
                      split_projector=config.get("split_projector", False),
                      projector_architecture=config.get("projector_architecture", "linear"))
        settings = SimpleNamespace(**config)
        payload = torch.load(task / "style_bank.pt", map_location="cpu", weights_only=True)
        sources = config["source_domain_order"]
        classes = sorted(p.name for p in (args.data_root / sources[0]).iterdir() if p.is_dir())
        prompt = PromptGenerator(classes, clip_model, sources, target, settings, style_bank=payload)
        prompt.load_state_dict(saved["prompt"], strict=True)
        prompt.count = saved["target_feature_count"]
        tokens = clip.tokenize([f"A photo of a {name}" for name in classes]).cuda()
        output = args.output / target
        output.mkdir()
        probe = TrajectoryProbe(output, prompt, encoder, clip_model, tokens, settings)
        probe.snapshot(saved["step"], saved["running_means"].cuda(), saved["running_count"].cuda())
        # Reconstruct the NEXT training batch at the final state. This is a new
        # diagnostic probe, not a claim to recover the historical step1000 gradient.
        target_loader = load_pseudo_label_data(args.data_root / target, preprocess, clip_model, settings, classes)
        dataset = MultiSourceDataset(str(args.data_root), sources, preprocess)
        sampler = RandomDomainSampler(dataset.data, settings.batch_size, len(sources))
        source_loader = DataLoader(dataset, batch_size=settings.batch_size, sampler=sampler,
                                   num_workers=settings.num_workers, pin_memory=True)
        source_stream, target_stream = ReplayableLoader(source_loader), ReplayableLoader(target_loader)
        source_stream.load_state_dict(saved["source_stream"])
        target_stream.load_state_dict(saved["target_stream"])
        restore_rng(saved["rng"])
        target_images, _ = target_stream.next()
        source_images, labels, domains = source_stream.next()
        means, counts = saved["running_means"].cuda().clone(), saved["running_count"].cuda().clone()
        valley = LossValley(); valley.load_state_dict(saved["valley"])
        values = spl_step(prompt, encoder, clip_model, source_images.cuda(), labels.cuda(), domains.cuda(),
                          target_images.cuda(), tokens, means, counts, valley, saved["step"] + 1,
                          settings, return_objectives=True)
        summary = objective_gradient_probe(prompt, values[3], output / "mechanism/final_next_batch_gradients.pt", saved["step"])
        summary["batch_semantics"] = "Next replayable batch at final checkpoint; no update; NOT historical step1000 gradient"
        (output / "mechanism/final_next_batch_gradients.json").write_text(json.dumps(summary, indent=2) + "\n")
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        assert before == after
        protected[str(path)] = before
        print(f"{target}: final read-only probes completed", flush=True)
        del probe, prompt, values, source_stream, target_stream, source_loader, target_loader
    (args.output / "provenance.json").write_text(json.dumps({"training_updates": 0, "unchanged_input_checkpoints": protected}, indent=2) + "\n")


if __name__ == "__main__":
    main()
