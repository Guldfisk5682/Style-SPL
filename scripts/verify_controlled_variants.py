"""Real CLIP acceptance gate before running any full-budget ablation."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from clip_custom import clip
from controlled_diagnostics import load_frozen_bank, objective_gradient_probe
from dataset import MultiSourceDataset
from model import PromptGenerator, Custom_Clip
from runtime import fix_random_seed, gradient_check, rng_state
from spl import LossValley, spl_step
from style import UnlabelledImages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--bank_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fix_random_seed(1)
    clip_model, preprocess = clip.load("RN50", device="cuda")
    clip_model.float().eval().requires_grad_(False)
    encoder = Custom_Clip(clip_model).eval()
    sources, target = ["clipart", "product", "real_world"], "art"
    bank, bank_hash = load_frozen_bank(args.bank_root, target, sources, args.data_root, preprocess)
    classes = sorted(p.name for p in (args.data_root / sources[0]).iterdir() if p.is_dir())
    dataset = MultiSourceDataset(str(args.data_root), sources, preprocess)
    # Ten labelled examples per source; no target GT access.
    selected = []
    for domain in range(3):
        ids = [i for i, item in enumerate(dataset.data) if item.domain == domain][:10]
        selected += [dataset[i] for i in ids]
    source_images = torch.stack([x[0] for x in selected]).cuda()
    labels = torch.tensor([x[1] for x in selected], device="cuda")
    domains = torch.tensor([x[2] for x in selected], device="cuda")
    unlabelled = UnlabelledImages(args.data_root / target, preprocess)
    target_images = torch.stack([unlabelled[i] for i in range(30)]).cuda()
    base_tokens = clip.tokenize([f"A photo of a {name}" for name in classes]).cuda()
    report = {"bank_sha256": bank_hash, "variants": {}}
    ref_class = ref_rng = ref_source = ref_pooled = ref_initial_loss = None
    for name, independent, split, arch in [("r1_shared", False, False, "linear"),
        ("r2_pooled", True, False, "linear"), ("r3_split", True, True, "linear"),
        ("r4_silu32", True, True, "silu")]:
        config = SimpleNamespace(device="cuda", seed=1, M1=16, M2=16, style_spl_enabled=1,
            independent_pooled=independent, split_projector=split, projector_architecture=arch,
            bottleneck_dim=32, w_scale=10., t_weight=.5)
        fix_random_seed(1)
        prompt = PromptGenerator(classes, clip_model, sources, target, config, style_bank=bank)
        state_rng = rng_state()
        if ref_class is None:
            ref_class, ref_rng = prompt.ctx_cls.detach().clone(), state_rng["torch"]
            ref_source = {k: v.clone() for k, v in prompt.style_projector.state_dict().items()}
            ref_pooled = prompt.domain_tokens("pooled").detach().clone()
        else:
            torch.testing.assert_close(prompt.ctx_cls, ref_class, rtol=0, atol=0)
            torch.testing.assert_close(state_rng["torch"], ref_rng, rtol=0, atol=0)
            if arch == "linear":
                for k, v in prompt.style_projector.state_dict().items():
                    torch.testing.assert_close(v, ref_source[k], rtol=0, atol=0)
        if independent:
            torch.testing.assert_close(prompt.ctx_source_combined, ref_pooled, rtol=0, atol=0)
            torch.testing.assert_close(prompt.domain_tokens("pooled", descriptor_index=0),
                                       prompt.domain_tokens("pooled", descriptor_index=4), rtol=0, atol=0)
        if split:
            for k, v in prompt.style_projector.state_dict().items():
                torch.testing.assert_close(v, prompt.target_style_projector.state_dict()[k], rtol=0, atol=0)
            assert prompt.style_projector.expansion.data_ptr() != prompt.target_style_projector.expansion.data_ptr()
        bank_before = {k: v.clone() for k, v in prompt.style_bank.state_dict().items()}
        means = torch.zeros(3, 65, 1024, device="cuda")
        counts = torch.zeros(3, 65, device="cuda")
        result = spl_step(prompt, encoder, clip_model, source_images, labels, domains, target_images,
                          base_tokens, means, counts, LossValley(), 1, config, return_objectives=True)
        if ref_initial_loss is None:
            ref_initial_loss = result[0].detach().clone()
        elif arch == "linear":
            torch.testing.assert_close(result[0], ref_initial_loss, rtol=0, atol=0)
        summary = objective_gradient_probe(prompt, result[3], args.output / f"{name}_gradients.pt", 1)
        expansion_before = {n: p.clone() for n, p in prompt.named_parameters() if n.endswith("expansion")}
        optimizer = torch.optim.AdamW(prompt.parameters(), lr=.005)
        result[0].backward()
        norms = gradient_check(prompt, {"loss": result[0]})
        assert all(v > 0 for v in norms.values()), norms
        assert all(p.grad is None for p in clip_model.parameters())
        optimizer.step()
        expansion_changes = {n: (p != expansion_before[n]).sum().item() for n, p in prompt.named_parameters() if n in expansion_before}
        assert all(v > 0 for v in expansion_changes.values())
        for k, v in prompt.style_bank.state_dict().items():
            torch.testing.assert_close(v, bank_before[k], rtol=0, atol=0)
        report["variants"][name] = {"loss": result[0].item(), "all_gradients_nonzero": True,
            "expansion_changed_entries": expansion_changes, "bank_unchanged": True,
            "init_rms": prompt.init_diagnostics["after"]["global_rms"],
            "objective_gradient_summary": summary, "class_initialization_and_rng_match": True}
        del result, prompt, optimizer
    (args.output / "acceptance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"loss": v["loss"], "valid": True} for name, v in report["variants"].items()}))


if __name__ == "__main__":
    main()
