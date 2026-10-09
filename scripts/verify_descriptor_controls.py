"""Actual-run acceptance: two updates per input condition before full queues."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from descriptor_controls import descriptor_control, bank_digest
from scripts.verify_resume import compare
from style import FixedStyleBank, prompt_diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = args.workspace / "Style-SPL"
    fixed = base / "runs/s1_protocol_checked_20261009_seed1"
    previous = base / "runs/controlled_20261009_seed1/r4_silu32"
    report = {"valid": True, "reference_architecture": "R4 SiLU r=32", "conditions": {}}
    saved = {}
    for mode in ("real", "shuffled", "random"):
        output = args.output / mode
        cmd = [sys.executable, "-u", "train.py", "--data_root", str(args.workspace / "da_lab/data/office_home"),
               "--output_dir", str(output), "--fixed_bank_root", str(fixed), "--diagnostic_image_cache",
               str(base / "runs/analysis_round_20261009/image_embeddings"), "--targets", "art",
               "--independent_pooled", "--split_projector", "--projector_architecture", "silu",
               "--bottleneck_dim", "32", "--descriptor_mode", mode, "--descriptor_seed", "20261010",
               "--mechanism_step", "100", "--counterfactual_step", "200", "--stop_after", "2", "--smoke"]
        subprocess.run(cmd, check=True)
        root = output / "art"
        initial = torch.load(root / "mechanism/step0000.pt", map_location="cpu", weights_only=True)["prompt_state"]
        last = torch.load(root / "checkpoints/last.pth", map_location="cpu", weights_only=True)
        gradient = torch.load(root / "mechanism/gradient0001.pt", map_location="cpu", weights_only=True)["summary"]
        config = json.loads((root / "config.json").read_text())
        assert config["trainable_parameters"] == 1167744
        if mode == "real":
            old = torch.load(previous / "art/mechanism/step0000.pt", map_location="cpu", weights_only=True)["prompt_state"]
            compare(initial, old, "R4 original/new real initialization")
            saved[mode] = (initial, last)
        else:
            reference, ref_last = saved["real"]
            compare(initial["ctx_cls"], reference["ctx_cls"], "class initialization")
            compare(initial["ctx_source_combined"], reference["ctx_source_combined"], "pooled initialization")
            for key in ("rng", "source_stream", "target_stream", "running_means", "running_count", "scheduler"):
                compare(last[key], ref_last[key], key)
            for key, value in initial.items():
                if "projector" in key:
                    if mode == "shuffled" or ".0." in key or key.endswith("expansion"):
                        compare(value, reference[key], key)
                    else:
                        a = json.loads((root / "init_diagnostics.json").read_text())["factor"]
                        b = json.loads((args.output / "real/art/init_diagnostics.json").read_text())["factor"]
                        torch.testing.assert_close(value, reference[key] * (a / b), rtol=1e-6, atol=1e-8)
        assert gradient["style_projector"]["target_norm"] == 0
        assert gradient["target_style_projector"]["source_norm"] == 0
        updates = {}
        for key, value in initial.items():
            if key.startswith("style_bank."):
                compare(value, last["prompt"][key], "frozen descriptor buffers")
            elif key.endswith("expansion"):
                updates[key] = (last["prompt"][key] != value).sum().item()
                assert updates[key] > 0
        for key, gradients in gradient["per_parameter"].items():
            expected = "target_norm" if key.startswith("target_style_projector.") else "source_norm"
            assert gradients[expected] > 0, (mode, key)
        first_loss = list(csv.DictReader((root / "train_metrics.csv").open()))[0]["train/loss_total"]
        snapshot = json.loads((root / "mechanism/step0000.json").read_text())
        report["conditions"][mode] = {"initial_loss": float(first_loss),
            "trainable_parameters": config["trainable_parameters"],
            "class_and_pooled_initialization_matched": True,
            "training_rng_and_streams_matched": True,
            "frozen_descriptors": True, "split_gradients_isolated": True,
            "stage_gradients_nonzero": True, "expansion_changes": updates,
            "initial_prompt_rms_by_role": snapshot["prompt"]["domain_prompt_rms"],
            "calibration": json.loads((root / "init_diagnostics.json").read_text()),
            "descriptor_control": config.get("descriptor_control")}
    mappings = {}
    for target in ("art", "clipart", "product", "real_world"):
        bank = FixedStyleBank(torch.load(fixed / target / "style_bank.pt", map_location="cpu", weights_only=True))
        changed, metadata = descriptor_control(bank, target, "shuffled")
        mappings[target] = metadata["domain_to_descriptor"]
    assert all(value == mappings["art"] for value in mappings.values())
    report["fixed_global_shuffle_mapping"] = mappings["art"]
    (args.output / "acceptance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid": True, "mapping": mappings["art"],
                      "losses": {k: v["initial_loss"] for k, v in report["conditions"].items()}}))


if __name__ == "__main__":
    main()
