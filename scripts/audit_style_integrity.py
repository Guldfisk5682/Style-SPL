"""CPU-only audit of real run artifacts; never changes or resumes training."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from style import DomainStyleProjector, FixedStyleBank, calibrate_initial_output


@torch.no_grad()
def pairwise(projector, bank, names):
    prompts = torch.stack([projector(bank.entry(i)) for i in range(5)])
    if prompts.shape != (5, 16, 512) or not torch.isfinite(prompts).all():
        raise RuntimeError("Invalid style prompts")
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            delta = prompts[i] - prompts[j]
            rms = delta.square().mean().sqrt().item()
            if rms == 0:
                raise RuntimeError(f"Identical prompts: {names[i]}, {names[j]}")
            pairs.append({"a": names[i], "b": names[j], "difference_rms": rms,
                          "max_abs_difference": delta.abs().max().item()})
    return {"shape": list(prompts.shape), "all_ten_pairs_distinct": True,
            "min_difference_rms": min(p["difference_rms"] for p in pairs), "pairs": pairs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.task_root / "config.json").read_text())
    payload = torch.load(args.task_root / "style_bank.pt", map_location="cpu", weights_only=True)
    bank = FixedStyleBank(payload)
    names = [*payload["metadata"]["source_domain_order"], "pooled_source", payload["metadata"]["target_domain"]]
    # Exact CPU initialization seed used before .to(cuda) by PromptGenerator.
    torch.manual_seed(config["style_initialization_seed"])
    projector = DomainStyleProjector()
    calibrate_initial_output(projector, bank)
    report = {"device": "cpu", "target": config["target_domain"], "domain_order": names,
              "initial_prompts": pairwise(projector, bank, names),
              "bank_registered_only_as_buffers": not list(bank.parameters()),
              "bank_requires_grad": any(b.requires_grad for b in bank.buffers()),
              "checkpoints": []}
    snapshots = []
    for path in sorted((args.task_root / "checkpoints").glob("*.pth")):
        # Writers atomically rename .tmp to .pth. Each loaded file is complete.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        for name, original in bank.state_dict().items():
            torch.testing.assert_close(saved["prompt"]["style_bank." + name], original, rtol=0, atol=0)
        projector.load_state_dict({k.removeprefix("style_projector."): v for k, v in saved["prompt"].items()
                                   if k.startswith("style_projector.")}, strict=True)
        initial_expansion = torch.load(args.task_root / "expansion/step0000.pt", map_location="cpu", weights_only=True)
        delta = projector.expansion.detach() - initial_expansion
        if delta.abs().max().item() == 0:
            raise RuntimeError("Expansion was not updated")
        report["checkpoints"].append({"path": str(path), "step": saved["step"],
             "all_eight_bank_buffers_exactly_unchanged": True,
             "expansion_update_l2": delta.norm().item(),
             "expansion_changed_entries": (delta != 0).sum().item(),
             "prompts": pairwise(projector, bank, names)})
        snapshots.append(saved["step"])
    if not snapshots:
        raise RuntimeError("No training checkpoint available yet")
    metrics_path = args.task_root / "train_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open() as file:
            rows = list(csv.DictReader(file))
        keys = [*[f"style/grad_norm_stage_{i}" for i in range(1, 5)], "style/grad_norm_expansion"]
        report["gradient_log"] = {"logged_steps": len(rows), "last_logged_step": int(rows[-1]["step"]),
                                  "minimum_norms": {key: min(float(r[key]) for r in rows) for key in keys}}
        if not all(v > 0 for v in report["gradient_log"]["minimum_norms"].values()):
            raise RuntimeError("Zero stage or Expansion gradient in logged updates")
    report["bank_check_scope"] = "Exact at saved checkpoints, plus no bank parameters; not a per-update buffer checksum."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"target": report["target"], "latest_checkpoint": max(snapshots),
                      "initial_min_pair_rms": report["initial_prompts"]["min_difference_rms"],
                      "checkpoints": [{"step": r["step"], "bank_unchanged": r["all_eight_bank_buffers_exactly_unchanged"],
                                      "expansion_update_l2": r["expansion_update_l2"],
                                      "prompt_min_pair_rms": r["prompts"]["min_difference_rms"]} for r in report["checkpoints"]],
                      "gradient_log": report.get("gradient_log")}, indent=2))


if __name__ == "__main__":
    main()
