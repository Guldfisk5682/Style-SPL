"""Office-Home B0/S1 runner. Default configuration comes from verified B0."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from clip_custom import clip
from dataloader import load_data, load_pseudo_label_data
from dataset import MultiSourceDataset
from samplers import RandomDomainSampler
from model import Custom_Clip, PromptGenerator
from runtime import ReplayableLoader, fix_random_seed, gradient_check, parameter_check, rng_state, restore_rng
from style import build_style_bank
from spl import LossValley, spl_step, evaluate
from logging_utils import MetricLogger, style_metrics, write_json

DOMAIN_ORDER = ["art", "clipart", "product", "real_world"]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, required=True, help="Direct office_home root")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, default=Path("cache/style_banks"))
    p.add_argument("--style_spl_enabled", type=int, choices=(0, 1), default=1)
    p.add_argument("--targets", nargs="+", choices=DOMAIN_ORDER, default=DOMAIN_ORDER)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--M1", type=int, default=16)
    p.add_argument("--M2", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prompt_iteration", type=int, default=1000)
    p.add_argument("--prompt_learning_rate", type=float, default=0.005)
    p.add_argument("--evaluation_step", type=int, default=200)
    p.add_argument("--logging_step", type=int, default=10)
    p.add_argument("--checkpoint_step", type=int, default=200)
    p.add_argument("--style_batch_size", type=int, default=30)
    p.add_argument("--w_scale", type=float, default=10)
    p.add_argument("--t_weight", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--backbone", choices=("RN50",), default="RN50")
    p.add_argument("--training_mode", choices=("multi-source",), default="multi-source")
    p.add_argument("--OT_clustering", type=int, choices=(0,), default=0)
    p.add_argument("--enhanced_pseudo_label", type=int, choices=(1,), default=1)
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke", action="store_true", help="Allow absent temporal cache only for short diagnostics")
    p.add_argument("--stop_after", type=int, help="Stop a diagnostic run early, retaining the 1000-step scheduler")
    return p


def checkpoint(path, prompt, optimizer, scheduler, valley, running_means, running_count,
               source_stream, target_stream, step, args, config, best_instant=0):
    payload = {"prompt": prompt.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "step": step, "seed": args.seed,
               "rng": rng_state(), "target_feature_count": prompt.count,
               "valley": valley.state_dict(), "running_means": running_means,
               "running_count": running_count, "source_stream": source_stream.state_dict(),
               "target_stream": target_stream.state_dict(), "config": config,
               "best_instant": best_instant}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_target(args, target, clip_model, preprocess, classnames):
    root = args.output_dir / target
    if root.exists() and args.resume is None:
        raise FileExistsError(f"Preserve existing run: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "expansion").mkdir(exist_ok=True)
    sources = [d for d in DOMAIN_ORDER if d != target]
    bank = None
    if args.style_spl_enabled:
        bank = build_style_bank(clip_model, preprocess, args.data_root, sources, target,
                                args.cache_dir, args.style_batch_size, args.num_workers)
        torch.save(bank, root / "style_bank.pt")
    # Keep B0 loader construction and sampler initialization order.
    target_train = load_pseudo_label_data(args.data_root / target, preprocess, clip_model, args, classnames)
    target_test = load_data(args.data_root / target, preprocess, args)
    source_dataset = MultiSourceDataset(str(args.data_root), sources, preprocess)
    sampler = RandomDomainSampler(source_dataset.data, args.batch_size, len(sources))
    source_train = DataLoader(source_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    prompt = PromptGenerator(classnames, clip_model, sources, target, args, style_bank=bank)
    optimizer = torch.optim.AdamW(list(prompt.parameters()), lr=args.prompt_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.prompt_iteration)
    encoder = Custom_Clip(clip_model).eval()
    tokens = clip.tokenize([f"A photo of a {name}" for name in classnames]).to(args.device)
    valley = LossValley(20)
    running_means = torch.zeros(len(sources), len(classnames), 1024, device=args.device)
    running_count = torch.zeros(len(sources), len(classnames), device=args.device)
    source_stream, target_stream = ReplayableLoader(source_train), ReplayableLoader(target_train)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"target_domain": target, "source_domain_order": sources,
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "style_bank_metadata": bank["metadata"] if bank else None,
                   "evaluation": f"batch{args.batch_size}/shuffle=True/drop_last=True; temporal target text average",
                   "optimizer": "AdamW default weight_decay=0.01, shared lr for all prompt parameters",
                   "scheduler": "CosineAnnealingLR T_max=1000, step every prompt_iteration/20 updates",
                   "style_initialization_seed": args.seed + 1009,
                   "resume_contract": "fixed deterministic CLIP transforms; replay epoch RNG + consumed batch cursor"})
    config["runtime_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in ("train.py", "model.py", "style.py", "spl.py", "runtime.py", "dataloader.py", "dataset.py", "samplers.py", "logging_utils.py", "clip_custom/model.py")}
    config["dependencies"] = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}
    config["trainable_parameters"] = sum(p.numel() for p in prompt.parameters())
    write_json(root / "config.json", config)
    if args.style_spl_enabled:
        write_json(root / "init_diagnostics.json", prompt.init_diagnostics)
        torch.save(prompt.style_projector.expansion.detach().cpu(), root / "expansion/step0000.pt")
    else:
        write_json(root / "init_diagnostics.json", {"mode": "B0", "combined_source_init": "Normal(0,0.02)"})
    initial = {name: p.detach().cpu().clone() for name, p in prompt.named_parameters()}
    start_step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = saved["config"]
        for key in ("target_domain", "source_domain_order", "style_spl_enabled", "seed", "prompt_iteration", "M1", "M2", "batch_size", "prompt_learning_rate", "w_scale", "t_weight"):
            if previous[key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if args.style_spl_enabled:
            for key in ("cache_identity", "counts"):
                if previous["style_bank_metadata"][key] != config["style_bank_metadata"][key]:
                    raise ValueError(f"Resume Style Bank mismatch: {key}")
        prompt.load_state_dict(saved["prompt"], strict=True)
        prompt.count = saved["target_feature_count"]
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        valley.load_state_dict(saved["valley"])
        running_means.copy_(saved["running_means"])
        running_count.copy_(saved["running_count"])
        source_stream.load_state_dict(saved["source_stream"])
        target_stream.load_state_dict(saved["target_stream"])
        restore_rng(saved["rng"])
        start_step = saved["step"]
    logger = MetricLogger(root, [*sources, "pooled", target], append=args.resume is not None)
    checked_updates = 0
    best_instant = saved.get("best_instant", 0) if args.resume else 0
    end_step = min(args.stop_after or args.prompt_iteration, args.prompt_iteration)
    if end_step <= start_step:
        raise ValueError("Requested stop must be after the resumed update")
    try:
        for step in range(start_step + 1, end_step + 1):
            target_data, _pseudo = target_stream.next()  # pseudo, never GT
            source_data, source_label, source_domain = source_stream.next()
            optimizer.zero_grad()
            loss, target_text, metrics = spl_step(prompt, encoder, clip_model,
                source_data.to(args.device), source_label.to(args.device), source_domain.to(args.device),
                target_data.to(args.device), tokens, running_means, running_count, valley, step, args)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite total SPL loss")
            loss.backward()
            norms = gradient_check(prompt, {"total": loss})
            optimizer.step()
            parameter_check(prompt)
            checked_updates += 1
            if step % (args.prompt_iteration / 20) == 0:
                scheduler.step()
            if valley.is_converged:
                prompt.store_txt_features(target_text)
            metrics["train/lr"] = optimizer.param_groups[0]["lr"]
            # Keep optional routing columns present before centroid coverage.
            metrics.setdefault("routing/source_weight_entropy", 0.0)
            metrics.setdefault("routing/source_weight_max_mean", 0.0)
            if step == 1 or step % args.logging_step == 0:
                metrics.update(style_metrics(prompt, norms))
                logger.write("train", step, metrics)
                print(f"{target} step={step} loss={loss.item():.6f} style_rms={metrics.get('style/prompt_rms_global', 0):.5f}", flush=True)
            if args.style_spl_enabled and (step in (100, 500) or step == end_step):
                torch.save(prompt.style_projector.expansion.detach().cpu(), root / f"expansion/step{step:04d}.pt")
            if step % args.evaluation_step == 0:
                # Upstream evaluates the text computed before this step's update.
                accuracy = evaluate(target_test, encoder, target_text, args.batch_size, args.device)
                best_instant = max(best_instant, accuracy)
                logger.write("eval", step, {"eval/target_top1_instant": accuracy})
                print(f"{target} step={step} instant_accuracy={accuracy:.6f}", flush=True)
            if step % args.checkpoint_step == 0:
                checkpoint(root / "checkpoints/latest.pth", prompt, optimizer, scheduler, valley,
                           running_means, running_count, source_stream, target_stream, step, args, config, best_instant)
        parameter_check(prompt)
        deltas = {name: torch.linalg.vector_norm(p.detach().cpu() - initial[name]).item() for name, p in prompt.named_parameters()}
        result = {"target": target, "style_spl_enabled": bool(args.style_spl_enabled),
                  "seed": args.seed, "steps": end_step, "planned_steps": args.prompt_iteration,
                  "checked_updates_this_invocation": checked_updates,
                  "temporal_feature_count": prompt.count, "cache_valid": prompt.count > 0,
                  "best_instant_diagnostic_only": best_instant, "parameter_update_norms": deltas,
                  "training_valid": True, "final_accuracy": None}
        if prompt.count > 0:
            result["final_accuracy"] = evaluate(target_test, encoder, prompt.target_features, args.batch_size, args.device)
            # Separate final-only CSV avoids altering instant curve semantics.
            write_json(root / "final_eval.json", {"step": end_step, "eval/target_top1_final": result["final_accuracy"]})
            logger.writer.add_scalar("eval/target_top1_final", result["final_accuracy"], end_step)
        elif not args.smoke:
            raise RuntimeError("Temporal target text cache was never written; cannot report final accuracy")
        checkpoint(root / "checkpoints/last.pth", prompt, optimizer, scheduler, valley,
                   running_means, running_count, source_stream, target_stream, end_step, args, config, best_instant)
        write_json(root / "result.json", result)
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        write_json(root / "failure.json", {"error": repr(exc), "checked_updates": checked_updates})
        raise
    finally:
        logger.close()


def main():
    args = parser().parse_args()
    if args.prompt_iteration <= 0 or args.batch_size % 3 != 0:
        raise ValueError("Positive training budget and batch divisible by 3 required")
    if args.resume is not None and len(args.targets) != 1:
        raise ValueError("Resume a single target task at a time")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.pin_memory = True
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    fix_random_seed(args.seed)
    clip_model, preprocess = clip.load("RN50", device=args.device)
    clip_model.float().eval().requires_grad_(False)
    classnames = sorted(p.name for p in (args.data_root / DOMAIN_ORDER[0]).iterdir() if p.is_dir())
    if len(classnames) != 65:
        raise ValueError("Office-Home requires 65 source-known classes")
    results = [run_target(args, target, clip_model, preprocess, classnames) for target in args.targets]
    complete = len(results) == 4 and all(r["cache_valid"] and r["steps"] == r["planned_steps"] for r in results)
    scores = [r["final_accuracy"] for r in results if r["final_accuracy"] is not None]
    write_json(args.output_dir / "summary.json", {"complete_four_targets": complete, "tasks": results,
               "mean_accuracy": sum(scores) / len(scores) if scores else None})


if __name__ == "__main__":
    main()
