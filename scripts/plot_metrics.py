"""Rebuild all specified scientific plots from CSV and expansion snapshots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def rows(path):
    if not path.exists():
        return []
    with path.open() as file:
        return [{k: float(v) if v else float("nan") for k,v in row.items()} for row in csv.DictReader(file)]


def plot(root, baseline=None):
    destination = root/"plots"
    destination.mkdir(exist_ok=True)
    train = rows(root/"train_metrics.csv")
    base_train = rows(baseline/"train_metrics.csv") if baseline else []
    panels = {"loss": ["train/loss_total", "train/loss_source_pooled", "train/loss_source_domain_avg", "train/loss_target_soft"],
              "prompt_rms": ["style/prompt_rms_global", *[f"style/prompt_rms_domain_{i}" for i in range(5)]],
              "gradients": [*[f"style/grad_norm_stage_{i}" for i in range(1,5)], "style/grad_norm_expansion", "style/grad_norm_class_prompt"],
              "stage_rms": [f"style/stage_token_rms_{i}" for i in range(1,5)],
              "teacher": ["train/teacher_entropy", "train/teacher_confidence_max_mean", "train/source_centroid_coverage"]}
    for name, keys in panels.items():
        fig,ax = plt.subplots(figsize=(9,5))
        for key in keys:
            if train and key in train[0]:
                ax.plot([r["step"] for r in train], [r[key] for r in train], label=key)
            if base_train and key in base_train[0]:
                ax.plot([r["step"] for r in base_train], [r[key] for r in base_train], linestyle="--", label="B0 "+key)
        if name=="prompt_rms":
            ax.axhline(.02, color="black", linestyle=":", label="initial RMS=.02")
        if name=="teacher":
            enabled = next((r["step"] for r in train if r["train/weighted_teacher_enabled"]), None)
            if enabled is not None:
                ax.axvline(enabled, color="black", linestyle=":", label="weighted teacher enabled (logged step)")
        ax.set_xlabel("Optimizer update"); ax.set_title(root.name+" / "+name)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(destination/(name+".png"),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for path,label in [(root,"S1"), (baseline,"B0")]:
        if path is None: continue
        ev=rows(path/"eval_metrics.csv")
        if ev: ax.plot([r["step"] for r in ev],[r["eval/target_top1_instant"]*100 for r in ev],marker="o",label=label)
    ax.set_xlabel("Optimizer update");ax.set_ylabel("Instant target top-1 (%)")
    if ax.get_legend_handles_labels()[0]: ax.legend()
    fig.tight_layout();fig.savefig(destination/"accuracy.png",dpi=150);plt.close(fig)
    snapshots=sorted((root/"expansion").glob("*.pt"))
    if snapshots:
        matrices=[torch.load(p,map_location="cpu",weights_only=True).numpy() for p in snapshots]
        low=min(a.min() for a in matrices);high=max(a.max() for a in matrices)
        fig,axes=plt.subplots(1,len(snapshots),figsize=(4*len(snapshots),6),squeeze=False)
        for axis,path,matrix in zip(axes[0],snapshots,matrices):
            artist=axis.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=low,vmax=high)
            axis.set_title(path.stem);axis.set_xlabel("Stage");axis.set_ylabel("Domain token")
        fig.colorbar(artist,ax=axes[0].tolist(),label="Unconstrained expansion weight")
        fig.savefig(destination/"expansion.png",dpi=150,bbox_inches="tight");plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("run_root",type=Path);parser.add_argument("--baseline",type=Path)
    args=parser.parse_args();table=[]
    for root in sorted(p for p in args.run_root.iterdir() if p.is_dir() and (p/"train_metrics.csv").exists()):
        base=args.baseline/root.name if args.baseline else None
        plot(root,base)
        for candidate,label in [(root,"S1"),(base,"B0")]:
            if candidate is None or not (candidate/"result.json").exists():continue
            result=json.loads((candidate/"result.json").read_text())
            table.append({"model":label,"target":root.name,"final_temporal_accuracy":result["final_accuracy"],
                          "best_instant_diagnostic_only":result["best_instant_diagnostic_only"]})
    if table:
        with (args.run_root/"comparison.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        means={model:sum(r["final_temporal_accuracy"] for r in table if r["model"]==model and r["final_temporal_accuracy"] is not None)/sum(r["model"]==model and r["final_temporal_accuracy"] is not None for r in table)
               for model in {r["model"] for r in table} if any(r["model"]==model and r["final_temporal_accuracy"] is not None for r in table)}
        (args.run_root/"comparison_means.json").write_text(json.dumps(means,indent=2)+"\n")


if __name__=="__main__":main()
