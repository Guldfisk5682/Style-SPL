"""Scientific PNG/PDF figures for the paired controlled trajectories."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["r1_shared", "r2_pooled", "r3_split", "r4_silu32"]
LABELS = ["Shared S1", "+ Independent pooled", "+ Split source/target", "+ SiLU r=32"]
COLORS = ["#455a64", "#e69f00", "#0072b2", "#009e73"]
TARGETS = ["art", "clipart", "product", "real_world"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    args = p.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    output = args.root / "figures"; output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    def trajectory_plot(key, ylabel, filename, log=False, transform=lambda x: x):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for target, ax in zip(TARGETS, axes.flat):
            for arm, label, color in zip(ARMS, LABELS, COLORS):
                task = summary["arms"][arm]["tasks"].get(target, {})
                rows = [r for r in task.get("trajectory", []) if r.get(key) is not None]
                if rows:
                    ax.plot([r["step"] for r in rows], [transform(r[key]) for r in rows], color=color,
                            label=label, marker=".", linewidth=1.4)
            ax.set_title(target.replace("_", " ").title()); ax.set_ylabel(ylabel)
            ax.grid(alpha=.2); ax.set_xlabel("Optimizer update")
            if log:
                ax.set_yscale("log")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.025))
        fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, filename)
    trajectory_plot("source_target_text_cosine", "1 - same-class cosine (source/target)", "text_domain_separation", True, lambda x: max(1e-9, 1-x))
    trajectory_plot("source_source_text_cosine", "1 - same-class cosine (source/source)", "source_text_diversity", True, lambda x: max(1e-9, 1-x))
    trajectory_plot("source_target_interval_movement_cosine", "Source/target interval movement cosine", "prompt_comovement")
    trajectory_plot("kl_teacher_to_student", "KL(Teacher || Student), full target pool", "student_teacher_kl", True)
    trajectory_plot("target_to_class_rms", "Target domain / shared class RMS", "domain_class_rms", True)
    trajectory_plot("teacher_combined_accuracy", "Combined teacher accuracy (%)", "teacher_quality", False, lambda x: 100*x)
    trajectory_plot("teacher_pooled_accuracy", "Pooled teacher accuracy (%)", "pooled_teacher_quality", False, lambda x: 100*x)
    trajectory_plot("teacher_weighted_source_accuracy", "Weighted source teacher accuracy (%)", "weighted_teacher_quality", False, lambda x: 100*x)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for target, ax in zip(TARGETS, axes.flat):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            task = summary["arms"][arm]["tasks"].get(target, {})
            rows = task.get("instant_accuracy", [])
            if rows:
                ax.plot([r["step"] for r in rows], [100*r["eval/target_top1_instant"] for r in rows],
                        color=color, label=label, marker=".")
        ax.set_title(target.replace("_", " ").title()); ax.set_ylabel("Instant accuracy (%)")
        ax.set_xlabel("Optimizer update"); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.025))
    fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, "instant_accuracy")
    swap_rows = list(csv.DictReader((args.root / "descriptor_swaps.csv").open())) if (args.root / "descriptor_swaps.csv").exists() else []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for target, ax in zip(TARGETS, axes.flat):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            rows = [r for r in swap_rows if r["arm"] == arm and r["target"] == target and r["recipient"] == target]
            steps = sorted(set(int(r["step"]) for r in rows))
            mean = [np.mean([100*float(r["prediction_flip_rate"]) for r in rows if int(r["step"]) == s]) for s in steps]
            if steps:
                ax.plot(steps, mean, color=color, label=label, marker=".")
        ax.set_title(target.replace("_", " ").title()); ax.set_ylabel("Target descriptor swap: mean prediction flips (%)")
        ax.set_xlabel("Optimizer update"); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.025))
    fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, "descriptor_swap_response")
    if summary["all_complete"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        means = [100*summary["arms"][a]["mean_formal_accuracy"] for a in ARMS]
        bars = ax.bar(LABELS, means, color=COLORS)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, value+.12, f"{value:.2f}", ha="center")
        ax.set_ylim(min(means)-2, max(means)+1); ax.set_ylabel("Four-domain mean temporal accuracy (%)")
        ax.grid(axis="y", alpha=.2); fig.tight_layout(); save(fig, "final_accuracy")


if __name__ == "__main__":
    main()
