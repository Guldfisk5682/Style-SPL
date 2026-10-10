"""Standalone scientific figures for frozen-teacher reliability diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ("art", "clipart", "product", "real_world")
NAMES = ("Art", "Clipart", "Product", "Real World")
ARMS = ("b0", "r4_real", "r4_random")
COLORS = ("#265b91", "#d17b29", "#29846f")


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    report = json.loads((args.root / "summary.json").read_text())
    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    metrics = [*[f"cosine_stage{i}" for i in range(1, 5)], "cosine_mean4",
               *[f"source_z_l2_stage{i}" for i in range(1, 5)], "source_z_l2_mean4"]
    labels = [*[f"Cosine S{i}" for i in range(1, 5)], "Cosine mean4",
              *[f"Source-z L2 S{i}" for i in range(1, 5)], "Source-z L2 mean4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    fields = ("within_image_similarity_vs_negative_nll_pearson", "class_source_adjusted_within_image_pearson")
    for ax, field, title in zip(axes, fields, ("Same image, different source teachers", "Class x source averages removed")):
        data = np.array([[report["tasks"][f"b0/{d}"]["metrics"][m][field] for d in DOMAINS] for m in metrics])
        image = ax.imshow(data, cmap="RdBu", vmin=-.3, vmax=.3, aspect="auto")
        ax.set_xticks(range(4), NAMES, rotation=20)
        ax.set_yticks(range(10), labels)
        ax.set_title(title)
        for i in range(10):
            for j in range(4):
                ax.text(j, i, f"{data[i,j]:+.3f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(data[i,j]) > .22 else "black")
    fig.colorbar(image, ax=axes.tolist(), shrink=.8, label="Similarity vs. -NLL correlation (positive = useful)")
    fig.suptitle("Frozen SPL B0: does closer style predict a better source teacher?", y=1.02)
    save(fig, figures, "b0_stage_reliability")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ("cosine_mean4", "source_z_l2_mean4"), ("Mean4 cosine", "Mean4 source-only standardized L2")):
        for aidx, arm in enumerate(ARMS):
            stats = [report["tasks"][f"{arm}/{d}"]["metrics"][metric]["selection"] for d in DOMAINS]
            mean = np.array([s["nll_reduction_vs_uniform_teacher_expectation"] for s in stats])
            ci = np.array([s["nll_reduction_vs_uniform_teacher_expectation_class_cluster_ci95"] for s in stats])
            ax.errorbar(np.arange(4) + (aidx-1)*.14, mean,
                        yerr=np.maximum(0, np.stack((mean-ci[:,0], ci[:,1]-mean))),
                        marker="o", capsize=3, linestyle="none", color=COLORS[aidx], label=arm)
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set_xticks(range(4), NAMES, rotation=20)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Style-nearest NLL reduction\nvs. uniform hard teacher")
    fig.suptitle("95% class-cluster bootstrap intervals; fixed trained models", fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig, figures, "style_selection_nll")
    with (args.root / "per_class.csv").open() as file:
        classes = list(csv.DictReader(file))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, target, name in zip(axes.flat, DOMAINS, NAMES):
        for metric, color, label in (("cosine_mean4", COLORS[0], "Mean4 cosine"),
                                     ("source_z_l2_mean4", COLORS[1], "Mean4 source-z L2")):
            values = [float(r["class_source_adjusted_within_image_pearson"]) for r in classes
                      if r["arm"] == "b0" and r["target"] == target and r["distance"] == metric
                      and r["class_source_adjusted_within_image_pearson"]]
            ax.hist(values, bins=np.linspace(-1, 1, 21), histtype="step", linewidth=1.8, color=color, label=label)
        ax.axvline(0, color="gray", linewidth=.8)
        ax.set_title(name)
        ax.set_xlabel("Within-class adjusted correlation")
        ax.set_ylabel("Number of classes")
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, figures, "b0_per_class_association")
    with (args.root / "semantic_strata.csv").open() as file:
        strata = list(csv.DictReader(file))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, target, name in zip(axes.flat, DOMAINS, NAMES):
        for metric, color, label in (("cosine_mean4", COLORS[0], "Cosine"), ("source_z_l2_mean4", COLORS[1], "Source-z L2")):
            rows = [r for r in strata if r["arm"] == "b0" and r["target"] == target and r["distance"] == metric]
            values = [float(next(r for r in rows if r["stratum"] == s)["nll_reduction_vs_semantic"])
                      for s in ("weak_relative", "middle_relative", "strong_relative")]
            ax.plot(range(3), values, marker="o", color=color, label=label)
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set_xticks(range(3), ("Weak", "Middle", "Strong"))
        ax.set_title(name)
        ax.set_ylabel("Style NLL reduction\nvs. semantic hard choice")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Frozen B0 teachers; semantic strength from R4 Real reference\nBase CLIP predicted class; relative entropy terciles (exploratory)", fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91))
    save(fig, figures, "b0_semantic_strata")


if __name__ == "__main__":
    main()
