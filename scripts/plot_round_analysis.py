"""Generate PNG/PDF figures from the immutable round-analysis metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ["art", "clipart", "product", "real_world"]
LABELS = {"art": "Art", "clipart": "Clipart", "product": "Product", "real_world": "Real World", "pooled": "Pooled"}


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def heatmaps(entries, matrices, labels, titles, destination, filename, suptitle, digits=5, vmin=None, vmax=1.):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    if vmin is None:
        vmin = min(float(np.array(m).min()) for m in matrices)
    for ax, matrix, names, title in zip(axes.flat, matrices, labels, titles):
        matrix = np.array(matrix)
        artist = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                fraction = (matrix[i,j]-vmin) / max(vmax-vmin, 1e-12)
                ax.text(j, i, f"{matrix[i,j]:.{digits}f}", ha="center", va="center", fontsize=8,
                        color="white" if fraction < .48 else "black")
    fig.suptitle(suptitle, fontsize=13)
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7, label="Cosine similarity (magnified range)")
    save(fig, destination, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_root", type=Path)
    args = parser.parse_args()
    root = args.analysis_root
    report = json.loads((root / "analysis.json").read_text())
    entries = [report["targets"][d] for d in DOMAINS]
    out = root / "figures"; out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, domain in zip(axes.flat, DOMAINS):
        for filename, label, color, marker, line in (("baseline_eval_metrics.csv", "Corrected SPL", "#555555", "o", "--"),
                                                    ("eval_metrics.csv", "Style-SPL", "#1673b1", "D", "-")):
            values = rows(root / domain / filename)
            x = [int(r["step"]) for r in values]; y = [100*float(r["eval/target_top1_instant"]) for r in values]
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=line)
            ax.annotate(f"{y[-1]:.2f}%", (x[-1], y[-1]), xytext=(-5, 8 if label=="Corrected SPL" else -16),
                        textcoords="offset points", ha="right", color=color)
        ax.set_title(LABELS[domain]); ax.set_xlabel("Optimizer update"); ax.set_ylabel("Instant target accuracy (%)")
        ax.set_xticks([200,400,600,800,1000]); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("Recorded Instant Target Accuracy | pre-update text, shuffled dropped-tail evaluation", fontsize=12)
    save(fig, out, "instant_target_accuracy")
    labels = [[LABELS[n] for n in e["domain_order"]] for e in entries]
    titles = ["Target task: "+LABELS[d] for d in DOMAINS]
    for key, filename, title, vmin in (("flattened_cosine", "domain_prompt_cosine", "Final domain-prompt cosine | flattened 16 x 512 tokens", None),
                                      ("centered_cosine", "domain_prompt_centered_cosine", "Domain-prompt cosine after subtracting common prompt", -1.)):
        heatmaps(entries, [e["prompt_final"][key] for e in entries], labels, titles, out, filename, title, vmin=vmin)
    heatmaps(entries, [e["b0_prompt_final"]["flattened_cosine"] for e in entries], labels, titles, out,
             "baseline_domain_prompt_cosine", "Corrected SPL | final domain-prompt cosine", digits=3)
    for key, filename, title in (("text_final", "same_class_text_cosine", "Style-SPL | same-class cross-domain text cosine, mean over 65 classes"),
                                 ("b0_text_final", "baseline_same_class_text_cosine", "Corrected SPL | same-class cross-domain text cosine, mean over 65 classes")):
        heatmaps(entries, [e[key]["mean"] for e in entries], labels, titles, out, filename, title)
    cross = report["cross_target_models"]
    heatmaps(entries, [cross[k]["mean"] for k in ("post_update", "temporal", "b0_post_update", "b0_temporal")],
             [[LABELS[d] for d in DOMAINS]]*4,
             ["Style-SPL: final post-update", "Style-SPL: temporal average", "Corrected SPL: final post-update", "Corrected SPL: temporal average"],
             out, "cross_target_model_text_cosine", "Same-class TARGET text cosine across separately trained models", digits=4)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(4)
    for offset, key, label, color in ((-.24, "b0_prompt_final", "SPL target / shared class RMS", "#555555"),
                                    (0, "prompt_final", "Style target / shared class RMS", "#1673b1")):
        values = [e[key]["domain_to_shared_class_rms_ratio"][4] for e in entries]
        bars = axes[0].bar(x+offset, values, width=.24, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    values = [e["prompt_final"]["centered_domain_to_shared_class_rms_ratio"][4] for e in entries]
    bars = axes[0].bar(x+.24, values, width=.24, label="Style target residual / shared class RMS", color="#dc8d28")
    axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axes[0].set_yscale("log"); axes[0].set_ylim(.35, 650); axes[0].set_ylabel("RMS ratio (log scale)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
    axes[0].set_xticks(x, [LABELS[d] for d in DOMAINS]); axes[0].set_title("Domain vs shared CLASS context (16 tokens each)")
    percentages = [100*e["prompt_final"]["common_style_energy_fraction"] for e in entries]
    bars = axes[1].bar(x, percentages, color="#1673b1")
    axes[1].bar_label(bars, fmt="%.3f%%", padding=3)
    axes[1].set_ylim(0, 108); axes[1].set_ylabel("Fraction of total prompt energy (%)")
    axes[1].set_xticks(x, [LABELS[d] for d in DOMAINS]); axes[1].set_title("Common component across five bank entries")
    save(fig, out, "prompt_rms_ratios")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, key, title in zip(axes, ("signed_shares", "standalone_energy_shares"),
                              ("Exact signed contribution (cross terms included)", "Standalone stage energy (cross terms excluded)")):
        for i in range(4):
            values = [100*e["stage_domain_difference"]["source_target_pairs"][key][i] for e in entries]
            bars = ax.bar(x+(i-1.5)*.2, values, width=.2, label=f"Stage {i+1}")
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(x, [LABELS[d] for d in DOMAINS]); ax.set_ylabel("Share (%)"); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("Stage contributions to source-target DOMAIN PROMPT differences | final checkpoint")
    save(fig, out, "stage_domain_difference")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    teacher_names = ["base", "pooled", "weighted_source", "combined", "student_post_update", "student_temporal"]
    teacher_labels = ["Base", "Pooled", "Weighted", "Combined", "Target live", "Target avg"]
    for ax, e, domain in zip(axes.flat, entries, DOMAINS):
        values = [100*e["teacher_diagnostic"]["accuracy"][k] for k in teacher_names]
        bars = ax.bar(range(6), values, color=["#999999", "#6295ad", "#45809d", "#235f80", "#dc8d28", "#aa6817"])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_xticks(range(6), teacher_labels, rotation=20); ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)"); ax.set_title(f"{LABELS[domain]} | n={e['teacher_diagnostic']['n']}")
    fig.suptitle("Read-only final-state diagnostic | all target images, no dropped tail, no updates")
    save(fig, out, "teacher_accuracy")
    print(f"Created {len(list(out.glob('*.png')))} figures, each with PDF export.")


if __name__ == "__main__":
    main()
