"""Fixed-sign conflict checks and confidence-conditional AUROC; no fitting."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.analyze_style_reliability import auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    args = p.parse_args()
    out = {"parameters_used_for_routing": False,
           "conditional_auc_protocol": "20 predetermined equal-count confidence rank bins; pair-count-weighted within-bin AUROC. Bins depend on confidence only. Descriptive conditional association, not a fitted router or proof of improvement.",
           "tasks": {}}
    states = torch.load(args.prepared / "b0_static_state.pt", map_location="cpu", weights_only=True)
    for path in sorted(args.root.glob("per_image_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        task = path.stem.removeprefix("per_image_")
        mode, target = task.split("_", 1)
        labels = saved["labels_offline_diagnostic_only"].numpy()
        prediction = saved["branch_predictions"]["combined"].numpy()
        correct = prediction == labels
        scores = {k: v.numpy() for k, v in saved["combined_scores"].items()}
        confidence = scores["teacher_confidence"]
        margin = scores["centroid_class_margin"]
        order = np.argsort(confidence, kind="stable")
        bins = np.array_split(order, 20)
        total_pairs = 0
        weighted_auc = {k: 0. for k in ("centroid_class_margin", "centroid_negative_candidate_distance", "teacher_confidence")}
        bin_rows = []
        for i, indexes in enumerate(bins):
            positives = int(correct[indexes].sum())
            negatives = len(indexes) - positives
            pairs = positives * negatives
            row = {"bin": i, "n": len(indexes), "correct": positives, "errors": negatives,
                   "confidence_min": float(confidence[indexes].min()), "confidence_max": float(confidence[indexes].max()), "scores": {}}
            for name in weighted_auc:
                value = auc(scores[name][indexes], correct[indexes])
                row["scores"][name] = value
                if value is not None:
                    weighted_auc[name] += pairs * value
            total_pairs += pairs
            bin_rows.append(row)
        entry = {"conditional_pair_support": total_pairs,
                 "conditional_auroc": {k: v/total_pairs if total_pairs else None for k, v in weighted_auc.items()},
                 "confidence_bins": bin_rows, "conflict": {}}
        class_distance = saved["distance_matrix_N_source_class"].min(1).values
        true_distance = class_distance[torch.arange(len(labels)), torch.from_numpy(labels)]
        entry["true_class_nearer_than_uniform_wrong_class_fraction"] = float(
            ((class_distance > true_distance[:, None]).sum(1).double() / (class_distance.shape[1]-1)).mean())
        # Parameter-free static aggregation references, not new trained models.
        state = states[target]
        cache = torch.load(args.runs / "analysis_round_20261009/image_embeddings" / f"{target}.pt", map_location="cpu", weights_only=True)
        image = cache["normalized"]
        source = torch.stack([image @ t.T for t in state["source_texts"]], 1)
        base, pooled = image @ state["base_text"].T, image @ state["pooled_text"].T
        weighted = (saved["source_weights"] * source).sum(1)
        combined = (base + pooled + weighted) / 3
        reference_logits = {"uniform_source_logits": source.mean(1), "base_pooled_only": (base + pooled)/2,
                            "base_pooled_uniform_source": (base + pooled + source.mean(1))/3,
                            "combined": combined}
        references = {}
        for name, logits in reference_logits.items():
            pred = logits.argmax(-1).numpy()
            references[name] = {"accuracy": float((pred == labels).mean()),
                                "rescued_combined_errors": int(((pred == labels) & ~correct).sum()),
                                "spoiled_combined_correct": int(((pred != labels) & correct).sum())}
        logits_pool = torch.cat((source, combined[:, None, :]), 1)
        most_confident = (state["logit_scale"] * logits_pool).softmax(-1).max(-1).values.argmax(1)
        pred = logits_pool.argmax(-1)[torch.arange(len(labels)), most_confident].numpy()
        references["max_confidence_source_or_combined"] = {
            "accuracy": float((pred == labels).mean()), "rescued_combined_errors": int(((pred == labels) & ~correct).sum()),
            "spoiled_combined_correct": int(((pred != labels) & correct).sum())}
        entry["static_aggregation_references"] = references
        q4 = np.zeros(len(labels), bool)
        q4[np.array_split(order, 4)[3]] = True
        for name, mask in (("all", np.ones(len(labels), bool)), ("confidence_ge_0.9", confidence >= .9),
                           ("highest_confidence_quartile", q4), ("source_disagreement", saved["source_disagreement"].numpy())):
            conflict = margin < 0
            flagged = mask & conflict
            errors = mask & ~correct
            good = mask & correct
            entry["conflict"][name] = {"n": int(mask.sum()), "errors": int(errors.sum()),
                "flagged": int(flagged.sum()), "flagged_errors": int((flagged & errors).sum()),
                "flagged_correct": int((flagged & good).sum()),
                "error_prevalence": float(errors.sum()/mask.sum()) if mask.any() else None,
                "error_precision": float((flagged & errors).sum()/flagged.sum()) if flagged.any() else None,
                "error_recall": float((flagged & errors).sum()/errors.sum()) if errors.any() else None,
                "correct_rejection_rate": float((flagged & good).sum()/good.sum()) if good.any() else None}
        out["tasks"][task] = entry
        print(task, entry["conditional_auroc"], entry["conflict"]["confidence_ge_0.9"], flush=True)
    (args.root / "supplement.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
