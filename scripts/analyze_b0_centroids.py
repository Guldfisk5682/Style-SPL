"""Frozen B0: class verification and hard-teacher oracle headroom.

All distances, scores, candidate labels and selections are computed without
target labels. Labels are read only for metrics, strata and labelled oracles.
No fitting, target-label parameter selection, or new training is performed.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.stats import rankdata
from scripts.analyze_style_reliability import auc, cluster_ci

DOMAINS = ("art", "clipart", "product", "real_world")


def roc_curve(score, positive):
    """Group tied thresholds; no target-label-selected operating threshold."""
    order = np.argsort(-score, kind="stable")
    s, y = score[order], positive[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    fpr, tpr = np.r_[0., fp / (~y).sum()], np.r_[0., tp / y.sum()]
    display = np.unique(np.linspace(0, len(fpr)-1, min(201, len(fpr))).astype(int))
    return {"fpr": fpr[display].tolist(), "tpr": tpr[display].tolist(),
            "display_only_downsampled": len(display) != len(fpr)}


def auc_bootstrap(scores, correct, labels, repetitions=399):
    """Paired class-cluster intervals, conditional on these frozen teachers."""
    groups = [np.flatnonzero(labels == cls) for cls in np.unique(labels)]
    rng = np.random.default_rng(2041)
    rows = {k: [] for k in scores}
    for _ in range(repetitions):
        chosen = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        for name, score in scores.items():
            rows[name].append(auc(score[idx], correct[idx]))
    result = {}
    reference = np.asarray(rows["teacher_confidence"], dtype=float)
    for name, values in rows.items():
        values = np.asarray(values, dtype=float)
        result[name] = {"ci95": np.nanquantile(values, [.025, .975]).tolist(),
                        "delta_auc_vs_confidence_ci95": np.nanquantile(values - reference, [.025, .975]).tolist()}
    return result


def verification_scores(distance, weights, predictions, probability):
    class_distance = distance.min(1).values
    row = torch.arange(len(distance))
    selected = class_distance[row, predictions]
    alternatives = class_distance.clone()
    alternatives[row, predictions] = float("inf")
    nearest_other = alternatives.min(1).values
    source_predictions = distance.argmin(-1)
    routed = weights[row, :, predictions]
    values = {
        "centroid_class_margin": nearest_other - selected,
        "centroid_relative_margin": (nearest_other - selected) / (nearest_other + selected).clamp_min(1e-8),
        "centroid_negative_candidate_distance": -selected,
        "centroid_negative_candidate_rank": -(class_distance < selected[:, None]).sum(1).float(),
        "centroid_source_class_agreement": (source_predictions == predictions[:, None]).float().mean(1),
        "source_weight_max_for_candidate": routed.max(1).values,
        "source_weight_negative_entropy_for_candidate": (routed * routed.clamp_min(1e-38).log()).sum(1),
        "teacher_confidence": probability.max(-1).values,
        "teacher_probability_margin": probability.topk(2, -1).values.diff(dim=-1).neg().squeeze(-1),
        "teacher_negative_entropy": (probability * probability.clamp_min(1e-38).log()).sum(-1),
    }
    return {k: v.double().numpy() for k, v in values.items()}


def oracle_headroom(predictions, combined, labels, mask):
    if not mask.any():
        return {"n": 0}
    y = labels[mask]
    current = combined[mask] == y
    candidate_correct = predictions[mask] == y[:, None]
    any_correct = candidate_correct.any(1)
    rescue = (~current) & any_correct
    return {"n": len(y), "combined_accuracy": float(current.mean()),
            "candidate_only_hard_top1_oracle_accuracy": float(any_correct.mean()),
            "keep_combined_or_select_candidate_oracle_accuracy": float((current | any_correct).mean()),
            "recoverable_combined_errors": int(rescue.sum()), "combined_errors": int((~current).sum()),
            "headroom_pp_with_combined_fallback": float(rescue.mean() * 100),
            "recoverable_fraction_of_combined_errors": float(rescue.sum() / (~current).sum()) if (~current).any() else None,
            "headroom_pp_class_cluster_ci95": [v * 100 for v in cluster_ci(rescue.astype(float), labels[mask])],
            "combined_correct_all_candidates_wrong": int((current & ~any_correct).sum())}


def candidate_centroid_selection(class_distance, candidates):
    return candidates.gather(1, class_distance.gather(1, candidates).argmin(1, keepdim=True)).squeeze(1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    states = torch.load(args.prepared / "b0_static_state.pt", map_location="cpu", weights_only=True)
    report = {"training_updates": 0, "parameters_selected_with_target_labels": False,
              "main_protocol": "B0 final post-update frozen source/Base/Pooled teachers; replayed source-sampling centroids; original raw squared-L2 and w_scale=10",
              "sensitivity_protocol": "Full-source class centroids with same B0 texts and same distance/weight algebra; no metric or temperature search",
              "target_teacher_definition": "Combined Teacher supplies target soft pseudo-labels. Target Student post-update/temporal classifiers are separately evaluated and not conflated with Combined Teacher.",
              "primary_verification_score": "Nearest-other-class distance minus predicted-class distance, each class minimized across source domains; positive means prototype supports candidate over competitors",
              "auc_orientation": "Positive score predicts correct pseudo-label. Identical AUROC obtained with negated score predicting errors.",
              "oracle_limit": "Hard top1 selection among a specified finite teacher pool, with optional Combined fallback. NOT an upper bound for arbitrary class-dependent logit fusion, new labels, or retrained models. All oracles use target GT offline only.",
              "preparation": json.loads((args.prepared / "preparation.json").read_text()), "tasks": {}}
    class_rows, strata_rows, coverage_rows, auc_rows = [], [], [], []
    for target in DOMAINS:
        state = states[target]
        cache = torch.load(args.runs / "analysis_round_20261009/image_embeddings" / f"{target}.pt", map_location="cpu", weights_only=True)
        assert cache["classes"] == state["classes"]
        images, raw = cache["normalized"], cache["raw"]
        source_logits = torch.stack([images @ t.T for t in state["source_texts"]], 1)
        base_logits = images @ state["base_text"].T
        pooled_logits = images @ state["pooled_text"].T
        student_logits = images @ state["student_text"].T
        temporal_logits = images @ state["student_temporal_text"].T
        scale = state["logit_scale"]
        source_predictions = source_logits.argmax(-1)
        for mode in ("replay", "full"):
            centroids = state[f"{mode}_centroids"]
            distance = torch.stack([raw.square().sum(1, keepdim=True) + c.square().sum(1) - 2 * raw @ c.T for c in centroids], 1)
            weights = (-10 * distance).softmax(1)
            weighted_logits = (weights * source_logits).sum(1)
            combined_logits = (base_logits + pooled_logits + weighted_logits) / 3
            branches = {"combined": combined_logits, "base": base_logits, "pooled": pooled_logits,
                        "weighted_source": weighted_logits, "student_post_update": student_logits,
                        "student_temporal": temporal_logits,
                        **{f"source{i}": source_logits[:, i] for i in range(3)}}
            probabilities = {k: (scale * v).softmax(-1) for k, v in branches.items()}
            predictions = {k: v.argmax(-1) for k, v in branches.items()}
            scores = {k: verification_scores(distance, weights, predictions[k], probabilities[k]) for k in branches}
            class_distance = distance.min(1).values
            nearest_class = class_distance.argmin(1)
            candidate_source = candidate_centroid_selection(class_distance, source_predictions)
            candidate_with_combined = candidate_centroid_selection(class_distance, torch.cat((source_predictions, predictions["combined"][:, None]), 1))
            # Structural counterfactual: class-common offsets cancel in source
            # softmax. They can still change prototype-based class decisions.
            d64 = distance.double()
            offsets = torch.arange(distance.shape[-1], dtype=torch.float64) * 100
            shifted = d64 + offsets[None, None, :]
            shifted_weights = (-10 * shifted).softmax(1)
            invariant_error = (shifted_weights - (-10 * d64).softmax(1)).abs().max().item()
            assert invariant_error < 1e-9
            # The first access to target GT: diagnostics start here.
            labels = cache["labels"].numpy()
            y = torch.from_numpy(labels)
            n = len(labels)
            row = torch.arange(n)
            combined = predictions["combined"].numpy()
            correct = combined == labels
            disagreement = (source_predictions != source_predictions[:, :1]).any(1).numpy()
            branch_preds = torch.stack([predictions[k] for k in ("base", "pooled", "weighted_source")], 1).numpy()
            branch_disagreement = (branch_preds != branch_preds[:, :1]).any(1)
            truth_distance = class_distance[row, y]
            other_distance = class_distance.clone()
            other_distance[row, y] = float("inf")
            nearest_wrong = other_distance.min(1).values
            true_rank = 1 + (class_distance < truth_distance[:, None]).sum(1)
            entry = {"n": n, "source_order": state["source_order"], "classes": state["classes"], "logit_scale": scale,
                     "teacher_accuracy": {k: float((v.numpy() == labels).mean()) for k, v in predictions.items()},
                     "teacher_nll": {k: float(-(scale * v).log_softmax(-1)[row, y].mean()) for k, v in branches.items()},
                     "source_prediction_disagreement_fraction": float(disagreement.mean()),
                     "branch_prediction_disagreement_fraction": float(branch_disagreement.mean()),
                     "nearest_centroid_class_accuracy": float((nearest_class.numpy() == labels).mean()),
                     "true_class_strictly_nearer_than_any_wrong": float((truth_distance < nearest_wrong).double().mean()),
                     "true_class_rank_median": float(true_rank.double().median()),
                     "true_class_rank_top3": float((true_rank <= 3).double().mean()),
                     "true_class_rank_top5": float((true_rank <= 5).double().mean()),
                     "true_vs_nearest_wrong_distance_margin_quantiles": torch.quantile((nearest_wrong - truth_distance).double(), torch.tensor([.05, .25, .5, .75, .95], dtype=torch.float64)).tolist(),
                     "source_weight_class_offset_invariance_max_error": invariant_error,
                     "class_offset_changes_nearest_class_fraction": float((shifted.min(1).values.argmin(1) != nearest_class).double().mean()),
                     "source_weight_mass_per_class": {"min": weights.sum(1).min().item(), "max": weights.sum(1).max().item()},
                     "verification": {}, "oracle": {}, "candidate_class_selection": {}}
            for branch, branch_scores in scores.items():
                branch_correct = predictions[branch].numpy() == labels
                item = {"accuracy": float(branch_correct.mean()), "scores": {}}
                for name, score in branch_scores.items():
                    macro = [auc(score[labels == cls], branch_correct[labels == cls]) for cls in np.unique(labels)]
                    macro = [v for v in macro if v is not None]
                    pred_array = predictions[branch].numpy()
                    pred_macro = [auc(score[pred_array == cls], branch_correct[pred_array == cls]) for cls in np.unique(pred_array)]
                    pred_macro = [v for v in pred_macro if v is not None]
                    sr = {"auroc": auc(score, branch_correct), "true_class_macro_auroc": float(np.mean(macro)) if macro else None,
                          "true_classes_supported": len(macro), "predicted_class_macro_auroc": float(np.mean(pred_macro)) if pred_macro else None,
                          "predicted_classes_supported": len(pred_macro), "roc": roc_curve(score, branch_correct)}
                    item["scores"][name] = sr
                    auc_rows.append({"mode": mode, "target": target, "teacher": branch, "score": name,
                                     **{k: v for k, v in sr.items() if k != "roc"}})
                if branch == "combined":
                    for name, uncertainty in auc_bootstrap({k: branch_scores[k] for k in ("centroid_class_margin", "centroid_negative_candidate_distance", "teacher_confidence")}, branch_correct, labels).items():
                        item["scores"][name].update(uncertainty)
                entry["verification"][branch] = item
            primary_scores = scores["combined"]
            confidence = primary_scores["teacher_confidence"]
            # Predetermined unlabeled confidence rank quartiles and absolute bins.
            order = np.argsort(confidence, kind="stable")
            bins = np.empty(n, dtype=int)
            for i, idx in enumerate(np.array_split(order, 4)):
                bins[idx] = i
            masks = {f"confidence_q{i+1}": bins == i for i in range(4)}
            masks.update({"confidence_ge_0.9": confidence >= .9, "confidence_ge_0.99": confidence >= .99,
                          "source_disagreement": disagreement, "source_agreement": ~disagreement,
                          "branch_disagreement": branch_disagreement})
            for stratum, mask in masks.items():
                if mask.sum() < 5:
                    continue
                for name, score in primary_scores.items():
                    strata_rows.append({"mode": mode, "target": target, "stratum": stratum, "score": name,
                                        "n": int(mask.sum()), "teacher_accuracy": float(correct[mask].mean()),
                                        "confidence_min": float(confidence[mask].min()), "confidence_max": float(confidence[mask].max()),
                                        "auroc": auc(score[mask], correct[mask])})
            for cls, class_name in enumerate(state["classes"]):
                mask = labels == cls
                for name, score in primary_scores.items():
                    class_rows.append({"mode": mode, "target": target, "class": class_name, "score": name,
                                       "n": int(mask.sum()), "teacher_accuracy": float(correct[mask].mean()),
                                       "nearest_centroid_accuracy": float((nearest_class.numpy()[mask] == labels[mask]).mean()),
                                       "auroc": auc(score[mask], correct[mask])})
            for name in ("centroid_class_margin", "centroid_negative_candidate_distance", "teacher_confidence"):
                order = np.argsort(-primary_scores[name], kind="stable")
                for coverage in (.25, .5, .75, .9, 1.):
                    selected = order[:max(1, int(np.ceil(n * coverage)))]
                    coverage_rows.append({"mode": mode, "target": target, "score": name, "coverage": coverage,
                                          "n": len(selected), "retained_pseudo_label_accuracy": float(correct[selected].mean())})
            candidate_pools = {"three_source_teachers": source_predictions.numpy(),
                               "three_teacher_branches": branch_preds,
                               "source_plus_base_pooled_weighted": np.concatenate((source_predictions.numpy(), branch_preds), 1)}
            for pool, candidate_predictions in candidate_pools.items():
                entry["oracle"][pool] = {stratum: oracle_headroom(candidate_predictions, combined, labels, mask)
                                          for stratum, mask in (("all", np.ones(n, bool)), ("source_disagreement", disagreement),
                                                                ("branch_disagreement", branch_disagreement))}
            source_nll = -(scale * source_logits).log_softmax(-1)[row, :, y]
            combined_nll = -(scale * combined_logits).log_softmax(-1)[row, y]
            entry["oracle"]["nll_source_plus_combined"] = {
                "combined_nll": combined_nll.mean().item(), "source_only_min_nll": source_nll.min(1).values.mean().item(),
                "keep_combined_or_source_min_nll": torch.minimum(source_nll.min(1).values, combined_nll).mean().item(),
                "nll_headroom": (combined_nll - torch.minimum(source_nll.min(1).values, combined_nll)).mean().item()}
            for selection_name, chosen in (("nearest_centroid_all_classes", nearest_class),
                                             ("centroid_among_source_predicted_classes", candidate_source),
                                             ("centroid_among_source_and_combined_classes", candidate_with_combined)):
                chosen_correct = chosen.numpy() == labels
                entry["candidate_class_selection"][selection_name] = {}
                for stratum, mask in (("all", np.ones(n, bool)), ("source_disagreement", disagreement)):
                    rescued = (~correct) & chosen_correct & mask
                    spoiled = correct & ~chosen_correct & mask
                    entry["candidate_class_selection"][selection_name][stratum] = {
                        "n": int(mask.sum()), "accuracy": float(chosen_correct[mask].mean()),
                        "rescued_combined_errors": int(rescued.sum()), "spoiled_combined_correct": int(spoiled.sum()),
                        "net_gain_pp": float((chosen_correct[mask].mean() - correct[mask].mean()) * 100)}
            unsupported = (~correct) & ~(source_predictions.numpy() == labels[:, None]).any(1)
            entry["centroid_correct_when_combined_and_all_sources_wrong"] = int((unsupported & (nearest_class.numpy() == labels)).sum())
            report["tasks"][f"{mode}/{target}"] = entry
            torch.save({"paths": cache["paths"], "source_order": state["source_order"], "classes": state["classes"],
                        "distance_matrix_N_source_class": distance, "source_weights": weights,
                        "source_predictions": source_predictions, "branch_predictions": predictions,
                        "combined_scores": {k: torch.from_numpy(v) for k, v in primary_scores.items()},
                        "labels_offline_diagnostic_only": y, "nearest_centroid_class": nearest_class,
                        "source_disagreement": torch.from_numpy(disagreement)}, args.output / f"per_image_{mode}_{target}.pt")
            print(json.dumps({"task": f"{mode}/{target}", "accuracy": entry["teacher_accuracy"],
                              "centroid_accuracy": entry["nearest_centroid_class_accuracy"],
                              "verification_auc": {k: v["auroc"] for k, v in entry["verification"]["combined"]["scores"].items()},
                              "source_oracle": entry["oracle"]["three_source_teachers"]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, rows in (("auroc.csv", auc_rows), ("per_class.csv", class_rows),
                       ("confidence_disagreement_strata.csv", strata_rows), ("coverage.csv", coverage_rows)):
        with (args.output / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    (args.output / "analysis_provenance.json").write_text(json.dumps({"training_updates": 0,
        "prepared_state_sha256": hashlib.sha256((args.prepared / "b0_static_state.pt").read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
