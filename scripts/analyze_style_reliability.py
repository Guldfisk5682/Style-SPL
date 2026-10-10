"""Offline frozen-teacher reliability, with class and semantic-route controls.

No target labels enter style distances, normalization, inference-class routing,
or teacher selection. Labels enter evaluation and explicitly marked statistical
controls only. No fitted router, temperature search, or training is performed.
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
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

DOMAINS = ("art", "clipart", "product", "real_world")


def corr(x, y):
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-14 else None


def auc(score, correct):
    count = int(correct.sum())
    if count == 0 or count == len(correct):
        return None
    ranks = rankdata(score)
    return float((ranks[correct].sum() - count * (count + 1) / 2) / (count * (len(correct) - count)))


def class_source_residual(values, labels):
    """Remove class x source averages, then remove each image's common offset."""
    out = np.asarray(values, dtype=np.float64).copy()
    for cls in np.unique(labels):
        mask = labels == cls
        out[mask] -= out[mask].mean(0, keepdims=True)
    return out - out.mean(1, keepdims=True)


def cluster_ci(delta, labels, seed=831, repetitions=999):
    """Resample whole classes; three teacher rows of one image stay together."""
    classes = np.unique(labels)
    sums = np.array([delta[labels == cls].sum() for cls in classes])
    counts = np.array([(labels == cls).sum() for cls in classes])
    draw = np.random.default_rng(seed).integers(len(classes), size=(repetitions, len(classes)))
    values = sums[draw].sum(1) / counts[draw].sum(1)
    return np.quantile(values, [.025, .975]).tolist()


def association(distance, nll, correct, labels, permutation=False):
    # Positive correlations mean closer style predicts better teacher.
    similarity = -np.asarray(distance, dtype=np.float64)
    reliability = -np.asarray(nll, dtype=np.float64)
    a = similarity - similarity.mean(1, keepdims=True)
    b = reliability - reliability.mean(1, keepdims=True)
    residual_a = class_source_residual(similarity, labels)
    residual_b = class_source_residual(reliability, labels)
    concordant = nll_comparisons = correct_wins = correct_comparisons = 0
    for i in range(3):
        for j in range(i + 1, 3):
            ds, dr = similarity[:, i] - similarity[:, j], reliability[:, i] - reliability[:, j]
            valid = (np.abs(ds) > 1e-10) & (np.abs(dr) > 1e-5)
            concordant += ((ds[valid] * dr[valid] > 0).sum())
            nll_comparisons += valid.sum()
            dc = correct[:, i].astype(float) - correct[:, j].astype(float)
            valid = (np.abs(ds) > 1e-10) & (dc != 0)
            correct_wins += ((ds[valid] * dc[valid] > 0).sum())
            correct_comparisons += valid.sum()
    out = {"n": len(labels), "within_image_similarity_vs_negative_nll_pearson": corr(a, b),
           "class_source_adjusted_within_image_pearson": corr(residual_a, residual_b),
           "nll_pair_concordance": float(concordant / nll_comparisons) if nll_comparisons else None,
           "nll_pair_support": int(nll_comparisons),
           "correctness_pair_concordance": float(correct_wins / correct_comparisons) if correct_comparisons else None,
           "correctness_pair_support": int(correct_comparisons),
           "mean_within_image_nll_range": float(np.ptp(nll, axis=1).mean()),
           "prediction_correctness_disagreement_images": int((np.ptp(correct.astype(int), axis=1) > 0).sum())}
    if permutation and out["class_source_adjusted_within_image_pearson"] is not None:
        # Within-image source permutations preserve difficulty and all NLLs.
        permutations = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
        rng = np.random.default_rng(921)
        observed = float((residual_a * residual_b).sum())
        exceed = 0
        for _ in range(499):
            order = permutations[rng.integers(6, size=len(labels))]
            null = np.take_along_axis(residual_a, order, axis=1)
            exceed += float((null * residual_b).sum()) >= observed
        out["adjusted_positive_association_permutation_p"] = (exceed + 1) / 500
    return out


def selections(distance, nll, correct, semantic, labels, uncertainty=False):
    selected = np.asarray(distance).argmin(1)
    row = np.arange(len(labels))
    losses = {"style_nearest": nll[row, selected], "uniform_teacher_expectation": nll.mean(1),
              "semantic_hard_predicted_class": nll[row, semantic], "oracle_min_nll": nll.min(1)}
    accuracies = {"style_nearest": correct[row, selected].astype(float),
                  "uniform_teacher_expectation": correct.mean(1),
                  "semantic_hard_predicted_class": correct[row, semantic].astype(float),
                  "oracle_any_correct": correct.any(1).astype(float)}
    out = {"n": len(labels), "accuracy": {k: float(v.mean()) for k, v in accuracies.items()},
           "nll": {k: float(v.mean()) for k, v in losses.items()},
           "style_selection_fraction": np.bincount(selected, minlength=3).astype(float).__truediv__(len(labels)).tolist(),
           "style_semantic_selection_agreement": float((selected == semantic).mean())}
    # Post-hoc labelled-target constants are explanatory references, NOT a
    # deployable baseline selected without target ground truth.
    best_loss = int(nll.mean(0).argmin())
    best_acc = int(correct.mean(0).argmax())
    adjusted = class_source_residual(nll, labels)
    out["posthoc_best_constant_teacher_offline_only"] = {
        "nll_source_index": best_loss, "nll": float(nll[:, best_loss].mean()),
        "accuracy_source_index": best_acc, "accuracy": float(correct[:, best_acc].mean()),
        "style_nll_reduction": float((nll[:, best_loss] - losses["style_nearest"]).mean()),
        "style_accuracy_gain": float((accuracies["style_nearest"] - correct[:, best_acc]).mean())}
    out["class_source_adjusted_nll_reduction_vs_uniform_offline_only"] = float(
        (adjusted.mean(1) - adjusted[row, selected]).mean())
    if uncertainty:
        out["class_source_adjusted_nll_reduction_vs_uniform_ci95"] = cluster_ci(
            adjusted.mean(1) - adjusted[row, selected], labels)
    for baseline in ("uniform_teacher_expectation", "semantic_hard_predicted_class"):
        acc_delta = accuracies["style_nearest"] - accuracies[baseline]
        nll_delta = losses[baseline] - losses["style_nearest"]
        out[f"accuracy_gain_vs_{baseline}"] = float(acc_delta.mean())
        out[f"nll_reduction_vs_{baseline}"] = float(nll_delta.mean())
        if uncertainty:
            out[f"accuracy_gain_vs_{baseline}_class_cluster_ci95"] = cluster_ci(acc_delta, labels)
            out[f"nll_reduction_vs_{baseline}_class_cluster_ci95"] = cluster_ci(nll_delta, labels)
    return out


def distances(blocks, bank, all_styles, sources):
    scores, normalization = {}, []
    for stage, (block, entries) in enumerate(zip(blocks, zip(*bank["sources"]))):
        reference = torch.stack([torch.cat(pair) for pair in entries]).double()
        x = block.double()
        cosine = 1 - F.normalize(x, dim=-1) @ F.normalize(reference, dim=-1).T
        # Source-only channel standard deviation, weighted by source image count.
        samples = torch.cat([all_styles[s]["blocks"][stage] for s in sources]).double()
        for i, source in enumerate(sources):
            torch.testing.assert_close(all_styles[source]["blocks"][stage].double().mean(0).float(),
                                       reference[i].float(), atol=3e-5, rtol=3e-5)
        mean = samples.mean(0)
        std = samples.std(0, correction=0)
        floor = 1e-6
        z = (x - mean) / std.clamp_min(floor)
        z_reference = (reference - mean) / std.clamp_min(floor)
        # Root mean squared standardized coordinate distance, no dim-size bias.
        l2 = ((z[:, None, :] - z_reference[None, :, :]).square().mean(-1)).sqrt()
        scores[f"cosine_stage{stage + 1}"] = cosine.numpy()
        scores[f"source_z_l2_stage{stage + 1}"] = l2.numpy()
        normalization.append({"stage": stage + 1, "source_images": len(samples),
                              "source_domains": sources, "std_floor": floor,
                              "floored_channels": int((std < floor).sum()),
                              "std_median": std.median().item(), "target_labels_used": False})
    for family in ("cosine", "source_z_l2"):
        scores[f"{family}_mean4"] = np.mean([scores[f"{family}_stage{i}"] for i in range(1, 5)], axis=0)
    return scores, normalization


def routing(raw, images, base, snapshot, w_scale):
    centers = snapshot["running_means"]
    assert (snapshot["running_count"] > 0).all()
    ds = torch.stack([raw.square().sum(1, keepdim=True) + c.square().sum(1) - 2 * raw @ c.T for c in centers])
    weights = (-w_scale * ds).softmax(0)
    predicted = (images @ base.T).argmax(-1)
    route = weights[:, torch.arange(len(images)), predicted].T.numpy()
    return weights, route, predicted.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--styles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    b0 = torch.load(args.styles / "b0_texts.pt", map_location="cpu", weights_only=True)
    all_styles = {d: torch.load(args.styles / f"{d}.pt", map_location="cpu", weights_only=True) for d in DOMAINS}
    report = {"training_updates": 0, "target_labels": "offline correctness/NLL, class strata and statistical controls only",
              "distance_protocol": "Per-stage concatenated spatial mean/std cosine FIRST; source-only per-channel population-standardized RMS L2 SECOND; equal-stage mean sensitivity; no tuned temperature",
              "main_metrics": ["cosine_mean4", "source_z_l2_mean4"],
              "logit_scale": b0["art"]["logit_scale"],
              "semantic_protocol": "Predicted class from frozen Base CLIP, then saved SPL class-dependent weights. B0 has no saved centroids: R4 Real reference ONLY, never called original B0 routing.",
              "limits": ["Frozen final post-update source teachers; not temporal target-model evaluation or a new trained router.",
                         "Single trained seed; CIs resample classes conditional on fixed models, not training randomness.",
                         "Raw distance is expected positively associated with NLL and negatively with correctness; reported similarity=-distance correlations use positive=better reliability.",
                         "Uniform expectation means random hard teacher, not averaging logits/probabilities.",
                         "Stage 4 spatial statistics may contain class information; class stratification reduces but does not prove removal of semantic confounding.",
                         "Stage/family analyses are exploratory multiple comparisons; target labels do not select a preferred distance for deployment.",
                         "Relative semantic terciles use descending entropy; tied entropy values use stable path order. Absolute confidence strata separately disclose saturation."],
              "extraction_validation": {d: all_styles[d]["metadata"] for d in DOMAINS}, "tasks": {}}
    class_rows, teacher_rows, strata_rows, confidence_rows, crossed_rows = [], [], [], [], []
    for target in DOMAINS:
        cache = torch.load(args.runs / "analysis_round_20261009/image_embeddings" / f"{target}.pt", map_location="cpu", weights_only=True)
        assert cache["paths"] == all_styles[target]["paths"]
        assert cache["classes"] == b0[target]["classes"]
        images, raw = cache["normalized"], cache["raw"]
        labels = cache["labels"].numpy()
        bank_path = args.runs / "s1_protocol_checked_20261009_seed1" / target / "style_bank.pt"
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        sources = bank["metadata"]["source_domain_order"]
        scores, normalization = distances(all_styles[target]["blocks"], bank, all_styles, sources)
        real_root = args.runs / "descriptor_20261010_seed1/r4_real" / target
        real = torch.load(real_root / "mechanism/step1000.pt", map_location="cpu", weights_only=True)
        real_config = json.loads((real_root / "config.json").read_text())
        common_weights, common_route, base_prediction = routing(raw, images, b0[target]["base_text"], real, real_config["w_scale"])
        for arm in ("b0", "r4_real", "r4_random"):
            if arm == "b0":
                texts = b0[target]["source_texts"]
                weights, route = common_weights, common_route
                snapshot_sha = b0[target]["checkpoint_sha256"]
                original_route = False
            else:
                task_root = args.runs / "descriptor_20261010_seed1" / arm / target
                path = task_root / "mechanism/step1000.pt"
                snapshot = torch.load(path, map_location="cpu", weights_only=True)
                assert snapshot["domain_order"][:3] == sources
                texts = snapshot["text_features"][:3]
                config = json.loads((task_root / "config.json").read_text())
                weights, route, pred = routing(raw, images, b0[target]["base_text"], snapshot, config["w_scale"])
                assert np.array_equal(pred, base_prediction)
                snapshot_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                original_route = True
            logits = torch.stack([images @ text.T for text in texts]) * b0[target]["logit_scale"]
            nll = -logits.log_softmax(-1)[:, torch.arange(len(images)), torch.from_numpy(labels)].T.numpy()
            correct = (logits.argmax(-1).T.numpy() == labels[:, None])
            predictions = logits.argmax(-1).T.numpy()
            semantic = route.argmax(1)
            confidence = route.max(1)
            entropy = -(route * np.log(np.maximum(route, 1e-38))).sum(1)
            # Tercile strength without target labels; entropy breaks softmax saturation.
            order = np.argsort(-entropy, kind="stable")
            terciles = np.empty(len(images), dtype=int)
            for idx, split in enumerate(np.array_split(order, 3)):
                terciles[split] = idx
            true_route = weights[:, torch.arange(len(images)), torch.from_numpy(labels)].T.numpy()
            entry = {"source_domain_order": sources, "n": len(images), "normalization": normalization,
                     "snapshot_sha256": snapshot_sha, "original_semantic_route_available": original_route,
                     "teacher_accuracy": correct.mean(0).tolist(), "teacher_nll": nll.mean(0).tolist(),
                     "all_prediction_agreement": float((predictions == predictions[:, :1]).all(1).mean()),
                     "semantic_predicted_class_correct_fraction": float((base_prediction == labels).mean()),
                     "semantic_confidence_quantiles": np.quantile(confidence, [0, .25, .5, .75, 1]).tolist(),
                     "semantic_entropy_quantiles": np.quantile(entropy, [0, .25, .5, .75, 1]).tolist(),
                     "semantic_saturated_entropy_lt_1e_6_fraction": float((entropy < 1e-6).mean()),
                     "metrics": {}}
            # Actual per-class weighted-logit source aggregation retained as reference.
            weighted_logits = (weights * logits).sum(0)
            entry["semantic_classwise_logit_mixture"] = {
                "accuracy": float((weighted_logits.argmax(-1).numpy() == labels).mean()),
                "nll": float(-weighted_logits.log_softmax(-1)[torch.arange(len(images)), torch.from_numpy(labels)].mean()),
                "is_original_arm_routing": original_route}
            uniform_logits = logits.mean(0)
            entry["uniform_logit_mixture"] = {
                "accuracy": float((uniform_logits.argmax(-1).numpy() == labels).mean()),
                "nll": float(-uniform_logits.log_softmax(-1)[torch.arange(len(images)), torch.from_numpy(labels)].mean())}
            for name, distance in scores.items():
                main_metric = name.endswith("mean4")
                metrics = association(distance, nll, correct, labels, permutation=main_metric)
                metrics["selection"] = selections(distance, nll, correct, semantic, labels, uncertainty=main_metric)
                metrics["per_teacher"] = []
                for i, source in enumerate(sources):
                    item = {"source": source, "distance_vs_nll_spearman": float(spearmanr(distance[:, i], nll[:, i]).statistic),
                            "similarity_predicts_correctness_auc": auc(-distance[:, i], correct[:, i])}
                    class_aucs = [auc(-distance[labels == cls, i], correct[labels == cls, i]) for cls in np.unique(labels)]
                    supported_aucs = [v for v in class_aucs if v is not None]
                    item["class_macro_correctness_auc"] = float(np.mean(supported_aucs)) if supported_aucs else None
                    item["classes_with_correct_and_wrong_images"] = len(supported_aucs)
                    metrics["per_teacher"].append(item)
                    teacher_rows.append({"arm": arm, "target": target, "distance": name, **item})
                macro = []
                for cls, class_name in enumerate(cache["classes"]):
                    mask = labels == cls
                    cm = association(distance[mask], nll[mask], correct[mask], labels[mask])
                    cs = selections(distance[mask], nll[mask], correct[mask], semantic[mask], labels[mask])
                    class_rows.append({"arm": arm, "target": target, "distance": name, "class": class_name,
                                       **cm, "style_accuracy": cs["accuracy"]["style_nearest"],
                                       "style_nll": cs["nll"]["style_nearest"],
                                       "style_accuracy_gain_vs_uniform": cs["accuracy_gain_vs_uniform_teacher_expectation"],
                                       "style_nll_reduction_vs_uniform": cs["nll_reduction_vs_uniform_teacher_expectation"],
                                       "style_adjusted_nll_reduction_vs_uniform": cs["class_source_adjusted_nll_reduction_vs_uniform_offline_only"],
                                       "style_nll_reduction_vs_semantic": cs["nll_reduction_vs_semantic_hard_predicted_class"]})
                    macro.append(cm["class_source_adjusted_within_image_pearson"])
                    if main_metric:
                        for idx, stratum in enumerate(("weak_relative", "middle_relative", "strong_relative")):
                            cross = mask & (terciles == idx)
                            if cross.sum() < 5:
                                continue
                            xm = association(distance[cross], nll[cross], correct[cross], labels[cross])
                            xs = selections(distance[cross], nll[cross], correct[cross], semantic[cross], labels[cross])
                            crossed_rows.append({"arm": arm, "target": target, "distance": name,
                                                 "class": class_name, "stratum": stratum, **xm,
                                                 "nll_reduction_vs_uniform": xs["nll_reduction_vs_uniform_teacher_expectation"],
                                                 "nll_reduction_vs_semantic": xs["nll_reduction_vs_semantic_hard_predicted_class"]})
                finite_macro = [v for v in macro if v is not None]
                metrics["class_macro_adjusted_pearson"] = float(np.mean(finite_macro)) if finite_macro else None
                metrics["classes_positive_adjusted_pearson"] = sum(v > 0 for v in finite_macro)
                metrics["classes_supported"] = len(finite_macro)
                for idx, stratum in enumerate(("weak_relative", "middle_relative", "strong_relative")):
                    mask = terciles == idx
                    sm = association(distance[mask], nll[mask], correct[mask], labels[mask])
                    ss = selections(distance[mask], nll[mask], correct[mask], semantic[mask], labels[mask])
                    strata_rows.append({"arm": arm, "target": target, "distance": name, "stratum": stratum,
                                        "semantic_confidence_mean": float(confidence[mask].mean()),
                                        "semantic_entropy_mean": float(entropy[mask].mean()), **sm,
                                        "accuracy_gain_vs_uniform": ss["accuracy_gain_vs_uniform_teacher_expectation"],
                                        "nll_reduction_vs_uniform": ss["nll_reduction_vs_uniform_teacher_expectation"],
                                        "accuracy_gain_vs_semantic": ss["accuracy_gain_vs_semantic_hard_predicted_class"],
                                        "nll_reduction_vs_semantic": ss["nll_reduction_vs_semantic_hard_predicted_class"]})
                # Fixed thresholds additionally disclose absolute saturation.
                for stratum, mask in (("weak_abs_max_lt_0.6", confidence < .6),
                                      ("middle_abs", (confidence >= .6) & (confidence < .9)),
                                      ("strong_abs_max_ge_0.9", confidence >= .9),
                                      ("base_class_correct", base_prediction == labels),
                                      ("base_class_wrong", base_prediction != labels)):
                    if mask.sum() < 5:
                        continue
                    sm = association(distance[mask], nll[mask], correct[mask], labels[mask])
                    ss = selections(distance[mask], nll[mask], correct[mask], semantic[mask], labels[mask])
                    confidence_rows.append({"arm": arm, "target": target, "distance": name, "stratum": stratum,
                                            **sm, "nll_reduction_vs_semantic": ss["nll_reduction_vs_semantic_hard_predicted_class"]})
                # Ground-truth-class route is an OFFLINE explanatory upper reference.
                metrics["true_class_semantic_selection_offline_only"] = {
                    "accuracy": float(correct[np.arange(len(images)), true_route.argmax(1)].mean()),
                    "nll": float(nll[np.arange(len(images)), true_route.argmax(1)].mean())}
                entry["metrics"][name] = metrics
            report["tasks"][f"{arm}/{target}"] = entry
            # Preserve all per-image diagnostics remotely, no bulky tensors in git.
            torch.save({"paths": cache["paths"], "labels_offline_only": torch.from_numpy(labels),
                        "source_order": sources, "nll": torch.from_numpy(nll),
                        "correct": torch.from_numpy(correct), "source_predictions": torch.from_numpy(predictions),
                        "semantic_predicted_class_weights": torch.from_numpy(route),
                        "semantic_tercile": torch.from_numpy(terciles),
                        "distances": {k: torch.from_numpy(v) for k, v in scores.items()}},
                       args.output / f"per_image_{arm}_{target}.pt")
            print(json.dumps({"task": f"{arm}/{target}", "agreement": entry["all_prediction_agreement"],
                              "primary": {k: v for k, v in entry["metrics"].items() if k.endswith("mean4")}}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, rows in (("per_class.csv", class_rows), ("per_teacher.csv", teacher_rows),
                       ("semantic_strata.csv", strata_rows), ("absolute_strata.csv", confidence_rows),
                       ("class_semantic_strata.csv", crossed_rows)):
        with (args.output / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
