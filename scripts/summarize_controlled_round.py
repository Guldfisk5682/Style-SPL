"""Collect controlled arm trajectories/results without launching experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

ARMS = ["r1_shared", "r2_pooled", "r3_split", "r4_silu32"]
TARGETS = ["art", "clipart", "product", "real_world"]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def pair_mean(matrix, pairs):
    return sum(matrix[i][j] for i, j in pairs) / len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"arms": {}, "protocol": "One-seed controlled screening; fixed bank, lr, sampling and losses",
        "limitations": ["Formal temporal accuracy retains original shuffled/drop-tail evaluation; diagnostic accuracy covers all cached target images.",
                        "Source/Target stage and Expansion are separated in r3/r4; class context is still shared.",
                        "r4-r3 changes stage architecture and parameterization capacity; it is not a pure activation-function comparison."]}
    replay = args.run_root / "replay_verification.json"
    if replay.exists():
        summary["replay_verification"] = json.loads(replay.read_text())
    trajectory_rows, gradient_rows, swap_rows = [], [], []
    for arm in ARMS:
        arm_result = {"tasks": {}, "complete": (args.run_root / arm / "summary.json").exists()}
        for target in TARGETS:
            task = args.run_root / arm / target
            if not task.exists():
                continue
            records = []
            config_path = task / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            for path in sorted((task / "mechanism").glob("step*.json")):
                record = json.loads(path.read_text())
                text, prompts, teacher = record["text"]["mean"], record["prompt"], record.get("teacher", {})
                same_pairs = [(0, 1), (0, 2), (1, 2)]
                source_target_pairs = [(i, 4) for i in range(3)]
                row = {"arm": arm, "target": target, "step": record["step"],
                    "source_source_text_cosine": pair_mean(text, same_pairs),
                    "source_target_text_cosine": pair_mean(text, source_target_pairs),
                    "source_pooled_text_cosine": pair_mean(text, [(i, 3) for i in range(3)]),
                    "pooled_target_text_cosine": text[3][4],
                    "target_domain_rms": prompts["domain_prompt_rms"][4],
                    "class_rms": prompts["shared_class_prompt_rms"],
                    "target_to_class_rms": prompts["domain_to_shared_class_rms_ratio"][4],
                    "common_energy_fraction": prompts["common_style_energy_fraction"],
                    "kl_teacher_to_student": teacher.get("kl_teacher_to_student"),
                    "kl_student_to_teacher": teacher.get("kl_student_to_teacher"),
                    "teacher_student_agreement": teacher.get("student_teacher_prediction_agreement")}
                if "interval_prompt_movement_cosine" in record:
                    row["source_target_interval_movement_cosine"] = pair_mean(record["interval_prompt_movement_cosine"], source_target_pairs)
                    row["source_source_interval_movement_cosine"] = pair_mean(record["interval_prompt_movement_cosine"], same_pairs)
                row.update({f"teacher_{key}_accuracy": value for key, value in teacher.get("accuracy_offline_only", {}).items()})
                if "live_domain_accuracy_offline_only" in record:
                    row["student_live_accuracy"] = record["live_domain_accuracy_offline_only"][4]
                trajectory_rows.append(row)
                records.append(row)
                swaps = record.get("counterfactual_descriptor_swap", [])
                for swap in swaps:
                    swap_rows.append({"arm": arm, "target": target, "step": record["step"], **swap})
                dump(args.output / "raw_trajectory" / arm / target / path.name, record)
            gradients = []
            for path in sorted((task / "mechanism").glob("gradient*.pt")):
                cache = args.output / "gradient_summaries" / arm / target / path.with_suffix(".json").name
                if cache.exists():
                    report = json.loads(cache.read_text())
                else:
                    report = torch.load(path, map_location="cpu", weights_only=True)["summary"]
                    dump(cache, report)
                gradients.append(report)
                for role in ("style_projector", "target_style_projector"):
                    if role in report:
                        gradient_rows.append({"arm": arm, "target": target, "step": report["step"],
                            "projector": role, **report[role],
                            "matched_role_cosine": report.get("matched_source_target_projector_gradient_cosine"),
                            "shared_class_gradient_cosine": report["per_parameter"]["ctx_cls"]["cosine"]})
            result_path = task / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            eval_path = task / "eval_metrics.csv"
            instant = list(csv.DictReader(eval_path.open())) if eval_path.exists() else []
            for row in instant:
                row["step"] = int(row["step"])
                row["eval/target_top1_instant"] = float(row["eval/target_top1_instant"])
            arm_result["tasks"][target] = {"result": result, "trajectory": records,
                "instant_accuracy": instant, "trainable_parameters": config.get("trainable_parameters"),
                "frozen_bank_input_sha256": config.get("frozen_bank_input_sha256"),
                "runtime_sha256": config.get("runtime_sha256"), "git_commit": config.get("git_commit"),
                "final_gradient_summary": gradients[-1] if gradients else None}
        scores = [task["result"]["final_accuracy"] for task in arm_result["tasks"].values()
                  if task["result"] and task["result"]["final_accuracy"] is not None]
        arm_result["completed_targets"] = len(scores)
        arm_result["mean_formal_accuracy"] = sum(scores) / len(scores) if len(scores) == 4 else None
        summary["arms"][arm] = arm_result
    summary["all_complete"] = all(arm["complete"] for arm in summary["arms"].values())
    dump(args.output / "summary.json", summary)
    write_csv(args.output / "trajectory.csv", trajectory_rows)
    write_csv(args.output / "gradient_directions.csv", gradient_rows)
    write_csv(args.output / "descriptor_swaps.csv", swap_rows)
    original_probes = args.run_root / "final_checkpoint_probe"
    for path in original_probes.rglob("*.json"):
        dump(args.output / "original_final_probes" / path.relative_to(original_probes), json.loads(path.read_text()))
    print(json.dumps({k: {"completed": v["completed_targets"], "mean": v["mean_formal_accuracy"]}
                      for k, v in summary["arms"].items()}))


if __name__ == "__main__":
    main()
