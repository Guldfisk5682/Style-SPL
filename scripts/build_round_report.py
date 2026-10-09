"""Summarize existing diagnostic JSON/CSV; no model forward or training."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

DOMAINS = ["art", "clipart", "product", "real_world"]
DISPLAY = {"art": "Art", "clipart": "Clipart", "product": "Product", "real_world": "Real World"}


def csv_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def off_diagonal(matrix):
    return [matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))]


def main():
    p = argparse.ArgumentParser(); p.add_argument("analysis_root", type=Path); p.add_argument("--publish", type=Path, required=True)
    args = p.parse_args(); root = args.analysis_root
    raw = json.loads((root / "analysis.json").read_text())
    output = args.publish; output.mkdir(parents=True, exist_ok=True)
    summary = {"training_updates": 0, "protocol": raw["protocol"], "limits": raw["limits"],
               "checkpoints_unchanged": raw["checkpoints_unchanged_after_analysis"], "targets": {}, "cross_target_models": raw["cross_target_models"]}
    for target in DOMAINS:
        e = raw["targets"][target]; prompt = e["prompt_final"]
        predictions = csv_rows(root / target / "teacher_predictions.csv")
        disagree = sum(row["pooled"] != row["weighted_source"] for row in predictions)
        instant = csv_rows(root / target / "eval_metrics.csv")
        baseline = csv_rows(root / target / "baseline_eval_metrics.csv")
        target_text = [e["text_final"]["mean"][i][4] for i in range(3)]
        b0_target_text = [e["b0_text_final"]["mean"][i][4] for i in range(3)]
        s = {"target_prompt_rms": prompt["domain_prompt_rms"][4],
             "shared_class_prompt_rms": prompt["shared_class_prompt_rms"],
             "target_to_shared_rms_ratio": prompt["domain_to_shared_class_rms_ratio"][4],
             "b0_target_to_shared_rms_ratio": e["b0_prompt_final"]["domain_to_shared_class_rms_ratio"][4],
             "target_residual_to_shared_rms_ratio": prompt["centered_domain_to_shared_class_rms_ratio"][4],
             "common_prompt_energy_fraction": prompt["common_style_energy_fraction"],
             "initial_common_energy_fraction": e["prompt_initial"]["common_style_energy_fraction"],
             "residual_common_rms_ratio": prompt["centered_to_common_style_rms_ratio"],
             "prompt_source_target_cosine_range": [min(prompt["flattened_cosine"][i][4] for i in range(3)), max(prompt["flattened_cosine"][i][4] for i in range(3))],
             "same_class_source_target_cosine_range": [min(target_text), max(target_text)],
             "b0_same_class_source_target_cosine_range": [min(b0_target_text), max(b0_target_text)],
             "different_class_target_mean_cosine": e["text_final"]["different_class_within_domain_mean_cosine"][4],
             "stage_signed_share_source_target": e["stage_domain_difference"]["source_target_pairs"]["signed_shares"],
             "teacher_accuracy": e["teacher_diagnostic"]["accuracy"],
             "b0_diagnostic_accuracy": e["b0_diagnostic_accuracy"],
             "pooled_weighted_prediction_disagreement": {"n": disagree, "total": len(predictions), "fraction": disagree / len(predictions)},
             "routing": e["teacher_diagnostic"]["routing"], "teacher_vs_base": e["teacher_diagnostic"]["vs_base"],
             "formal_accuracy": e["formal_final_accuracy"], "formal_baseline_accuracy": e["formal_b0_accuracy"],
             "instant_accuracy_step1000": float(instant[-1]["eval/target_top1_instant"]),
             "baseline_instant_accuracy_step1000": float(baseline[-1]["eval/target_top1_instant"]),
             "instant_steps": instant}
        bank_path = root / target / "style_bank.pt"
        if bank_path.exists():
            bank = torch.load(bank_path, map_location="cpu", weights_only=True)
            entries = [*bank["sources"], bank["pooled"], bank["target"]]
            inputs = []
            for stage in range(4):
                descriptor = torch.stack([torch.cat(entry[stage]) for entry in entries]).double()
                normalized = F.normalize(descriptor, dim=-1)
                cosine = normalized @ normalized.T
                common = descriptor.mean(0)
                inputs.append({"cosine": cosine.tolist(),
                               "source_target_range": [cosine[:3,4].min().item(), cosine[:3,4].max().item()],
                               "common_energy_fraction": common.square().mean().item() / descriptor.square().mean().item()})
            s["bank_input_stage_geometry"] = inputs
        summary["targets"][target] = s
    accuracies = {branch: sum(s["teacher_accuracy"][branch] for s in summary["targets"].values()) / 4
                  for branch in ("base", "pooled", "weighted_source", "combined", "student_post_update", "student_temporal")}
    summary["teacher_four_domain_macro_accuracy"] = accuracies
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(root / "analysis.json", output / "raw_analysis.json")
    shutil.copytree(root / "figures", output / "figures", dirs_exist_ok=True)
    for target in DOMAINS:
        destination = output / "per_class"; destination.mkdir(exist_ok=True)
        for name in ("same_class_cross_domain_cosine.csv", "teacher_per_class_accuracy.csv"):
            shutil.copy2(root / target / name, destination / f"{target}_{name}")
    for source in root.glob("cross_target_*_cosine.csv"):
        shutil.copy2(source, output / "per_class" / source.name)
    s = summary["targets"]
    lines = ["# Style-SPL：2026-10-09 首轮训练诊断", "",
             "本报告只读取已完成的 seed=1、RN50、1000-step、OT-off 实验。没有新训练、消融、参数更新或标签调参。原检查点 SHA256 在分析前后完全一致。", "",
             "## 核心判断", "",
             "本轮出现的是**同一模型内部的域条件文本表示趋同**：源域、合并源域和目标域的同类别文本特征接近相同。它不是全类别语义塌缩，因为不同类别仍有明显差异。域提示的共同分量占绝大部分能量；改变 Source 路由几乎不改变教师预测。", "",
             "四套目标模型之间仍有较大差异，但它们各自训练了类别提示和 Projector。这不能单独证明共享模型能识别域，也不能证明这些差异有益。", "",
             "## 评测与指标口径", "",
             "- Instant Accuracy 图直接来自训练日志，使用当步更新前的文本特征，以及原基线 shuffle=True / drop_last=True 的评测。", 
             "- 教师诊断使用最终更新后的参数和保存的源域质心，覆盖完整目标集合：Art 2427、Clipart 4365、Product 4439、Real World 4357；shuffle=False、drop_last=False。教师比较共享同一批图像和标签。", 
             "- 因此教师诊断分数与原正式最终准确率不是完全相同的评测口径。历史教师准确率没有记录，不能由最终检查点回推。", 
             "- 五组 Domain Prompt 是 3 个 Source、1 个 Pooled Source、1 个 Target。Pooled 不是第五个真实域。", 
             "- Domain Prompt Cosine 对对应位置的 16×512 token 展平计算；另存逐 token 对齐余弦及减去共同提示后的矩阵。", 
             "- Shared Prompt 指 65 类共享于各域的 `ctx_cls`；RMS 在其全部元素上计算。域均值共同分量另行统计，不能与 Shared Class Prompt 混称。", 
             "- 同类别跨域 Text Cosine 先对每个类别比较，再在 65 类上求均值，未将所有类别平均成一个原型；逐类别结果也已保存。", 
             "- 跨目标模型比较分别展示最终即时特征与时间平均特征。计算余弦时才做单位化；时间平均特征的分类评测不额外归一化。", "",
             "## Instant Target Accuracy", "", "![Instant Accuracy](figures/instant_target_accuracy.png)", ""]
    lines.append(table(["目标域", "SPL 即时1000", "Style 即时1000", "SPL 正式最终", "Style 正式最终"],
                       [[DISPLAY[t], *[f"{100*s[t][k]:.2f}%" for k in ("baseline_instant_accuracy_step1000", "instant_accuracy_step1000", "formal_baseline_accuracy", "formal_accuracy")]] for t in DOMAINS]))
    lines += ["", "Style-SPL 在五个已记录的评测时点均未超过基线。Clipart 在 step400 后下降，Product 在 step600→800 明显下降；Real World 总体缓慢改善，中间有小幅波动。时间平均特征提高了最终分数，但这种提高不等于最后一步模型已经恢复。只有五个评测时点，不能据此推断未记录的逐步变化。", "",
              "## 域提示共同分量与 RMS", "", "![RMS](figures/prompt_rms_ratios.png)", ""]
    lines.append(table(["目标域", "Target / Class RMS", "SPL Target / Class RMS", "Target 残差 / Class RMS", "共同提示能量占比"],
                       [[DISPLAY[t], f"{s[t]['target_to_shared_rms_ratio']:.1f}", f"{s[t]['b0_target_to_shared_rms_ratio']:.3f}",
                         f"{s[t]['target_residual_to_shared_rms_ratio']:.2f}", f"{100*s[t]['common_prompt_energy_fraction']:.3f}%"] for t in DOMAINS]))
    lines += ["", "五组提示均值构成共同分量，各提示减去该均值构成域残差。初始化时共同能量已约99.56%；训练后增至99.80–99.85%。残差/共同分量 RMS 从约6.6%降至约3.9–4.5%。所以‘各域提示不同’成立，但大部分提示能量并不用于域间差异。域残差本身仍大于类别提示，不能把它误称为绝对接近零。", "",
              "![Domain Prompt Cosine](figures/domain_prompt_cosine.png)", "",
              "[去共同分量后的矩阵](figures/domain_prompt_centered_cosine.png)与[SPL 原始提示矩阵](figures/baseline_domain_prompt_cosine.png)用于检查共同偏移对原始余弦的影响。矩阵采用放大的色标，数值均直接标注。", "",
              "## 同一模型内的同类别文本差异", "", "![Text Cosine](figures/same_class_text_cosine.png)", ""]
    lines.append(table(["目标模型", "Style Source→Target 同类别余弦范围", "SPL 对应范围", "Style Target 不同类别平均余弦"],
                       [[DISPLAY[t], "–".join(f"{v:.6f}" for v in s[t]["same_class_source_target_cosine_range"]),
                         "–".join(f"{v:.6f}" for v in s[t]["b0_same_class_source_target_cosine_range"]),
                         f"{s[t]['different_class_target_mean_cosine']:.4f}"] for t in DOMAINS]))
    lines += ["", "Style-SPL 的域间同类别方向几乎重合；SPL 仍保留明显的同类别跨域方向差异。Style 不同类别的余弦约0.41–0.48，说明类别语义并没有一同塌缩。因此更准确的表述是‘域条件的语义效果趋同’，而不是‘整个文本编码器失效’。", "",
              "[SPL 同类别矩阵](figures/baseline_same_class_text_cosine.png)、`raw_analysis.json` 的类别分位数及 `per_class/` 的逐类 CSV 可检查均值是否掩盖少数异常类别。", "",
              "## 不同目标任务的 Target Feature", "", "![Cross Target](figures/cross_target_model_text_cosine.png)", "",
              "四套 Style-SPL 模型最终即时 Target Feature 的同类别跨模型余弦为0.6010–0.7596，时间平均为0.6102–0.8441；SPL 对应为0.8321–0.8670与0.9293–0.9414。Style 的模型间差异确实更大，但与同模型内的0.999级域间相似度形成对照。", "",
              "这提示当前结构可能主要学习了每个训练任务各自的整体提示，而没有在同一 Projector 内形成足够的域角色差异。它是与现有指标相符的解释，尚不是因果验证。Clipart 与另外三套目标模型的差异尤其大。", "",
              "## 各 Stage 对 Domain Difference 的贡献", "", "![Stages](figures/stage_domain_difference.png)", ""]
    lines.append(table(["目标域", "Stage 1", "Stage 2", "Stage 3", "Stage 4"],
                       [[DISPLAY[t], *[f"{100*v:.2f}%" for v in s[t]["stage_signed_share_source_target"]]] for t in DOMAINS]))
    lines += ["", "将源域与目标域 Stage Token 的差异乘以该 Stage 的 Expansion 列，得到四个贡献向量。四向量之和精确等于总 Domain Prompt 差异；归因保留交叉项及抵消，按差异能量聚合三个 Source→Target 配对。另图同时给出忽略交叉项的单 Stage 能量比例。", "",
              "Stage 4 占93–96%，前三级合计只有4–7%。四个 Stage 梯度非零，并不意味着它们提供了相近的域差异。原始 Bank 的 Stage4 输入本身比前三级更有域间差异，所以贡献集中未必就是故障。这是原始提示空间的精确归因；Text Encoder 是非线性的，不能将这些百分比直接解释为最终文本特征的因果贡献。", "",
              "## Base / Pooled / Weighted Source Teacher 准确率", "", "![Teachers](figures/teacher_accuracy.png)", ""]
    branches = ("base", "pooled", "weighted_source", "combined", "student_post_update", "student_temporal")
    lines.append(table(["目标域", "Base", "Pooled", "Weighted Source", "Combined", "Target即时", "Target平均"],
                       [[DISPLAY[t], *[f"{100*s[t]['teacher_accuracy'][k]:.2f}%" for k in branches]] for t in DOMAINS] +
                       [["四域均值", *[f"{100*accuracies[k]:.2f}%" for k in branches]]]))
    lines += ["", "Weighted Source 按原 SPL 的逐图像×逐类别负平方距离 softmax 加权源域余弦 logits；不重新归一化混合文本，不改为平均概率。Combined 对 Base、Pooled、Weighted 的 logits 做三项平均。", ""]
    lines.append(table(["目标域", "Pooled 与 Weighted 预测不同", "占目标图像比例", "路由最大权重均值", "相对Base：Combined纠错 / 破坏正确预测"],
                       [[DISPLAY[t], f"{s[t]['pooled_weighted_prediction_disagreement']['n']}/{s[t]['pooled_weighted_prediction_disagreement']['total']}",
                         f"{100*s[t]['pooled_weighted_prediction_disagreement']['fraction']:.3f}%", f"{s[t]['routing']['mean_max_source_weight']:.3f}",
                         f"{s[t]['teacher_vs_base']['combined']['rescued_base_errors']} / {s[t]['teacher_vs_base']['combined']['spoiled_base_correct']}"] for t in DOMAINS]))
    lines += ["", "路由最大权重均值0.82–0.89，说明路由计算并非均匀无效；但候选源域文本特征几乎相同，切换权重难以改变输出。Weighted 相对 Pooled 的准确率变化仅约−0.21至+0.02个百分点。Base 与训练后的类别表示仍有互补，Combined 相对 Base 提高约1.53–3.23个百分点。", "",
              "官方 SPL 检查点只保存 Prompt，没有质心历史，因此其 Weighted/Combined 教师不能精确恢复。可恢复的 Pooled 教师同口径结果为：" +
              "，".join(f"{DISPLAY[t]} {100*s[t]['b0_diagnostic_accuracy']['pooled']:.2f}%" for t in DOMAINS) + "。不能用 Style 的质心补入 SPL 来冒充原教师。", "",
              "## 当前可支持的机制线索与边界", "",
              "1. 输入 Bank 不同且固定，Stage / Expansion 真实更新；当前问题不是分支没训练。", 
              "2. 共模提示能量随训练占比上升，同模型内域条件文本方向趋同。", 
              "3. 候选源域教师趋同后，距离路由虽然正常工作，却几乎失去提供额外预测差异的空间。", 
              "4. 因此最值得深入的问题是：为什么共享 Projector 更倾向生成每个任务的共同提示，而不是保留输入中的域差异。特别是原始 Stage4 描述子 Source→Target 余弦约0.90–0.98，输入中已有差异；生成提示与最终文本却更接近。大 RMS、输入统计的共同分量和 Stage4 主导是待检验的解释，不应直接宣布其中任意一项为唯一根因。", 
              "5. 本轮为单种子，不能据此宣布所有配置下都会发生该现象，也不能将差异越大简单等同于适配越有效。", "",
              "所有图提供同名 PDF；`raw_analysis.json` 包含完整矩阵、均值/标准差/分位数、逐 Stage 配对能量及检查点 SHA256；`summary.json` 为紧凑摘要。没有启动新消融。", ""]
    if all("bank_input_stage_geometry" in s[t] for t in DOMAINS):
        lines += ["## 附：输入 Bank 的 Stage 描述子余弦", "",
                  "直接比较固定的拼接 mean/std 描述子，用于观察投影之前已经有多少共同分量。它不依赖目标真实类别标签。", ""]
        lines.append(table(["目标域", "Stage1 Source→Target", "Stage2", "Stage3", "Stage4"],
                           [[DISPLAY[t], *["–".join(f"{v:.5f}" for v in stage["source_target_range"]) for stage in s[t]["bank_input_stage_geometry"]]] for t in DOMAINS]))
    (output / "analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output/"analysis.md"), "teacher_macro_accuracy": accuracies, "protected_checkpoint_unchanged": summary["checkpoints_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
