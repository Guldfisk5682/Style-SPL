"""Scientific report of completed R4 input-necessity controls."""
import argparse
import csv
import json
from pathlib import Path

ARMS = ["r4_real", "r4_shuffled", "r4_random"]
LABELS = {"r4_real": "R4 / Real", "r4_shuffled": "R4 / Shuffled", "r4_random": "R4 / Random"}
TARGETS = ["art", "clipart", "product", "real_world"]


def average(values):
    return sum(values) / len(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    assert summary["all_complete"]
    paired = json.loads((args.root / "paired_predictions.json").read_text())
    acceptance = json.loads((args.root / "acceptance.json").read_text())
    pairing = json.loads((args.root / "pairing_verification.json").read_text())
    assert acceptance["valid"] and pairing["matched"]
    swaps = list(csv.DictReader((args.root / "descriptor_swaps.csv").open()))
    lines = ["# R4：真实 Style Descriptor 的必要性验证", "",
        "固定 R4 的网络结构、损失、路由、优化器、学习率、调度、采样和评测，只替换固定 Descriptor。Office-Home 四个 3→1 任务，seed=1，每任务 1000 次更新；共新增八次完整训练。Real 引用已完成的原 R4，不重复训练或挑选 checkpoint。", "",
        "## 控制条件", "",
        "Shuffled 在 Art、Clipart、Product、Real World 四个真实域之间固定置换，四个 Stage 的均值和标准差整组移动；无固定点，各任务采用同一实际域名映射。", "",
        "固定映射：" + "；".join(f"{a} → {b}" for a, b in acceptance["fixed_global_shuffle_mapping"].items()) + "。", "",
        "Random 使用独立 CPU RNG（20261010）生成固定随机域代码。每个 Stage 的 mean/std 输入块分别匹配真实四域所有通道的全局均值和总体标准差；各域代码共享相同标量尺度，但通道排列、各域真实统计值与域间几何关系不保留。随机数不消耗训练 RNG；所有时点输入保持不变。这些输入是随机代码，不再解释为物理图像统计。", "",
        "独立 Pooled Tokens 始终复制原真实 Linear Bank 的校准输出，三组初值逐位相同；Pooled 不读取控制后的 Descriptor。未使用的 Pooled Bank 行保留真实参考值，随机输入和输出尺度校准均已记录。", "",
        "R4 两层 Linear 原始初始化相同，Expansion 保留 one-hot 初始化。Shuffled 保持原 R4 的校准参数逐位一致；Random 沿用 R4 策略，仅按实际固定输入输出的初始全局 RMS=.02 缩放第二层 Weight/Bias，因此其第二层校准系数与 Real 可以不同。没有训练期 RMS 限制。这是相同架构/初始化规则的输入控制，不是三组初始 Text Features 完全相同。", "",
        "17 项单元测试与真实两步训练验收通过；原 Real 的初始模型逐位匹配旧 R4。完整训练后再次核对类别/Pooled 初值、原 Bank SHA、数据流、训练 RNG、源域 Centroids/Counts 和 Scheduler，全四域均一致；有效 Descriptor 全程冻结，生成器交叉梯度严格为零。", "",
        "## 正式性能", "",
        "以下为原 SPL 的时间平均 Target Text Features，评测保留 shuffle=True/drop_last=True。", "",
        "| 条件 | Art | Clipart | Product | Real World | Mean % | 相对 Real pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    real_mean = summary["arms"]["r4_real"]["mean_formal_accuracy"]
    for arm in ARMS:
        entry = summary["arms"][arm]
        scores = [100 * entry["tasks"][t]["result"]["final_accuracy"] for t in TARGETS]
        mean = 100 * entry["mean_formal_accuracy"]
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{v:.2f}" for v in scores + [mean]) + f" | {mean - 100*real_mean:+.3f} |")
    lines += ["", "已修正数据协议的 SPL B0 四域均值为 76.376%。本轮不以追平 B0 作为 Style 有效性的证据。", "",
        "![Formal accuracy](figures/final_accuracy.png)", "",
        "![Instant accuracy](figures/instant_accuracy.png)", "",
        "## 同一完整目标池上的配对预测", "",
        "各模型使用最终保存的时间平均文本，在同一完整缓存目标图像上重新分类；不丢弃尾批。因此它与正式性能的分母略有不同。", "",
        "| 目标 | Real % | Shuffled % | Random % | Real−Shuffled pp | Real−Random pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for target in TARGETS:
        entry = paired["targets"][target]
        scores = [100*entry["arms"][arm]["accuracy"] for arm in ARMS]
        differences = [entry["paired"][arm]["real_minus_control_pp"] for arm in ARMS[1:]]
        lines.append("| " + target + " | " + " | ".join(f"{v:.3f}" for v in scores + differences) + " |")
    lines += ["", "配对图像 Bootstrap 只反映固定已训练模型下的样本波动，不覆盖训练种子或随机代码/置换种子的波动，不能据此声称跨种子显著。", ""]
    for arm in ARMS[1:]:
        value = paired["comparisons"][arm]
        low, high = value["conditional_image_bootstrap_95pct_pp"]
        lines.append(f"- Real−{LABELS[arm]}：完整池四域平均 {value['mean_real_minus_control_pp']:+.3f} pp；条件图像区间 [{low:+.3f}, {high:+.3f}] pp。")
    lines += ["", "## Teacher 与输入条件响应", "",
        "Teacher 使用最后一次更新后的当前 Prompt、完整缓存目标图像，不能与时间平均 Student 的正式精度直接混为同一指标。Descriptor 替换保持角色及模型参数不变，仅在四个实际域的输入之间互换；以下 Target 翻转率排除了未使用的 Pooled Descriptor。", "",
        "| 条件 | Combined Teacher Mean % | Weighted Source Mean % | Source–Source Text Cos 范围 | Target 输入替换翻转率范围 % |",
        "| --- | ---: | ---: | --- | --- |"]
    mechanisms = {}
    for arm in ARMS:
        teacher, weighted, source_cos, flip = [], [], [], []
        for target in TARGETS:
            row = summary["arms"][arm]["tasks"][target]["trajectory"][-1]
            teacher.append(row["teacher_combined_accuracy"])
            weighted.append(row["teacher_weighted_source_accuracy"])
            source_cos.append(row["source_source_text_cosine"])
            selected = [r for r in swaps if r["arm"] == arm and r["target"] == target and int(r["step"]) == 1000
                        and r["recipient"] == target and r["donor"] != "pooled"]
            flip.append(100*average([float(r["prediction_flip_rate"]) for r in selected]))
        mechanisms[arm] = {"combined_teacher_mean": average(teacher), "weighted_teacher_mean": average(weighted),
                           "source_source_cosine": source_cos, "target_actual_descriptor_swap_pct": flip}
        lines.append(f"| {LABELS[arm]} | {100*average(teacher):.3f} | {100*average(weighted):.3f} | {min(source_cos):.6f}–{max(source_cos):.6f} | {min(flip):.3f}–{max(flip):.3f} |")
    lines += ["", "![Source text diversity](figures/source_text_diversity.png)", "",
        "![Target descriptor swap](figures/descriptor_swap_response.png)", "",
        "![Teacher quality](figures/teacher_quality.png)", "",
        "## 现有 Source Teacher 的路由空间", "",
        "此项只读检查没有提取逐图像 Style，也没有拟合或训练新路由。它测量最后状态三个 Source Teacher 的预测分歧，为下一步可靠性诊断提供背景。", "",
        "| 目标 | 三者预测一致率 % | 最佳单一 Source % | 任一 Source 正确的理想选择 % | 理想选择−最佳单一 pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    headroom = json.loads((args.root / "source_routing_headroom.json").read_text())
    for target, row in headroom["arms"]["r4_real"].items():
        best = max(row["source_accuracy"])
        oracle = row["hard_selection_oracle_accuracy"]
        values = [100*row["all_source_prediction_agreement"], 100*best, 100*oracle, 100*(oracle-best)]
        lines.append("| " + target + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "理想选择只约束直接选取一个 Teacher 的 top-1 预测，不是类别相关 Logits 混合的上限，也不排除置信度/NLL 互补。若保留当前 Source Teacher，Style 路由还需要面对预测同质化；不能只凭风格距离较近就假设该 Teacher 更可靠。", "",
        "## 判读边界", "",
        "若 Real 对 Random/Shuffled 都有一致优势，支持真实输入在当前固定架构和协议中的额外价值，随后仍需换训练种子和控制输入种子确认。若优势接近零或方向不一致，则当前损失和参数化尚未建立真实 Style 信息的必要性。Shuffled/Random 都保留可辨识的固定域代码，网络可以重新学习其关联；结果接近不等于没有任何输入响应，也不等于所有 Style 路线无效。", "",
        "随机输入改变通道结构，也改变函数空间的优化轨迹，第二层校准系数可以不同。因此即使 Random 较差，也不能单独归因于丢失正确风格语义。应同时结合 Shuffled、同模型文本差异、替换响应和 Teacher Quality。", "",
        "本轮没有实现 Style 路由，也没有启动 Stage4-only、r=64 或学习率变化。若转向 Style-guided 路由，应先对真实图像 Style Similarity 与 Source Teacher 正确率/NLL 做离线、按类别的可靠性诊断，再评估它是否补充现有语义路由；目标真实标签仅用于该诊断。"]
    (args.root / "mechanism_summary.json").write_text(json.dumps(mechanisms, indent=2) + "\n")
    (args.root / "analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
