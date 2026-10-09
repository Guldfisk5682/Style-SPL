"""Generate the evidence tables for completed controlled ablations."""
import argparse
import json
from pathlib import Path

ARMS = ["r1_shared", "r2_pooled", "r3_split", "r4_silu32"]
NAMES = {"r1_shared": "S1 Shared", "r2_pooled": "+ 独立 Pooled Tokens",
         "r3_split": "+ Source/Target 分离", "r4_silu32": "+ SiLU r=32"}
TARGETS = ["art", "clipart", "product", "real_world"]


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline_report", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    baseline = json.loads(args.baseline_report.read_text())
    lines = ["# Style-SPL 受控结构消融", "",
        "先解除 Pooled 的严格线性约束，再解除 Source/Target Projector 参数耦合，最后更换 Stage 映射。所有正式实验保持同一固定 Bank、种子 1、Office-Home 四个 3→1 任务、1000 次更新、原 SPL 损失及采样、学习率 .005 与原调度器。没有同时调整学习率、Extractor Stage 或 Bank。", "",
        "独立 Pooled Tokens 的初值严格复制 S1 初始 Linear Projector 的 Pooled 输出；r2/r3/r4 的 Pooled 初值一致。线性三组在真实 CLIP、同一批数据上的初始 Loss 完全一致。Source/Target 分离会同时分离其 Stage 参数和 Expansion 参数，二者从相同校准状态复制；共享 Class Prompt 保持原 SPL 设计。", "",
        "SiLU r=32 的两层 Linear 使用 PyTorch 默认初始化；仅将第二层 Weight/Bias 按固定 Bank 的初始全局 Prompt RMS 一次性缩放到 .02。保持 one-hot Expansion 初始化，后续允许正常学习；没有训练期 RMS 截断或重校准。", "",
        "随机初始化独立 Pooled Tokens 的早期先导实验已单独归档为 r2_pooled_random_init_pilot，因混入初始化差异而排除正式对比。", "",
        "## 复现和观测有效性", ""]
    verification = summary.get("replay_verification")
    if verification:
        lines.append("S1 重放逐位核对 Prompt、Optimizer、Scheduler、RNG、源/目标数据流、Centroids、Loss Valley、缓存累计次数及 Instant 最佳值；四域全部一致。" if verification["exact_replay"] else "重放校验失败；不能将轨迹视为原运行的历史恢复。")
    else:
        lines.append("S1 重放尚待四域最终验证。")
    lines += ["", "原运行只保留最终完整 checkpoint 和部分 Expansion，因此历史轨迹来自经过验证的重放。每 100 步保存五组 Prompt、同类别 Text Features、Bank/模型状态、Teacher KL 和损失分支梯度；每 200 步做描述符替换。梯度取自实际训练图的更新前状态，文本与 Prompt 快照为更新后状态；不能把两者视为完全同一状态。原 checkpoint 上的额外梯度来自最终状态的下一批数据，不冒称原 step1000 的历史梯度。", "",
        "## 正式性能", "", "正式性能使用原 SPL 的时间平均 Target Text Features，以及 shuffle=True/drop_last=True 评测。其余 Teacher 和替换诊断使用全部固定缓存 Target 图像，没有丢弃尾批。两种分母不同，应分别比较。", ""]
    rows = []
    b0 = [baseline["targets"][t]["formal_b0_accuracy"] for t in TARGETS]
    rows.append(["修正数据协议的 SPL B0", *[f"{100*x:.2f}" for x in b0], f"{100*sum(b0)/4:.2f}", "—"])
    last_mean = None
    for arm in ARMS:
        data = summary["arms"][arm]
        scores = [data["tasks"].get(t, {}).get("result") for t in TARGETS]
        values = [r["final_accuracy"] if r else None for r in scores]
        mean = data["mean_formal_accuracy"]
        rows.append([NAMES[arm], *[f"{100*x:.2f}" if x is not None else "待完成" for x in values],
                     f"{100*mean:.2f}" if mean is not None else "待完成",
                     f"{100*(mean-last_mean):+.2f}" if mean is not None and last_mean is not None else "—"])
        last_mean = mean
    lines += [table(["模型", "Art", "Clipart", "Product", "Real World", "Mean %", "相对上一组 pp"], rows), "",
              "![Instant Accuracy](figures/instant_accuracy.png)", "",
              "## 同模型内域条件差异与共同移动", "",
              "下表比较每个模型内同一类别的 Source/Target 文本，不比较独立训练模型之间的差异。平均范围为三个源域配对目标域。Prompt 移动量余弦比较相邻 100 步的差值；高共同移动和高文本余弦是观测关联，单凭这一关联不能确定唯一因果。", ""]
    rows = []
    for arm in ARMS:
        for target in TARGETS:
            task = summary["arms"][arm]["tasks"].get(target, {})
            records = task.get("trajectory", [])
            if not records:
                continue
            first, final = records[0], records[-1]
            rows.append([NAMES[arm], target, final["step"], f"{first['source_target_text_cosine']:.6f}",
                f"{final['source_target_text_cosine']:.6f}", f"{final['source_source_text_cosine']:.6f}",
                f"{final['pooled_target_text_cosine']:.6f}",
                f"{final['source_target_interval_movement_cosine']:.6f}" if "source_target_interval_movement_cosine" in final else "—"])
    lines += [table(["模型", "目标", "步数", "初始 S–T Text Cos", "当前 S–T Text Cos", "S–S Text Cos", "P–T Text Cos", "区间 S–T Prompt 移动 Cos"], rows), "",
              "![Text domain separation](figures/text_domain_separation.png)", "",
              "![Prompt co-movement](figures/prompt_comovement.png)", "",
              "## Projector 梯度耦合", "",
              "Source Objective 为 Pooled CE + 平均 Source CE；Target Objective 为 .5×Target Soft CE。Teacher 概率已 detach。梯度方向使用对应参数上的完整 FP32 向量，余弦无需平滑或 PCA。分离组要求 Source Projector 的 Target 梯度、Target Projector 的 Source 梯度严格为零；实际探测器一旦违反该条件便停止训练。共享 Class Prompt 的交叉作用仍存在，不能称为整个 Teacher/Student 完全解耦。", ""]
    rows = []
    for arm in ARMS:
        for target in TARGETS:
            gradient = summary["arms"][arm]["tasks"].get(target, {}).get("final_gradient_summary")
            if not gradient:
                continue
            source = gradient["style_projector"]
            target_projector = gradient.get("target_style_projector")
            rows.append([NAMES[arm], target, gradient["step"], f"{source['source_norm']:.5g}",
                f"{source['target_norm']:.5g}", f"{source['cosine']:.5f}" if source["cosine"] is not None else "无交叉梯度",
                f"{target_projector['source_norm']:.5g}" if target_projector else "共用"])
    lines += [table(["模型", "目标", "探测步数", "Source 参数上 Source 梯度范数", "Source 参数上 Target 梯度范数", "共享参数方向 Cos", "Target 参数上 Source 梯度范数"], rows), "",
              "## Teacher Quality 和蒸馏差异", "",
              "Base 为冻结模板 CLIP，Pooled 为混合源 Prompt，Weighted Source 按图像及类别执行原 SPL 的负平方距离 Softmax 路由，Combined 平均三路 Logits。KL 的方向为 KL(Teacher || Student)，在全部目标样本上计算。", ""]
    rows = []
    for arm in ARMS:
        for target in TARGETS:
            records = summary["arms"][arm]["tasks"].get(target, {}).get("trajectory", [])
            if not records:
                continue
            final = records[-1]
            rows.append([NAMES[arm], target, final["step"],
                *[f"{100*final[key]:.2f}" if final.get(key) is not None else "尚未启用" for key in
                   ["teacher_base_accuracy", "teacher_pooled_accuracy", "teacher_weighted_source_accuracy", "teacher_combined_accuracy", "student_live_accuracy"]],
                f"{final['kl_teacher_to_student']:.5f}"])
    lines += [table(["模型", "目标", "步数", "Base %", "Pooled %", "Weighted %", "Combined %", "Live Student %", "KL(T||S)"], rows), "",
              "![Teacher quality](figures/teacher_quality.png)", "",
              "![Teacher student KL](figures/student_teacher_kl.png)", "",
              "## 描述符反事实替换", "",
              "保持模型参数、Class Prompt、图像特征不变，仅将某角色的 Bank 输入改为其他域或 Pooled Descriptor，仍使用该角色自己的 Projector。汇报同类别 Text Cosine、原始及逐图像去均值后的 scaled-logit RMS 差、预测翻转率和分布 KL。去均值可以排除不会改变分类的公共 Logit 偏移。独立 Pooled Tokens 不读取 Descriptor，因此对该角色的描述符替换应严格无影响，这是预期行为。全部配对与每个时点位于 descriptor_swaps.csv。", "",
              "![Descriptor swap response](figures/descriptor_swap_response.png)", "",
              "## 解释与下一步边界", "",
              "r2−r1 检验独立 Pooled 参数的价值；r3−r2 检验生成器耦合影响；r4−r3 检验 r=32 非线性 Stage 映射的组合效果。最后一项同时改变了映射非线性、容量及优化参数化；若有收益，还需未来独立控制容量后才能声称收益来自 SiLU。", "",
              "这轮只有一个种子，不声称统计显著，也不使用目标标签选 checkpoint。域间 Text Feature 差异更大只是分支多样性证据；是否有用还要同时看 Teacher Quality、Student Accuracy、替换敏感性，不能只凭余弦数值给方法判优。Stage4-only、r=64、学习率调整未混入本轮。", ""]
    if summary["all_complete"]:
        means = [summary["arms"][a]["mean_formal_accuracy"] for a in ARMS]
        lines += ["所有正式实验已完成。", "",
                  f"逐步平均准确率变化：独立 Pooled {100*(means[1]-means[0]):+.3f} pp；分离生成器 {100*(means[2]-means[1]):+.3f} pp；SiLU r=32 {100*(means[3]-means[2]):+.3f} pp。", "",
                  "![Final accuracy](figures/final_accuracy.png)", ""]
    else:
        lines += ["当前报告为中间汇总，未完成组不得据此作最终结论。", ""]
    (args.root / "analysis.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
