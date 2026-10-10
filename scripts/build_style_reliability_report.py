"""Render the actual frozen-teacher reliability measurements, without fitting."""
import argparse
import csv
import json
from pathlib import Path

DOMAINS = ("art", "clipart", "product", "real_world")
NAMES = ("Art", "Clipart", "Product", "Real World")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(v) for v in row) + " |" for row in rows]]) + "\n"


def fmt(v):
    return "NA" if v is None else f"{v:+.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    report = json.loads((args.root / "summary.json").read_text())
    tasks = report["tasks"]
    text = ["# 冻结 SPL B0：逐图像 Style Similarity 与 Source Teacher 可靠性\n",
            "结论：**风格距离包含弱的逐图像 Teacher 可靠性信息，但当前距离不足以单独承担可靠的源 Teacher 路由。** B0 保留了充分的 Teacher 差异；控制类别×源 Teacher 平均优势后，两种四层平均距离的正向相关仍存在，但仅约 0.026–0.056。风格最近优于均匀随机选择的部分收益来自整体较强源域的偏好；四个任务都没有超过事后挑出的整体最强固定 Source Teacher。后者使用目标标签挑选，只是解释性参照，不能当作无标签部署基线。\n",
            "## 范围与数据完整性\n",
            "- OfficeHome 四个 leave-one-domain-out 任务，3 个有标签源域→1 个无标签目标域；完整目标池共 15,588 张图、每域 65 类。\n- 主要对象为已完成的、协议修正后的官方 SPL B0 最后一步 Source Teachers；全部参数冻结。补充 R4 Real / Random，使用同一真实物理 Style Bank，不用随机或打乱的输入 codes 充当物理风格。\n- 没有训练、更新 BN、修改网络结构或搜索路由温度。提取器没有读取目标类别标签；分析器只在离线 correctness/NLL、类别分层和统计控制中读取标签。\n- 图像路径、preprocess、数据 manifest 全部核对；四域 image embeddings 与旧缓存逐元素一致（max error=0），逐图像统计的平均值与已有 Style Bank 四层逐元素一致（max error=0）。所有输入 Teacher 状态 SHA256 在分析后保持不变，见 provenance.json。\n- 第一次提取因默认卷积后端与旧实验不同而未通过数值对齐检查，未生成正式缓存；改用原训练的 `fix_random_seed` 数值设置（cuDNN disabled）后重新提取并通过精确检查。失败日志保留于服务器，未用于指标。\n",
            "## 两种距离与可靠性\n",
            "每层把 RN50 空间 mean 和 per-image spatial std 拼接。**先计算逐 Stage cosine distance**；随后对拼接 descriptor 各坐标用该任务三个源域全部图像的 population std 标准化，计算坐标均方的平方根 L2。标准化均值、标准差与 floor=1e-6 均不使用目标数据或标签；源域样本数加权，银行是 per-image std 的平均。各 Stage 分别评估，四层距离等权平均只作为预先固定的聚合参照，没有根据目标标签选层。\n",
            f"NLL 用原始 CLIP RN50 `logit_scale.exp()={report['logit_scale']:.6f}`、归一化 image/text features 和稳定 log-softmax 计算。正确率用每个源 Teacher 的完整 65 类预测。应期待 **distance–NLL 正相关、similarity–可靠性正相关**；以下统一报告 similarity=−distance 与可靠性=−NLL，正值为有用方向。\n",
            "同一图像内比较 3 个 Teacher 可消除图像共有难度；进一步减去每个类别×源 Teacher 的平均值，检查是否只是在偏爱一个整体更强的源。类别控制只用于解释，不能成为依赖真实目标类别的推理策略。\n",
            "## B0 是否保留可检测的 Teacher 差异\n"]
    rows = []
    for target, name in zip(DOMAINS, NAMES):
        t = tasks[f"b0/{target}"]
        rows.append([name, ", ".join(t["source_domain_order"]), ", ".join(f"{v*100:.2f}" for v in t["teacher_accuracy"]),
                     ", ".join(f"{v:.3f}" for v in t["teacher_nll"]), f"{t['all_prediction_agreement']*100:.2f}%",
                     f"{t['metrics']['cosine_mean4']['mean_within_image_nll_range']:.3f}"])
    text.append(table(["目标", "源顺序", "各源准确率 %", "各源 NLL", "三源预测一致率", "平均单图 NLL range"], rows))
    text.append("B0 的三源一致率 56.77%–81.36%，而 R4 Real 为 98.06%–99.82%。因此本轮 B0 的弱关联不能简单归因于 Teacher 已趋同；B0 确有足够多的预测和 NLL 差异可供检测。这里是 Source Teacher 的最后一步性能，不是 B0 论文/正式实验的 Target Student temporal-average 准确率。\n")
    text.append("## 逐 Stage 关联：不是 Stage4 始终最有价值\n")
    for family, title in (("cosine", "Cosine distance"), ("source_z_l2", "Source-only 标准化 L2 distance")):
        text.append(f"### {title}\n\n下表是控制类别×源平均优势后的同图 Teacher 关联（正值为相似度更高、NLL 更低）。\n")
        rows = []
        for suffix in ("stage1", "stage2", "stage3", "stage4", "mean4"):
            rows.append([suffix, *[fmt(tasks[f"b0/{d}"]["metrics"][f"{family}_{suffix}"]["class_source_adjusted_within_image_pearson"]) for d in DOMAINS]])
        text.append(table(["Stage/聚合", *NAMES], rows))
    text.append("各层都出现小幅正相关，没有一致的 Stage4 优势。例如 Product 的 cosine 调整后相关从 S1 的 0.059 降至 S4 的 0.017；不能把此前 Stage4 对 Prompt Difference 的贡献最大解释成它对 Teacher Reliability 最有用。空间统计也可能含类别信息，类别控制不能证明已经完全剥离语义。\n\n![B0 stage reliability](figures/b0_stage_reliability.png)\n")
    text.append("## 选择最接近风格的 Teacher，会改善性能吗\n\n均匀参照是等概率选择一个 Teacher 的期望，不是平均 logits；原语义路由的类条件 logit mixture 另列于 summary.json。固定最强 Teacher 通过目标标签事后挑选，仅用于拆解总体优势。下表用四层平均 cosine：\n")
    rows = []
    for target, name in zip(DOMAINS, NAMES):
        m = tasks[f"b0/{target}"]["metrics"]["cosine_mean4"]
        s = m["selection"]
        best = s["posthoc_best_constant_teacher_offline_only"]
        rows.append([name, f"{s['accuracy']['style_nearest']*100:.2f}", f"{s['accuracy_gain_vs_uniform_teacher_expectation']*100:+.2f}",
                     fmt(s["nll_reduction_vs_uniform_teacher_expectation"]), f"{best['accuracy']*100:.2f}",
                     fmt(best["style_nll_reduction"]), f"{m['correctness_pair_concordance']*100:.2f}% / {m['correctness_pair_support']}"])
    text.append(table(["目标", "Style Accuracy %", "较均匀 Δpp", "较均匀 NLL 降低", "最佳固定源 Accuracy %", "较最佳固定源 NLL 降低", "一对正确/错误 Teacher 的风格排序胜率/支持数"], rows))
    text.append("Art 的 cosine 路由较均匀 +2.17pp、NLL 降低 0.0906，但固定 Real World Teacher 已更强；Clipart 的正确/错误 Teacher 风格排序接近随机且略低于 50%。不能把 Art 的收益解释为通用、强烈的逐图像风格路由能力。**硬选择一个 Teacher 的结果不是软加权或类条件 logits 融合的性能上界；本轮没有测试新的 soft mixture 或其训练效果。**\n\n进一步去除类别×源平均 NLL 后，逐图像选择仍有小幅收益：\n")
    rows = []
    for metric in ("cosine_mean4", "source_z_l2_mean4"):
        for target, name in zip(DOMAINS, NAMES):
            m = tasks[f"b0/{target}"]["metrics"][metric]
            s = m["selection"]
            ci = s["class_source_adjusted_nll_reduction_vs_uniform_ci95"]
            rows.append([metric, name, fmt(m["class_source_adjusted_within_image_pearson"]),
                         fmt(s["class_source_adjusted_nll_reduction_vs_uniform_offline_only"]),
                         f"[{ci[0]:+.4f}, {ci[1]:+.4f}]", f"{m['classes_positive_adjusted_pearson']}/{m['classes_supported']}"])
    text.append(table(["距离", "目标", "调整后 r", "调整后 NLL 降低", "95% 类别 cluster CI", "正相关类别数"], rows))
    text.append("标准化 L2 的校正后 NLL 收益在 Clipart / Product / Real World 的类 cluster 区间为正，提供了一些**有信息但较弱**的证据；cosine 的区间多数跨零。65 类中仅 36–44 类正相关，不是所有类别都遵循同一规律。固定模型的类别 bootstrap 只描述本轮数据构成的不确定性，不能代替多训练种子检验；十种距离/层分析涉及多重探索，不应凭小 p 值宣称方法有效。\n\n单个 Teacher 上跨图像的 correctness AUC 与同图源排序是不同问题。Product/Real World 的部分 cosine 按类别 AUC 低于 0.5，即“风格较近的图更容易分类”不普遍成立；详见 per_teacher.csv。\n\n![class association](figures/b0_per_class_association.png)\n\n![NLL selection](figures/style_selection_nll.png)\n")
    text.append("## 与现有语义路由是否互补\n\nB0 原检查点没有保存 source class centroids，不能重构其训练结束时的原始语义权重。B0 使用 R4 Real 已保存的路由作为共同参考；R4 两组使用各自真实保存的路由。可执行的类索引用冻结 Base CLIP 的预测，不用真实目标类别；真实类别切片仅以 `offline_only` 保存。\n\n语义强弱按该预测类别的路由 entropy 排为三等份（低 entropy 为强），阈值不看标签；还保存固定 max-weight<0.6 / ≥0.9 的绝对分层，防止把相对弱误叫成绝对低置信。按类别×语义强弱的交叉指标见 class_semantic_strata.csv。\n")
    with (args.root / "semantic_strata.csv").open() as file:
        strata = list(csv.DictReader(file))
    rows = []
    for target, name in zip(DOMAINS, NAMES):
        rows.append([name, *[fmt(float(next(r for r in strata if r["arm"] == "b0" and r["target"] == target and r["distance"] == "cosine_mean4" and r["stratum"] == s)["nll_reduction_vs_semantic"]))
                             for s in ("weak_relative", "middle_relative", "strong_relative")]])
    text.append(table(["目标", "语义弱：Style NLL 降低", "中", "强"], rows))
    text.append("风格相对语义选择在 Art 更有利、Clipart 弱路由区域有一些互补，但 Product 三层都受损；Real World 接近零。**没有支持统一的“语义弱就用 Style”规则。** 这是固定 B0 Teacher+R4 语义参考的离线比较，不能转述为原 B0 推理改进。\n\n![semantic strata](figures/b0_semantic_strata.png)\n")
    text.append("## 对下一步研究的判断\n\n1. Teacher 趋同确实降低可检测的路由收益，B0 是更合适的诊断起点；但恢复 Teacher 差异后，真实 Style Distance 仍只给出弱、不稳定的可靠性信号。\n2. 风格匹配与 Teacher 本身的类别能力是两个不同因素：近风格源不一定是最强源。均匀选择对照会混入固定源的总体优势，这是下一步机制叙事必须分清的问题。\n3. 尚不足以直接将 `softmax(-τ_s distance)` 全量替换/叠加到 SPL，再把性能波动当作 Style 路由的证据。若继续探索，更适合将风格作为受限的辅助信号，并明确验证何种源/类别/语义冲突场景存在增量信息；本轮没有启动这样的训练。\n4. Source-only 标准化 L2 较 raw cosine 更能保留一些类别控制后的信号，值得保留为候选，但它尚未在所有域的实际源选择准确率/NLL上稳定获益；不能据此宣称距离选择或方法已经定型。\n")
    text.append("## 文件与复现\n\n- summary.json：12 组任务×10 个预先固定距离指标；包含源顺序、Source Teacher 正确率/NLL、分层、路由协议、NLL scale 与提取完整性。\n- per_class.csv / per_teacher.csv：逐类别关联、类别内 AUC 和 Teacher 个体难度关联。\n- semantic_strata.csv / absolute_strata.csv / class_semantic_strata.csv：相对、绝对及交叉分层。\n- figures/：PNG 与 PDF 科学图。\n- provenance.json：代码和输入状态 SHA、零训练更新、任务退出状态。\n- 服务器 `~/workspace/Style-SPL/runs/style_reliability_20261010/styles` 保存四域逐图像统计与 B0 texts；`analysis_complete/per_image_*.pt` 保存路径、离线标签、每源 NLL/正确性、10 种距离和语义权重。大张量没有提交 git。\n\n运行：先 `scripts/extract_image_styles.py`，再 `scripts/analyze_style_reliability.py`，最后 `scripts/plot_style_reliability.py` 和本脚本。统计检查 tests/test_style_reliability.py 验证 source-only 标准化不读取 target 样本、固定源优势被移除、逐图像有效信号仍能保留。\n")
    (args.root / "analysis.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
