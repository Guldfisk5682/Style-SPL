"""Readable measured report and standalone scientific figures; no fitting."""
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


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]]) + "\n"


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    s = json.loads((args.root / "summary.json").read_text())
    supplement = json.loads((args.root / "supplement.json").read_text())
    tasks = [s["tasks"][f"replay/{t}"] for t in DOMAINS]
    extra = [supplement["tasks"][f"replay_{t}"] for t in DOMAINS]
    figs = args.root / "figures"
    figs.mkdir(exist_ok=True)
    colors = ("#245d91", "#d27624", "#2c8876", "#888888")
    score_names = ("centroid_class_margin", "centroid_negative_candidate_distance", "teacher_confidence", "source_weight_max_for_candidate")
    score_labels = ("Centroid class margin", "Negative centroid distance", "Combined confidence", "Source weight max")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, task, name in zip(axes.flat, tasks, NAMES):
        for key, label, color in zip(score_names, score_labels, colors):
            v = task["verification"]["combined"]["scores"][key]
            roc = v["roc"]
            # Reverse the correctness ROC to obtain error-detection orientation.
            ax.plot(1-np.asarray(roc["tpr"])[::-1], 1-np.asarray(roc["fpr"])[::-1],
                    color=color, label=f"{label}: {v['auroc']:.3f}")
        ax.plot([0,1], [0,1], linestyle="--", color="silver", linewidth=.8)
        ax.set_title(name)
        ax.set_xlabel("False positive rate (correct pseudo-label flagged)")
        ax.set_ylabel("True positive rate (error detected)")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Frozen B0 Combined Teacher: error detection ROC", fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    save(fig, figs, "verification_roc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(4)
    for ax, stratum, title in zip(axes, ("all", "source_disagreement"), ("Full target pool", "Source-teacher disagreement subset")):
        for idx, pool, label, color in ((0, "three_source_teachers", "3 source teachers", colors[0]),
                                        (1, "three_teacher_branches", "Base / Pooled / Weighted", colors[1])):
            values = [t["oracle"][pool][stratum] for t in tasks]
            mean = np.array([v["headroom_pp_with_combined_fallback"] for v in values])
            ci = np.array([v["headroom_pp_class_cluster_ci95"] for v in values])
            ax.bar(x+(idx-.5)*.32, mean, width=.3, label=label, color=color)
            ax.errorbar(x+(idx-.5)*.32, mean, yerr=np.maximum(0, np.stack((mean-ci[:,0],ci[:,1]-mean))),
                        fmt="none", capsize=3, color="black", linewidth=.8)
        ax.set_xticks(x, NAMES, rotation=15)
        ax.set_title(title)
        ax.set_ylabel("Hard-selection oracle headroom (pp)\nCombined fallback allowed")
        ax.legend(fontsize=8)
    fig.suptitle("95% class-cluster intervals; oracle uses labels offline only", fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig, figs, "teacher_oracle_headroom")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, conditional, title in zip(axes, (False, True), ("All target images", "Within 20 confidence rank bins")):
        for idx, score, label, color in ((0, "centroid_class_margin", "Centroid class margin", colors[0]),
                                         (1, "teacher_confidence", "Teacher confidence", colors[1])):
            values = [e["conditional_auroc"][score] if conditional else t["verification"]["combined"]["scores"][score]["auroc"]
                      for t, e in zip(tasks, extra)]
            ax.bar(x+(idx-.5)*.32, values, width=.3, label=label, color=color)
        ax.axhline(.5, linestyle="--", color="gray", linewidth=.8)
        ax.set_xticks(x, NAMES, rotation=15)
        ax.set_ylim(.35, 1.)
        ax.set_title(title)
        ax.set_ylabel("Error detection AUROC")
        ax.legend(fontsize=8)
    fig.suptitle("Conditional AUROC averages within-bin pairs; descriptive, no fitted router", fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig, figs, "confidence_conditional_verification")
    with (args.root / "coverage.csv").open() as file:
        coverage = list(csv.DictReader(file))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, target, name in zip(axes.flat, DOMAINS, NAMES):
        for score, label, color in zip(score_names[:3], score_labels[:3], colors[:3]):
            rows = [r for r in coverage if r["mode"] == "replay" and r["target"] == target and r["score"] == score]
            ax.plot([float(r["coverage"]) for r in rows], [1-float(r["retained_pseudo_label_accuracy"]) for r in rows],
                    marker="o", color=color, label=label)
        ax.set_title(name)
        ax.set_xlabel("Retained pseudo-label fraction (predefined ranks)")
        ax.set_ylabel("Error rate among retained pseudo-labels")
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, figs, "verification_risk_coverage")

    text = ["# 冻结 B0：Visual Centroid 类别验证与 Teacher Oracle Headroom\n",
            "**主要结论：跨类别质心间隔能够区分部分错误伪标签，但其整体 AUROC 低于 Combined Teacher 的置信度，控制置信度后增量信号偏弱。Teacher 选择的 Oracle 空间则明确存在。因此当前更支持把研究重点转向 Teacher Ensemble / Confidence Calibration / Source Knowledge Aggregation，Class Prototype 暂保留为辅助候选。** 没有启动任何训练。\n",
            "## 数据、状态与标签边界\n",
            "四个 OfficeHome 3-source→1-target 任务，完整目标池 15,588 张图、65 类。使用已训练 B0 最后一步保存的冻结 prompts、CLIP RN50。目标真实标签只用于诊断指标、类别分层和明确标记的 Oracle；距离、verification score、Teacher 预测、候选选择、置信度分层均不读取目标标签。未拟合可靠性模型，未选择路由参数，未搜索距离或温度。\n",
            "B0 官方检查点没有保存 running centroids。因此本轮回放原作者的 Python-RNG 源采样序列，基于冻结图像缓存恢复质心：4 个源数据顺序 manifest 哈希匹配原训练，每任务所有 100 个已记录覆盖率检查匹配，每源累计 10,000 次采样，Source Text Features 与之前导出逐元素一致。原始质心没有保存，不能宣称与历史质心逐元素一致；缓存编码 batch=64 与训练 batch=30 也可能有小数值差异。另以完整源数据质心做敏感性对照，主要结果保持一致。\n",
            "这里的 **Target Teacher** 指给目标样本提供 soft pseudo-label 的 **Combined Teacher**；Target Student 的最后一步及 temporal-average 分类器另存于 summary.json。Combined 是冻结最终状态下按原融合公式重建的教师，不是声称恢复历史 step1000 更新前的每一份伪标签。全池指标也不是作者 shuffle/drop_last 正式测试指标。\n",
            "## 1. 正确类别的最近源质心是否通常更近\n",
            "主分析保持原 SPL：未归一化视觉特征的 squared-L2、源维度 softmax、w_scale=10。每张目标图保留完整 [3 sources,65 classes] 距离矩阵；每类取最近源质心，然后比较正确类别与最近错误类别。\n"]
    text.append(table(["目标", "正确类胜过均匀随机错误类 %", "正确类胜过最近错误类 %", "真实类别进入 Top3 %", "Combined Accuracy %"],
                      [[name, f"{e['true_class_nearer_than_uniform_wrong_class_fraction']*100:.2f}",
                        f"{t['true_class_strictly_nearer_than_any_wrong']*100:.2f}", f"{t['true_class_rank_top3']*100:.2f}",
                        f"{t['teacher_accuracy']['combined']*100:.2f}"] for name,t,e in zip(NAMES,tasks,extra)]))
    text.append("质心通常能把真实类别排在大多数随机错误类别之前，说明它保留了粗粒度类别证据；但在最近竞争类的比较中，最近质心分类在四域均明显弱于 Combined，尤其 Clipart 只有约 31%。因此不能把 source class mean 当作目标类别正确性的充分证据，也不能直接替换教师伪标签。\n")
    text.append("## 2. Class Verification 的错误检测能力\n\n预先固定主分数：**最近竞争类别距离−Teacher 预测类别距离**；各类别距离都在源维度取最小值。分数越高，视觉原型越支持 Teacher 的类别。把伪标签正确作为正例计算 AUROC，等价于把分数取负、伪标签错误作为正例检测错误。\n")
    rows = []
    for name, t, e in zip(NAMES, tasks, extra):
        v = t["verification"]["combined"]["scores"]
        ci = v["centroid_class_margin"]["ci95"]
        rows.append([name, f"{v['centroid_class_margin']['auroc']:.4f}", f"[{ci[0]:.3f},{ci[1]:.3f}]",
                     f"{v['teacher_confidence']['auroc']:.4f}", f"{v['centroid_negative_candidate_distance']['auroc']:.4f}",
                     f"{v['source_weight_max_for_candidate']['auroc']:.4f}", f"{e['conditional_auroc']['centroid_class_margin']:.4f}"])
    text.append(table(["目标", "跨类质心间隔 AUROC", "95% 类 cluster CI", "Teacher Confidence AUROC", "仅近距离 AUROC", "原 Source Weight Max AUROC", "同置信度层内间隔 AUROC"], rows))
    text.append("跨类别比较明显强于仅看候选类别距离；但 Combined confidence 在四域都更强。进一步把样本按不看标签的 confidence 排为 20 个等样本量区间，只在各区间内比较正确/错误对，质心间隔的配对加权 AUROC 降到 0.512 / 0.473 / 0.559 / 0.548。**整体 AUROC 0.75 不能直接解释为提供了独立于置信度的强验证信号。** 条件 AUROC 是描述性关联，没有训练/拟合一个 confidence+centroid router，也不能证明两个分数组合毫无潜力。\n\n按真实类别计算的 macro AUROC 与每类支持数见 auroc.csv/per_class.csv；按置信度四分位、≥0.9、≥0.99、源/三路分歧分层见 confidence_disagreement_strata.csv。最高置信度四分位在 Art / Real World 仅各 2 个错误，在 Product 没有错误，因此其中某些漂亮的 AUROC 无法构成可靠证据。\n\n![ROC](figures/verification_roc.png)\n\n![conditional AUROC](figures/confidence_conditional_verification.png)\n")
    text.append("固定、不调阈值的冲突规则：质心主分数<0，即竞争类视觉原型更近。在 Combined confidence≥0.9 的样本中：\n")
    rows = []
    for name, e in zip(NAMES, extra):
        c = e["conflict"]["confidence_ge_0.9"]
        rows.append([name, str(c["errors"]), f"{c['flagged_errors']} / {c['flagged_correct']}",
                     f"{c['error_precision']*100:.2f}%", f"{c['error_recall']*100:.2f}%", f"{c['correct_rejection_rate']*100:.2f}%"])
    text.append(table(["目标", "高置信总错误数", "标出错误 / 误标正确", "错误检测 precision", "错误 recall", "正确样本误拒率"], rows))
    text.append("原型冲突能提高部分错误的浓度，但误拒大量正确伪标签。例如 Art 找到 8 个高置信错误，同时标出 150 个正确预测；Clipart 为 14 vs 234。不能将这一规则直接作为硬否决或标签纠正器。固定 coverage 排名对照也显示 confidence 更有效地保留高质量伪标签。\n\n![risk coverage](figures/verification_risk_coverage.png)\n")
    text.append("## 3. Teacher 分歧时，改良的 Oracle 空间\n\nOracle 只使用目标真标签离线计算。分别考察 3 个 Source Teachers，以及 Base/Pooled/Weighted 3 路教师。允许保留 Combined 的 Oracle 可避免把 Combined 正确、所有候选错误的样本强行替换；candidate-only oracle 另存于 summary.json。\n")
    rows = []
    for name, t in zip(NAMES, tasks):
        a = t["oracle"]["three_source_teachers"]["all"]
        d = t["oracle"]["three_source_teachers"]["source_disagreement"]
        b = t["oracle"]["three_teacher_branches"]["all"]
        rows.append([name, f"{a['combined_accuracy']*100:.2f}", f"{a['keep_combined_or_select_candidate_oracle_accuracy']*100:.2f}",
                     f"+{a['headroom_pp_with_combined_fallback']:.2f}", str(a["recoverable_combined_errors"]),
                     str(d["n"]), f"+{d['headroom_pp_with_combined_fallback']:.2f}", f"+{b['headroom_pp_with_combined_fallback']:.2f}"])
    text.append(table(["目标", "Combined %", "Source+Combined Oracle %", "全池 Source Headroom pp", "可修复 Combined 错误数", "源分歧样本数", "分歧子集 Source Headroom pp", "全池三路 Branch Headroom pp"], rows))
    source_headroom = np.mean([t["oracle"]["three_source_teachers"]["all"]["headroom_pp_with_combined_fallback"] for t in tasks])
    text.append(f"四域等权平均 Source+Combined Oracle Headroom 为 **{source_headroom:.2f}pp**；分歧子集约 16.77–23.65pp。空间不小，但它是假设能够识别错误并挑中正确意见的理想值，不能当作可达到的性能预测。该上界只针对指定候选的 hard top1 selection，不是新训练模型、软分布融合或创造新类别预测的上界。NLL 的 source+combined Oracle 也保留了 0.126–0.309 的改善空间。\n\n![oracle](figures/teacher_oracle_headroom.png)\n")
    text.append("但用当前质心证据在 Source Teachers 给出的候选类别之间直接选最近者，结果如下：\n")
    text.append(table(["目标", "全池较 Combined Δpp", "源分歧子集 Δpp", "分歧中救回错误", "分歧中破坏正确"],
                      [[name, f"{t['candidate_class_selection']['centroid_among_source_predicted_classes']['all']['net_gain_pp']:+.2f}",
                        f"{t['candidate_class_selection']['centroid_among_source_predicted_classes']['source_disagreement']['net_gain_pp']:+.2f}",
                        str(t['candidate_class_selection']['centroid_among_source_predicted_classes']['source_disagreement']['rescued_combined_errors']),
                        str(t['candidate_class_selection']['centroid_among_source_predicted_classes']['source_disagreement']['spoiled_combined_correct'])] for name,t in zip(NAMES,tasks)]))
    text.append("可供选择的正确意见确实存在，但当前质心规则没有找到它们的可靠选择方式。分歧区域的 centroid margin AUROC 仅约 0.558–0.638，低于 confidence 的 0.707–0.755。\n")
    text.append("## 4. 原 Source Weighting 已经包含哪些信息\n\n原权重是**类条件的相对源亲和度**，并非完全不含类别信息。但在同一候选类别下，对全部源距离加同一个数，源维度 softmax 不变；跨类别加不同公共偏移时，它也不保存这些公共偏移。数值反事实验证：给每类全部源距离加相同的预定类别偏移，source weights 的 max error<1e-9，而最近原型类别发生变化。改变的是距离矩阵的代数反事实，没有生成图像或声称这是现实中的物理变换。\n\n因此矩阵确实还包含原源路由没有显式保留的跨类别/公共距离信息；问题在于**这些额外信息在当前 CLIP+单质心表示下是否足够可靠**。本轮 global margin AUROC 有效，但置信度条件后的信号偏弱。Teacher text logits 本身仍包含类别判断，不能把 Source Weighting 的信息损失说成整个 SPL 完全不能判断类别。\n")
    text.append("## 5. 静态融合对照与研究定调\n\n所有对照都只在同一批冻结 B0 文本特征上计算，没有重新训练、学习权重或依据目标标签选择规则。\n")
    keys = ("combined", "uniform_source_logits", "base_pooled_only", "base_pooled_uniform_source", "max_confidence_source_or_combined")
    text.append(table(["目标", "原 Combined %", "均匀 Source Logits %", "Base+Pooled %", "Base+Pooled+UniformSource %", "Source/Combined 中选最高置信 %"],
                      [[name, *[f"{e['static_aggregation_references'][k]['accuracy']*100:.2f}" for k in keys]] for name,e in zip(NAMES,extra)]))
    text.append("Weighted Source 单路分别为 70.33 / 51.20 / 82.02 / 80.86%，均低于相同 Source Teachers 的均匀 logit 平均。固定 Base+Pooled 在 Art、Clipart、Real World 上也略高于三路 Combined，Product 相同。**这些是最终固定状态的对照，不能推断训练时删除 Weighted 分支就会得到相同增益**；但值得具体检查它在哪些类别/样本上救回或破坏已有知识。\n\n另一方面，直接选择最高 confidence 的 Source/Combined 在四域都低于 Combined。这说明“Combined confidence 能检测图像难度”与“不同 Teacher 的 confidence 能公平比较其可靠性”是不同问题，后者可能需要校准和类别能力控制。\n\n建议当前主线收敛到：Source Teachers 保留有价值的互补意见，但原类条件源路由与固定三路融合没有可靠地利用这些意见。优先进一步静态研究 source/branch confidence 的可比较性、冲突类别、错误相关性与 Pooled-source 的知识融合；Class Prototype 暂作为辅助诊断，不据本轮直接进入硬 verification 或新的训练机制。MINT 的伪标签/蒸馏经验可复用在这一步，但不能代替当前任务中的可靠性证据。\n")
    text.append("## 敏感性、限制与输出\n")
    text.append(table(["目标", "完整源质心 Combined Δpp", "完整源质心 Margin AUROC Δ"],
                      [[name, f"{(s['tasks']['full/'+target]['teacher_accuracy']['combined']-t['teacher_accuracy']['combined'])*100:+.3f}",
                        f"{s['tasks']['full/'+target]['verification']['combined']['scores']['centroid_class_margin']['auroc']-t['verification']['combined']['scores']['centroid_class_margin']['auroc']:+.4f}"] for target,name,t in zip(DOMAINS,NAMES,tasks)]))
    text.append("- 单一已训练种子。类 cluster bootstrap 只刻画冻结模型下的类/数据构成不确定性，不代表训练种子显著性。\n- 主要 score 提前固定为 class margin；其他分数为机制对照，没有据 AUROC 挑选部署分数或参数。\n- 原质心未保存，采样回放不是有原张量对照的逐元素复原；完整源质心敏感性提供了交叉检查。\n- Student 的正式论文性能不能由静态 Teacher Oracle 推导；本轮没有训练或做在线标签纠正。\n- auroc.csv：所有教师及分数的 micro/按真实类别/按预测类别 macro AUROC。\n- per_class.csv：65 类的验证和最近质心分类指标。\n- confidence_disagreement_strata.csv：固定置信度和分歧分层。\n- supplement.json：条件 AUROC、固定冲突规则误检/漏检、静态融合对照。\n- summary.json / preparation.json / provenance.json：主要指标、采样回放依据、代码与输入状态 SHA、零更新记录。\n- 服务器 `~/workspace/Style-SPL/runs/b0_centroid_20261010/analysis/per_image_*.pt` 保存逐图像 [source,class] 距离矩阵、原源权重、教师预测、verification scores 和离线标签；prepared/b0_static_state.pt 保存回放/完整源质心及冻结教师特征。大张量不进入 git。\n")
    (args.root / "analysis.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
