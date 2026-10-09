# Style-SPL ：工程实现与实验验证规格

> **用途**：交给 Codex，在已成功复现的 CRPL / SPL 分支上进行最小侵入式实现，并完成第一轮 Office-Home 多源 UDA 验证。  
> **状态**： 实验方案（不是已验证有效的方法）；日期：2026-10-09。  
> **关键原则**：仅修改 Domain Prompt 的来源；保留 Class Prompt、Source Visual Centroids、Source Routing、Teacher 融合以及原 SPL 监督路径。**首次实验关闭 OT**。

## 0. 目标与边界

### 研究假设

CRPL 将每个源域、合并源域和目标域的 Domain Prompt 作为彼此独立的自由可学习参数。我们希望验证：使用冻结 CLIP RN50 提取的**域级多尺度 Style Statistics（每层 Mean / Std）**，经**各层共享 Style Projector + 4→16 Token Expansion**生成 Domain Prompt，是否能改善 SPL 的跨域学习与最终目标域准确率。

 **不是**逐图像动态 Prompt，不引入 Content Projector、不修改源域类别 Visual Centroid、不使用 Target 真实标签、不增加 Style 对齐损失。先衡量对原 SPL 的净收益，再考虑低秩 Projector、Content Verification、Instance-conditioned Style 或 Style-aware Routing。

### 约定

- Backbone：**CLIP RN50**，冻结 Visual / Text Encoder；Visual `eval()`，尤其不得改变 BN Running Statistics。
- 第一轮：**Office-Home，multi-source UDA**；逐目标域跑四个 target domains，每个目标域对应 3 个 source domains。
- `N=3`（Source 数）、`K=65`（类别数）、`M1=16`（原 Class Token 数）、`M2=16`（Domain Token 数）。RN50 **Text Transformer 的 Token Width = 512**；CLIP **最终图文特征维度 = 1024**，二者不能混淆。
- 已复现的本地基线分支是比较基准。**不要直接覆盖已经修复的 Source-combined Prompt 初始化问题**；检查本地实际实现，保留已有数值一致性修正。
-  实验默认：Style Projector 使用 Full Linear，Mean/Std 分别保留并拼接，Expansion 循环 One-hot 初始化，**直接替换**原 Domain Prompt（不叠加残差，不另加可学习 Scale）；训练开始前执行一次输出 RMS 校准。

## 1. 与官方 CRPL 代码的对照

公开参考仓库：[VuongLong/Clustering-Reinforcement-Prompt-Learning](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning)。此处以公开 `main` 分支核对；**如本地代码已更改，以本地已复现分支为准**。

| 文件 / 对应位置 | 现有职责 |  处理 |
|---|---|---|
| [`clip_custom/model.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/clip_custom/model.py) `ModifiedResNet.forward()` | Stem → `layer1..layer4` → `attnpool` | 只为 Style 预处理读取各 Stage 的中间特征，**保持最终视觉 Embedding 不变**。优先通过安全的 Hook 或额外提取接口完成。 |
| [`model.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/model.py) `PromptGenerator.__init__()` | `ctx_cls`；每 Source 一个 `ctx_source`；`ctx_source_combined`；`ctx_target` |  新增 `style_projector` 及固定的 Style Bank；替换 Domain Prompt 的数据来源。**`ctx_cls` 保留。** |
| `model.py` `PromptGenerator.forward()` | 合并 Source + Target Prompt，全部 K 类 | 改为读取 `style_prompt[pooled]` 与 `style_prompt[target]`；拼接的 token 位置及类别名称不变。 |
| `model.py` `PromptGenerator.forward_source(i)` | 第 i 个源域的全部 K 类 Prompt | 改为读取 `style_prompt[source_i]`。 |
| [`main.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/main.py) `train()` | 每目标域建立模型、优化器、Source Centroids、Teacher / Loss / Evaluation | 每目标域开始训练前构建/加载 Style Bank；Projector 纳入原 Prompt 优化器；其余 SPL 数学逻辑不改。 |
| [`dataloader.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/dataloader.py) / [`samplers.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/samplers.py) | Target 数据、Source 均衡采样 | 训练 Loader 不改；**Style 预处理遍历完整允许的源域训练样本和无标签目标训练样本**，不采用每步随机 Source Batch 来估计最终固定 Bank。 |

**现有关键事实**：

- 原 Class Prompt `ctx_cls ∈ [K,16,512]` 由所有源域与目标域共享。原域 Prompt `ctx_source[i]`、`ctx_source_combined`、`ctx_target` 各为 `[1,16,512]`。
- 原 Source Visual Centroid 是**真实源域类别标签分组后的未归一化 CLIP 最终视觉特征 Running Mean**，`running_means ∈ [N,K,1024]`。**Class Prompt 不参与质心计算。**
- Teacher 包含 Frozen Base CLIP、Source-combined Text、按 Source Centroid 平方 L2 距离加权的 Source-specific Text Logits。第三路只在每个 Source×Class 的质心至少观察过一次后启用。
- Target Teacher Soft Labels `detach()`；对 Target Prompt 的 Soft CE 保持原状。关闭 OT 要同时关闭原代码中的 OT Cost 与该分支额外的 Hard Pseudo-label CE（`--OT_clustering 0`）。
- 官方 `main.py` 还具有 Final Evaluation 使用 `prompt_learner.target_features`（运行平均缓存）的行为。**对比试验使用完全一致的评估规则**，并检查该缓存是否真正被写入；可以额外记录即时 Target Text Features 的准确率，但不要混称为原指标。

## 2. 数据结构与形状

### 2.1 Style Bank（新增，预计算一次）

冻结 CLIP RN50 Stage 通道数：`C=[256,512,1024,2048]`。输入典型尺寸为 224×224 时，各 Stage 输出大致是：

| Stage | Feature Map `[B,C,H,W]` | `mu` / `std` 各自 `[B,C]` | 拼接 `[B,2C]` |
|---|---|---|---|
| 1 | `[B,256,56,56]` | `[B,256]` | `[B,512]` |
| 2 | `[B,512,28,28]` | `[B,512]` | `[B,1024]` |
| 3 | `[B,1024,14,14]` | `[B,1024]` | `[B,2048]` |
| 4 | `[B,2048,7,7]` | `[B,2048]` | `[B,4096]` |

**每张图像的 Style**：每层 `mean` 和 `std` 均沿 `H,W` 维计算：

$$
\mu_l(x)=\mathrm{mean}_{h,w}F_l(x),\qquad
\sigma_l(x)=\sqrt{\mathrm{mean}_{h,w}(F_l(x)-\mu_l(x))^2+\epsilon}.
$$

采用 **AD-CLIP 官方 RN50 `AdaIN` 的标准差定义（包含 `HW` 分母）**，同时采用 **STYLIP-style Mean/Std Concatenation**：每层保存**完整的两组 C 维统计量**，不要预先相加或取平均成 C 维。

**每个 Domain 的 Style**：对该域训练集中每张图像的 Style Statistics 分别求样本平均：

$$
S_{d,l}=\left[\frac{1}{n_d}\sum_x\mu_l(x)\,;\,\frac{1}{n_d}\sum_x\sigma_l(x)\right]\in\mathbb R^{2C_l}.
$$

这**不是**把全域所有像素/Feature Map 混在一起重新求 Std；它是“图像级 std 向量的均值”，需在文档和代码中明确。

一个目标域对应 `N+2` 个 Domain Style entries：

- `source_style[i]`：N 个独立源域；
- `pooled_source_style`：所有源域**按图像数加权**合并得到的整体均值（不能在各源域样本数不相等时简单平均域均值）；
- `target_style`：该 Target Domain 的**无标签训练样本**的整体均值。

**推荐存储**：每个 Domain 存四组 `[(mu1,std1), ..., (mu4,std4)]`，不强制物理拼接为一个 7680 维 Tensor。若拼接全部四层，逻辑总维度为 `2*(256+512+1024+2048)=7680`；这一数字仅指整套统计量，不指单层通道数。五个 Domain（Office-Home 的 N+2）共约 150 KiB FP32 统计值。

推荐缓存元数据：`backbone=RN50`、`dataset`、`target_domain`、`source_domain_order`、`preprocess_id`、样本数、每层 shape、数据清单/版本标识、提取精度（FP32）、生成时间、代码版本/commit。**验证 Cache 元数据后才能复用**，避免跨 Target Split / 变换 / Domain 顺序误用。

### 2.2 其余结构（完全保留）

- `ctx_cls: [K,16,512]`：原 Class Prompt，继续学习；
- `running_means: [N,K,1024]`：原 Source Visual Centroid，训练中按照原逻辑更新；
- `source_text_features: [N,K,1024]`：每步重算，仍是**Style-conditioned 类别文本特征**，不是“每个 Domain×Class 的 Mean/Std”；
- `pooled_text_features: [K,1024]`、`target_text_features: [K,1024]`：每步重算；
- `base_text_features: [K,1024]`：原 CLIP base 模板，不受 Style Projector 影响。

## 3. 固定 Domain Style Bank：预处理伪代码

> 以下是适配本地 CRPL 的**伪代码/接口规范**，不是声称官方 repo 已存在的函数。预处理必须使用 `model.eval()`、`torch.no_grad()`、固定图像变换，并对每个 ImageFolder **逐样本遍历一次**。不使用 Target GT Label。

```python
CHANNELS = (256, 512, 1024, 2048)

@torch.no_grad()
def extract_style(images, frozen_rn50):
    # images: [B,3,224,224]
    # 必须使用当前 CRPL 正在使用的 CLIP RN50 的同一组权重
    # 注意：自定义 CLIP ModifiedResNet 的 layer1..layer4 输出，位于 attnpool 之前
    maps = forward_rn50_stages(frozen_rn50, images)  # list of four [B,C,H,W]
    out = []
    for feat in maps:
        feat = feat.float()   # 均值/方差累计使用 FP32
        B, C, H, W = feat.shape
        mu = feat.mean(dim=(-2, -1))   # [B,C]
        # 与 AD-CLIP 同类的 population std，明确 HW 分母
        sigma = (((feat - mu[..., None, None]).square()
                   .sum(dim=(-2, -1)) + 1e-8) / (H * W)).sqrt()  # [B,C]
        out.append((mu, sigma))
    return out


def new_accumulator():
    return {
        'mu_sum': [torch.zeros(c, dtype=torch.float64) for c in CHANNELS],
        'std_sum': [torch.zeros(c, dtype=torch.float64) for c in CHANNELS],
        'count': 0,
    }


def update_accumulator(acc, stats):
    B = stats[0][0].shape[0]
    for l, (mu, std) in enumerate(stats):
        acc['mu_sum'][l] += mu.double().sum(dim=0).cpu()
        acc['std_sum'][l] += std.double().sum(dim=0).cpu()
    acc['count'] += B


def finalize(acc):
    assert acc['count'] > 0
    n = acc['count']
    return [
        (acc['mu_sum'][l].div(n).float(),
         acc['std_sum'][l].div(n).float())
        for l in range(4)
    ]

# source_accs[i]: 扫描该源域所有训练图像
# target_acc:  扫描对应目标域所有无标签训练图像
# 每个目标域训练前，得到 source_accs、target_acc
pooled_acc = new_accumulator()
for acc in source_accs:
    pooled_acc['count'] += acc['count']
    for l in range(4):
        pooled_acc['mu_sum'][l] += acc['mu_sum'][l]
        pooled_acc['std_sum'][l] += acc['std_sum'][l]

style_bank = {
    'sources': [finalize(acc) for acc in source_accs],  # 长度 N
    'pooled': finalize(pooled_acc),
    'target': finalize(target_acc),
}
```

**实现提示**：可对 `clip_custom/model.py` 的 `visual.layer1..layer4` 使用一次性的 Forward Hooks，或新增独立的 `forward_stages()` 路径。必须验证：提取操作不改变原 RN50 的普通 `encode_image` / `Custom_Clip.forward_img_both()` 输出；Hook 无泄漏，预处理无梯度、无 BN 统计变化。若启用 AMP，应先 `.float()` 再求均值/方差。

**协议注意**：Style Bank 只能使用该轮 UDA 允许访问的源域训练图像和目标域无标签训练图像。Office-Home 某些代码将整个目标域作为 transductive pool；第一轮应与已复现基线**保持完全相同的目标训练集合**。不得使用目标域测试标签进行 Bank 构建或训练调参。若训练/测试有严格划分，只使用目标训练图像。

## 4. Style Projector：Full Linear + One-hot Token Expansion

对某个域，四个 Stage-wise Projectors（**所有 Domain 共用同一套参数**）分别做：

$$
t_l=W_l[\mu_l;\sigma_l]+b_l\in\mathbb R^{512},\quad
T=[t_1;t_2;t_3;t_4]\in\mathbb R^{4\times512}.
$$

接着使用可学习的 `A ∈ R[16,4]`：

$$
\boxed{P_D=A T\in\mathbb R^{16\times512}}.
$$

- `W_l` 维度分别为 `512×512`、`512×1024`、`512×2048`、`512×4096`；四层合计 3,932,160 个 Weight，另有 2,048 个 Bias；`A` 64 个参数。
- **Token Expansion 初始化**：`A[j, j % 4] = 1`，其余为 0（最终是循环 One-hot，不是全零矩阵、也不是全一矩阵）。初始时四层 Token 各重复四次；训练中 `A` 可学习混合各层信息。
- 这个扩展**不会创建 16 份互相独立的原始 Style 信息**：输出 Token 在 Token 维上的秩 ≤ 4。这是  接受的结构限制；后续可单独研究高表达能力 Expansion。
- 与 AD-CLIP `domain_projector` 不同：**不先将 mean/std 平均为 C 维**，保留 `2C` 完整输入。
- 所有 Source / Pooled / Target 的 Style Prompt 都由同一个 Projector 生成，输出并按原 CRPL 的类别维度 `repeat(K,1,1)`。**不存在每张 Target Image 重算 K 条 Text Prompts**。

```python
class DomainStyleProjector(nn.Module):
    def __init__(self, token_width=512, n_tokens=16):
        super().__init__()
        channels = (256, 512, 1024, 2048)
        self.stage = nn.ModuleList([
            nn.Linear(2*c, token_width) for c in channels
        ])
        for layer in self.stage:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        A = torch.zeros(n_tokens, 4)
        for j in range(n_tokens):
            A[j, j % 4] = 1.0
        self.expansion = nn.Parameter(A)

    def forward(self, domain_style):
        # domain_style = [(mu_l[C_l],std_l[C_l])] * 4
        tokens = []
        for l, (mu, std) in enumerate(domain_style):
            x = torch.cat([mu, std], dim=-1)   # [2*C_l]
            tokens.append(self.stage[l](x))   # [512]
        T = torch.stack(tokens, dim=0)         # [4,512]
        return self.expansion @ T               # [16,512]
```

### 4.1 初始化：只做一次 Output RMS Calibration

直接替换 Domain Prompt 时，目标是在训练起点让新 Domain Tokens 的**整体 RMS 大约等于原 CRPL 的随机初始 std=0.02**；不通过 `bias=-10`、输入 Clamp 或额外可学习 Scale 实现。

```python
@torch.no_grad()
def calibrate_initial_output(projector, style_bank, target_rms=0.02):
    domains = [*style_bank['sources'], style_bank['pooled'], style_bank['target']]
    prompts = torch.stack([projector(S) for S in domains], dim=0)
    rms = prompts.float().square().mean().sqrt()
    assert torch.isfinite(rms) and rms > 1e-10
    factor = (target_rms / rms).item()
    for layer in projector.stage:
        layer.weight.mul_(factor)
        layer.bias.mul_(factor)
    # A 不缩放；第一次且仅第一次，在创建 optimizer 之前执行。
```

**必须记录**：校准前后全局 RMS、**每个 Domain** Prompt 的 RMS / mean / std / max-abs，以及 Stage Tokens 的 RMS。整体 RMS≈0.02 不确保每域都接近 0.02，若某域异常，先报告，不要自动按域调节（避免引入新的设计因素）。校准只约束初始化，训练后输出可自行变化。

## 5. 替换 CRPL PromptGenerator：接口与梯度

### 5.1 直接替换，不做残差

原始：

```text
ctx_cls[k] + ctx_source[i] / ctx_source_combined / ctx_target + class_name
```

：

```text
ctx_cls[k] + style_projector(style_bank[source_i / pooled / target]) + class_name
```

两处 `+` 表示 Prompt **Token 序列拼接**，不是数值相加。沿用原文本模板：

```python
# 这两个函数为接口示意，不是原 repo 现成 API

def domain_tokens(kind, source_index=None):
    if kind == 'source':
        S = style_bank['sources'][source_index]
    elif kind == 'pooled':
        S = style_bank['pooled']
    elif kind == 'target':
        S = style_bank['target']
    return style_projector(S).unsqueeze(0)  # [1,16,512]


def build_text_prompts(domain_ctx):
    return torch.cat([
        token_prefix,                 # [K,1,512]
        ctx_cls,                      # [K,16,512]
        domain_ctx.repeat(K, 1, 1),  # [K,16,512]
        token_suffix,                 # [K,*,512]
    ], dim=1)
```

- `PromptGenerator.forward()`：分别构造 Pooled-source 和 Target 全 K 类 Prompts。
- `PromptGenerator.forward_source(i)`：构造第 i 个 Source 全 K 类 Prompts。
- 删除/停用旧 `ctx_source / ctx_source_combined / ctx_target` 的**可学习**参数；避免它们保留在优化器里成为冗余参数。Class Prompt 完全保留。
- 将 Style Bank 注册为 `register_buffer`（建议以按域+阶段的结构化 Tensor Buffer 或不可训练 Module 持有；**不能**用 `nn.Parameter`）。预处理后固定，但设备移动、保存和恢复需兼容。
- **同一个前向计算图可以让共享 Projector 从所有域的 Source CE 和 Target Soft CE 获得梯度**；不得为加速而对输出 `style_prompt` 或 `target_txt_features` 执行 `detach()`，也不得将它们缓存为跨 Step 固定值。
- Teacher 的 Soft Label `detach()` 保持原逻辑。Target 分支的 Student Text Features 通过冻结 Text Encoder 反向传播至 Style Projector 和 Class Prompt；冻结权重并不等于将整个 Text Forward 包在 `no_grad()` 中。
- Projector 每一步 Forward 重新计算 Domain Prompts；**Style Bank 不更新、不重算**。Source Visual Centroid Bank 仍依原逻辑在线更新。不可混淆。

### 5.2 Projector 输出语义

`source_text_features[i,k]` 是第 i 个源域的 **Style-conditioned 类别文本特征**（1024 维），**不是**第 i 域第 k 类图像 Style Statistics 的 Mean/Std。Mean/Std 仅用于构造该 Domain 的公共 Prompt。原 Source Visual Centroid `[N,K,1024]` 才是按真实类别划分的图像特征均值。

## 6. 训练设置与公平对比

### 6.1 首轮实验矩阵

| ID | 模型 | Domain Prompt | OT | 用途 |
|---|---|---|---|---|
| B0 | 已复现 SPL-only | 原 CRPL 可学习 16 Tokens | **Off** | **必须首先验证的真正对照** |
| S1 | Style-SPL （主实验） | `Full Linear(2C→512) ×4 + One-hot Expansion(4→16)` **直接替换** | **Off** | 检验风格条件化的效用 |
| S2（可选） | Style-SPL Residual | 原 Domain Prompt + Style Prompt | Off | 若 S1 明显退化，作为诊断对照，不混入第一轮主要结论 |
| S3（后续） | Style-SPL Low-rank | 低秩 Stage Projectors，其他不变 | Off | 在  验证后再测试效率 / 参数量 |

**对比时固定**：已复现的 `seed`、Target Domain 顺序、数据划分与预处理、Source 采样策略、Class Prompt 初始化、Batch、训练 Step 数、学习率及调度、`w_scale`、`t_weight`、Teacher 可用条件、评估规则。正式汇报至少覆盖 Office-Home 四个 Target Domains；首个 Smoke Test 可选其中一个域。

**首次验证不应该把任何额外损失、Content Projector、Style Routing、Label-based Target Prototype、数据增强一起加入。**

### 6.2 有关学习率与优化器

公开仓库 `main.py` 的优化器为 `AdamW(prompt_learner.parameters(), lr=args.prompt_learning_rate)`；README 示例传入 `0.005`，而代码默认值为 `0.003`。**不要假设其中一个就是本地已复现的配置**，应读取已有实验日志，B0 与 S1 用相同设置。Style Projector 和 Expansion 均通过 `nn.ModuleList` / `nn.Parameter` 自动注册，加入同一个优化器，不额外引入独立学习率。

新增参数是否实际受训练：在第一个 Backward 后确认每个 Stage Projector 的 `weight.grad` 非空且有限，`expansion.grad` 非空且有限；保存并比较训练前后参数变化。若梯度爆炸才考虑后续干预。

## 7. 必须记录的训练指标

统一在固定 Step 间隔输出结构化日志（CSV 或 TensorBoard，推荐二者兼容），按 `run_id / seed / target_domain / step` 组织；训练日志中不允许依赖目标 GT。

### P0：训练正确性、损失与评估（每 Step 或每 10 Step）

| 日志键（建议） | 定义 / 必要性 |
|---|---|
| `train/loss_total` | 原 SPL 完整训练目标，检查 NaN / 损失尺度。 |
| `train/loss_source_pooled` | Source-combined CE。 |
| `train/loss_source_domain_avg` | 各 Source CE 的平均（与原 SPL 一致）。 |
| `train/loss_source_domain_i` | 各源域独立 CE，用于查看哪个源域困难。 |
| `train/loss_target_soft` | Target Soft-label CE（OT 关闭后不混入 Hard Target CE）。 |
| `train/teacher_entropy`, `train/teacher_confidence_max_mean` | Teacher 软标签熵、最大概率平均值，用于检查伪标签过度自信或均匀化。 |
| `train/student_teacher_agreement` | Student / Teacher 的 argmax 一致率（**不需要目标标签**）。 |
| `train/student_teacher_kl` | `KL(teacher || student)`，指定方向，与 Soft CE 监控互补。 |
| `train/source_centroid_coverage` | 有效 `[source,class]` 项数 / `[N×K]`，确保 weighted 分支启用条件一致。 |
| `train/weighted_teacher_enabled` | 当前是否启用第三路距离加权 Teacher。 |
| `train/lr` | 当前学习率（沿用原 scheduler 步频）。 |
| `eval/target_top1_instant` | 每次测试时**即时 Target Text Features** 的 Top-1；与 B0 同规则。 |
| `eval/target_top1_final` | 训练完成后按原模型的 Final Text Feature 缓存规则测得的 Top-1；若缓存未更新要显式告警。 |

### P1：Style Prompt 健康度（建议每 10–20 Step）

| 日志键（建议） | 定义 / 必要性 |
|---|---|
| `style/prompt_rms_global` | 所有 N+2 个 Domain Prompt 的 RMS；训练起点约 0.02。 |
| `style/prompt_rms_domain_i` | 每个源域、Pooled、Target 的 RMS。 |
| `style/prompt_mean_domain_i`, `style/prompt_std_domain_i` | 每域 Prompt 元素分布，检查严重偏移或塌缩。 |
| `style/prompt_absmax_domain_i` | 每域绝对值最大值，监控异常。 |
| `style/stage_token_rms_l` | 每个 Stage 投影后 Token RMS，检查某层完全主导。 |
| `style/grad_norm_stage_l` | 每层 Projector 梯度 L2 范数。 |
| `style/grad_norm_expansion` | Expansion 矩阵梯度 L2 范数。 |
| `style/grad_norm_class_prompt` | 原共享 Class Prompt 梯度范数。 |
| `style/expansion_row_entropy`（可选） | 归一化后 Expansion 的 Stage Mixing 分布熵；需说明使用 softmax 仅用于诊断，不在 Forward 强制 Softmax。 |
| `style/token_cosine_source_target`（可选） | 各 Source Prompt 与 Target Prompt 的余弦相似度，检查跨域表示塌缩。 |

**关于 `A` 的记录**：Forward 中 `A` 是无约束的可学习矩阵，因此其元素可能为负；不要把原始 `A` 行和或负值直接当作“概率”。可直接画 Heatmap，并记录行 L1/L2 norm；如要 entropy，需基于单独的诊断 softmax。

### P2：SPL 路由与伪标签诊断（可选，每 50–100 Step）

- `routing/source_weight_entropy`：平均 Source 权重熵；必须只在 `running_count` 已覆盖全 N×K 后计算。
- `routing/source_weight_max_mean`：每个 Target×Class 的最大 Source 权重平均值。
- `teacher/base_vs_pooled_agreement`、`teacher/pooled_vs_weighted_agreement`：各 Teacher 分支的类别预测一致率（无目标标签）。
- `eval/per_class_acc`、Confusion Matrix：仅在**显式评估阶段**使用标签，绝不反馈至训练、Bank 或超参数选择。

## 8. 必须绘制的可视化

一份 `plots/` 目录，脚本可从日志自动重建。不同 Target Domains 分开绘图；最终再汇总。

1. **B0 vs S1 目标域 Top-1 曲线**：`step → target_top1_instant`，横轴相同，四个 Target Domains 各一张；结果表列最终值、最好值、平均值，并明确 final-vs-best 的统计口径。
2. **Loss 曲线**：Total / Pooled Source CE / Avg Source CE / Target Soft CE，B0 与 S1 同坐标可比较。
3. **Style Prompt RMS 曲线**：Global + 每个 Domain，画出初始化约 0.02 的参考线；检查早期 100–200 Step 的变化。
4. **Projector / Expansion / Class Prompt 梯度范数曲线**：发现某模块始终无梯度、异常大梯度或快速趋零。
5. **Stage Token RMS 曲线**：Stage 1–4，判断高通道 Stage 是否主导生成内容。
6. **Token Expansion Heatmap**：初始化、训练中（例如 step 100/500）、训练结束的 `[16,4]` 权重；对比是否保持 One-hot 或学习到 Stage Mix。
7. **Teacher Confidence / Entropy + Centroid Coverage**：在启用 Weighted Source Teacher 的 Step 附近标注垂线，排查两路→三路切换造成的训练震荡。

可选：原始 Domain Style Descriptors 的 Stage-wise PCA / cosine similarity 以及 Style Tokens 域间相似度，仅作为分析；**不应把域区分度直接当成性能提升的证据**。

## 9. 运行工件、复现信息和断点恢复

建议每个 Run 目录：

```text
runs/style_spl_/<dataset>/<target_domain>/<seed>/<run_id>/
├── config.json                      # 全部真实运行参数 + git commit + style bank metadata
├── style_bank.pt                    # 不含标签的固定源域 / 合并源域 / 目标域统计
├── train_metrics.csv                # step 对齐的训练日志
├── eval_metrics.csv                 # 每次 eval 的结果
├── init_diagnostics.json            # 校准前/后 RMS、每域统计和 shape
├── checkpoints/
│   ├── last.pth                      # PromptGenerator / Projector / Expansion + 关键状态
│   └── best.pth                      # 可选；只在既定允许的验证集上评选
└── plots/
```

- 保存 `style_projector.state_dict()`、`ctx_cls`、`expansion`、Optimizer / Scheduler / Step / Seed / RNG 状态（用于真正断点续训）。保持原 SPL 的 Target Text Cache 行为可恢复；如果只是末轮推理，也可保存最终 Target Text Features。
- 原仓库存在基于 `LossValley` 更新 `target_features` 缓存的机制；必须保证不会在缓存全零、未初始化时静默报告 Final Accuracy。必要时增加**不改变训练**的即时 Feature 评估作为审计对照。
- 保存旧 Prompt 参数兼容策略：明确 B0 checkpoint 不含新 Style Projector；S1 需初始化新 Projector，不能直接用 `strict=True` 载入旧 Domain 参数。兼容处理**不得改写已复现基线权重**。
- 第一次实验优先使用现有成熟的训练脚本配置，关闭 OT（`--OT_clustering 0`），保留 `--training_mode multi-source --enhanced_pseudo_label 1 --backbone RN50`，其他 CLI 参数从 B0 实际记录继承，而非从 README 猜测。

## 10. 验收清单与测试（Codex 应先自测再跑完整训练）

### A. 预处理和 Bank

- [ ] 使用 RN50 同一权重、固定 preprocessing、eval + no_grad；四层特征 shape 匹配 Stage 1–4。
- [ ] Mean / Std 同一图像逐通道空间计算，与参考简单实现对齐（数值 tolerance 依 FP32 误差设置）。
- [ ] `std` 非负、有限；样本个数恰好覆盖全部允许的训练图像，无重复/遗漏。
- [ ] Pooled-source Bank 等于各 Source **按样本数**合并的结果；Target Bank 完全不读取真标签。
- [ ] Bank Cache 再加载后数值不变、Domain 顺序不变；数据划分/变换发生改变时 Cache 自动失效。

### B. Prompt / 梯度

- [ ] Style Bank 固定且 `requires_grad=False`；`ctx_cls`、四层 `stage_projectors`、`expansion` 可训练。
- [ ] 四层输入 `[2C]` → Token `[512]`，`stack` `[4,512]`，`A@T` `[16,512]`。
- [ ] One-hot 初始化后 `A@T` 顺序与预期一致，**不是零输出**；校准后各 Domain Prompt 输出有限、整体 RMS≈0.02。
- [ ] 所有 Domain 共享**同一 Projector**；不同 Domain 使用不同的 Bank 数据；`forward()` 和 `forward_source(i)` 均产生原 Shape 的 `[K, context_length,512]`。
- [ ] 从 Source CE 和 Target Soft CE 做 Backward，四层 Projectors、Expansion、Class Prompt 均有有限梯度；冻结 CLIP 权重无梯度。
- [ ] 冻结 Vision Encoder 的最后视觉 Embedding、Visual Centroid / Source Distance 数学逻辑与 B0 不变；Text Encoder 接口不变。

### C. Baseline 对齐与训练

- [ ] `style_spl_enabled=False` 时，与 B0 在同一 seed / batch 下输出与 loss 数值一致（或误差在既有浮点 tolerance 内）。
- [ ] `style_spl_enabled=True` 时一个完整 Forward/Backward/Optimizer Step 正常通过，无 NaN / Inf。
- [ ] 以单目标域做 20–50 Step Smoke Test：记录所有 P0、P1 指标；检查 RMS 曲线和梯度。
- [ ] 做单目标域既定 Step 数训练，无 silent target-cache failure，无 OOM（包括预处理）；然后四目标域完整实验。
- [ ] B0 vs S1 相同评估协议、相同数据和训练预算；S1 若准确率下降也如实记录，不根据 Target Test Labels 调整超参数。

## 11. Codex 实施顺序（优先级）

1. **建立实验分支**：记录 B0 代码版本、实际运行参数和已经修复的初始化逻辑；确认关闭 OT 后 B0 结果。
2. **实现并单测 Style Extractor**：从 `clip_custom/model.py` 的 RN50 四层取特征；核查 Mean/Std 值与 shape，保持普通 Encoder 不变。
3. **实现离线 Style Bank**：每个 Target Domain 生成 Source / Pooled / Target 三类 Entry；校验样本数、顺序和缓存元数据。
4. **实现 Style Projector 和一次性 Calibration**：Full Linear(2C→512) ×4、可学习循环 One-hot `[16,4]` Expansion；打印初始化前/后统计。
5. **最小修改 `PromptGenerator`**：直接替换三个 Domain Prompt 入口，保留 Class Prompt 和其余源域/目标域计算。
6. **增加训练日志 + 绘图脚本**：先保证最关键 P0/P1，后加 P2。修改日志不应改变原 Loss / 评估。
7. **单域 Smoke Test 与数值对齐** → 单域正式实验 → Office-Home 四个目标域 → B0/S1 结果表与曲线。

## 12. 第一版明确不做的事情

- 不从当前输入图像直接生成 Image-specific Style Prompt；**不**在每个 Step 更新固定 Style Bank。
- 不把 Source Text Feature Bank 改成类别风格统计，也不把 Style Bank 做成 `[N,K, ...]`。
- 不改变 Source Visual Centroid、L2 Distance、Softmax Source Weights、Teacher 结构、Target Soft CE。
- 不引入 Content Projector、Class Verification、Style Routing、对抗训练、新 OT 或额外正则化。
- 不首轮使用 LoRA-like Low-rank、Learnable Scale、`bias=-10` 或简单 Clamp Mean/Std。
- 不在没有独立证据时宣称 Style 与 Content 完全解耦； 仅测试**域级 Style Statistics 驱动 Prompt 的效用**。

---

## 参考及代码核查来源

- [CRPL 官方代码：`model.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/model.py)：`PromptGenerator` Prompt Shape、`forward()` / `forward_source()`。
- [CRPL 官方代码：`main.py`](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/main.py)：Source CE、Running Visual Centroids、Teacher / Soft CE、Optimizer、Final Evaluation。
- [CRPL 自定义 CLIP RN50](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/main/clip_custom/model.py)：`ModifiedResNet` 的 `layer1..layer4` 和 `attnpool`。
- [AD-CLIP 官方 RN50 实现](https://github.com/mainaksingha01/AD-CLIP/blob/master/trainers/adclip_rn50.py)：`AdaIN.mu/sigma` 与 `domain_projector`；本方案**只参考 Std 算法，不照搬其 Mean/Std Pooling**。
- [STYLIP（WACV 2024）正式 PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Bose_STYLIP_Multi-Scale_Style-Conditioned_Prompt_Learning_for_CLIP-Based_Domain_Generalization_WACV_2024_paper.pdf)：多尺度 Style Statistics / Style-conditioned Prompts 的研究动机。**本项目的 4→16 Token Expansion 以及 Domain-level Bank 注入 CRPL 是自拟工程方案，不是 STYLIP 原实现。**
