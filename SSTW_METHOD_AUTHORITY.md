# 仿射不变捕获与自校准分离的状态空间同步水印

> 文档状态：**FINAL FROZEN MINIMAL AUTHORITY**  
> 方法简称：**SSTW**  
> 方法形态：**生成式、零比特、单视频盲检测视频水印**  
> 当前方法结论：`TARGET_METHOD_FEASIBILITY = INSUFFICIENT_TO_DECIDE`  
> 当前推进位置：`S1 / relation_injector construction recovery`
> 权威性：本文件是 SSTW 方法身份、实现边界、验证路线与证据解释的唯一权威定义。任何代码、配置、历史实验、笔记本或讨论若与本文件冲突，只能作为历史实现或失败诊断，不得改变 SSTW 方法身份。

## 项目推进第一性原则

项目推进的唯一目的是验证冻结 SSTW 机制链：S0 construction、S1 真实 DiT relation、S2 saved-MP4 二维观察、S3 AISB/校准/Viterbi，以及 S4 极小样本盲检。不要求且禁止独立审计、各种 Gate、manifest/result gate、provenance/receipt/authorization、环境身份冻结、论文固定 FPR、大样本攻击、复现实验治理、claim ceiling，以及为未来正式实验建设基础设施。只允许为避免代码或接线错误被误判为科学结果而进行最小正确性检查。model revision、真实 relation interface、B1/B2、lambda、Flow support 和结果前冻结 construction 属于被测方法定义并必须保留；torch/CUDA/GPU 名称只作诊断记录，不作身份门。所有证据均为 `DIAGNOSTIC_ONLY`。

## 当前唯一进度快照（2026-08-10）

本节覆盖早期“尚未验证真实 relation interface”的旧描述。后续任务必须先读取本节，不得把历史代理结果、工程故障或某个 construction 的失败解释为整个方法已经成功或失败。

### 已完成的真实 S1 诊断

运行 `0d3223669982a6c3` 在真实 Wan pre-softmax Q/K relation 接口上完成了 exact20 transformer calls，得到：

```text
S1_NO_GO_THIS_CONSTRUCTION
```

该结论精确绑定当前 construction：

```text
block14
+ radius-1 horizontal/vertical single Patch-pairs
+ two local antisymmetric axes B1/B2
+ lambda = 1
+ Flow scheduler index 4 / timestep 749
```

已确认的正向事实：

- 真实 Wan relation interface 与 native sparse SDPA 接线成立；
- 局部反对称 logit 写入确实产生双轴 odd response；
- relation Jacobian 两个对角为正；
- cross leakage 约 `0.004--0.006`，axis cosine 很低，两个轴未在 relation 层塌缩；
- exact20 已完整进入真实 transformer，结果不是 OOM、device、shape、序列化或其他工程故障。

当前 construction 的决定性失败：

- relation odd/ULP 仅约 `1.20--5.89`，未达到 `8`；
- block odd/ULP 仅约 `0.0045--0.032`；
- velocity odd/ULP 仅约 `0.0666--0.491`；
- even/odd 约 `1.54--1.66`，未满足 `<0.25`；
- normalized temporal variation 约 `1.38--1.47`，未满足 `<0.10`；
- guidance velocity global relative RMS 约 `2.42%--2.46%`，超过 `1%` construction 质量预算。

因此失败位于“relation 信号向 block/velocity 的传播、common-mode、时序稳定性和质量预算无法同时闭合”，而不是 relation interface 完全不可写。

### 已完成的单-pair dictionary 诊断

随后运行 `79f92e73ef3a6223`，结果包 SHA-256 为 `fbedc7cccdb31889621dfc6e228888062a1d22978601c273d753674777bab229`。该运行在同一 block14、`lambda=1`、Flow step4 与真实 OFF Q/K 上完成 exact10 calls，解析 screen 水平/垂直半径 `1..8` 的 16 个单-pair 候选，得到：

```text
PAIR_DICTIONARY_NO_GO
```

该结果同样不是 instrumentation failure：L4、真实 Q/K、调用计数、source identity、有限数值、canonical stats 与结果包均完整。具体诊断为：

- 多数候选的 odd/ULP 已超过 `8`，说明改变 pair 半径可以提高 relation-level separation；
- 少数候选的 even/odd 也满足 `<0.25`；
- 但 16 个候选在 cond/uncond 的 normalized temporal variation 全部约为 `0.75--1.30`，没有候选接近 `<0.10`；
- 因此没有任何单轴候选 eligible，也不存在可进入双轴组合判定的 pair。

该结果排除的是 `block14 + lambda1 + Flow step4` 下的有限 radius1--8 单-pair字典。它进一步表明当前主要矛盾是时序稳定性，而不是单纯换一个更远的 Patch pair 即可解决。下一步不得扩大 radius 或重复单-pair扫描。

### propagation screen 首次执行的工程状态与最终结论

运行 `7f9f1484e9c30627` 在进入 Stage2 three-step changed-latent 路径时触发 `UniPCMultistepScheduler.this_order` 断言。根因是多个 condition 复用了同一个已经推进过内部历史的 scheduler，而不是每个 condition 从同一 step3/step4 scheduler state 独立分叉。

该运行没有生成 `audit.json`，不得解释为 `PROPAGATION_CONSTRUCTION_READY` 或 `PROPAGATION_CONSTRUCTION_NO_GO`，也不得改变任何方法判据。状态固定为：

```text
PROPAGATION_SCREEN_EXECUTION = ENGINEERING_SCHEDULER_FORK_INVALID
SCIENTIFIC_RESULT = NONE
NEXT_ACTION = FIX_SCHEDULER_STATE_FORK_AND_RERUN_SAME_FROZEN_SCREEN
```

修复只允许保存前缀对应的 scheduler state，并为每个 OFF/axis/sign/branch condition 独立恢复；候选、basis、CFG公式、Flow support、阈值、调用预算和排序必须零变化。

修复版运行 `ff43e8be9475cbdb` 已在 NVIDIA L4 上完整完成 `170` 次 transformer calls，正常返回：

```text
PROPAGATION_CONSTRUCTION_NO_GO
selected_construction = null
```

结果包为 `sstw-s1-propagation-ff43e8be9475cbdb.zip`，SHA-256 为 `42a313580214f397315734906c9c70662b87c09cdb37cf4ce4d0f1d193af228c`。这是修复后的有效有限 construction-screen 负结果，不是 scheduler、OOM、device、shape、序列化或调用预算失败。本机 Drive 挂载当前不可读，逐 block/basis/support 数值待结果包恢复可读时只读补录；不得据此重跑、扩大候选或改变 screen 定义。

依照预先冻结停止规则，有限 blocks 7/14/22、two-pair zero-sum basis、single/three-step support 和无参数 CFG weighting 中不存在可进入 exact20 的 construction。因此：

```text
UNTRAINED_WAN_PATCH_RELATION_CARRIER = NOT_FEASIBLE
S1_PROPAGATION_SCREEN = PROPAGATION_CONSTRUCTION_NO_GO
S2_S3_S4 = STOPPED_FOR_THIS_CARRIER_ROUTE
```

该结论否定的是当前**无训练 Wan Patch-relation carrier**路线；它不是对二维状态、AISB、public-only calibration、Viterbi 或未来另行定义的生成模型/训练式 carrier 路线的否定。禁止再扫描 lambda、半径、更多 Patch pair、更多 block 或更多 Flow support 以延续本路线。

### lambda-only 结论

不得把下一步设为单纯提高 `lambda`。在 `lambda=1` 附近的局部线性外推中：

- velocity odd 达到现有 separation 门槛约需放大 `16--120` 倍；
- guidance 质量预算反而要求 `lambda <= 0.406--0.413`；
- normalized even/odd 与 temporal variation 也不会因统一标量放大而得到结构性修复。

这不是对所有 `lambda` 的数学不可能性证明，但信号下界与质量上界已经不相交，足以排除 `lambda-only scan`。`lambda` 后续只能作为新 construction 中的从属能量参数，不能作为主要搜索轴。

### 六模块真实链状态

| 模块 | 当前状态 | 结论边界 |
| --- | --- | --- |
| `state_generator` | `IMPLEMENTED_PARTIALLY / NOT_CHAIN_VALIDATED` | 已有 keyed 二维状态公式；完整 trajectory、public burst interleave 与真实生成时序写入尚未验证。 |
| `relation_injector` | `REAL_VALIDATION_COMPLETED / SINGLE_PAIR_FAMILY_NO_GO_AT_CURRENT_BLOCK_FLOW` | 真实接口与双轴写入成立；block14/lambda1/step4 的 radius1 exact20 与 radius1--8 单-pair字典均失败，模块家族尚未被否定。 |
| `relation_observer_2d` | `NOT_VALIDATED` | 尚无真实 saved-MP4 固定二维 observer 的 rank/condition/survival 结论。 |
| `aisb_acquisition` | `NOT_VALIDATED_ON_TARGET_OBSERVATION` | 历史 pixel/synthetic 成功不是目标链证据。 |
| `affine_equalizer_2d` | `NOT_VALIDATED_ON_TARGET_OBSERVATION` | 尚未在真实 observer 与冻结 public ambiguity set 上验证。 |
| `viterbi_detector` | `NOT_VALIDATED_END_TO_END` | 尚未接入真实 observation、wrong-key/null 与最终单视频零比特判断。 |

冻结总状态：

```text
TARGET_METHOD_FEASIBILITY = INSUFFICIENT_TO_DECIDE
FIRST_FAILED_STAGE = S1_REAL_DIT_RELATION_PRIMITIVE
ACTIVE_MODULE = relation_injector
CURRENT_RESULT = PAIR_DICTIONARY_NO_GO
S2_S3_S4 = HOLD_UNTIL_S1_PASSES
```

---

## 1. 唯一研究问题与正式检测边界

本项目只回答一个问题：

> 能否在冻结的 Flow-Matching 视频 DiT 推理过程中，通过局部反对称 Patch-relation 写入密钥二维状态轨迹，并仅从处理后的单个 MP4 与系统密钥完成固定低 FPR 的零比特水印检测？

正式检测接口固定为：

```text
detect(processed_video, system_key) -> present | absent, confidence
```

检测器不得读取或要求：

- 原始未处理视频；
- matched OFF 视频；
- prompt、seed、初始 latent；
- DiT 内部 activation；
- 注入记录；
- 真实 source index；
- 人工 A/B 标签；
- payload、消息标签或用户 ID。

SSTW 是零比特检测器，不恢复任意消息。

---

## 2. 不可变方法身份

SSTW 的方法身份由以下四个不可删除机制共同定义：

1. **二维密钥状态轨迹**：水印不是逐帧独立比特，而是具有密钥条件状态动力学的二维连续轨迹；
2. **DiT 局部反对称 relation carrier**：水印在冻结 Flow-Matching 视频 DiT 的真实 relation 计算路径中生成时写入；
3. **仿射不变捕获与自校准严格分离**：先由公开 AISB 在未知 affine channel 下完成捕获并冻结有限公共歧义集，再进行候选内 public-only affine calibration；
4. **状态约束同步**：正式同步器必须利用密钥状态转移约束，而非仅依赖普通 edit-distance、DTW 或逐点相似度。

删除或替换其中任意一项，均不再属于本文件定义的 SSTW。

---

## 3. 最小充分端到端机制

唯一允许的正式主链冻结为：

```text
system key K
-> 2D keyed state trajectory
-> interleaved single-template public AISB pilots
-> fixed 2D local antisymmetric DiT Patch-relation bases B1/B2
-> fixed Flow-time relation injection
-> normal Transformer / scheduler / VAE / MP4 path
-> fixed key-independent 2D relation observation
-> public affine-invariant AISB capture
-> frozen bounded public ambiguity set
-> candidate-wise public 2D affine calibration and equalization
-> same-budget keyed state-constrained Viterbi synchronization
-> frozen null normalization and fixed-FPR threshold
-> present | absent, confidence
```

形式化地：

\[
K\rightarrow u_n\rightarrow \Delta R_{\tau,n}\rightarrow q_i\rightarrow \widehat u_i\rightarrow \operatorname{Viterbi}\rightarrow S_K.
\]

正式版本不再同时维护多种 carrier、observer、AISB family、同步器或聚合器。

---

## 4. 二维密钥状态轨迹

视频被划分为状态窗口 \(W_n\)。为消除检测接口中的 `nonce` 不闭合问题，冻结缩减版不使用 per-video nonce；仅使用域分离 PRF：

\[
s_n=\operatorname{Sign}(\operatorname{PRF}(K,\texttt{"SSTW-state"},n)),
\]

\[
\phi_{n+1}=\phi_n+\omega_0\Delta t_n+\delta_\phi s_n,
\]

\[
u_n=
\begin{bmatrix}
\cos\phi_n\\
\sin\phi_n
\end{bmatrix}\in\mathbb R^2.
\]

其中 \(\omega_0\)、\(\delta_\phi\)、窗口长度及初始相位规则均在 construction 阶段冻结。

二维是本方法的最小状态维度。正式方法不得退化为全局标量、单符号或逐帧独立 bit carrier。

---

## 5. 最小公开 AISB

### 5.1 唯一作用

AISB 只负责：

1. 在不知道密钥的情况下恢复公共时间对应候选；
2. 在未知二维 affine channel 下完成捕获；
3. 输出有限公共歧义集。

AISB 不负责 payload，不读取 owner key，不在捕获阶段估计自由 affine channel。

### 5.2 单模板冻结

正式版本只允许一个冻结的 public AISB template，不再保留多模板 ID、模板选择器或复杂协议族。

设公开 pilot 为：

\[
P_{\mathrm{AISB}}=\{p_1,p_2,p_3,p_4\},\qquad p_j\in\mathbb R^2,
\]

其中 \(p_1,p_2,p_3\) 非共线，且：

\[
p_4=\alpha p_1+\beta p_2+\gamma p_3,
\qquad
\alpha+\beta+\gamma=1.
\]

若观察满足：

\[
q_j=Ap_j+b,
\]

则：

\[
q_4=\alpha q_1+\beta q_2+\gamma q_3.
\]

因此该公开仿射重心关系可在未知 \(A,b\) 下用于 acquisition，而无需先拟合 affine channel。

AISB 只允许一个冻结的序列规则和一个冻结的 affine consistency / checksum 规则。

### 5.3 有限公共歧义集

AISB 输出：

\[
\mathcal A(Q)=\{\pi_1,\ldots,\pi_M\},\qquad M\le M_{\max}.
\]

不得强制单一路径；不得使用 owner key 从候选中选择 alignment。

---

## 6. 生成时载体：二维局部反对称 Patch-relation

### 6.1 冻结 carrier

Patch-pair dictionary 与传播敏感度 screen 仅允许作为 construction 工具，不属于运行时方法模块。正式方法只保留两个冻结 basis：

\[
B_1,B_2.
\]

每个 basis 必须作用于真实 DiT relation / attention-logit 计算接口。最小形式是在同一 query row 上对一对 Patch token 施加反对称偏置：

```text
Delta b(query, patch_a) = +delta
Delta b(query, patch_b) = -delta
```

若全部单-pair construction 不能同时满足 S1，允许在同一方法身份内将 basis 收紧为**有限稀疏、多 Patch-pair、零和、归一化的局部反对称方向**。这不是 output/latent/pixel 代理，也不得引入训练式 observer。其支持、系数、block、Flow support 与 CFG branch 配重必须在完整 S1 运行前由一个有限的、结果前冻结的 propagation-sensitivity screen 唯一选定。

两个 basis 必须线性独立、低相关、归一化，并满足 S1 的 odd-response 判据。

### 6.2 写入规则

对私有状态窗口，写入：

\[
\Delta R_{\tau,n}
=
\lambda w(\tau)
\left(
 u_{n,1}B_1+u_{n,2}B_2
\right).
\]

对 AISB 窗口，将 \(u_n\) 替换为对应公开 pilot \(p_j\)。

Flow-time schedule 不作为独立创新模块。正式版本只允许一个 construction 阶段冻结的 support：

\[
w(\tau)=
\begin{cases}
1,&\tau\in\mathcal T_{\mathrm{inject}},\\
0,&\text{otherwise}.
\end{cases}
\]

### 6.3 必须经过正常生成路径

relation perturbation 必须经正常：

```text
DiT blocks -> Flow/scheduler -> VAE decode -> one MP4 encode
```

最终形成 saved MP4。

---

## 7. 固定二维 saved-MP4 relation observer

### 7.1 正式观测接口

检测端只读取 saved MP4：

\[
Q=\operatorname{Observe}(V;P_{\mathrm{obs}}),
\qquad
Q\in\mathbb R^{T\times2}.
\]

正式缩减版固定 \(R=2\)，不以高维 observer 作为默认路线。

### 7.2 Observer admissible class

observer 必须属于以下固定、非训练式 relation-readout 类：

\[
e_i=\psi(V_{i-w:i+w}),
\]

\[
q_{i,r}
=
\operatorname{Pool}_{\mathcal P_{a,r}}(e_i)
-
\operatorname{Pool}_{\mathcal P_{b,r}}(e_i),
\qquad r\in\{1,2\}.
\]

其中：

- \(\psi\) 必须完全冻结；
- \(\psi\) 可以是预训练固定 encoder 或固定非学习变换，但不得针对水印任务训练；
- Patch / region pair、pooling rule、时间窗口及全部 observer 参数必须在 construction split 上冻结；
- 不允许 watermark-specific MLP、ridge adapter、linear probe、test-time learning 或 OFF-difference 输入。

### 7.3 目标通道模型

每个视频只要求存在视频内近似固定的二维 affine channel：

\[
q_i=A_xv_{\pi(i)}+b_x+\eta_i,
\]

其中 \(v\) 为私有状态 \(u\) 或公开 AISB pilot。

必须满足：

\[
\operatorname{rank}(A_x)=2,
\]

\[
\sigma_{\min}(A_x)>\tau_{\mathrm{rank}},
\qquad
\kappa(A_x)<\tau_{\mathrm{cond}}.
\]

正式缩减版不引入自由时变 \(A_{x,t}\) 或 test-time piecewise channel tracking。若视频内 channel drift 超过预冻结阈值，则该样本/路线在 S2 判定为观测模型不成立，而不是增加自适应自由度掩盖失败。

---

## 8. 捕获与自校准的不可交换顺序

唯一合法顺序：

```text
saved-MP4 observation
-> public AISB invariant scan
-> public sequence decoding
-> freeze bounded ambiguity set A(Q)
-> for each pi_m in A(Q):
     estimate affine channel from public pilots only
     equalize observation
     run keyed state-constrained Viterbi
-> fixed candidate aggregation
-> fixed null normalization
```

禁止：

```text
enumerate private/keyed alignment
-> fit free affine channel for each private hypothesis
-> select best private fit
```

该禁止项是永久性方法约束，不得以“更强优化”“更准确同步”或“test-time calibration”重新引入。

---

## 9. 二维 public-only affine calibration

对于每个公开候选 \(\pi_m\)，只使用该候选对应的 AISB pilot pairs：

\[
(\widehat A_m,\widehat b_m)
=
\arg\min_{A,b}
\sum_{j\in\mathcal P_m}
\|q_j-(Ap_j+b)\|_2^2
+\lambda_r\|A\|_F^2.
\]

\(\lambda_r\) 必须预先冻结；若数值条件允许，可固定为0。

随后：

\[
\widehat u_i^{(m)}
=
\widehat A_m^{-1}(q_i-\widehat b_m).
\]

仅当 \(\widehat A_m\) 满足预冻结 rank / condition Gate 时，该 candidate 才合法。

owner key、wrong keys 和 clean videos 必须使用完全相同的：

- ambiguity set；
- affine estimator；
- ridge；
-维度；
-合法性 Gate；
-路径预算；
-候选聚合规则。

不得 per-key channel fitting。

---

## 10. 唯一正式同步器：状态约束 Viterbi

### 10.1 隐状态路径

设 \(z_i\) 表示 observation window \(i\) 对应的源状态窗口索引。允许的转移集合 \(\mathcal D\) 和总 edit budget \(B_{\max}\) 在 construction 阶段冻结，用于表达：

- 正常推进；
-重复 / 插帧导致的停留；
-删除导致的有限跳跃；
-裁剪导致的起始偏移；
-非整数变速导致的局部 skip / stay 组合。

### 10.2 发射项

\[
\ell_{\mathrm{em}}(i,j;K)
=
-\frac{
\|\widehat u_i-u_j^K\|_2^2
}{\sigma_e^2}.
\]

### 10.3 状态转移一致性项

正式 SSTW 必须包含状态动力学项，而不是只有 edit penalty：

\[
\ell_{\mathrm{state}}
(i-1,i,j',j;K)
=
-\rho\,
 d_{\mathbb S^1}
\left(
\Delta\widehat\phi_i,
\Delta\phi^K_{j',j}
\right)^2.
\]

其中 \(d_{\mathbb S^1}\) 为圆周相位距离。

Viterbi 目标为：

\[
S_{K,m}^{\mathrm{raw}}
=
\max_{z_{1:T}}
\sum_i
\ell_{\mathrm{em}}(i,z_i;K)
+
\sum_{i>1}
\ell_{\mathrm{state}}(i-1,i,z_{i-1},z_i;K)
-
\operatorname{EditPenalty}(z_{1:T}).
\]

bounded-DTW / edit-distance 不再是第二套正式 synchronizer，只保留为必要 baseline。

---

## 11. 候选聚合、零比特分数与固定 FPR

对全部公共候选只允许一个冻结聚合规则，例如：

\[
S_K^{\mathrm{raw}}
=
\max_{\pi_m\in\mathcal A(Q)}
S_{K,m}^{\mathrm{raw}}.
\]

candidate search 带来的 multiple-testing inflation 必须由 clean/null development set 统一吸收，而不是测试时修正。

在与测试视频隔离的 clean/null development set 上冻结：

\[
\mu_0,\sigma_0,F_0,\theta_{\alpha}.
\]

定义：

\[
Z_K=\frac{S_K^{\mathrm{raw}}-\mu_0}{\sigma_0}.
\]

最终：

```text
present  iff  Z_K >= theta_alpha
absent   otherwise
confidence = frozen null-tail confidence
```

\(\theta_\alpha\) 对应预声明目标 FPR \(\alpha\)。单个测试视频不得自适应阈值。

---

## 12. 已成立、未决与已否定的路线证据

### 12.1 已成立或未被否定，可继承

以下结论可作为 SSTW 后续推进依据，但不得夸大其证明范围：

1. **二维状态、自校准与 keyed trajectory score 的基本数学链未被否定**：已知 public pilot correspondence 的 synthetic 条件下可完成二维 affine calibration、equalization 和 owner/wrong-key 区分；
2. **AISB-before-calibration 是必要顺序**：未知 label / clock 下 observation-only affine-LS acquisition 存在 gauge ambiguity，错误对应可被自由 affine fit 吸收；
3. **public ambiguity set 优于强制唯一 alignment**：synthetic 已支持先冻结有限公共候选，再在每个候选内公平 calibration / keyed scoring；
4. **owner 与 wrong key 必须共享同一 public search space 和同一预算**；
5. **exact / bounded temporal dynamic programming 可作为同步可行性参考**，但不能替代正式状态空间贡献验证。

上述证据只支持机制方向，不证明真实 saved-MP4 observer、真实 carrier 或最终低 FPR 性能。

### 12.2 已明确否定，不得重新成为主线

以下路线已经被方法性或实验性证据判定为不适合作为 SSTW 正式主线：

- observation-only affine-LS pilot acquisition；
- 强制唯一 public alignment；
- owner-specific alignment；
- per-key affine calibration；
- owner/wrong-key 不同搜索预算；
- final transformer model-output residual carrier；
-已测试的 block-level attention-output residual carrier；
- scheduler/output residual 代理；
- initial-noise、final-VAE latent、decoder 或 post-decode pixel/chroma 写入作为 SSTW carrier；
- matched OFF difference 作为正式 detector 输入；
-内部 activation 或真实 path label 作为正式 detector 输入；
- 30D-to-MLP、train-only ridge、linear probe 或任意 lightweight learned adapter 作为正式 observer；
- payload / PRC / user-ID recovery 任务；
- unlimited per-video Procrustes、自由 test-time carrier/readout selection 或 test-time threshold selection；
- 将历史 direct-output、attention-output、MLP、VAE、pixel 诊断重新命名为 SSTW 成功证据。

需要指出的是，某个历史 carrier/readout candidate 的失败，只否定该 candidate，不等价于否定局部反对称 DiT relation carrier 本身。

---

## 13. 最小验证路线与诊断阶段

正式路线压缩为：

\[
S0\rightarrow S1\rightarrow S2\rightarrow S3\rightarrow S4.
\]

### S0：冻结 construction 对象

必须记录并冻结：

- DiT revision 与真实 relation interface；
- \(B_1,B_2\)；
- Patch / relation positions；
- \(\lambda\) 与 \(\mathcal T_{\mathrm{inject}}\)；
-状态参数；
-唯一 AISB template 与序列规则；
- \(M_{\max}\)；
- observer admissible implementation \(P_{\mathrm{obs}}\)；
- affine estimator；
- Viterbi transition set / edit budget；
- null protocol 与 threshold development split。

未冻结不得进入对应方法诊断。

### S1：真实 DiT relation primitive

只验证：

\[
B_1/B_2
\rightarrow
\text{attention/relation response}
\rightarrow
\text{block response}
\rightarrow
\text{velocity response}.
\]

必须证明：

- 两个 basis 都存在非零 odd response；
- common-mode 足够低；
-两个 basis response 低相关；
-扰动处于预声明质量预算内。

若失败，只允许在 relation interface、Patch pair、basis selection、\(\lambda\) 或 Flow support 内定位。不得切换到 output/latent/pixel 代理。

#### 当前 S1 construction recovery 顺序

当前只允许按以下有限顺序推进；不得并行多路线探索：

1. **单-pair dictionary screen（已完成）**：block14、`lambda=1`、Flow step4、半径 `1..8` 的 exact10 结果为 `PAIR_DICTIONARY_NO_GO`；禁止重复或扩大半径扫描。
2. **小型传播敏感度 screen（当前下一 GPU 节点）**：只允许少量预冻结 block、局部邻域与单步/三步 Flow support；构造两个稀疏多-pair零和 basis，并允许固定 CFG-aware cond/uncond 配重。选择目标是 relation→block→velocity odd gain、低 common-mode、低 temporal variation、双轴低相关与质量预算同时成立。
3. `lambda` 在该 screen 中仅作固定总能量下的从属线性检查，不得重新成为主搜索轴。多 Flow-step 使用固定总能量，不得通过无限累积扩大扰动。
4. 唯一胜出 construction 才运行一次完整 S1；若该有限 screen 仍无 construction 通过，则结论为 `UNTRAINED_WAN_PATCH_RELATION_CARRIER_NOT_FEASIBLE`，停止当前无训练 Wan relation-carrier 路线，不得继续堆高 `lambda` 或用历史代理补位。

### S2：真实 saved-MP4 二维可观测性

验证：

\[
\text{relation injection}
\rightarrow
\text{saved MP4}
\rightarrow
q_i\in\mathbb R^2.
\]

必须验证：

- rank 2；
-条件数合格；
-两个状态轴均可观察；
-视频内 affine channel drift 不超过预冻结阈值；
-信号经过一次正常 MP4 编码后仍存在。

matched OFF 只允许作为因果诊断，不进入 detector。

若 S2 在 admissible fixed observer class 内失败，不得通过训练 adapter、增加 OFF 输入或读取内部 tensor 挽救。

### S3：真实观察背景中的 output-side oracle closure

在真实 saved-MP4 observation 背景上叠加受控二维状态响应，只验证：

\[
\text{AISB capture}
\rightarrow
\mathcal A(Q)
\rightarrow
\text{public calibration}
\rightarrow
\text{equalization}
\rightarrow
\text{state Viterbi}
\rightarrow
\text{zero-bit score}.
\]

该阶段必须同时比较：

1. correct key；
2. wrong keys；
3. clean/null；
4. stateless edit-distance / bounded-DTW baseline；
5. state-constrained Viterbi。

状态空间贡献定义为：

\[
\Delta_{\mathrm{SS}}
=
\mathrm{Performance}_{\mathrm{state\text{-}Viterbi}}
-
\mathrm{Performance}_{\mathrm{stateless}}.
\]

必须满足预声明的：

\[
\Delta_{\mathrm{SS}}>0.
\]

否则不能宣称“状态空间同步”贡献成立。

### S4：端到端单视频盲检与鲁棒性

对未参与 construction、observer selection 或 threshold selection 的新 prompt / seed：

```text
real relation injection
-> MP4 save
-> optional temporal attack
-> detect(MP4, K)
```

必须同时包含：

- correct key；
- wrong keys；
- clean/OFF；
-删除；
-裁剪；
-重复；
-连续删除；
-非整数变速；
-插帧。

至少报告：

- AISB acquisition coverage；
- ambiguity size；
- channel rank / condition；
- calibration held-out error；
- correct-vs-wrong key margin；
- \(\Delta_{\mathrm{SS}}\)；
- TPR@fixed FPR；
-视觉质量 / generation quality 约束。

任何失败必须回到首个失败 Gate，不得以新代理路线跨 Gate。

---

## 14. 正式方法不需要的模块

为避免工程堆叠，以下内容不属于冻结版 SSTW 首要实现：

- 多 AISB template；
- template classifier；
-多 Flow schedule ensemble；
-高维 relation bank；
- learned observer；
- learned calibration；
- neural state model；
- Kalman / RTS / BCJR / Mamba 等并行同步器；
-多个 candidate aggregator；
- payload encoder / decoder；
-消息纠错码；
-复杂用户身份系统。

正式实现只保留：

```text
1. state_generator
2. relation_injector
3. relation_observer_2d
4. aisb_acquisition
5. affine_equalizer_2d
6. viterbi_detector
```

---

## 15. 训练边界

冻结版 SSTW 本身不需要 watermark-specific training。

允许：

-使用冻结的预训练 DiT；
-使用冻结的预训练 observer encoder \(\psi\)；
-construction split 上进行有限的参数 / basis /固定 observer 选择；
-clean development set 上冻结 null model 与 threshold。

禁止：

-为水印检测训练 adapter、MLP、ridge probe 或 decoder；
-利用 test video 更新 observer、carrier、channel model 或 threshold。

---

## 16. 方法成立条件

只有当以下四项全部通过时，才能把：

```text
TARGET_METHOD_FEASIBILITY
```

从：

```text
INSUFFICIENT_TO_DECIDE
```

更新为：

```text
SUPPORTED
```

四项分别为：

\[
C_1:
\text{2D relation carrier produces genuine controllable DiT response},
\]

\[
C_2:
\text{saved MP4 preserves a usable 2D fixed-observer affine channel},
\]

\[
C_3:
\text{AISB-before-calibration closes blind public acquisition and self-calibration},
\]

\[
C_4:
\text{state-constrained synchronization gives positive gain under temporal edits at fixed FPR}.
\]

其中任何一项失败，都不得由历史禁止代理补位。

---

## 17. 当前唯一下一步

当前尚没有同一条真实链同时满足：

```text
2D local antisymmetric DiT relation generation-time writing
+ saved-MP4-only fixed 2D observation
+ affine-invariant public AISB capture
+ bounded public ambiguity
+ public-only 2D self-calibration
+ state-constrained Viterbi
+ correct-key / wrong-key / clean fixed-FPR zero-bit decision
```

真实链已完成 exact20、single-pair dictionary 与 propagation-sensitivity screen；最后一项有效返回 `PROPAGATION_CONSTRUCTION_NO_GO`。当前无训练 Wan Patch-relation carrier 路线结束，不再存在可执行的 S1 construction recovery。S2/S3/S4 不得启动；也不得继续 direct-output、attention-output residual、MLP observer、VAE carrier、pixel carrier、payload 或完整攻击评测路线。

下一动作只能是：在不继承失败 construction 的前提下，先定义一个新的方法版本或明确停止整个项目。任何新路线必须重新冻结 carrier family、模型/训练边界和最小单视频盲检问题，不能把本路线的历史 proxy 结果包装为正证据。

---

## 18. 版本控制规则

以下改变属于**同一 SSTW 内允许的 construction 变更**，但必须在对应方法诊断前冻结：

- \(B_1,B_2\) 的具体 Patch pair；
- 单-pair失败后的有限稀疏多-pair零和局部支持与系数；
- \(\lambda\)；
-固定 Flow support；
-固定 CFG cond/uncond branch 配重；
-状态窗口长度与相位参数；
-唯一 AISB template 的具体坐标；
-固定 observer encoder / pair definition；
-Viterbi 的有限 transition / edit budget；
-null threshold。

以下改变属于**方法身份变化**，不得静默修改本文件后继续称为同一冻结 SSTW：

- 2D 状态退化为 scalar / payload；
- relation carrier 改为 latent / output / pixel carrier；
- acquisition 改为 affine-fit-driven private search；
-加入 per-key calibration；
-正式 observer 改为 learned detector；
-检测器需要原视频、OFF、内部 tensor 或 prompt/seed；
-删除状态转移约束，仅保留普通 DTW；
-从零比特改为消息恢复任务。

发生上述变化时，必须建立新的方法版本，而不是继续修改 SSTW 的证据解释。

---

## 19. 最终冻结结论

本文件冻结后的 SSTW 可以概括为：

\[
\boxed{
\text{2D keyed state}
+
\text{2D antisymmetric DiT relation writing}
+
\text{2D saved-MP4 relation observation}
+
\text{minimal public AISB}
+
\text{public 2D affine self-calibration}
+
\text{state-constrained Viterbi}
+
\text{fixed-FPR zero-bit decision}
}
\]

其方法身份仍然严格属于：

> **仿射不变捕获与自校准分离的状态空间同步水印。**

当前证据已经用尽 S1 内预先冻结的有限 construction recovery；未授权进入 S2，也不足以宣称真实端到端方法已经成立。因此冻结状态更新为：

```text
TARGET_METHOD_FEASIBILITY = NOT_FEASIBLE_FOR_UNTRAINED_WAN_PATCH_RELATION_ROUTE
FIRST_FAILED_STAGE = S1_REAL_DIT_RELATION_PRIMITIVE
CURRENT_S1_RESULT = S1_NO_GO_THIS_CONSTRUCTION
CURRENT_CONSTRUCTION_SCREEN_RESULT = PROPAGATION_CONSTRUCTION_NO_GO
NEXT_DIAGNOSTIC = NONE_PENDING_NEW_METHOD_VERSION_OR_PROJECT_STOP
LAST_EXECUTION = PROPAGATION_CONSTRUCTION_NO_GO
```
