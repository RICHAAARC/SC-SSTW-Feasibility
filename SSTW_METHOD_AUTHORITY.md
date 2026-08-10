# SSTW-v2：Flow-Guided Observer Carrier

> 文档状态：**CURRENT METHOD AUTHORITY**
> 方法名称：**仿射不变捕获与自校准分离的状态空间同步水印**
> 当前版本：`SSTW-v2`
> 产品形态：生成时嵌入、零比特、单视频盲检测视频水印
> 证据类别：`DIAGNOSTIC_ONLY`
> 当前状态：`FLOW_GUIDANCE_PRIMITIVE_PENDING`

本文件是活跃仓库的方法身份、允许实现和下一科学问题的唯一权威。历史 Q/K relation 代码与实验仅用于解释已经否定的 `SSTW-v1`，不得继续驱动新实验。

## 1. 第一性原则

项目只有一个目标：**以最短路径判断当前方法路线是否可行**。

不建设或要求：治理平面、独立审计、protocol/implementation/manifest/result Gate、authorization、provenance/receipt、环境身份冻结、论文级 fixed-FPR、大样本攻击、payload、PRC、owner 协议或未来生产设施。

只保留避免工程错误被误判为科学结果的最小检查。CUDA/BF16 是能力要求；torch/CUDA/GPU 型号和软件小版本只记录，不作阻断。

## 2. v1 已结束的路线

`SSTW-v1` 使用真实 Wan Q/K、RoPE 后、softmax 前的局部反对称 Patch-relation carrier。有效结果为：

1. relation interface 可以写入两个低相关 odd 轴；
2. `block14 + radius1 single-pair + lambda1 + Flow step4` exact20 为 `S1_NO_GO_THIS_CONSTRUCTION`；
3. block14、radius1..8 single-pair dictionary 为 `PAIR_DICTIONARY_NO_GO`；
4. 修复 UniPC condition-local scheduler forks 后，blocks 7/14/22、two-pair zero-sum basis、single/three-step Flow support、无参数 CFG weighting 的完整 170-call screen 为：

```text
run = ff43e8be9475cbdb
status = PROPAGATION_CONSTRUCTION_NO_GO
selected_construction = null
archive_sha256 = 42a313580214f397315734906c9c70662b87c09cdb37cf4ce4d0f1d193af228c
```

因此：

```text
UNTRAINED_WAN_PATCH_RELATION_CARRIER = NOT_FEASIBLE
```

禁止继续扫描 Q/K pair、block、半径、lambda、Flow support 或 CFG weighting。该结论不否定二维状态、AISB、public calibration 或 Viterbi；它只终止 v1 生成端。

## 3. v2 唯一研究问题

> 能否在同一次 Wan Flow sampling 中，用固定二维 saved-video observer 的梯度闭环推动生成轨迹，使两个 keyed state 轴在正常 VAE/MP4 输出中可观测？

目标检测接口仍为：

```text
detect(processed_video, system_key) -> present | absent, confidence
```

检测器不得读取原视频、matched OFF、prompt、seed、initial latent、内部 activation、注入记录或真值路径。

## 4. 唯一机制链

```text
system key
-> 2D keyed state trajectory + public AISB pilots
-> fixed differentiable 2D observer objective
-> gradient-guided Wan Flow sampling
-> normal remaining scheduler / VAE / first MP4 encode
-> same fixed key-independent 2D saved-MP4 observer
-> public affine-invariant AISB capture
-> frozen bounded public ambiguity set
-> candidate-wise public-only affine calibration
-> state-constrained Viterbi
-> present / absent + confidence
```

v2 不是 post-generation watermark：禁止在 VAE decode 或 MP4 保存后修改像素。observer 可以作用于 generation 中的可微 VAE RGB 以提供梯度，但写入动作必须发生在 Flow latent/sampling state 中。

## 5. 活跃六模块

| 模块 | 职责 | v2 状态 |
| --- | --- | --- |
| `state_generator` | 生成二维 keyed circular trajectory | 保留；未改数学身份。 |
| `flow_guidance_embedder` | 由固定 observer loss 对 Flow latent 求梯度并作有界更新 | 新活动模块；当前 G0。 |
| `relation_observer_2d` | generation 内可微版本与 saved-MP4 检测版本共享同一二维公式 | v2 收紧；G0/G1 验证。 |
| `aisb_acquisition` | key-independent public affine-invariant capture | 保留，G0/G1 不运行。 |
| `affine_equalizer_2d` | 对冻结 public candidates 做同规则 public-only affine fit | 保留，G0/G1 不运行。 |
| `viterbi_detector` | keyed state-constrained synchronization与零比特分数 | 保留，G0/G1 不运行。 |

`relation_injector` 及 S1 Q/K runner 是历史 v1 代码，不属于活跃六模块。

## 6. 固定二维 observer v2.0

输入 RGB 取值统一为浮点 `[−1,1]` 或等价线性范围。对输出视频的 49 帧，固定采样中心：

```text
frame_indices = [0,4,8,12,16,20,24,28,32,36,40,44,48]
```

定义两个 opponent channels：

\[
c_1=R-G,\qquad c_2=B-\frac{R+G}{2}.
\]

定义零均值、单位 RMS 的低频空间方向：

\[
h(x)=\operatorname{Norm}\{\cos(2\pi(x+1/2)/W)\},
\]

\[
v(y)=\operatorname{Norm}\{\cos(2\pi(y+1/2)/H)\}.
\]

每个中心帧输出：

\[
q_{n,1}=\operatorname{mean}_{x,y}[c_1(y,x)h(x)],
\quad
q_{n,2}=\operatorname{mean}_{x,y}[c_2(y,x)v(y)].
\]

该公式：

- 不使用 key、OFF、prompt、seed 或生成内部记录；
- generation 内以 torch 计算并保持可微；
- detector 端对解码 MP4 使用逐字相同的 frame/channel/basis/pooling 语义；
- 不训练，不进行 test-time fitting。

历史 pixel/chroma 结果只支持选择这种低频、codec-friendly observer class；不作为 v2 成功证据。

## 7. Flow guidance 写入

对状态目标 (u_n\in\mathbb R^2) 与当前可微 decoded observation (q_n(x_\tau))，定义：

\[
\mathcal L_{wm}(x_\tau,u)=-\frac1N\sum_n q_n(x_\tau)^\top u_n.
\]

计算：

\[
g_\tau=\nabla_{x_\tau}\mathcal L_{wm}.
\]

以相对 RMS 约束的 normalized update 改变 Flow state：

\[
\Delta x_\tau=-\epsilon\,\operatorname{RMS}(x_\tau)
\frac{g_\tau}{\operatorname{RMS}(g_\tau)},
\qquad x_\tau\leftarrow x_\tau+\Delta x_\tau.
\]

若 gradient 非有限或 RMS 为零，该 construction 直接为不足/不可行，禁止伪造方向。

## 8. G0：最快 guidance primitive

G0 只回答：固定 observer 是否能在 Wan Flow state 上提供两个独立、方向正确、质量受限的控制方向。

### 8.1 固定生成对象

```text
model = Wan-AI/Wan2.1-T2V-1.3B-Diffusers
revision = 0fad780a534b6463e45facd96134c9f345acfa5b
prompt = locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts
negative_prompt = text, watermark, logo, camera motion, cuts, multiple subjects, flicker
seed = 1275
guidance_scale = 5.0
inference_steps = 8
frames = 49
height = 320
width = 512
dtype = bfloat16 model / float32 observer-gradient statistics
```

### 8.2 固定 guidance construction

```text
guidance_boundary = after scheduler index 5
normal_steps_remaining = [6,7]
relative_update_rms = 0.005
conditions = OFF_R1, OFF_R2, PLUS_G1, MINUS_G1, PLUS_G2, MINUS_G2
state_targets:
  OFF_R1/OFF_R2 = (0,0)
  PLUS_G1/MINUS_G1 = (+1,0)/(-1,0)
  PLUS_G2/MINUS_G2 = (0,+1)/(0,-1)
```

所有 condition 从相同 latent 和完整 UniPC scheduler snapshot 独立 fork。OFF 不求梯度、不更新 latent；四个 active condition 使用同一个基点计算的两个 axis gradients，只改变符号。然后各自继续相同的剩余 scheduler steps。

G0 不写 MP4、不运行 AISB/calibration/Viterbi；只比较最终 VAE RGB observer 和 OFF 图像质量差异。

### 8.3 最小判据

必须逐项满足：

1. 两个 axis gradient finite 且 RMS > 0；
2. gradient absolute cosine `< 0.5`；
3. 每轴 odd response (O=(q_+-q_-)/2) 在目标坐标方向为正；
4. cross/own response ratio `<= 0.5`；
5. even/odd ratio `<= 0.5`；
6. 每个 active final RGB 相对 OFF RMS `<= 0.02`；
7. OFF_R1/OFF_R2 的确定性差异只作数值噪声地板记录，不以平均掩盖任一轴。

终态只允许：

```text
FLOW_GUIDANCE_PRIMITIVE_FEASIBLE
FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE
INSTRUMENTATION_INSUFFICIENT
```

`FEASIBLE` 只允许进入 G1 saved-MP4 exact8；不是完整 SSTW 成功。

## 9. 后续最短路线

### G1：saved-MP4 exact8

2 个异质内容，每组 `OFF_R1/OFF_R2/A/B`，同 prompt/seed/initial latent/非 carrier 参数。A/B 使用完整 13-window state trajectory与 public pilots。只验证：

- normal VAE + first MP4 encode 后二维 effect 超过 OFF floor；
- 两轴不塌缩；
- 基础质量可接受。

### G2：单视频 observer + AISB/calibration

只读取各自 MP4，验证正确 public alignment进入 bounded ambiguity set，并对每 candidate 做 public-only affine calibration。禁止 matched OFF 进入检测。

### G3：Viterbi zero-bit tiny closure

correct key、wrong keys、clean/OFF 共用全部 public candidates、fits 和路径预算；比较 state-Viterbi 与 stateless baseline。

## 10. 停止与 fallback

若 G0 失败，先确认不是 OOM、autograd断链、VAE scaling/shape 或 scheduler fork 工程错误。工程闭合后仍失败，则：

```text
FLOW_GUIDED_OBSERVER_CARRIER = NOT_FEASIBLE
```

随后才允许建立独立的新版本 `SSTW-noise`，测试 structured initial-noise 2D carrier + inversion observation。不得并行实现 noise、trained decoder 或多个 guidance observer。

trained VAE/decoder carrier 当前不在项目范围。Dynamics-level trained velocity-field model watermark 也不是 MP4 content watermark，不作为直接 fallback。

## 11. 版本边界

以下属于 v2 construction 参数，必须在一次 GPU 运行前冻结：observer formula、guidance boundary、relative update RMS、Flow continuation、状态时钟和最小质量判据。

以下属于方法版本变化：改用 initial noise/inversion、训练 decoder、训练 observer、post-generation pixel embedding、per-key calibration、删除 AISB 或删除 state-constrained Viterbi。发生这些改变必须创建新方法版本，不能静默继承 v2 结论。

## 12. 当前状态

```text
SSTW_V1_UNTRAINED_WAN_PATCH_RELATION = NOT_FEASIBLE
SSTW_V2_FLOW_GUIDED_OBSERVER = INSUFFICIENT_TO_DECIDE
ACTIVE_MODULE = flow_guidance_embedder
CURRENT_STAGE = G0_FLOW_GUIDANCE_PRIMITIVE
NEXT_ACTION = IMPLEMENT_AND_RUN_ONE_G0_GPU_DIAGNOSTIC
S2_AISB_CALIBRATION_VITERBI = NOT_YET_ACTIVE
```
