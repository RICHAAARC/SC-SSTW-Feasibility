# SSTW-v2：Flow-Guided Observer Carrier

> 文档状态：**CURRENT METHOD AUTHORITY**
> 方法名称：**仿射不变捕获与自校准分离的状态空间同步水印**
> 当前版本：`SSTW-v2`
> 产品形态：生成时嵌入、零比特、单视频盲检测视频水印
> 证据类别：`DIAGNOSTIC_ONLY`
> 当前状态：`SSTW_V2_FLOW_GUIDED_OBSERVER_NOT_FEASIBLE`

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

## 8. G0：guidance primitive（已完成）

G0 只回答：固定 observer 是否能在 Wan Flow state 上提供两个独立、方向正确、质量受限的控制方向。

### 8.1 固定生成对象

```text
model = Wan-AI/Wan2.1-T2V-1.3B-Diffusers
revision = 0fad780a534b6463e45facd96134c9f345acfa5b
prompt = locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts
negative_prompt = text, watermark, logo, camera motion, cuts, multiple subjects, flicker
seed = 1275
guidance_scale = 5.0
max_sequence_length = 226
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

### 8.3 原始预注册判据与结果

必须逐项满足：

1. 两个 axis gradient finite 且 RMS > 0；
2. gradient absolute cosine `< 0.5`；
3. 每轴 odd response (O=(q_+-q_-)/2) 在目标坐标方向为正；
4. 每轴 own odd RMS `>= 8 × max(OFF repeat observation RMS, numeric floor)`，其中 `numeric floor = 32 × float32 epsilon × max(1, OFF baseline observation RMS)`；
5. cross/own response ratio `<= 0.5`；
6. even/odd ratio `<= 0.5`；
7. 每个 active final RGB 相对 OFF RMS `<= 0.02`；
8. OFF_R1/OFF_R2 的确定性差异只作数值噪声地板记录，不以平均掩盖任一轴。

终态只允许：

```text
FLOW_GUIDANCE_PRIMITIVE_FEASIBLE
FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE
INSTRUMENTATION_INSUFFICIENT
```

`FEASIBLE` 只允许进入 G1 saved-MP4 exact8；不是完整 SSTW 成功。

唯一有效运行：

```text
run_id = 53b555d04484fa20
implementation = dbb80cdb0456dedb8d6172474b6397332450f0c4
delivery = 4dbd8208d456069dced8c6c8c010580a2bb5f357
archive_sha256 = 4bc54f566af93c6bb4e27f0ae083657ebb2aa22bb050aa3f33de93ef757cf845
status = FLOW_GUIDANCE_PRIMITIVE_NOT_FEASIBLE
transformer_calls = 36
vae_decodes = 8
```

该运行完整经过真实 Wan、两个可微 VAE VJP、六个独立 UniPC forks、终态 VAE 与指标计算；不是工程失败。原判据仅有 `G1/G2 cross_to_own <= 0.5` 两项失败，其余十三项通过：

```text
G1 own RMS = 1.513589e-3
G1 cross/own = 0.554935
G2 own RMS = 1.078376e-3
G2 cross/own = 0.710919
odd/floor = 396.78, 282.69
even/odd = 0.0561, 0.0789
final RGB relative RMS = 0.00399 .. 0.00424
gradient cosine = -0.03659
```

### 8.4 方法解释纠正

原始 `cross/own <= 0.5` 要求 raw observer 坐标近似对角。它适合筛选无需校准的控制轴，但不是本方法“二维仿射通道 + public calibration”的必要条件。实际 odd-response 传递矩阵为：

\[
M=\begin{bmatrix}
0.001512672 & -0.000766078\\
-0.000838704 & 0.001077700
\end{bmatrix},
\]

其 `det(M)=9.87693e-7`、`cond_2(M)=4.58067`，满足 calibration 模块既有的 `condition <= 10` 可逆二维通道边界。两个轴所有13个时间点目标方向均为正；以该单一 public matrix 做 equalization 后，两列均恢复对应单位轴，逐帧最大偏差分别约 `0.198/0.135`。

因此必须同时登记：

```text
RAW_AXIS_DIAGONALITY_G0 = NOT_FEASIBLE
FLOW_GUIDANCE_2D_AFFINE_SUBSPACE_ON_ONE_CONTENT = FEASIBLE
FULL_FLOW_GUIDED_WATERMARK = INSUFFICIENT_TO_DECIDE
```

这不是放宽阈值或结果后挑指标，而是删除与既定 affine-calibration 方法结构冲突的多余坐标对齐要求。原始状态、数值和失败项永久保留。新判据在任何 fresh 内容生成前冻结为：二维 response matrix finite、det 非零、`condition <= 10`、两条轨迹 effect 超过 OFF/numeric floor、固定fit residual与质量门。G0 的±轴even/odd结果继续作为局部线性证据记录，但G1 exact8没有±轴条件，不能伪造同一指标。不得用 matched OFF 或该矩阵进入最终单视频检测；它只用于生成端因果诊断。

## 9. 后续最短路线

### G1：fresh saved-MP4 exact8（已完成）

2 个异质内容，每组 `OFF_R1/OFF_R2/A/B`，同 prompt/seed/initial latent/非 carrier 参数。A/B 使用完整 13-window state trajectory与 public pilots。只验证：

- normal VAE + first MP4 encode 后二维 effect 超过 OFF floor；
- 每个内容的二维 response matrix 满秩且 `condition <= 10`；不再要求 raw 坐标近似对角；
- 基础质量可接受。

唯一闭合结果：

```text
run_id = 12307e2866c50f8d
implementation = 785129e7c184720413f00aaeb60be2bec1d2bb01
delivery = bc44f1792727a2ba72e84773cb46f914a6fcff4a
archive_sha256 = 0bad50104f7a6197a9cfd049e529e696a5102f5c19208428616367d8da5cb05a
status = FLOW_GUIDANCE_SAVED_MP4_NOT_FEASIBLE
```

执行链完整：第一内容复用已完成的四个 MP4，第二内容生成四个 MP4；总计8个可用MP4，续跑执行28次Transformer、6次VAE，无工程不足。两个内容均满足effect高于floor、response matrix满秩、`condition<=10`：lighthouse condition=`8.1210`，glass-garden condition=`1.6309`。但两者fit residual分别为`0.9761/0.6886`，均超过`0.5`；saved-MP4相对OFF质量变化分别约`0.317/0.320`与`0.0891/0.0889`，均超过`0.02`。

因此当前construction确实能推动可观测二维子空间，但不能在fresh内容上保持预定13点状态轨迹与基础质量。不得通过放宽阈值、缩小结果后强度、改终态latent或继续扫描guidance boundary来挽救本版本。

### G2：单视频 observer + AISB/calibration

只读取各自 MP4，验证正确 public alignment进入 bounded ambiguity set，并对每 candidate 做 public-only affine calibration。禁止 matched OFF 进入检测。

### G3：Viterbi zero-bit tiny closure

correct key、wrong keys、clean/OFF 共用全部 public candidates、fits 和路径预算；比较 state-Viterbi 与 stateless baseline。

## 10. 停止与 fallback

fresh G1 已因轨迹fit与质量失败，正式登记：

```text
FLOW_GUIDED_OBSERVER_CARRIER = NOT_FEASIBLE
```

下一且唯一活动版本为独立 `SSTW-noise`：测试 structured initial-noise 2D carrier + 单MP4固定VAE-inversion observation。不得并行实现trained decoder或多个observer，不得把历史final-latent/VAE-latent注入重新命名为该路线。

`SSTW-noise G0` 在看到新结果前冻结为：一个fresh prompt/seed；`OFF_R1/OFF_R2/A/B` exact4；四条件共享同一真实初始noise和全部非carrier生成参数；A/B仅在scheduler step 0之前加入相对initial-noise RMS=`0.03`的二维结构。第一坐标写入latent channel `0/1`的水平单周期反对称cosine，第二坐标写入channel `2/3`的垂直单周期反对称cosine，完整13点keyed state trajectory随latent时间轴写入。之后执行未经修改的8步Wan Flow、官方VAE decode与首次H264 MP4。

首次工程尝试`run_id=8ebdd045ca397a98`在四个Flow条件全部生成后，于首个OFF VAE decode因调用方遗漏`torch.inference_mode()`而为冻结VAE错误构建完整反向图，最终OOM；没有生成任何saved MP4、observation或科学指标，登记为`INSTRUMENTATION_INSUFFICIENT / SCIENTIFIC_RESULT_NONE`。修复只恢复终态VAE decode的纯推理语义，不改变本段任何construction或判据；该次不计作SSTW-noise科学尝试。

读取端每次仅接受单个saved MP4：固定解码49帧，调用同一冻结Wan VAE的posterior `mode()`，按官方latent mean/std反归一回normalized latent，再从相同两个反对称channel pair/cosine basis得到`13x2` observation。matched OFF只用于本次carrier诊断的噪声地板和质量比较，不进入未来blind detector接口。唯一G0同时要求A/B效应超过OFF/numeric floor、二维response满秩、condition不超过10、trajectory fit residual不超过0.5、saved-MP4相对质量变化不超过0.02；任何一项失败即只否定该structured-noise construction，不得结果后改变强度、basis、prompt或判据。

trained VAE/decoder carrier 当前不在项目范围。Dynamics-level trained velocity-field model watermark 也不是 MP4 content watermark，不作为直接 fallback。

## 11. 版本边界

以下属于 v2 construction 参数，必须在一次 GPU 运行前冻结：observer formula、guidance boundary、relative update RMS、Flow continuation、状态时钟和最小质量判据。

以下属于方法版本变化：改用 initial noise/inversion、训练 decoder、训练 observer、post-generation pixel embedding、per-key calibration、删除 AISB 或删除 state-constrained Viterbi。发生这些改变必须创建新方法版本，不能静默继承 v2 结论。

## 12. 当前状态

```text
SSTW_V1_UNTRAINED_WAN_PATCH_RELATION = NOT_FEASIBLE
SSTW_V2_FLOW_GUIDED_OBSERVER = NOT_FEASIBLE
SSTW_NOISE_STRUCTURED_INITIAL_CARRIER = INSUFFICIENT_TO_DECIDE
ACTIVE_MODULE = structured_initial_noise_embedder
CURRENT_STAGE = SSTW_NOISE_G0_SAVED_MP4
NEXT_ACTION = FREEZE_AND_RUN_ONE_STRUCTURED_NOISE_G0
S2_AISB_CALIBRATION_VITERBI = NOT_YET_ACTIVE
```
