# S1 non-method blocker premortem

Scope: minimal implementation checks that prevent wiring or numeric failures from being mistaken for the frozen S1 scientific outcome. This note changes no construction, threshold, or GO/NO_GO meaning. Evidence remains `DIAGNOSTIC_ONLY`.

## Current run disposition

Do not run delivery `4c07fac79fb608cb5127a145fade38f96f8ab323` / RUN_ID `025f2346a837106c`. A legitimate zero-signal or single-axis-collapse result can produce a non-finite derived ratio in that implementation and fail canonical JSON instead of publishing `S1_NO_GO_THIS_CONSTRUCTION`.

## Exact chain contract

| Node | Required value | Device/dtype and optional fields |
|---|---|---|
| Wan processor inputs | Q/K/V `[1,8320,12,128]`; selected additive bias `[1,1,13,8320]` | Q/K/V and bias stay on the active device in Wan dtype; manual relation logits/probabilities are FP32 diagnostics; inference enabled and grad disabled |
| Processor record | relation `[13,12,2]`; 13 selected rows; OFF replacement count 0, active count 13 | relation is detached FP32 CPU/list; bias count/pair sum/backend/finite flags and lambda-zero numeric differences are diagnostics |
| `one_call` capture | block `[13,1536]`; velocity `[13,16,2,2]` | both detached FP32 CPU; full block `[1,8320,1536]` and full velocity `[1,16,13,40,64]` are detached FP32 CPU only for frozen global RMS |
| `numeric_probe` | exact six conditions × two branches × relation/block/velocity | NumPy FP64 conversion; exact shapes above, axis 0 exactly 13, all finite before O/E/N |
| Guidance | uncond + 5 × (cond − uncond), same three shapes | derived only after cond/uncond contract passes; no implicit batch/head/time axis |
| O/E/N evaluator | every layer × cond/uncond/guidance × B1/B2 | zero denominators serialize as `null` plus denominator/status and force the affected predicate false; no NaN/Infinity and no threshold change |
| Structural/status | exact20, sparse-native/bias/replacement/inference checks | all-pass science+structure → `S1_GO`; otherwise valid finite science → `S1_NO_GO_THIS_CONSTRUCTION` |
| Package/CLI | canonical stats/audit/config/plan/command plus readable checksums | GO exit 0; NO_GO exit 3; instrumentation error exit 2 with single JSON reason on stdout |
| Notebook | reads audit for GO/NO_GO and prints runner stdout before reporting missing audit | L4 is usable; the first Drive-mount cell and real-reason display remain unchanged |

Optional processor diagnostics must never be used as substitute science signals. Missing required condition/branch/layer fields, shape drift, non-finite probe input, invalid device indexing, or autograd retention is `INSTRUMENTATION_INSUFFICIENT`.

## S1 to S2 precheck

Only `S1_GO` may hand off the construction identified by the frozen config: block 14, scheduler index 4/timestep 749, lambda 1, and the B1/B2 relation geometry. `S1_NO_GO_THIS_CONSTRUCTION` stops that construction.

S2 is not implemented here. Its future boundary is one saved MP4 as input and one key-independent, fixed, non-learning `T×2` relation observation as output. It must not consume S1 internal tensors, OFF reference, prompt, seed, condition truth, or injection records. Frame decoding (49 frames) and any 49-to-13 window aggregation must be explicit in S2 rather than inferred from tensor axes.

## Remaining GPU-only uncertainty

CPU checks establish shape, numeric, serialization, status, and package wiring only. The replacement Notebook must still be run once on CUDA+BF16 to obtain the real S1 DiT relation primitive outcome. No CPU test predicts `S1_GO` or `S1_NO_GO_THIS_CONSTRUCTION`.
