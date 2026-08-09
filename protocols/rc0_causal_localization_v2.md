# RC0 causal-localization V2 preregistration

Status: protocol design only. This document defines an oracle-paired causal
localization screen. It is not a blind detector, watermark validation, an RC1
pass, method validation, or authorization for any execution stage.

## A. Frozen question and hypothesis order

The sole primary question is whether the frozen block-29 `attn1`
output-residual carrier, at relative RMS 0.03, causes a reproducible difference
in the actually saved and independently decoded MP4 beyond the repeat floor of
two otherwise identical OFF attempts. Prompt, seed, the actual initial latent,
model, sampling, runtime, and encoding are matched within each content group.

`H_P` is the four-cell claim that both schedule A and schedule B exceed the
absolute pixel-effect floor and the matched OFF-repeat floor in each of two
content groups. `H_R` is conditional: only after all four `H_P` cells pass, the
frozen 30-dimensional public extractor's OFF-paired difference must retain the
schedule-specific A/B affine relation.

The immutable state machine is:

1. integrity and evidence sufficiency;
2. Level P for `2 groups × {A,B}` with all four cells required;
3. Level R only if all four Level P cells pass;
4. one terminal state from the priority table below.

Computing or reporting Level R after any Level P cell fails is an integrity
violation. Neither level uses A1, A2, a selected candidate, the old MLP, the
train-only ridge, a new observer family, or any existing formal feature input.

### Why observer search is deferred

The original fixed-30D-to-MLP route and train-only ridge route were previously
contradicted. The formal L1-v2 result is `FORMAL_G0_STOP_VALID`: both A1 and A2
failed training absolute gates and `selected_candidate=null`. The evidence
ledger also lacks matched carrier ON/OFF and same-content repeat controls.
Searching a third observer before closing that gap would mix the upstream
carrier-to-pixel question with downstream observation capacity and invite
overfitting to 41001–41006. Those six IDs are historical negative evidence
only and are not read or used for thresholds here. IDs 41007–41008 remain
sealed.

## B. Independent RC0 identity and matched quartets

The machine-readable identities are
`configs/rc0_causal_localization_v2.json` and
`plans/rc0_matched_quartets_v2.json`. They guard, but do not modify, the frozen
RC1 sources. Schedule A is read from
`sc_sstw_feasibility.learned_observation_l1_v2.TEMPORAL_POINTS`; schedule B is
derived only by exchanging indices 4 and 5. No second schedule-A table is
introduced.

The two groups reuse the frozen heterogeneous RC1 content grammars and prompts:

| order | group | grammar | seed | quartet order |
|---:|---|---|---:|---|
| 1 | `orbital_glass` | multi-object counter-rotation | 52001 | OFF_R1, OFF_R2, A, B |
| 2 | `articulated_paper` | articulated surface deformation | 52002 | OFF_R1, OFF_R2, A, B |

Within a group, one actual initial latent tensor is created once. Four clones
are made from those exact bytes before any condition executes. Clone hashes,
shape, and dtype must match the source latent and one another. OFF_R1 and
OFF_R2 are two real generation attempts with identical OFF semantics; they are
not duplicate files or feature-level perturbations. The only allowed quartet
difference is condition label and its corresponding frozen carrier schedule:
OFF has no carrier effect, A uses schedule A, and B uses schedule B.

The maximum and planned budget is exactly eight started attempts in the table's
order. Starting an attempt consumes it. An attempt failure stops execution;
there is no retry, replacement, additional sample, relabel, changed seed,
changed prompt, or changed parameter. A stopped partial plan is reported as
`INSUFFICIENT_EVIDENCE` if the failure is honestly recorded and identities
remain valid, otherwise `INVALID_EXPERIMENT`.

All generation and encoding settings are copied exactly from the guarded RC1
config: Wan2.1-T2V-1.3B-Diffusers revision
`0fad780a534b6463e45facd96134c9f345acfa5b`, eight steps, guidance 5.0,
320×512, 49 frames, 8 fps, bfloat16, the frozen negative prompt, and H.264
yuv420p via `export_to_video` at quality 5.0 and macro-block size 16. Encoder
fallback is forbidden.

## C. Level P: sole pixel gate

### Decode and alignment

Level P uses only each actually saved MP4, reopened after save and decoded by a
frozen FFmpeg RGB24 reference path. The package must independently verify MP4,
H.264, yuv420p, 49 decoded frames, 8 fps, 320×512 geometry, and presentation
order. The decoded array is exactly `49×320×512×3`, channel order RGB, integer
range 0–255, interpreted as FFmpeg RGB24 sRGB nonlinear code values.

There is no resize, crop, color conversion after RGB24, motion compensation,
temporal shift, interpolation, registration, or result-driven alignment.
Frames pair by the fixed decoded presentation index 0 through 48. Any mismatch
is `INVALID_EXPERIMENT`; a decoder/runtime failure with intact identity but no
complete evidence is `INSUFFICIENT_EVIDENCE`.

### Exact formula and symmetric OFF reference

Let `X_t` be frame `t` converted to float64 by division by 255. For two same-
shape frames define

`r_t(X,Y) = sqrt(mean_{height,width,RGB}((X_t - Y_t)^2))`.

Define the unique symmetric OFF reference

`O_ref = (OFF_R1 + OFF_R2) / 2`

in float64 normalized RGB. For carrier condition `C∈{A,B}`:

`e_t(C) = r_t(C, O_ref)`

`n_t = 0.5 * r_t(OFF_R1, OFF_R2)`.

The factor one-half makes `n_t` the exact distance from either OFF realization
to their midpoint. Aggregate without frame selection:

`E(C) = sqrt(mean_t(e_t(C)^2))`

`N = sqrt(mean_t(n_t^2))`.

This normalized decoded-RGB RMS is the only Level P statistic. LPIPS, SSIM,
optical flow, DCT coefficients, perceptual selection, or choosing the most
favorable OFF baseline are forbidden.

For each group and each C, all of the following inclusive checks are required:

1. OFF control sufficiency: `N ≤ 1/255 + 1e-12` normalized RGB RMS;
2. absolute carrier effect: `E(C) + 1e-12 ≥ 2/255` normalized RGB RMS;
3. repeat-floor separation: `E(C) + 1e-12 ≥ 3*N`.

The first check applies once per group and is a sufficiency gate. If it fails,
the run is `INSUFFICIENT_EVIDENCE`, not a carrier failure. Checks 2–3 define
each of the four scientific cells. Every cell passes or Level P fails; no
cross-group average, cross-condition average, majority vote, or favorable OFF
selection is permitted.

## D. Conditional Level R: fixed no-training relation projection

Level R runs only after all four Level P cells pass. Each saved MP4 is processed
by the existing frozen public extractor
`sc_sstw_feasibility.learned_observation.extract_feature_matrix`, including its
fixed 49-to-13 frame grouping and 30 analytic features. It yields a finite
`13×30` matrix `F(X)`. No new MLP, ridge, candidate, normalizer, envelope,
feature selection, threshold fit, or content-direction swap is allowed.

For group `g`, define the same symmetric OFF rule elementwise:

`F_off,g = (F(OFF_R1,g) + F(OFF_R2,g))/2`

`D_g,C = F(C,g) - F_off,g`

`Q_g = (F(OFF_R1,g) - F(OFF_R2,g))/2`.

Center each matrix over its 13 time rows, independently for each feature:
`center(X)=X-mean_time(X)`. Define `S_g,C=||center(D_g,C)||_F` and
`N_g,F=||center(Q_g)||_F`. Before relation checks, require strictly
`S_g,C > 1e-12` and inclusively `S_g,C + 1e-12 ≥ 3*N_g,F`. Failure is a Level R
failure, locating the frozen public observation as insufficient even though
Level P passed.

Normalize `Z_g,C=center(D_g,C)/S_g,C`. This removes feature magnitude but does
not fit a readout. For a six-row observation window `W` and a six-point 2D
template `T`, the first three non-collinear schedule anchors determine fixed
barycentric weights for points 3–5. If
`A_T=[T_0,T_1,T_2; ones]^T`, each later template point has unique affine
weights `w_j=A_T^{-1}[T_j;1]`. The residual is

`R(W,T) = sum_{j=3..5} ||W_j - w_j·W_0:2||_2^2 /
          (sum_{j=0..5} ||W_j-mean(W)||_2^2 + 1e-9)`.

This is an analytic projection test, not a learned cross-content readout. Its
only nuisance freedom is the affine embedding fixed by the three anchors in
the window; there are no fitted thresholds or parameters carried from one
content group to the other. Consequently the leakage-prone group1-fit/group2-
apply and reverse directions are eliminated rather than selected. The exact
same fixed formula is applied independently to both groups, so both contents
are out-of-sample with respect to parameter fitting.

Each group and condition is scanned with templates A then B at starts 0 through
7, with equal budget. For A, only A/start0 may satisfy `R≤0.25`; A/start1–7 and
all B windows must strictly satisfy `R>0.25`. B is symmetric. The relation
threshold 0.25 is the already-frozen G0 absolute `max_residual` at source commit
`51b4d4d1e5c0a52139b3f8a61748decff6d52931`, config
`configs/learned_observation_l1_v2_development.json`; it is not re-estimated
from the failed G0 inputs. Every group×condition cell must pass. There is no
averaging or majority vote.

The formula-only schedule preflight gives same-template/start0 residual zero,
both A↔B cross residuals `0.761311945935564 ± 1e-9`, and every analytic
wrong-window residual strictly above 0.25. That statement concerns schedule
geometry only and is not saved-MP4 evidence.

## E. Threshold register and falsifiability

| threshold | formula/unit | boundary | grid | sole prior source | future-video independence | degenerate/fail-closed behavior |
|---|---|---|---|---|---|---|
| pixel epsilon | `1e-12`, normalized RGB RMS | additive only | all P comparisons | float64 engineering tolerance | fixed before generation | non-finite input invalid |
| OFF repeat maximum | `N≤1/255+epsilon` | inclusive upper | each group | one decoded 8-bit code value and preregistered OFF repeat | no result calibration | exceedance insufficient |
| absolute pixel effect | `E+epsilon≥2/255` | inclusive lower | group×A/B | two decoded 8-bit code values above quantization unit | no result calibration | miss is P failure |
| relative pixel floor | `E+epsilon≥3N` | inclusive lower | group×A/B | preregistered threefold engineering separation over matched repeat | multiplier fixed before generation | miss is P failure |
| feature norm epsilon | `S>1e-12`, extractor units | strict lower | group×A/B | float64 degeneracy tolerance | fixed before generation | miss is R failure |
| relative feature floor | `S+1e-12≥3N_F` | inclusive lower | group×A/B | preregistered threefold engineering separation over paired OFF features | no result calibration | miss is R failure |
| relation denominator epsilon | `1e-9`, squared normalized-feature units | additive only | all R windows | existing frozen affine-residual engineering tolerance | copied before generation | non-finite/degenerate anchors invalid |
| relation accept/reject | own/start0 `≤0.25`; all others `>0.25` | complementary inclusive/strict | both groups, both conditions, both templates, starts 0–7 | G0 frozen absolute max_residual plus analytic A/B geometry | not derived from G0 data or future video | any ambiguous equality is accept only for own/start0, therefore cross equality fails |
| analytic cross check | `|R-0.761311945935564|≤1e-9` | inclusive | A×B and B×A formula preflight | existing synthetic-only schedule geometry | no video involved | mismatch invalidates protocol implementation |

The factors 2 and 3 are deliberately simple engineering screening margins,
not estimates of a sampling distribution or claims of statistical confidence.
With only two OFF attempts, this protocol does not estimate a variance or
p-value. The all-cell rule makes the screen falsifiable without post-result
metric or threshold selection.

CPU preflight may test formulas, schemas, analytic A/B separation, exact
boundaries, and fail-closed injection. It must label every result
`synthetic_formula_preflight_only_non_evidence`. It may not fabricate MP4
evidence or use 41001–41006. Future eight-video values may only be compared to
the frozen predicates; they may not change them.

## F. Terminal outcome matrix

Priority is first matching row:

| priority | status | exact predicate | maximum claim | permitted next question | forbidden inference/action |
|---:|---|---|---|---|---|
| 1 | `INVALID_EXPERIMENT` | identity, same-latent, condition-only difference, budget, codec, decode, schema, or state-machine violation | experiment integrity invalid; no science interpretation | repair implementation/integrity under a new gate | treating as carrier/readout failure or success |
| 2 | `INSUFFICIENT_EVIDENCE` | valid identities but incomplete plan/output, recorded attempt failure, decoder/runtime evidence unavailable, or OFF repeat exceeds 1/255 | frozen screen lacks sufficient evidence | decide whether a separately authorized exact rerun is scientifically justified | silent retry, replacement, extra samples, or science conclusion |
| 3 | `P_FAIL_CURRENT_CARRIER_STOP` | complete valid evidence and at least one of four P cells misses an effect predicate | current frozen carrier fails this saved-MP4 pixel bridge screen | whether to stop this carrier route | whole-SSTW failure, observer diagnosis, or automatic rerun |
| 4 | `P_PASS_R_FAIL_READOUT_SWITCH_REQUIRED` | all P cells pass; Level R executes; at least one R cell fails | carrier-to-pixel bridge is present under the screen; current frozen public relation readout is blocked | whether a separately designed observer question is warranted | claiming a new observer works, blind detection, or RC1 pass |
| 5 | `P_PASS_R_PASS_PAIRED_RELATION_PRESENT` | all four P and all four R cells pass | oracle-paired saved-MP4 relation bridge is present under this frozen RC0 protocol | design, but do not start, a separate blind-observer protocol | watermark/method validation, RC1 pass, robustness, FPR, attack, or paper claim |

Every status has `formal_result=false` and
`stage_progression_allowed=false`. No status automatically authorizes RC1,
fresh heldout access, a GPU rerun, added samples, key/wrong-key work, sync,
attack/FPR studies, Flow trajectory, or a paper claim.

## G. Integrity, budget, and implementation boundary

Future implementation must fail closed on any missing/duplicate/mislabelled
condition, extra condition, wrong attempt order, retry, budget drift, prompt or
seed mismatch, latent-source or clone mismatch, model/revision/scheduler/runtime
mismatch, carrier-hook mismatch, save failure, codec mismatch, path escape,
decode failure, or feature-extractor identity mismatch. A partial group never
enters Level P. Level R must be unreachable unless Level P has four passing
cells.

The config and plan are closed schemas, not extensible request objects. Future
consumers must recursively require every object to have the exact frozen keys
in the frozen order, every array to have the exact length/order, and every
scalar to have its exact JSON type and value. Unknown, missing, renamed, or
reordered fields are invalid. In particular, JSON booleans are not integers;
numeric/string/object/array/null substitutions fail even when Python equality
would otherwise compare them as equal. Threshold, candidate, schedule, retry,
claim, or extra-input fields cannot be injected.

The validator's authority is sealed at module initialization as closure-local
canonical JSON strings. The mutable construction objects and one-time factory
are then removed from the module namespace. Every validation call decodes a
fresh internal expected object and compares against it; neither caller input,
disk files, environment variables, nor writable module attributes supply the
expected truth. There is no public expected-object setter, binder, factory,
mapping constructor, or mutable `FROZEN_CONFIG`/`FROZEN_PLAN` export.

This authority boundary is deliberately limited. It covers ordinary imports,
module attribute access and rebinding, in-place mutation of caller copies,
copy/deepcopy and JSON-roundtrip copies, and repeated or interleaved validator
calls. It does not attempt to resist arbitrary code execution already inside
the same Python process, explicit closure/cell or function-internals
reflection, validator monkeypatching, debugger injection, or direct memory
modification. Resisting those would require a separately authorized process or
external trust boundary. This protocol makes no claim of cryptographic
immutability, attestation, hardware origin, or remote proof.

State classification validates its complete cell schema before evidence
sufficiency. P and R cell maps contain exactly the four frozen group/condition
keys and literal Boolean values. `r_was_executed=false` requires no R map;
`r_was_executed=true` requires a complete R map. Running R after any P-cell
failure is invalid. Only after these checks may an excessive OFF repeat floor
produce `INSUFFICIENT_EVIDENCE`.

The frozen 30D extractor identity is more than a symbol name. It is
`extract_feature_matrix` in
`src/sc_sstw_feasibility/learned_observation.py`, bound at source commit
`fe8bc36461fdf40db917a3772a30ce6969a6c3a8`, tree
`90e50f107685c768600686239c922242e660d20a`, Git blob
`6288d954a1bdaded5fd2f92ed78b463bc11a6a18`, and raw SHA-256
`9c7fd37995d49344c2200a4855eaf6a547ea27336fdfe4fc652c62ac327b9866`.
The file contains the extractor and its frame-group/DCT definitions, so it is
the minimal source closure. A future implementation must verify all of these
identities before extracting features; matching only the import path or symbol
is insufficient.

The minimal future implementation should reuse the independently reviewed
GenerationReceipt same-process control-flow boundary, same-latent records,
saved-MP4 decode/cache cross-check, and fail-closed package infrastructure.
That is an implementation requirement, not work performed by this design.
Any missing support is a later implementation gap.

This design adds only protocol, config, plan, a pure in-memory formula module,
and protocol/formula tests. It does not add or modify generation, GPU backend,
runner, notebook, manifest, Drive integration, provenance architecture, RC1,
or G0 scientific semantics. No formal input or video is accessed.
