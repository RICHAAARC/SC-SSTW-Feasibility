## RC1 saved-MP4 relation validation preparation protocol

RC1 is an implementation and synthetic-preflight package. Its G0 construction
baseline is commit `3d7913f01f0315094a223a930bd71239f5ea56a5`, but that
commit has not received an independent `IMPLEMENTATION_GATE_PASS`. Nothing in
this protocol represents an audited G0 pass, a formal run, or stage admission.

### Frozen A/B discriminator

Schedule A is never copied into RC1. It is read from the existing
`sc_sstw_feasibility.learned_observation_l1_v2.TEMPORAL_POINTS` symbol, whose
first six values define the existing `burst_alpha`. Schedule B is derived only
by swapping indices 4 and 5. Thus both schedules have 13 points, share the
first three affine anchors and points 6–12, and differ only by exchanging
`(0.75,0.20)` and `(0.20,0.80)` at indices 4 and 5.

The analytic cross-template residual is frozen at
`0.761311945935564 ± 1e-9` in both directions. Same-template residual is zero.
This is only a deterministic protocol/synthetic preflight; it is not saved-MP4
or method evidence.

### Prerequisite and identity boundary

Before any generation or execution-package access, the RC1 runner must consume
an independent authorization manifest and self-record actual HEAD, tree, dirty
state, and raw SHA-256 for the RC1 config, protocol, plan, runner, library,
GPU-generation helper, notebook, and tests. A formal authorization manifest is
not part of this preparation task and must not be embedded in the source
commit.

Generation also requires a frozen L1-v2 selected-candidate package. The runner
verifies its checksum file, audit schema/protocol/status, source/config/input
identities, clean source state, selected candidate, frozen frontend definition,
readout bytes, and thresholds. A missing qualifying package yields
`PREREQUISITE_NOT_MET` before generation/input access. A present but damaged or
contradictory package yields `INVALID_EXPERIMENT`. RC1 never guesses A1/A2,
retrains on fresh data, recalibrates thresholds, or selects a candidate after
fresh access.

### Matched triplets

Two content-heterogeneous groups are frozen in
`plans/rc1_matched_triplets.json`: multi-object counter-rotation and articulated
surface deformation. Every group contains exactly OFF, A, and B. Prompt, seed,
the actual initial latent tensor and its SHA-256, model revision, scheduler,
sampler, steps, guidance, geometry, frame count, FPS, dtype, negative prompt,
encoder, codec/container, save parameters, and execution path are identical
within a group. Only carrier enablement and A/B schedule may differ.

Missing, duplicate, relabelled, retried-with-changed-parameters, or latent-
mismatched conditions are invalid. Video, logs, command, environment, config,
and integrity records are independently hashed. This preparation task creates
no formal video.

### Blind evaluator

The frozen selected frontend/readout maps each single saved MP4 to one 13×2
observation. Every video is scanned with both A and B templates at starts 0–7
under exactly the frozen prerequisite thresholds and equal budget. The six
condition/template combinations are A×A, A×B, B×B, B×A, OFF×A, and OFF×B.

An A video passes only if A/start0 passes and all A/start1–7 plus every B
window reject. B is symmetric. OFF must reject both templates at every start.
Every case in both triplet groups must pass individually; averaging, pooled
scores, and majority vote are forbidden.

### States and conclusion ceiling

The only states are `RC1_VALID_PASS`, `RC1_VALID_FAIL`,
`INVALID_EXPERIMENT`, and `PREREQUISITE_NOT_MET`. Every state explicitly keeps
`formal_result=false` and `stage_progression_allowed=false`.

The maximum wording for `RC1_VALID_PASS` is exactly:

> causal saved-MP4 public relation mechanism screen passed under this frozen RC1 protocol

It does not state method validation or open GPU, L2, robustness, private-key,
fixed-FPR, or paper claims. Local tests and notebook dry-runs remain
implementation evidence only.
