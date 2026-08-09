# RC0-DIAG-FAST

## Boundary

RC0-DIAG-FAST is a user-run GPU diagnostic. Every package is marked
`DIAGNOSTIC_ONLY`. The notebook constructs a self-identifying bootstrap only
after checking out an exact clean commit.

It answers only two questions, each with the exact values `FEASIBLE`,
`NOT_FEASIBLE`, or `INSUFFICIENT_TO_DECIDE`:

1. Step 1: does the current carrier survive into saved MP4 pixels?
2. Step 2: does a frozen paired relation readout remain observable?

## Frozen execution

The diagnostic executes exactly eight attempts in this order:

1. `orbital_glass`: `OFF_R1`, `OFF_R2`, `A`, `B`.
2. `articulated_paper`: `OFF_R1`, `OFF_R2`, `A`, `B`.

Within each group one actual initial latent is created once and four
byte-identical, storage-independent clones are made before any condition runs.
The OFF attempts have no carrier hook. A and B use the frozen schedules at
`transformer.blocks[29].attn1` output residual with target relative RMS 0.03.
The Wan model revision, prompts, seeds, scheduler, inference parameters, saved
H264/yuv420p geometry, and all non-carrier settings remain frozen by the RC0 V2
config and quartet plan. A started attempt consumes its budget. There is no
retry, replacement, extra attempt, result-dependent tuning, or post-result
metric selection.

## Diagnostic observations

The exact machine-readable formulas live in `configs/rc0_diag_fast.json`.

- Raw saved-MP4 effect uses independently decoded RGB24 only. OFF noise is half
  the OFF_R1/OFF_R2 normalized RMS. A and B are each compared with the
  elementwise OFF mean. Every group-condition cell must satisfy the frozen
  absolute and relative floors.
- Paired public 30D uses the frozen extractor recomputed from each saved MP4.
  It evaluates `F(condition) - mean(F(OFF_R1), F(OFF_R2))` against both frozen
  templates at starts 0 through 7 with the existing all-cell rule.
- VAE re-encode relation is frozen as a prospective paired latent diagnostic:
  deterministic same-revision VAE re-encode of decoded RGB24, OFF-mean latent
  subtraction, two fixed channel-half RMS coordinates, and the same A/B
  template/window budget. It is `available=false` because this delivery has no
  reviewed saved-MP4 VAE adapter/extractor identity. This absence does not block
  raw or 30D diagnosis; a branch that requires VAE evidence is
  `DIAGNOSTIC_INSUFFICIENT`.

## Decisions

- Raw effect failure: Step 1 `NOT_FEASIBLE`, Step 2
  `INSUFFICIENT_TO_DECIDE`, route `CURRENT_CARRIER_NOT_FEASIBLE`.
- Raw and paired-30D relation pass: both steps `FEASIBLE`, route
  `KEEP_CARRIER_BUILD_BLIND_READOUT`.
- Raw passes, 30D fails, and a future frozen VAE branch passes:
  Step 1 `FEASIBLE`, Step 2 `FEASIBLE`, route
  `KEEP_CARRIER_SWITCH_TO_VAE_READOUT`.
- Raw passes while both frozen relation readouts fail:
  Step 1 `FEASIBLE`, Step 2 `NOT_FEASIBLE`, route
  `REDESIGN_RELATION_CARRIER`.
- Identity/integrity/noise insufficiency, or an unavailable branch needed to
  distinguish decisions: affected steps `INSUFFICIENT_TO_DECIDE`, route
  `DIAGNOSTIC_INSUFFICIENT`. In the current delivery, a 30D failure has Step 1
  `FEASIBLE` and Step 2 `INSUFFICIENT_TO_DECIDE`; the minimum missing evidence
  is the frozen saved-MP4 VAE re-encode adapter and extractor identity.

Only the user runs the prepared GPU notebook. This delivery itself does not run
GPU, Colab, Drive, G0, or RC1.
