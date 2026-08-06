# GPU carrier propagation protocol v2

## Frozen predecessor

Protocol v1 remains an immutable, reproducible failure.  Its exact repository
commit is `17eb0cab4e0a0dc9ec4a6abce0692c13a18695fb`, and the repository config SHA256
is `a567f80248197c64f6de3d0138dd5fde1b9da7678c1aee4ad7d85955a88b67a2`.
The preserved Drive run is
`gpu_carrier_propagation/20260806T163104Z_17eb0cab`; its package manifest SHA256
entries include `metrics.json=221bce5c1e3aedc26db0701a7f6667ff90256d03ccc7a6ae44416aff94e40b78`
and `gate_decision.json=b5e3bacb0546144f533f955e490c4b2a9761aaabd2b273656caa908368a0ca9a`.
The run and archive are at:

- https://drive.google.com/drive/folders/18Ld8YLG2jIWLTV-3FCMZ2KFbSq9qvykW
- https://drive.google.com/file/d/1YcRjOgJAwBBqMtYg8shWXWzv0oO4puYR/view

The independent decision is `GATE_FAIL`: 5/16 actual BF16 residuals violated
`0.03 +/- 0.00005`, and paired saved-MP4 AISB residual was
`0.39369622983548647 > 0.25`.  The method conclusion is not determined.  v1
must not be overwritten, rerun under the same protocol id, or reinterpreted as
a pass.

## v2 scope

v2 is an implementation-correction protocol, not a new carrier search.  It
keeps the exact v1 model revision, prompt, seed, generation parameters, tensor
shape, thirteen temporal points, public 49-to-13 grouping, x/y cosine basis,
both CFG branches, and all scientific thresholds.  It changes only:

1. deterministic quantization-aware scalar gain correction, measured on the
   actual BF16/FP16 residual that is added to the transformer return; and
2. a paired RGB-before-MP4 diagnostic using the same public readout, so loss can
   be localized between latent, VAE/RGB, and MP4 stages.

Scheduler mutation and in-place tensor mutation remain forbidden.

## Frozen decisions

- Every one of the 16 actual injected residuals must have relative RMS
Before model loading, the formal CLI runs a fail-closed 16-case CUDA BF16
preflight over the exact `modified = sample + candidate` and
`effective_delta = modified - sample` path.  Its raw records are packaged;
failure forbids model loading and generation.
  `0.03 +/- 0.00005`; averages and pre-cast values cannot satisfy this check.
- Residual two-dimensional reconstruction MSE remains at most `0.0001`.
- AISB residual remains at most `0.25`; centered condition number remains at
  most `10.0`.
- Paired RGB and paired saved-MP4 observations are diagnostic only.  They may
  localize signal loss but can never satisfy the blind relation gate.
- The blind observation accepts only the single saved watermarked MP4 and the
  public frozen frame groups.  It receives no clean video, key, message,
  source index, alignment, latent, or injection record.
- Overall v2 `GATE_PASS` requires both execution integrity and the blind
  single-MP4 AISB/condition checks.  A pass would establish only a public
  relation-propagation bridge, not calibration, held-out, owner/wrong-key,
  temporal-edit, or method PASS.

GPU execution is forbidden until an independent auditor signs v2 Gate 0
`GATE_PASS` for the frozen config, implementation, tests, and thin notebook.
