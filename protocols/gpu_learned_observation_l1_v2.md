## Observation L1-v2 bounded development protocol

This protocol preserves the original L1 'Contradicted' result. It is an
implementation and CPU-development-candidate protocol only. A local test or a
valid development package is not a formal result and does not authorize stage
progression.

The immutable partitions are 41001–41004 for training leave-one-video-out
(LOO), 41005–41006 for development diagnosis, and 41007–41008 as forbidden
fresh held-out inputs. The runner must not probe the forbidden paths or IDs.

Exactly two candidates are admitted in one fixed order. A1 applies per-video,
per-feature median/MAD normalization with scale 1.4826, absolute MAD floor
1e-6, and clipping to [-6, 6]. Only if A1 fails may A2 run. A2 applies A1
and then the frozen length-preserving high-pass. If A2 fails, processing stops;
there is no candidate injection, third candidate, order swap, deletion variant,
or threshold override.

Every correct training LOO window must first pass all frozen absolute limits:
residual <= 0.25, global second singular value >= 0.10, fitted affine
second singular value >= 0.05, fitted affine condition number <= 10, and
public-calibration held-out MSE <= 0.02. Any failure ends that candidate
before development metrics are computed. Only four passing LOO values may
derive the envelope; every derived bound must equal or tighten its absolute
limit.

The development gate requires the correct start-0 window of both 41005 and
41006 to pass the frozen envelope and every start 1–7 window to fail it. The
only successful state is 'READY_TO_PREREGISTER_FRESH_GPU_GATE'; it means only
that this code/CPU-development candidate may later be considered for a separate
preregistration decision. It grants no run authorization. The scientific
failure state is 'STOP_DEVELOPMENT_GATE_FAILED'. Any integrity, provenance,
access, configuration, or runtime anomaly is 'INVALID_EXPERIMENT', never a
scientific pass or failure.

Before reading any feature input, the runner consumes an independent read-only
authorization manifest and records actual Git HEAD/tree/dirty state, raw
SHA-256 for the runner/config/protocol/library/test files, config SHA-256, and
all six input SHA-256 values. Unreadable Git, a dirty worktree, any identity
mismatch, any extra/forbidden input, or a caller-declared commit mismatch fails
closed. The authorization manifest is created only after the authorized source
commit and is never embedded in that commit, avoiding self-reference.

Normal packages record manifest/config/source/input identity, candidate order,
per-gate decisions, and the terminal state. Invalid packages contain a stable
reason code and integrity context but no development metrics or conclusion.
