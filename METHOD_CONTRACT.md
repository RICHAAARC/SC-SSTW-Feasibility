# SC-SSTW mechanism-feasibility contract

This repository tests one existence claim only:

> Under one frozen Wan configuration, a non-scheduler multi-step trajectory
> residual can produce a two-dimensional relation observation that is read
> from one saved MP4, acquired without a candidate key, calibrated with public
> states only, and then used to rank the owner state trajectory above frozen
> wrong keys under one minimal temporal edit.

This is not a fixed-FPR, attack-suite, cross-model, large-sample, or paper
claim. Synthetic results never count as saved-video evidence.

## Frozen evidence chain

1. `public/private state schedule -> DiT model output residual` at every
   declared denoising step. Direct final velocity/model-output residual is the
   primary carrier. Internal output/value is the only challenger, guidance is
   diagnostic, and scheduler residual is a positive control only.
2. The final observation front-end accepts one saved MP4 path and returns only
   finite two-dimensional `q` values. It must not accept a key, message,
   injected states, source indices, clean counterpart, internal activation,
   alignment, or owner label.
3. Public AISB acquisition consumes only `q` and public templates. Its one
   deterministic accepted candidate set is canonically serialized and hashed
   before calibration or private scoring.
4. `A_x,b_x` are fitted only from all retained occurrences of the declared
   public calibration logical IDs. Held-out and private occurrences never enter
   fitting. Held-out error aggregates all retained occurrences of the two
   declared held-out logical IDs.
5. Owner and all predeclared wrong keys reuse identical observation,
   ambiguity-set, and calibration digests. Exact IDs, public split, prompt,
   seeds, RNG, draw order, edits, metrics, ties, and search budgets are in the
   single JSON configuration and cannot be selected after a run.

The historical prompt-only geometric-control plus patch-brightness GPU probe is
an existing `NO_GO` for that exact observation path. It is not trajectory
injection and cannot satisfy any Gate below.

## Gate decisions

Each Gate submission includes changed files, exact commands, raw output,
artifact paths/checksums, and a threshold-by-threshold comparison. The
independent auditor alone issues `GATE_PASS`, `GATE_CONDITIONAL`, or
`GATE_FAIL`. Only `GATE_PASS` admits the next Gate.

- Gate 0: this contract and its single configuration are internally consistent
  and machine-checked. Gate 1 code rejects missing fields and has no hidden
  defaults.
- Gate 1: the frozen two-dimensional synthetic chain passes AISB
  deletion/duplication, independent null, public-only calibration, held-out
  validation, and same-evidence owner/wrong-key scoring. It is synthetic only.
- Gate 2: a real GPU run proves non-scheduler direct model-output injection at
  all eight declared steps. Each residual is finite, nonzero, and within the
  relative-RMS tolerance. Hook execution proves injection only.
- Gate 3: a single-saved-MP4 front-end produces a full-rank, held-out
  calibratable two-dimensional observation.
- Gate 4: public acquisition is frozen before public-only calibration and the
  owner ranks above all wrong keys on identical evidence.
- Gate 5: the identical blind chain remains closed under one minimal temporal
  edit.

Thresholds and claims stay fixed after failure. Oracle, paired, internal, or
scheduler results cannot be renamed as requested evidence.

## GPU execution boundary

GPU Gates use a thin Colab notebook that only mounts Drive, checks out an exact
commit, installs locked dependencies, reads this configuration, calls the
repository CLI, prints provenance/decision summaries, and copies the package.
It contains no method or metric implementation.

Every GPU run directory and archive contains `environment.json`,
`git_state.json`, resolved `config.json`, `command.txt`, `stdout.log`,
`stderr.log`, `metrics.json`, `gate_decision.json`, `artifacts/`,
`checksums.sha256`, and `README.md`. Missing evidence cannot pass.
