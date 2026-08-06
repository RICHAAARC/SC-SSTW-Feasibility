# SC-SSTW mechanism-feasibility protocol v2

Protocol v2 is independent from v1 and does not revise its evidence.
`protocols/protocol_v1_no_go.json` remains the immutable interpretation record:
v1 is `NO_GO` because its unconstrained random channel generator produced an
inadmissible frozen case; the method conclusion remains undetermined.

V2 changes exactly one scientific input rule before any v2 method result is
observed: each two-dimensional affine channel is constructed as

```text
A = R(theta_1) diag(s_1, s_2) R(theta_2)
```

with the rotation convention, eight case mappings, angles, positive scales,
biases, and draw order explicitly recorded in the v2 JSON. The v1 master seed,
seed derivation, and every per-case noise/null random stream are retained
exactly; only the removed `synthetic.channel` random domain is replaced by the
explicit constructed parameters. Because
rotations are orthogonal, the singular values are exactly `s_1,s_2` and the
condition number is `max(s_1,s_2)/min(s_1,s_2)`. There is no rejection,
resampling, or result-dependent channel selection.

## Gate split

### Gate 0 v2

Only contract/configuration consistency is checked. No AISB, held-out error,
owner score, or null rejection is run. The independent auditor alone may admit
Gate 1A.

### Gate 1A: input admissibility

Gate 1A reads the frozen v2 configuration and checks only that all eight
channels and all declared inputs are complete, two-dimensional, finite,
invertible, deterministic, and have condition number at most 10. It records
matrix and input digests. It must not call acquisition, calibration, DTW, or
owner/wrong-key scoring. Only `GATE_PASS` admits Gate 1B.

### Gate 1B: synthetic mechanism

Gate 1B consumes the exact v2 configuration and channel digests admitted by
Gate 1A. It then runs the unchanged two-dimensional chain:

```text
AISB acquisition
-> canonical ambiguity serialization/hash/readback
-> all-retained public-only calibration
-> held-out public validation
-> owner plus fixed 31 wrong keys on identical evidence
-> independent null rejection
```

The eight edit cases, eight null cases, public split, keys, search budget, RNG,
and metrics remain predeclared. V1 thresholds are unchanged. Gate 1B PASS is
synthetic bridge evidence only and cannot admit a GPU claim by itself.

GPU work remains forbidden until both Gate 1A and Gate 1B receive independent
`GATE_PASS`. Any later GPU run must use the thin Colab notebook, repository CLI,
complete Drive result package, and single-saved-MP4 blind observation boundary.
