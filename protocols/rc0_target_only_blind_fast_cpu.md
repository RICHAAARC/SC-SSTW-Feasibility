# RC0 target-only blind observation fast CPU diagnostic

Status: frozen before the sole CPU execution. Evidence class: `DIAGNOSTIC_ONLY`.
This diagnostic asks only whether the already generated pixel-chroma MP4s can be
read without a matched OFF reference. It is not a Gate, deployment validation,
robustness result, or paper claim.

## Bound input and separation

The input is the existing two matched quartets, in the order `orbital_glass`,
`articulated_paper`, each ordered `OFF_R1`, `OFF_R2`, `A`, `B`. Every MP4 is
bound by its existing raw SHA-256. The runner freshly decodes each target MP4.
It never subtracts, averages, or otherwise reads an OFF video while observing an
A or B video. OFF videos are independently processed by the identical function.
There is no paired difference, training, MLP, ridge candidate search, learned
parameter, result-dependent threshold, generation, or encoding.

The repository's earlier single-video cosine observer
`sc_sstw_feasibility.gpu_carrier.saved_video_observation` is a useful existing
target-only pattern, but it projects luma at half-pixel coordinates. The frozen
pixel carrier is orthogonal zero-sum chroma at integer coordinates, so that luma
observer is not the matching sufficient statistic. This diagnostic reuses the
existing decoder, AISB residual/scanner, pilot calibration, equalizer, and
dynamic-time-sync functions while freezing the following chroma projection.

## Unique target-only observation

Decode to RGB24 uint8 with exact shape `[49,320,512,3]`, then normalize as
`I = uint8/255` in float64. For each frame independently subtract one scalar
mean over all `H*W*RGB` samples. Coordinates include every decoded pixel; there
is no crop, resize, padding, alignment, window selection, or edge special case.

Let

```
phi_x(x) = cos(2*pi*x/512),          x = 0..511
phi_y(y) = cos(2*pi*y/320),          y = 0..319
u = (1,-1,0)/sqrt(2)
v = (1,1,-2)/sqrt(6)
Bx[y,x,c] = phi_x(x) * u[c]
By[y,x,c] = phi_y(y) * v[c]
```

For centered frame `Z`, the signed minimal sufficient projections are

```
qx = sum(Z*Bx) / (sum(Bx*Bx) + 1e-15)
qy = sum(Z*By) / (sum(By*By) + 1e-15)
```

Positive sign is the sign of the frozen basis above. The epsilon is only a
numeric denominator guard and is not a scientific threshold. Frame 0 forms
point 0. Frames `1..48` form twelve consecutive groups of four; each point is
the arithmetic mean of its member frame projections. Output is finite `13x2`.

## Presence and downstream path

Presence is label-free and is evaluated before calibration or phase recovery.
Let `O` be the `13x2` observation, `Oc = O - mean_rows(O)`, and
`E = sqrt(mean(Oc^2))`. Enumerate the fixed 16 AISB candidates
`{A,B} x start0..7` over six-point windows. Let `Rmin` be their minimum frozen
affine residual. Presence passes iff both `E >= 2/255` and `Rmin <= 0.25`.
Both bounds are inherited unchanged from the existing RC0 pixel-effect/AISB
diagnostics. Equality passes. No condition label selects a template.

Only a presence pass continues. Capture keeps exactly the lower-residual A/B
candidate at each start, producing K=8 from the 16-candidate universe. Every
captured candidate is independently calibrated using points 0,1,2,3 and scored
on points 4,5. The unique strict held-out-MSE winner is K2=1. Equalization uses
the existing ridge `1e-4`. Its template identity and full 13-point equalized
trajectory enter the existing primary dynamic-time-sync path and bounded score
cross-check with the already frozen penalties, tie rules, and four perturbations:
`identity`, `delete6_duplicate12`, `local_phase_plus1`,
`local_phase_minus1`.

Observation, presence, capture, calibration, selection, and phase functions do
not receive OFF/A/B truth. After all outputs are frozen, the audit alone checks:

- each of the four OFF targets rejected presence and did not run downstream;
- each of the four A/B targets passed presence, uniquely selected its own
  schedule at start 0, and passed all four pre-existing phase perturbations.

All twenty requirements must pass. There is no averaging or majority vote. A
failure reports the first stage and stops this readout route; no second readout,
parameter scan, or threshold change is authorized. Outcomes are `FEASIBLE`,
`NOT_FEASIBLE`, or `INSUFFICIENT_TO_DECIDE`. All records keep
`formal_result=false` and `stage_progression_allowed=false`.
