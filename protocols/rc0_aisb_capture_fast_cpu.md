# RC0 AISB capture fast CPU diagnostic

## Sole question

This `DIAGNOSTIC_ONLY` experiment asks only Step 3: does the repository's
existing affine-invariant AISB capture put the correct `own schedule / start 0`
window into a finite candidate set? It does not perform self-calibration,
owner/wrong-key scoring, attack testing, threshold fitting, generation, video
encoding, or any GPU work.

The input is exactly the eight saved MP4s already closed by pixel-chroma
evaluation audit SHA-256
`9b08c88aad5378c2b0c94574585906e96a9608d094002248fddd6761b5e1dead`.
Each video is SHA-bound and redecoded; the frozen 30D extractor is recomputed.
For each group and carrier condition the observation is
`F(condition) - mean(F(OFF_R1),F(OFF_R2))`. No MLP, ridge, fitting, or
calibration is allowed.

## Existing formula and finite budget

The sole capture score is the existing `affine_burst_residual_v1` implemented
by `sc_sstw_feasibility.aisb.affine_burst_residual`: template points 0,1,2 are
public affine anchors; barycentric weights predict checksum points 3,4,5; the
sum of checksum squared errors is divided by total six-point observation
scatter plus `1e-9`. It estimates no affine channel while capturing.

Both A and B first-six-point templates are scored at starts 0 through 7. The
unpruned universe is therefore exactly 16 candidates (`A/B × starts 0–7`). The
existing `top_k_per_start=1` is retained: at every start only the lower-residual
of A and B survives, so the finite returned budget is exactly `K=8`, not the
whole universe. There is no residual acceptance threshold in this Step 3
screen. At a residual tie, Python's stable ordering preserves the frozen
template order A before B. Report ordering is residual, start, then template
order. Correct truth is own template/start0; the cross template and every wrong
start are negatives. A cell passes only when own/start0 beats cross/start0 and
therefore enters the eight-candidate set.

## Frozen time perturbation family

Two cases are fixed before execution:

1. `identity` leaves the 13-point paired matrix unchanged.
2. `private_tail_delete6_duplicate12` deletes original propagation-probe index
   6 and appends one extra copy of original index 12. This reuses the project's
   existing private deletion/duplication edit semantics, keeps length 13, and
   leaves original indices 0–5 byte-for-byte unchanged. Its preregistered truth
   mapping therefore remains own template/start0; no post-edit remapping is
   inferred from results.

No perturbation touches the truth window or changes K. Both groups × A/B × both
perturbations must pass; there is no averaging or majority vote.

## Outcomes

- All eight cells capture truth: Step 3 `FEASIBLE`, route
  `PROCEED_STEP4_SELF_CALIBRATION_CPU`.
- Any complete valid cell omits truth: Step 3 `NOT_FEASIBLE`, route
  `AISB_CAPTURE_NOT_FEASIBLE`; this rejects only this frozen capture rule.
- Any identity, decode, feature, budget, or structural failure: Step 3
  `INSUFFICIENT_TO_DECIDE`, route `DIAGNOSTIC_INSUFFICIENT`.

Every output remains `formal_result=false` and
`stage_progression_allowed=false`. Step 4 is not implemented here.
