# RC0 candidate-wise self-calibration fast CPU diagnostic

## Sole question and frozen inputs

This `DIAGNOSTIC_ONLY` experiment asks only Step 4: are affine-invariant
capture and candidate-wise self-calibration genuinely separated? It consumes
the immutable Step-3 candidate sets from audit SHA-256
`6e424acbd21a36c0c965581e475a7e03440051ad62925c37fceecb3ac1c8ba6f`
and independently redecodes the same eight SHA-bound saved MP4s. It performs
no generation, encoding, candidate expansion, MLP/ridge training, threshold
selection, GPU work, or Step-5 phase recovery.

Capture is complete before calibration. For every group × carrier condition ×
fixed Step-3 time perturbation cell, its exact `K=8` candidate identities,
windows, templates, and capture residuals are copied from the bound Step-3
audit. Calibration cannot add, remove, reorder, or modify that capture set.
The capture-set canonical SHA-256 is checked before and after calibration.

## Existing candidate-wise formula

Each of all eight captured candidates is calibrated independently. A candidate
contains only its captured template (`A` or `B`) and start index. The six-row
30D observation window is selected from the redecoded paired feature matrix.
For that candidate alone, template points 0–3 provide exactly four public
pilot pairs. The existing
`sc_sstw_feasibility.calibration.calibrate_from_pilot_pairs` unregularized
least-squares fit estimates

`observation = matrix(30×2) * q(2) + bias(30)`.

Thus the fitted nuisance dimension is exactly 90 scalar coefficients per
candidate. No coefficient, observation, or label is shared between candidates.
The candidate's rows 4–5 are then mapped back to 2D with the existing
`equalize_observations` inverse and its frozen `ridge=1e-4`. Its sole score is
the mean squared error across the four resulting scalar coordinates against
that candidate's template points 4–5. There is no acceptance threshold.

All eight candidates, including seven capture decoys, receive the identical
independent fit. The actual condition (`own A` or `own B`) is not supplied to
the fitting or ranking function; it is consulted only after the complete
ranking has been frozen for truth audit.

## K2, ranking, perturbations, and falsifiability

No existing project constant names a smaller `K2`. The lowest-freedom value
`K2=1` is frozen from the existing downstream requirement that calibration
consume one frozen accepted candidate. Candidates rank by held-out MSE. A
winner exists only if the lowest score is strictly smaller than the runner-up;
an exact score tie is a failure, not a favorable tie break. For deterministic
report serialization only, equal scores order by capture residual, start index,
then template order A before B.

The Step-3 perturbation family is unchanged: `identity` and
`private_tail_delete6_duplicate12`. The latter leaves the six-row truth window
at start 0 unchanged, so its preregistered correct mapping remains own
template/start0. No new amplitude or bias perturbation is added because the
project contains no unique pre-existing fixed magnitude for such a diagnostic.

A cell passes only when its unique `K2=1` winner is own template/start0. All
eight group × condition × perturbation cells must pass. There is no averaging,
majority vote, result-dependent K2, or feedback into capture.

## Outcomes

- Eight of eight cells pass: Question 4 `FEASIBLE`, route
  `PROCEED_STEP5_PHASE_RECOVERY_CPU`.
- Any complete valid cell fails: Question 4 `NOT_FEASIBLE`, status
  `SELF_CALIBRATION_NOT_FEASIBLE`. This rejects only this frozen independent
  four-pilot/two-held-out self-calibration rule.
- Any identity, decode, feature, candidate-set, or numerical failure: Question
  4 `INSUFFICIENT_TO_DECIDE`, route `DIAGNOSTIC_INSUFFICIENT`.

All evidence is `DIAGNOSTIC_ONLY`; `formal_result=false` and
`stage_progression_allowed=false`. Step 5 is not implemented here.
