# RC0 tiny end-to-end fast CPU diagnostic

## Sole question

This final `DIAGNOSTIC_ONLY` CPU experiment asks only Question 6: can the
already-frozen Step-1 through Step-5 mechanisms close as one tiny matched-
control pipeline? The sample is exactly two matched quartets and eight existing
SHA-bound MP4s: `OFF_R1`, `OFF_R2`, `A`, and `B` for each of
`orbital_glass` and `articulated_paper`. No sample, generation, encoding,
training, threshold, candidate, or formula may be added or changed.

The run binds the accepted diagnostic audits from Step 1/2, Step 3, Step 4, and
Step 5 by raw SHA-256. It redecodes all eight videos and recomputes the frozen
30D frontend rather than trusting a feature cache.

## Thin pipeline

The fixed per-group OFF reference remains
`0.5 * (OFF_R1 + OFF_R2)`. Every original condition is first scored by the
existing Level-P saved-pixel formula:

- OFF controls must be rejected by carrier presence and stop immediately;
- A and B must pass carrier presence before any downstream calculation.

For each presence-positive condition only, the pipeline is:

1. compute the paired 13x30 matrix against the symmetric OFF reference;
2. call the existing AISB capture over A/B x starts 0-7, retaining one template
   per start and exactly K=8 candidates;
3. independently self-calibrate all K=8 candidates with the existing four-
   pilot/two-held-out rule, then retain strict unique K2=1;
4. use only that selected candidate's recorded affine fit to project the full
   13-point paired matrix to 2D;
5. apply the existing Step-5 primary DP to identity and the three frozen
   synthetic 2D timing edits.

Identity corresponds directly to the saved MP4 observation. The other three
cases (`delete6_duplicate12`, `local_phase_plus1`, and
`local_phase_minus1`) are explicitly synthetic timing diagnostics applied only
after the saved-video 30D trajectory has been projected to 2D.

The thin layer does not reproduce scientific formulas. It invokes the existing
Step-1 through Step-5 functions and compares each frozen stage output with the
bound prior audit. A capture helper runs the existing capture function under
both A and B audit labels and requires the returned K8 sets to be identical;
only that label-independent set enters calibration. Calibration and DP receive
the selected candidate identity, never the expected condition. Expected OFF/A/B
truth is consulted only after all pipeline outputs have been frozen.

## Success and scope

Success requires all eight original conditions to be correct individually:
four OFF controls rejected with no downstream output, and four A/B conditions
accepted with unique own-schedule/start-0 K2=1. All sixteen positive-condition
phase cells must also reproduce their exact preregistered paths and pass own
versus cross. No averaging or majority vote is allowed.

- all pass: Question 6 `FEASIBLE`, overall
  `FEASIBLE_FOR_TINY_MATCHED_CONTROL_DIAGNOSTIC`, route
  `METHOD_FEASIBILITY_CORE_CLOSED`;
- any complete stage fails: `TINY_E2E_NOT_FEASIBLE`, reporting the first failed
  stage in the frozen order presence, capture, calibration, phase;
- identity, decode, prior-audit, or structural failure:
  `INSUFFICIENT_TO_DECIDE` with the single concrete gap.

A feasible result means only that this exact tiny matched-control diagnostic
pipeline closes. It does not answer blind public deployment, robustness,
attacks, FPR, held-out generalization, or any paper claim. All output remains
`formal_result=false` and `stage_progression_allowed=false`.
