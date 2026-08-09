# RC0 phase-recovery fast CPU diagnostic

## Sole question and immutable input

This `DIAGNOSTIC_ONLY` experiment asks only Step 5: can the project's existing
Viterbi-style dynamic temporal synchronization close phase recovery under a
minimal, preregistered time-edit family? It binds Step-4 audit SHA-256
`572cc1a37a25e8a425125ee31db39e1c864c3e67604cbd964d6dc876740b3a8c`
and the same eight read-only saved MP4 artifacts. It does not generate or encode
video, search capture candidates, refit a calibration, train a model, tune a
threshold, run Step 6, or use GPU/Notebook/Drive.

For each group and condition, only Step 4's frozen `K2=1`, own-template/start-0
candidate is accepted. Its recorded 30x2 matrix and 30D bias are consumed
directly. The eight videos are redecoded and the frozen 30D extractor is
recomputed; the complete 13-point paired matrix is mapped to 2D with the
recorded calibration and the existing equalizer ridge `1e-4`. No truth label
is supplied to calibration and no calibration parameter is refitted.

## Existing primary path and exact recurrence

The sole primary path is `sc_sstw_feasibility.sync.dynamic_time_sync`. Its DP
state `(i,j)` is the minimum accumulated cost after consuming `i` observed
states and `j` template states. Local match cost is squared Euclidean distance
in the frozen 2D space. Transitions, in exact tie priority, are:

1. `MATCH (i-1,j-1)`: add local match cost;
2. `SKIP_TEMPLATE (i,j-1)`: add fixed skip penalty `0.22`;
3. `REPEAT_TEMPLATE (i-1,j)`: add fixed repeat penalty `0.14` plus local cost
   against template `j-1`.

Initialization is `(0,0)=0`; leading template skips are permitted by row zero,
leading observed repeats are not; the terminal state is exactly `(13,13)`.
The implementation has no Sakoe-Chiba band, so the frozen band is the full
13x13 rectangle. Its score is negative final cost divided by the number of
diagonal matches. `dynamic_time_sync_score_bounded` is not a selectable second
method: it only cross-checks the same exact score with `min_score_to_beat=-inf`
and must not abandon.

The existing full-path result exposes diagonal matches only. A local diagnostic
backtracker duplicates the same recurrence solely to record all MATCH/SKIP/
REPEAT operations; its total cost, diagonal path, average cost, and score must
equal the primary implementation. This is bookkeeping, not another winner.

## Frozen perturbations and truth paths

Each perturbation is an exact mapping from the 13 original projected states to
13 observed states. Its expected operation path is derived from the listed
mapping before execution. No interpolation or value change occurs.

- `identity`: `[0,1,2,3,4,5,6,7,8,9,10,11,12]`; thirteen MATCH operations.
- `delete6_duplicate12`: `[0,1,2,3,4,5,7,8,9,10,11,12,12]`; MATCH 0-5,
  SKIP template 6, MATCH 7-12, then REPEAT template 12.
- `local_phase_plus1`: `[0,1,2,3,4,5,7,7,8,9,10,11,12]`; skip phase 6,
  consume phase 7 twice, then return to identity phase.
- `local_phase_minus1`: `[0,1,2,3,4,5,5,7,8,9,10,11,12]`; consume phase 5
  twice, skip phase 6, then return to identity phase.

The latter two are the fixed `+1` and `-1` local phase-jitter cases. Identity
has edit budget zero; every edited case has exactly one SKIP plus one REPEAT.
The global maximum edit budget is two. The DP itself is unchanged and
unconstrained; exceeding or differing from the preregistered exact path fails
the cell.

## Falsifiable result

For each of two groups x A/B x four perturbations, all conditions must hold:

- the recovered complete operation path exactly equals its preregistered truth;
- edit count and diagonal match path agree with truth and remain within budget;
- primary and complete-backtracker scores agree within `1e-12` engineering
  tolerance;
- bounded score-only cross-check is exact and does not abandon;
- own-template score is strictly greater than cross-template score.

All sixteen cells must pass; there is no averaging or majority vote.

- all pass: Question 5 `FEASIBLE`, route `PROCEED_STEP6_TINY_E2E_CPU`;
- any complete valid cell fails: `PHASE_RECOVERY_NOT_FEASIBLE`, rejecting only
  this frozen recovery rule;
- identity, decode, calibration, DP, or numerical failure:
  `INSUFFICIENT_TO_DECIDE`, route `DIAGNOSTIC_INSUFFICIENT`.

All outputs remain `formal_result=false` and
`stage_progression_allowed=false`. Step 6 is not implemented.
