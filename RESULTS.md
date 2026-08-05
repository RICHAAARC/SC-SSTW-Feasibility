# CPU-only Synthetic Feasibility Results

Status: synthetic relation-channel probe only. No video, GPU, Wan runtime, saved
video observation, or paper claim.

## 2026-08-05 pass 245: burst16 noise0.62 threshold0.01251953125 diagnostic-pruned full layer

Command family:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.01251953125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,...,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std 0.62, residual_threshold 0.01251953125, burst_count 16:
  pass = 64 / 64
  truth_sequence_covered_by_ambiguity = 64 / 64
  owner_global_recovery = 64 / 64
  full_layer_min_margin = 0.04831724
  full_layer_mean_margin = 0.11320780
  max_ambiguity_sequence_count = 192
  max_exact_score_count_after_diagnostic_screen = 29957
```

Interpretation:

- residual_threshold 0.01251953125 still passes the full checked noise_std 0.62
  layer.
- The checked boundary is now between failing 0.0125 and passing 0.01251953125.
- Since the same synthetic evidence chain remains CPU-only and all failures are
  threshold/acquisition-search effects, this still does not require GPU/video.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
video, GPU, fixed-FPR calibration, or paper claim.

## 2026-08-04 pass 244: burst16 noise0.62 threshold0.0125390625 diagnostic-pruned full layer

Command family:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.0125390625 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,...,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std 0.62, residual_threshold 0.0125390625, burst_count 16:
  pass = 64 / 64
  truth_sequence_covered_by_ambiguity = 64 / 64
  owner_global_recovery = 64 / 64
  full_layer_min_margin = 0.04831724
  full_layer_mean_margin = 0.11305243
  max_ambiguity_sequence_count = 192
  max_exact_score_count_after_diagnostic_screen = 29957
```

Interpretation:

- residual_threshold 0.0125390625 remains sufficient for the full checked
  noise_std 0.62 layer.
- This narrows the checked public-acquisition residual boundary to the interval
  between failing 0.0125 and passing 0.0125390625.
- The result is still CPU-only synthetic diagnostic evidence; it does not imply
  video robustness, fixed-FPR calibration, or a paper-ready claim.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
video, GPU, fixed-FPR calibration, or paper claim.

## 2026-08-04 pass 243: burst16 noise0.62 threshold0.012578125 diagnostic-pruned full layer

Command family:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.012578125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,...,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std 0.62, residual_threshold 0.012578125, burst_count 16:
  pass = 64 / 64
  truth_sequence_covered_by_ambiguity = 64 / 64
  owner_global_recovery = 64 / 64
  full_layer_min_margin = 0.04831724
  full_layer_mean_margin = 0.11307260
  max_ambiguity_sequence_count = 192
  max_exact_score_count_after_diagnostic_screen = 29957
```

Interpretation:

- residual_threshold 0.012578125 remains sufficient for the full checked
  noise_std 0.62 layer, narrowing the boundary to the interval between the
  known failing 0.0125 and passing 0.012578125 settings.
- Public truth coverage and owner/global recovery remain complete under the
  current diagnostic-pruned ambiguity-set scoring route.
- This is still a CPU-only synthetic boundary result; no GPU, video, fixed-FPR,
  or paper claim is implied.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
video, GPU, fixed-FPR calibration, or paper claim.

## 2026-08-04 pass 242: burst16 noise0.62 threshold0.01265625 diagnostic-pruned full layer

Command family:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.01265625 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,...,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std 0.62, residual_threshold 0.01265625, burst_count 16:
  pass = 64 / 64
  truth_sequence_covered_by_ambiguity = 64 / 64
  owner_global_recovery = 64 / 64
  full_layer_min_margin = 0.04831724
  full_layer_mean_margin = 0.11303899
  max_ambiguity_sequence_count = 192
  max_exact_score_count_after_diagnostic_screen = 29957

8-case segment minimum margins:
  cases 0-7:   0.08909555
  cases 8-15:  0.09989343
  cases 16-23: 0.06739550
  cases 24-31: 0.08837956
  cases 32-39: 0.07272589
  cases 40-47: 0.04831724
  cases 48-55: 0.06031150
  cases 56-63: 0.06733981
```

Interpretation:

- residual_threshold 0.01265625 still recovers the full checked noise_std 0.62
  layer above the failing 0.0125 setting.
- Owner/global recovery and public truth coverage remain complete, and the same
  full-layer margin floor is preserved.
- The observed exact-score burden decreases relative to the 0.0128125 and
  0.013125 layers in this run, but this remains a CPU synthetic mapping result,
  not evidence from GPU/video or fixed-FPR calibration.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
video, GPU, fixed-FPR calibration, or paper claim.

## 2026-08-04 pass 241: burst16 noise0.62 threshold0.0128125 diagnostic-pruned full layer

Command family:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.0128125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,...,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std 0.62, residual_threshold 0.0128125, burst_count 16:
  pass = 64 / 64
  truth_sequence_covered_by_ambiguity = 64 / 64
  owner_global_recovery = 64 / 64
  full_layer_min_margin = 0.04831724
  full_layer_mean_margin = 0.11294394
  max_ambiguity_sequence_count = 288
  max_exact_score_count_after_diagnostic_screen = 44356

8-case segment minimum margins:
  cases 0-7:   0.08909555
  cases 8-15:  0.09989343
  cases 16-23: 0.06739550
  cases 24-31: 0.08525248
  cases 32-39: 0.07058426
  cases 40-47: 0.04831724
  cases 48-55: 0.06156125
  cases 56-63: 0.06733981
```

Interpretation:

- The viable residual-threshold interval above the failing 0.0125 setting is
  narrowed again: 0.0128125 still passes the full checked noise_std 0.62 layer.
- The weakest point remains the same margin floor seen at the looser recovered
  thresholds, and owner/global recovery remains complete.
- The worst exact-score burden is unchanged at 44356, so further threshold
  tightening is now primarily a CPU exact-search cost question, not a GPU or
  video-model requirement.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
video, GPU, fixed-FPR calibration, or paper claim.

## 2026-07-30 pass 1: oracle source indices

Command:

```bash
python3 experiments/run_synthetic_probe.py
```

Result:

```text
oracle.feasibility_pass = true
oracle.score_margin = 0.2668
oracle.channel_condition_number = 1.312
oracle.pilot_reconstruction_mse = 0.00067
```

Interpretation:

- If crop/deletion source indices, hence pilot labels, are known, the synthetic
  chain works:

```text
pilot calibration -> equalization -> temporal sync -> owner/wrong-key separation
```

This supports the mathematical plausibility of the self-calibrated affine
relation-channel model under the toy assumptions.

## 2026-07-30 pass 2: observation-only pilot re-acquisition

Command:

```bash
python3 experiments/run_synthetic_batch.py
```

Result summary:

```text
cardinal_four_direction_cycle:
  oracle_pass_count = 4 / 4
  acquired_pass_count = 2 / 4

asymmetric_five_direction_cycle:
  oracle_pass_count = 4 / 4
  acquired_pass_count = 1 / 4
```

Interpretation:

- The owner/wrong-key separation is robust when calibration uses correct public
  pilot labels.
- The current observation-only periodic beam acquisition is not robust enough.
- The immediate bottleneck is not least-squares channel calibration itself; it
  is public-pilot re-acquisition and absolute pilot-label alignment after
  crop/deletion.

Current scientific risk:

```text
public pilots can fit a low-error affine channel while still inducing a wrong
state-space orientation or clock alignment for key scoring.
```

Next probe:

- Design a pilot code with explicit synchronization structure rather than a
  simple repeated direction cycle.
- Score acquisition using key-independent constraints:
  - pilot fit;
  - unit-state manifold consistency;
  - monotone clock consistency;
  - robustness to missing pilots.

## 2026-07-30 pass 3: explicit public sync pilot code

Command:

```bash
python3 experiments/run_synthetic_batch.py
```

Result summary:

```text
cardinal_four_direction_cycle:
  oracle_pass_count = 4 / 4
  acquired_pass_count = 2 / 4

asymmetric_five_direction_cycle:
  oracle_pass_count = 4 / 4
  acquired_pass_count = 1 / 4

public_sync_noncyclic_code:
  oracle_pass_count = 4 / 4
  acquired_pass_count = 0 / 4
```

Additional diagnostic:

```text
failed acquired cases often have acquired_label_alignment near 0.0
```

Interpretation:

- A longer non-cyclic public pilot code does not solve the problem by itself.
- The failure mode is now sharper: the acquisition objective can select a pilot
  label sequence that fits an affine channel with low reconstruction error but
  does not match the true public label alignment.
- This is an identifiability issue in the synthetic model, not a GPU/runtime
  issue.

Current method-design implication:

```text
SC-SSTW should not rely on unlabeled affine pilot fitting alone.
```

The next method variant should add at least one of:

- a pilot acquisition statistic invariant to the unknown affine channel;
- a public synchronization marker that is not freely absorbable by affine
  calibration;
- a two-stage calibration where clock/pilot alignment is frozen before affine
  channel estimation;
- local pilot neighborhoods or bursts whose relative geometry survives affine
  fitting better than isolated pilot points.

## 2026-08-01 pass 4: AISB synthetic acquisition switch

Command:

```bash
python3 experiments/run_aisb_probe.py
```

Result summary:

```text
true AISB cases:
  pass_count = 8 / 8
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  state_reconstruction_mse_mean = 0.000138

random non-burst cases:
  false_pass_count = 8 / 8
  accepted_count = 0 for every case
```

Interpretation:

- The AISB switch fixes the specific synthetic identifiability failure exposed
  by affine least-squares pilot acquisition.
- Acquisition uses affine-invariant burst geometry and does not estimate
  `A_x,b_x` while locating public sync bursts.
- In the current toy setup, true bursts produce residuals around `0.0008` to
  `0.0033`, while random non-burst observations have best residuals around
  `0.08` to `0.23`.
- After AISB freezes public alignment, affine calibration and equalization
  recover the synthetic states with mean MSE around `1.38e-4`.

Current boundary:

- crop and deletion occur outside burst interiors;
- burst length is fixed at 6;
- templates use three anchors plus three checksum states;
- this is still synthetic affine-channel evidence only.

Next probe:

- Allow one deletion inside a burst and recover by enumerating missing checksum
  positions.
- Add key-conditioned secret-state scoring around AISB public bursts.
- Add multiple-testing threshold reporting for sliding-window false
  acquisition.


## 2026-08-01 pass 5: AISB checksum-deletion tolerance and mixed scoring

Command:

```bash
python3 experiments/run_aisb_probe.py
```

Result summary:

```text
complete AISB cases:
  pass_count = 8 / 8
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  state_reconstruction_mse_mean = 0.000138

single checksum deletion inside burst:
  pass_count = 8 / 8
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  state_reconstruction_mse_mean = 0.000137

random non-burst cases with deletion-aware scanner:
  false_pass_count = 8 / 8
  accepted_count_total = 0

AISB public alignment + secret-state owner/wrong-key mixed sequence:
  pass_count = 6 / 6
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  score_margin_mean = 0.504
```

Interpretation:

- AISB can recover public synchronization bursts when one checksum point inside
  each burst is deleted. The scanner enumerates the missing template position and
  scores only public affine geometry; it still does not estimate `A_x,b_x` during
  acquisition.
- Once AISB freezes the public alignment, the same public burst observations can
  calibrate the synthetic affine channel, equalize the edited sequence, and
  separate the owner key from wrong keys on mixed public-burst plus secret-state
  sequences.
- Random non-burst observations remain rejected under the deletion-aware scan in
  this synthetic batch.

Boundary:

- This pass covers single checksum deletion, not deletion of one of the three
  primary anchor points. With the current 6-point template, anchor deletion can
  remove part of the affine coordinate frame.

## 2026-08-01 pass 6: redundant AISB arbitrary-deletion diagnostic

Command:

```bash
python3 experiments/run_aisb_probe.py
```

Latest result summary:

```text
complete AISB cases:
  pass_count = 8 / 8

single checksum deletion inside burst:
  pass_count = 8 / 8

mixed public AISB + secret-state owner/wrong-key scoring:
  pass_count = 6 / 6

redundant exact-anchor arbitrary one-point deletion:
  pass_count = 17 / 18
  false_positive = 1
  false_negative = 1

redundant threshold development diagnostic:
  test_positive_pass_count = 17 / 18
  diagnostic_pass = false
```

Interpretation:

- The earlier checksum-deletion and mixed-scoring results remain positive.
- The exact redundant-anchor construction does **not** establish unique public
  AISB alignment for arbitrary one-point deletion.
- The failure is a shifted-window ambiguity: exact public anchor copies allow an
  adjacent window to score within the same low-residual band while assigning a
  different missing template index.
- This is a method-mechanism identifiability issue in the synthetic CPU model,
  not a runtime issue and not a reason to run GPU tests.

Current boundary:

- arbitrary burst-internal deletion with exact redundant anchor copies is a
  negative result;
- owner/wrong-key scoring after AISB-frozen alignment remains positive in the
  tested synthetic tiers;
- future CPU work should either use a stronger public sync burst code or treat
  public AISB acquisition as a bounded ambiguity set scored fairly for owner and
  wrong keys.

## 2026-08-01 pass 7: bounded ambiguity-set scoring diagnostic

Command:

```bash
python3 experiments/run_aisb_ambiguity_probe.py
```

Result summary:

```text
ambiguity payload cases:
  pass_count = 12 / 12
  owner_message_recovery_count = 12 / 12
  mean owner-vs-best-wrong margin = 0.141
  maximum ambiguity sequence count in this batch = 1

targeted shifted-window ambiguity cases:
  pass_count = 6 / 6
  ambiguity sequence count = 3 for every case
  mean owner-vs-best-wrong margin = 0.222

random non-burst cases:
  accepted ambiguity sequence total = 0
```

Interpretation:

- The scorer now supports the correct fairness boundary for a future bounded
  AISB ambiguity set: owner and wrong keys search the same public candidate
  alignments and the same fixed message space.
- In the random payload batch, acquisition remained unique
  (`max ambiguity sequence count = 1`).
- A targeted shifted-window batch explicitly creates three public alignment
  hypotheses per case. Owner and wrong keys search the same set, and owner
  still separates in 6/6 cases.
- This does not make exact redundant AISB a unique-alignment solution. It shows
  that bounded public ambiguity-set scoring is a viable CPU-only fallback to
  test further.

Boundary:

- no fixed-FPR calibration;
- no video observation;
- no GPU;
- no paper claim;
- no claim that exact redundant anchors solve arbitrary burst-internal deletion.

## 2026-08-01 pass 8: AISB payload capacity diagnostic

Command:

```bash
python3 experiments/run_aisb_capacity_probe.py
```

Result summary:

```text
messages_8_wrong_12:
  pass_count = 8 / 8
  minimum owner-vs-best-wrong margin = 0.108

messages_16_wrong_24:
  pass_count = 8 / 8
  minimum owner-vs-best-wrong margin = 0.106

messages_32_wrong_24:
  pass_count = 8 / 8
  minimum owner-vs-best-wrong margin = 0.0734
```

Interpretation:

- After AISB public alignment and one shared affine calibration, the toy
  key/message-conditioned state trajectory remains discriminative when the
  fixed message space is expanded to 32 messages and wrong-key count to 24.
- Owner and wrong keys use the same message search space; wrong keys are not
  given a freer or larger fit.
- Margin decreases as message space grows, which is the expected scientific
  risk to track next.

Boundary:

- this is still CPU-only synthetic affine-channel evidence;
- no multiple-testing calibration or fixed-FPR result;
- no real video observation or GPU result;
- no paper claim.

## 2026-08-02 pass 9: AISB capacity under non-affine mismatch

Command:

```bash
python3 experiments/run_aisb_capacity_mismatch_probe.py
```

Result summary:

```text
gamma_0.5_messages_8_wrong_12:
  pass_count = 6 / 6
  minimum owner-vs-best-wrong margin = 0.0950

gamma_0.5_messages_16_wrong_24:
  pass_count = 6 / 6
  minimum owner-vs-best-wrong margin = 0.0779

gamma_0.5_messages_32_wrong_24:
  pass_count = 6 / 6
  minimum owner-vs-best-wrong margin = 0.0606

gamma_1.0_messages_16_wrong_24:
  pass_count = 4 / 6
```

Interpretation:

- The combined pressure test passes at the previously validated non-affine tier
  `gamma = 0.5` even with a 32-message owner/wrong-key search.
- `gamma = 1.0` is a real margin boundary: it introduces one alignment false
  negative and one owner/wrong-key margin failure.
- Full-state reconstruction MSE increases under non-affine mismatch and is
  tracked as a diagnostic; the pass criterion here is the actual mechanism
  question: public alignment, owner message recovery, and owner-vs-wrong
  trajectory margin under the same message search space.

Boundary:

- still CPU-only synthetic evidence;
- no fixed-FPR calibration;
- no real video observation;
- no GPU;
- no paper claim.

## 2026-08-02 pass 10: edit-stress payload under non-affine mismatch

Command:

```bash
python3 experiments/run_aisb_stress_mismatch_payload_probe.py
```

Result summary:

```text
gamma_0.5_messages_16_wrong_24:
  pass_count = 6 / 6
  minimum owner-vs-best-wrong margin = 0.0324

gamma_0.5_messages_32_wrong_24:
  pass_count = 6 / 6
  minimum owner-vs-best-wrong margin = 0.0452

gamma_0.8_messages_16_wrong_24:
  pass_count = 6 / 6
  minimum owner-vs-best-wrong margin = 0.0248
```

Interpretation:

- The mechanism remains positive when combining crop, non-burst deletions,
  non-burst repeats, one missing point in every retained burst, non-affine
  mismatch, and a larger shared message search.
- The margin is now close to the decision floor at `gamma = 0.8`; this is a
  concrete CPU-side robustness boundary.
- The result still depends on synthetic relation observations and public AISB
  acquisition. It does not imply real saved-video observability.

Boundary:

- no fixed-FPR calibration;
- no real video observation;
- no GPU;
- no paper claim.

## 2026-08-02 pass 11: long-sequence payload diagnostic

Command:

```bash
python3 experiments/run_aisb_long_sequence_probe.py
```

Result summary:

```text
bursts_10_gamma_0.0_messages_32_wrong_16:
  pass_count = 2 / 2
  minimum owner-vs-best-wrong margin = 0.138

bursts_10_gamma_0.5_messages_32_wrong_16:
  pass_count = 2 / 2
  minimum owner-vs-best-wrong margin = 0.119

bursts_12_gamma_0.5_messages_32_wrong_16:
  pass_count = 2 / 2
  minimum owner-vs-best-wrong margin = 0.116

bursts_12_gamma_0.8_messages_16_wrong_16:
  pass_count = 2 / 2
  minimum owner-vs-best-wrong margin = 0.101
```

Interpretation:

- Longer sequences with 10-12 public bursts and larger 32-message search remain
  positive in this CPU synthetic probe.
- Increasing the number of public calibration bursts improves margin compared
  with the shorter high-edit stress cases, consistent with better affine
  calibration and stronger trajectory evidence.
- Runtime grows with `wrong_key_count * message_space_size * sequence_length`;
  this is a CPU cost boundary for exhaustive synthetic diagnostics, not a GPU
  requirement.

Boundary:

- only 2 cases per tier, used as a quick mechanism/margin diagnostic;
- no fixed-FPR calibration;
- no real video observation;
- no GPU;
- no paper claim.

## 2026-08-02 pass 12: pruned-search diagnostic

Command:

```bash
python3 experiments/run_aisb_pruned_search_probe.py
```

Result summary:

```text
exhaustive_check_bursts_10_messages_32_wrong_16:
  pass_count = 0 / 1
  pruned_matches_exhaustive = false
  pruned owner-vs-best-wrong margin = 0.0069
  candidate_count = 544
  full_scored_count_after_pruning = 150

pruned_bursts_12_messages_48_wrong_48:
  pass_count = 1 / 1
  candidate_count = 2352
  full_scored_count_after_pruning = 167
  margin = 0.128

pruned_bursts_12_messages_48_wrong_48_gamma_0.8:
  pass_count = 1 / 1
  candidate_count = 2352
  full_scored_count_after_pruning = 180
  margin = 0.117
```

Interpretation:

- The cheap decimated-DTW screening heuristic is **not** a valid replacement for
  exhaustive scoring: in the calibration tier, it changes the exhaustive winner
  and loses the decision margin.
- Larger pruned smoke tiers are positive, but they are not accepted as method
  evidence because the pruning rule failed the exhaustive equivalence check.
- The actionable conclusion is CPU-side: exhaustive owner/wrong-key/message
  scoring is expensive, and unsafe screening can fabricate a different decision.

Boundary:

- do not use this pruning heuristic for scientific pass/fail claims;
- larger search-space claims need either exact scoring, a proven-safe bound, or
  a diagnostic explicitly labelled non-exhaustive;
- no GPU is required by this finding.

## 2026-08-02 pass 13: long-sequence targeted ambiguity diagnostic

Command:

```bash
python3 experiments/run_aisb_long_ambiguity_probe.py
```

Result summary:

```text
bursts_10_gamma_0.0_messages_16_wrong_12:
  pass_count = 4 / 4
  ambiguity sequence count = 2 to 3
  minimum owner-vs-best-wrong margin = 0.210

bursts_10_gamma_0.5_messages_16_wrong_12:
  pass_count = 4 / 4
  ambiguity sequence count = 2 to 3
  minimum owner-vs-best-wrong margin = 0.181
```

Interpretation:

- The bounded public ambiguity-set route remains positive in a longer sequence
  with one targeted shifted-window ambiguity cluster.
- Owner and wrong keys search the same alignment hypotheses and message space.
- This supports the ambiguity-set fallback more directly than the random
  ambiguity probe because each case actually contains multiple public alignment
  hypotheses.

Boundary:

- only one targeted ambiguity cluster per sequence;
- no fixed-FPR calibration;
- no real video observation;
- no GPU;
- no paper claim.

## 2026-08-02 pass 14: multi-ambiguity-set diagnostic

Command:

```bash
python3 experiments/run_aisb_multi_ambiguity_probe.py
```

Result summary:

```text
bursts_10_gamma_0.0_messages_8_wrong_8:
  pass_count = 3 / 3
  ambiguity sequence count = 3 to 9
  multi-cluster-like cases = 1 / 3
  minimum owner-vs-best-wrong margin = 0.245

bursts_10_gamma_0.5_messages_8_wrong_8:
  pass_count = 3 / 3
  ambiguity sequence count = 3 to 6
  multi-cluster-like cases = 2 / 3
  minimum owner-vs-best-wrong margin = 0.174
```

Interpretation:

- The bounded public ambiguity-set route remains positive when the public
  acquisition set is non-unique in every case.
- Some cases produce multiple ambiguity combinations, but the targeted
  construction does not guarantee two independent ambiguity clusters in every
  case. This is tracked as a diagnostic rather than hidden.
- Owner and wrong keys still search the same alignment hypotheses and the same
  message set.

Boundary:

- this supports bounded ambiguity-set scoring, not unique AISB acquisition;
- no fixed-FPR calibration;
- no real video observation;
- no GPU;
- no paper claim.
  be ambiguous with adjacent filler and needs a stronger burst code or extra
  boundary evidence before being claimed.
- There is still no multiple-testing calibrated threshold, no real video
  observation, no fixed-FPR result, and no paper claim.


## 2026-08-01 pass 6: redundant AISB arbitrary single deletion

Command:

```bash
python3 experiments/run_aisb_probe.py
```

Result summary for the added redundant AISB tier:

```text
redundant_any_single_deletion_cases:
  pass_count = 18 / 18
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  state_reconstruction_mse_mean = 0.0000869

redundant_random_non_burst_threshold_diagnostic:
  case_count = 128
  diagnostic_threshold = 0.006
  accepted_count_total = 0
  best_residual_min = 0.0746
  best_residual_median = 0.2398
  fixed_fpr_claim = false

redundant_mixed_sequence_owner_wrong_key_cases:
  pass_count = 9 / 9
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  score_margin_mean = 0.450
```

Interpretation:

- The shorter 6-point AISB template is not generally robust to anchor deletion.
  Its validated deletion tier remains checksum deletion only.
- A 9-point redundant-anchor AISB template closes arbitrary single-point deletion
  in this synthetic affine-channel setting. The construction uses three primary
  anchors, three checksum points, and three public redundant anchor copies.
- The 128-case random non-burst scan shows a large diagnostic residual gap under
  the current toy distribution. This is useful feasibility evidence but is not a
  calibrated fixed-FPR result.
- The redundant AISB alignment also closes the mixed public-burst plus
  secret-state owner/wrong-key loop under arbitrary one-point burst deletion in
  the current synthetic setup.

Current next scientific gap:

- Stress the redundant AISB design under harder edit patterns and higher noise.
- Add an explicit threshold-development split before any fixed-FPR language.
- Only after the synthetic observation layer remains stable should the project
  define a real video observation adapter.


## 2026-08-01 pass 7: redundant AISB stress edits

Command:

```bash
python3 experiments/run_aisb_stress_probe.py
```

Result summary:

```text
stress_cases at noise_std = 0.016:
  pass_count = 12 / 12
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  score_margin_mean = 0.148
  state_reconstruction_mse_mean = 0.000138

random_non_burst_cases:
  pass_count = 64 / 64
  accepted_count_total = 0
  best_residual_min = 0.117

noise_margin_diagnostic:
  noise_0.012 pass_count = 6 / 6
  noise_0.016 pass_count = 6 / 6
  noise_0.020 pass_count = 4 / 6, false_negative = 2
```

Interpretation:

- The redundant AISB construction still recovers public alignment after combined
  crop, deterministic non-burst deletions, occasional non-burst repeats, variable
  burst spacing, and one arbitrary missing point inside every retained burst.
- After AISB freezes public alignment, the same calibration/equalization path
  continues to separate the owner key from wrong keys in the stress batch.
- Random non-burst sequences remain rejected at the existing diagnostic
  threshold in this CPU-only synthetic distribution.
- The noise margin is finite: at `noise_std = 0.020`, the current threshold starts
  producing false negatives. This is a useful method boundary, not a parameter
  tuning authorization and not a fixed-FPR result.

Evidence boundary remains unchanged: synthetic affine-channel diagnostics only;
no real video observations, no GPU, no fixed-FPR claim, and no paper claim.


## 2026-08-02 pass 9: exact CPU scoring throughput and stress continuation

Commands:

```bash
python3 experiments/run_aisb_stress_mismatch_payload_probe.py
python3 experiments/run_aisb_long_sequence_probe.py
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 500 --filler-multiplier 7 --near-tie-ratio 5 \
  --message-space-size 24 --wrong-key-count 12 \
  --start-index 24 --case-count 1 \
  --scoring-mode ordered_bounded_global_c
```

Result summary:

```text
stress mismatch payload:
  gamma_0.5_messages_16_wrong_24 pass = 6 / 6, min_margin = 0.032449
  gamma_0.5_messages_32_wrong_24 pass = 6 / 6, min_margin = 0.045161
  gamma_0.8_messages_16_wrong_24 pass = 6 / 6, min_margin = 0.024822

long sequence:
  bursts_10_gamma_0.0_messages_32_wrong_16 pass = 2 / 2, min_margin = 0.137839
  bursts_10_gamma_0.5_messages_32_wrong_16 pass = 2 / 2, min_margin = 0.118830
  bursts_12_gamma_0.5_messages_32_wrong_16 pass = 2 / 2, min_margin = 0.116410
  bursts_12_gamma_0.8_messages_16_wrong_16 pass = 2 / 2, min_margin = 0.101269

case 24 reduced exact diagnostic:
  message_space_size = 24
  wrong_key_count = 12
  synthetic_construction_pass = true
  truth sequence exact/covered = true / true
  owner/global recovery = true / true
  exact score_margin = 0.1472903461
```

Implementation-only CPU scorer updates:

- `ordered_bounded_c` can score public ambiguity sequences in parallel and uses
  prepared native arrays plus reusable native DP workspaces.
- `ordered_bounded_global_c` globally orders all sequence/candidate pairs by a
  cheap diagnostic score, then still uses exact bounded native DTW.
- `margin_proof_c` distinguishes exact owner recovery from a wrong-score upper
  bound and lower-bound margin proof.
- The bounded scorer now uses a stronger safe lower bound based on unavoidable
  future skip cost and the maximum possible final diagonal-match count.

Full-tier case 24 remains unresolved as a local CPU throughput issue:

```text
gamma = 500
filler_multiplier = 7
message_space_size = 64
wrong_key_count = 64
public ambiguity sequences = 24
workload = 24 * 4160 exact DTW candidates
```

The full-tier case still did not finish inside a 3-minute local CPU window with
the exact-preserving modes above. Public acquisition for that case is already
positive: the exact truth sequence is present and covered. The reduced same-seed
diagnostic also passes. This is therefore not an observed scientific failure and
not a GPU/video boundary; it is an exhaustive exact CPU scoring throughput
boundary.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
real video observations, no GPU, no fixed-FPR claim, and no paper claim.


## 2026-08-02 pass 10: AISB stress grid

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py
```

Result summary:

```text
gamma_0.5_noise_0.012 pass = 4 / 4, alignment = 4 / 4, owner = 4 / 4,
  min_margin = 0.032449
gamma_0.5_noise_0.016 pass = 4 / 4, alignment = 4 / 4, owner = 4 / 4,
  min_margin = 0.032303
gamma_0.8_noise_0.012 pass = 4 / 4, alignment = 4 / 4, owner = 4 / 4,
  min_margin = 0.024822
gamma_0.8_noise_0.016 pass = 3 / 4, alignment = 3 / 4, owner = 4 / 4,
  min_margin = 0.024728
gamma_1.0_noise_0.012 pass = 2 / 4, alignment = 3 / 4, owner = 4 / 4,
  min_margin = 0.016005
gamma_1.0_noise_0.016 pass = 2 / 4, alignment = 3 / 4, owner = 4 / 4,
  min_margin = 0.015912

ambiguity-set scoring reduced grid:
  gamma_0.5_noise_0.012 pass = 3 / 3, covered = 3 / 3, owner = 3 / 3,
    min_margin = 0.269424
  gamma_0.5_noise_0.016 pass = 3 / 3, covered = 3 / 3, owner = 3 / 3,
    min_margin = 0.269351
  gamma_0.8_noise_0.012 pass = 3 / 3, covered = 3 / 3, owner = 3 / 3,
    min_margin = 0.232391
  gamma_0.8_noise_0.016 pass = 3 / 3, covered = 3 / 3, owner = 3 / 3,
    min_margin = 0.232690
  gamma_1.0_noise_0.012 pass = 3 / 3, covered = 3 / 3, owner = 3 / 3,
    min_margin = 0.202500
  gamma_1.0_noise_0.016 pass = 3 / 3, covered = 3 / 3, owner = 3 / 3,
    min_margin = 0.202411
```

Interpretation:

- The checked crop/delete/repeat plus burst-internal deletion plus deterministic
  non-affine mismatch grid remains positive through `gamma = 0.8` at
  `noise_std = 0.012`.
- At `gamma = 0.8, noise_std = 0.016`, the first failure mode is public
  alignment, not owner-message recovery: owner recovery stays 4 / 4 while
  alignment and pass count drop to 3 / 4.
- At `gamma = 1.0`, owner recovery still remains 4 / 4, but the diagnostic
  margin falls below 0.02 in some cases and alignment remains 3 / 4. This is a
  finite synthetic robustness boundary for the current public acquisition and
  scoring stack.
- The reduced ambiguity-set scoring grid closes that boundary in the checked
  cases: truth coverage and owner recovery remain 3 / 3 through `gamma = 1.0`
  and `noise_std = 0.016`, with minimum margin above 0.20. The CPU-side method
  direction is therefore to freeze a public ambiguity set and score owner/wrong
  keys over it, rather than forcing one best public alignment before scoring.
- This grid is diagnostic only. It is not threshold calibration, fixed-FPR
  evidence, real observation evidence, or a paper claim.

Evidence boundary remains unchanged: CPU-only synthetic diagnostics only; no
real video observations, no GPU, no fixed-FPR claim, and no paper claim.


## Exact score-only search-scale diagnostic

Command:

```bash
python3 experiments/run_aisb_exact_search_scale_probe.py
```

Purpose:

- The prior pruned-search probe is a negative result: decimated screening can
  change the exhaustive winner.
- This diagnostic keeps exhaustive owner/wrong/message scoring and only swaps
  the path-producing DTW implementation for an exact score-only scorer.
- This is a CPU-cost diagnostic, not GPU evidence, not fixed-FPR calibration, and
  not a paper claim.

Result summary:

```text
10 bursts, gamma 0.5, 16 messages, 12 wrong keys:
  pass 2 / 2
  max candidates = 208
  min margin = 0.0826

12 bursts, gamma 0.5, 24 messages, 12 wrong keys:
  pass 1 / 1
  max candidates = 312
  margin = 0.1366

12 bursts, gamma 0.8, 16 messages, 12 wrong keys:
  pass 1 / 1
  max candidates = 208
  margin = 0.0663

one-off larger exact CPU scale checks:
  12 bursts, gamma 0.5, 48 messages, 48 wrong keys:
    pass, candidates = 2352, margin = 0.0944, elapsed = 32.9s, max RSS = 23.7MB
  12 bursts, gamma 0.8, 48 messages, 48 wrong keys:
    pass, candidates = 2352, margin = 0.0751, elapsed = 34.7s, max RSS = 23.4MB
  12 bursts, gamma 0.5, 64 messages, 64 wrong keys:
    pass, candidates = 4160, margin = 0.0863, elapsed = 59.3s, max RSS = 23.8MB
  12 bursts, gamma 0.8, 64 messages, 64 wrong keys:
    pass, candidates = 4160, margin = 0.0423, elapsed = 58.4s, max RSS = 23.8MB
```

Interpretation:

- Exact score-only DTW matches the existing full-path scorer in unit tests and
  in the owner-best verification inside this probe.
- The score-only path now inlines the same two-dimensional squared distance,
  reducing the larger 48-message / 48-wrong-key one-off runtime from about 79s
  to about 33-35s without changing the selected candidate or score.
- Exhaustive scoring remains viable for moderate CPU-only search sizes without
  unsafe pruning.
- The larger one-off 48-message / 48-wrong-key and 64-message / 64-wrong-key
  checks remain positive. Runtime is still CPU-feasible after the score-only
  optimization, but the 64-message / 64-wrong-key `gamma = 0.8` margin is only
  0.0423, so the next local risk is synthetic robustness margin, not GPU.


## AISB residual-threshold margin diagnostic

Command:

```bash
python3 experiments/run_aisb_threshold_margin_probe.py
```

Purpose:

- The 64-message / 64-wrong-key `gamma = 0.8` exact search found one case where
  owner/message scoring remained correct but public alignment was not exact.
- This diagnostic checks whether that failure is a narrow AISB residual margin
  issue or a broader acquisition collapse.
- This is not fixed-FPR calibration and does not authorize a formal threshold
  change by itself.

Result summary:

```text
same seed family as the exact-search high-mismatch cases:
  threshold 0.006:
    alignment exact 3 / 4
    false negatives = 1
    false positives = 0
  threshold 0.00625:
    alignment exact 4 / 4
    false negatives = 0
    false positives = 0
  max true residual = 0.006108
  best false residual min = 0.08208

random non-burst, gamma 0.8, 64 cases:
  threshold 0.00625 accepted total = 0
  best residual min = 0.08006
  best residual min case = 22

exact owner/wrong/message scoring with threshold 0.00625:
  12 bursts, gamma 0.8, 64 messages, 64 wrong keys:
    pass 12 / 12
    alignment exact 12 / 12
    min margin = 0.0423
    elapsed = 175.3s for cases 0-2
    elapsed = 183.6s for cases 3-5
    cases 6-8 interactive run completed; elapsed is not recorded in JSON
    cases 9-11 interactive run completed; elapsed is not recorded in JSON
```

Interpretation:

- The observed false negative is a narrow public-AISB residual-threshold margin
  issue under the current synthetic high-mismatch construction.
- The current synthetic distribution has a large gap between the borderline
  true burst residual and the nearest false burst residual, but this is still
  diagnostic evidence only.
- The predeclared diagnostic threshold candidate 0.00625 preserves exact
  owner/wrong/message separation across the first 12 high-mismatch 64-message /
  64-wrong-key cases. Random non-burst rejection is positive across 64 cases.
  This still does not require GPU.

Additional stress diagnostic:

```text
gamma 1.0, noise 0.012, 12 bursts, 64 messages, 64 wrong keys, cases 0-2:
  threshold 0.00625 exact scoring:
    pass = 2 / 3
    alignment exact = 2 / 3
    owner message recovery = 3 / 3
    minimum owner-vs-wrong margin = 0.0289
    failure mode: case 1 keeps owner/message separation but public AISB
    alignment is not exact.

  threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.00625: alignment exact 3 / 4, false negatives = 3,
      false positives = 0
    threshold 0.0075: alignment exact 4 / 4, false negatives = 0,
      false positives = 0

  random non-burst, gamma 1.0, 64 cases:
    threshold 0.0075 accepted total = 0
    best residual min = 0.07735

  threshold 0.0075 exact scoring, cases 0-11:
    pass = 12 / 12
    alignment exact = 12 / 12
    owner message recovery = 12 / 12
    minimum owner-vs-wrong margin = 0.0289
```

Interpretation:

- The first gamma 1.0 failure is an acquisition/alignment threshold boundary,
  not an owner/wrong-key scoring collapse.
- A wider diagnostic residual candidate, 0.0075, restores the first gamma 1.0
  exact-scoring cases while preserving rejection for the 64 random non-burst
  diagnostic cases.
- This is still threshold-margin diagnosis, not fixed-FPR calibration and not a
  paper claim.

Further stress diagnostic:

```text
gamma 1.3, noise 0.012, 12 bursts, 64 messages, 64 wrong keys:
  threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.01: alignment exact 4 / 4, false negatives = 0,
      false positives = 0

  random non-burst, gamma 1.3, 256 cases:
    threshold 0.01 accepted total = 0
    best residual min = 0.05406
    best residual min case = 188

  threshold 0.01 exact scoring, cases 0-5:
    pass = 6 / 6
    alignment exact = 6 / 6
    owner message recovery = 6 / 6
    minimum owner-vs-wrong margin = 0.0229

gamma 1.2, noise 0.012, 12 bursts, 64 messages, 64 wrong keys:
  threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.0075: alignment exact 3 / 4, false negatives = 1,
      false positives = 0
    threshold 0.01: alignment exact 4 / 4, false negatives = 0,
      false positives = 0

  random non-burst, gamma 1.2, 256 cases:
    threshold 0.01 accepted total = 0
    best residual min = 0.05306
    best residual min case = 188

  threshold 0.01 exact scoring, cases 0-11:
    pass = 12 / 12
    alignment exact = 12 / 12
    owner message recovery = 12 / 12
    minimum owner-vs-wrong margin = 0.0241
```

Interpretation: the same pattern persists through gamma 1.3. Stronger synthetic
mismatch primarily pushes the public AISB residual threshold upward; owner /
wrong-key scoring still separates on the checked exact cases once public
alignment is recovered. The expanded 256-case random non-burst check still
shows no accepted random bursts at threshold 0.01, but this remains diagnostic
threshold-margin evidence only.

Current synthetic separability boundary:

```text
gamma 1.4, noise 0.012, 12 bursts, 64 messages, 64 wrong keys:
  threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.0075: alignment exact 3 / 4, false negatives = 3,
      false positives = 0
    threshold 0.01: alignment exact 4 / 4, false negatives = 0,
      false positives = 0

  random non-burst, gamma 1.4, 256 cases:
    threshold 0.01 accepted total = 0
    best residual min = 0.05503
    best residual min case = 188

  threshold 0.01 exact scoring, cases 0-2:
    pass = 1 / 3
    alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum owner-vs-wrong margin = 0.0165
    failed pass condition: owner-vs-wrong margin <= 0.02 in cases 0 and 1

  fine transition probes, threshold 0.01, cases 0-2:
    gamma 1.35: pass = 3 / 3, minimum owner-vs-wrong margin = 0.02006
    gamma 1.36: pass = 1 / 3, minimum owner-vs-wrong margin = 0.01902

  public-burst-count probe, threshold 0.01, gamma 1.36, cases 0-2:
    burst_count 16: pass = 2 / 3, minimum margin = 0.0167
    burst_count 20: pass = 2 / 3, minimum margin = -0.0033
    interpretation: simply adding more public AISB bursts does not reliably
      restore owner/wrong separability and can make a wrong key globally win.

  secret-state filler-span probe, threshold 0.01, cases 0-2:
    gamma 1.36, burst_count 12, filler_multiplier 2:
      pass = 3 / 3, minimum margin = 0.0895
    gamma 1.4, burst_count 12, filler_multiplier 2, cases 0-11:
      pass = 12 / 12, alignment exact = 12 / 12
      owner message recovery = 12 / 12
      minimum margin = 0.0710
    gamma 1.6, burst_count 12, filler_multiplier 2, cases 0-11:
      pass = 12 / 12, alignment exact = 12 / 12
      owner message recovery = 12 / 12
      minimum margin = 0.0593

  pruned-screen diagnostic, gamma 1.6, burst_count 12, filler_multiplier 2,
  cases 0-2:
    pass = 1 / 3
    alignment exact = 3 / 3
    owner message recovery = 1 / 3
    interpretation: cheap decimated screening is not reliable in this
      high-mismatch / long-secret-span regime; exact scoring remains the
      active CPU evidence path.

  gamma 1.8, threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.01: alignment exact 4 / 4, false negatives = 0,
      false positives = 0
    random non-burst, 256 cases:
      threshold 0.01 accepted total = 0
      best residual min = 0.05850

  gamma 1.8, threshold 0.01 exact scoring, burst_count 12,
  filler_multiplier 2, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0581

  gamma 2.0, threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.01: alignment exact 4 / 4, false negatives = 0,
      false positives = 0
    random non-burst, 256 cases:
      threshold 0.01 accepted total = 0
      best residual min = 0.05996

  gamma 2.0, threshold 0.01 exact scoring, burst_count 12,
  filler_multiplier 2, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0508

  gamma 2.5, threshold margin sweep, high-mismatch cases 0-3:
    threshold 0.01: alignment exact 3 / 4, false negatives = 2,
      false positives = 0
    random non-burst, 256 cases:
      threshold 0.01 accepted total = 0
      best residual min = 0.06282
    interpretation: under the current diagnostic threshold set, gamma 2.5
      hits a public AISB acquisition boundary before owner/wrong exact scoring.

  gamma 2.5, extended threshold diagnostic:
    threshold 0.0125: alignment exact 4 / 4, false negatives = 0,
      false positives = 0
    random non-burst, 256 cases:
      threshold 0.0125 accepted total = 0

  gamma 2.5, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 2, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0465

  gamma 3.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 2, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0448

  gamma 5.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 2, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0372

  gamma 8.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 2, case 0:
    pass = 1 / 1
    owner message recovery = 1 / 1
    margin = 0.0298

  gamma 10.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 2, case 0:
    pass = 0 / 1
    owner message recovery = 1 / 1
    margin = 0.0187
    failed pass condition: owner-vs-wrong margin <= 0.02

  gamma 10.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 3, cases 0-11:
    pass = 12 / 12, alignment exact = 12 / 12
    owner message recovery = 12 / 12
    minimum margin = 0.0407

  gamma 15.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 3, message_space_size 64, wrong_key_count 64,
  workers 4, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0406

  gamma 15.0, public threshold-margin probe:
    threshold 0.0125 / 0.02 high-mismatch alignment = 4 / 4
    random non-burst false positives = 0 / 256
    threshold 0.03 produces 6 / 256 random false positives

  gamma 15.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 3, message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0384

  gamma 20.0, public threshold-margin probe:
    threshold 0.0125 high-mismatch alignment = 4 / 4
    threshold 0.0125 random non-burst false positives = 0 / 256
    threshold 0.02 produces 1 / 256 random false positives

  gamma 20.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 3, message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0293

  gamma 30.0, public threshold-margin probe:
    threshold 0.0105 high-mismatch alignment = 4 / 4
    threshold 0.0105 random non-burst false positives = 0 / 256
    threshold 0.0125 produces 2 / 256 random false positives
    threshold 0.01 produces 3 true-burst false negatives

  gamma 30.0, threshold 0.0105 exact scoring, burst_count 12,
  filler_multiplier 3, message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0286

  gamma 50.0, public threshold-margin probe:
    no checked single residual threshold simultaneously keeps all 4
    high-mismatch true-burst cases and rejects all 256 random non-burst cases
    threshold 0.006 rejects random non-bursts but creates 12 true-burst false negatives
    threshold 0.008 creates 8 true-burst false negatives and 1 random false positive
    threshold 0.0125 keeps all true bursts but creates 2 random false positives

  gamma 50.0, public sequence-consistency probe:
    min_sequence_support = 12 public template-cycle bursts
    threshold 0.0125 sequence alignment = 4 / 4
    threshold 0.0125 residual random false positives = 2 / 256
    threshold 0.0125 sequence random false positives = 0 / 256
    threshold 0.02 residual random false positives = 20 / 256
    threshold 0.02 sequence random false positives = 0 / 256

  gamma 50.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 3, message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0301

  gamma 50.0, sequence-supported exact scoring, threshold 0.0125,
  burst_count 12, filler_multiplier 3, min_sequence_support 12,
  message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0562

  gamma 100.0, public sequence-consistency probe:
    min_sequence_support = 12 public template-cycle bursts
    threshold 0.0125 sequence alignment = 4 / 4
    threshold 0.0125 residual random false positives = 6 / 256
    threshold 0.0125 sequence random false positives = 0 / 256
    threshold 0.03 residual random false positives = 52 / 256
    threshold 0.03 sequence random false positives = 0 / 256

  gamma 100.0, threshold 0.0125 exact scoring, burst_count 12,
  filler_multiplier 3, message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0357

  gamma 100.0, sequence-supported exact scoring, threshold 0.0125,
  burst_count 12, filler_multiplier 3, min_sequence_support 12,
  message_space_size 24, wrong_key_count 12, cases 0-2:
    pass = 3 / 3, alignment exact = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0585

  gamma 50.0, sequence-supported exact scoring, same tier, cases 0-11:
    pass = 9 / 12, alignment exact = 9 / 12
    owner message recovery = 12 / 12
    minimum margin among owner-positive cases = 0.0534
    failure mode = public alignment non-uniqueness, not owner/wrong scoring

  gamma 50.0, sequence-ambiguity exact scoring, threshold 0.0125,
  near_tie_ratio 3.0, burst_count 12, filler_multiplier 3,
  min_sequence_support 12, message_space_size 24, wrong_key_count 12,
  cases 0-11:
    pass = 12 / 12
    truth sequence in public ambiguity set = 12 / 12
    owner message recovery = 12 / 12
    ambiguity sequence count range = 1 .. 3
    minimum margin = 0.0431

  gamma 100.0, sequence-ambiguity exact scoring, same tier:
    pass = 12 / 12
    truth sequence in public ambiguity set = 12 / 12
    owner message recovery = 12 / 12
    ambiguity sequence count range = 1 .. 4
    minimum margin = 0.0473

  gamma 50.0, sequence-ambiguity exact scoring, same 24-message /
  12-wrong-key tier, cases 0-23:
    pass = 24 / 24
    truth sequence in public ambiguity set = 24 / 24
    owner message recovery = 24 / 24
    ambiguity sequence count range = 1 .. 4
    minimum margin = 0.0240

  gamma 100.0, sequence-ambiguity exact scoring, same 24-message /
  12-wrong-key tier, cases 0-23:
    pass = 24 / 24
    truth sequence in public ambiguity set = 24 / 24
    owner message recovery = 24 / 24
    ambiguity sequence count range = 1 .. 4
    minimum margin = 0.0227

  gamma 50.0, sequence-ambiguity exact scoring, full 64-message /
  64-wrong-key tier, cases 0-2, workers 4:
    pass = 3 / 3
    truth sequence in public ambiguity set = 3 / 3
    owner message recovery = 3 / 3
    ambiguity sequence count range = 1 .. 2
    minimum margin = 0.0465

  gamma 100.0, sequence-ambiguity exact scoring, full 64-message /
  64-wrong-key tier, cases 0-2, workers 4:
    pass = 3 / 3
    truth sequence in public ambiguity set = 3 / 3
    owner message recovery = 3 / 3
    ambiguity sequence count range = 1 .. 2
    minimum margin = 0.0466

  gamma 50.0, sequence-ambiguity exact scoring, full 64-message /
  64-wrong-key tier, cases 3-5, workers 4, filler_multiplier 3:
    pass = 2 / 3
    truth sequence in public ambiguity set = 3 / 3
    owner message recovery = 3 / 3
    failure mode = case 5 owner-vs-wrong margin 0.0194 below diagnostic 0.02

  gamma 50.0, same full tier, cases 3-5, workers 4, filler_multiplier 4:
    pass = 3 / 3
    truth sequence in public ambiguity set = 3 / 3
    owner message recovery = 3 / 3
    minimum margin = 0.0206

  gamma 100.0, same full tier, cases 3-5, workers 4, filler_multiplier 4:
    pass = 2 / 3
    truth sequence in public ambiguity set = 3 / 3
    owner message recovery = 3 / 3
    failure mode = case 3 owner-vs-wrong margin 0.0195 below diagnostic 0.02

  gamma 100.0, full tier, case 3, filler_multiplier 5:
    current exact Python scorer exceeded 5 minutes and was manually interrupted
    before result; this is a CPU runtime bottleneck for longer sequence evidence,
    not a GPU/video boundary

  gamma 100.0, full tier, case 3, filler_multiplier 5, ordered_bounded scorer:
    exact-preserving cheap ordering plus bounded-DTW abandon again exceeded
    5 minutes and was manually interrupted before result

  gamma 100.0, full tier, case 3, filler_multiplier 5, ordered_bounded scorer
  after pure-Python DP hot-path optimization:
    again exceeded 5 minutes and was manually interrupted before result

  gamma 100.0, full tier, cases 0-11, filler_multiplier 5,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 12 / 12
    truth sequence in public ambiguity set = 12 / 12
    owner message recovery = 12 / 12
    minimum margin = 0.0359

  gamma 150.0, full tier, cases 0-2, filler_multiplier 5,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 3 / 3
    minimum margin = 0.0491

  gamma 200.0, full tier, cases 0-2, filler_multiplier 5,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 3 / 3
    minimum margin = 0.0491

  gamma 500.0, full tier, cases 0-11, filler_multiplier 5,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 12 / 12
    truth sequence in public ambiguity set = 12 / 12
    owner message recovery = 12 / 12
    minimum margin = 0.0336

  gamma 500.0, full tier, cases 12-14, filler_multiplier 5:
    case 12 owner/message recovered and truth is in ambiguity set, but margin
    is 0.0165 below the diagnostic 0.02 line

  gamma 500.0, full tier, cases 12-23, filler_multiplier 6,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 12 / 12
    truth sequence covered by public ambiguity set = 12 / 12
    exact truth sequence in public ambiguity set = 11 / 12
    owner message recovery = 12 / 12
    minimum margin = 0.0201
    note = case 20 has one extra public false burst in the covering sequence,
           but owner/wrong scoring over that same contaminated public alignment
           still separates with margin 0.1059

  gamma 500.0, full tier, cases 24-63, filler_multiplier 6,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 39 / 40
    owner message recovery = 40 / 40
    global owner recovery = 40 / 40
    truth sequence covered by public ambiguity set = 39 / 40
    exact truth sequence in public ambiguity set = 34 / 40
    minimum passing margin = 0.0283
    failing case = case 54
    failure mode = owner/message/global owner all recover, but the public
                   ambiguity set does not cover the true burst sequence

  gamma 500.0, full tier, cases 52-55, filler_multiplier 7,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 4 / 4
    truth sequence covered by public ambiguity set = 4 / 4
    exact truth sequence in public ambiguity set = 3 / 4
    owner message recovery = 4 / 4
    minimum margin = 0.0388
    case 54 margin = 0.0756

  gamma 500.0, full tier, cases 0-12, filler_multiplier 7,
  near_tie_ratio 5.0, ordered_bounded_c:
    pass = 13 / 13
    truth sequence covered by public ambiguity set = 13 / 13
    owner message recovery = 13 / 13
    minimum margin = 0.0351

  gamma 500.0, full tier, completed cases 0-63 except case 24,
  filler_multiplier 7, near_tie_ratio 5.0, ordered_bounded_c:
    completed cases = 63 / 64
    pass among completed cases = 63 / 63
    owner message recovery among completed cases = 63 / 63
    global owner recovery among completed cases = 63 / 63
    truth sequence covered by public ambiguity set = 63 / 63
    exact truth sequence in public ambiguity set = 53 / 63
    minimum completed margin = 0.0284
    uncompleted case = case 24
    uncompleted mode = local CPU single-case runtime exceeded 5 minutes before
                       JSON result; no synthetic failure was observed

  follow-up case 24 runtime diagnostic:
    public acquisition scanned candidates = 1221
    public ambiguity sequence count = 24
    candidate count per sequence set = 4160
    truth sequence exact in ambiguity = true
    truth sequence covered by ambiguity = true
    score workload = 24 * 4160 exact DTW candidates
    ordered_bounded_c with sequence-level workers=8, prepared native C arrays,
    and per-sequence reusable native DP workspaces still exceeded a 3-minute
    local CPU window before JSON result; this remains a CPU scoring throughput
    issue, not an observed scientific or GPU/video failure.
    ordered_bounded_global_c, which globally orders all sequence/candidate
    pairs before the same exact bounded native DTW, also exceeded a 3-minute
    local CPU window before JSON result after matching the exact winner in the
    small regression case.
    stronger bounded-DTW lower bound:
      accounts for unavoidable future skip cost and maximum possible final
      diagonal-match count;
      matches Python/native exact-scorer regressions;
      margin_proof_c still exceeded a 3-minute local CPU window on full-tier
      case 24.
    reduced case 24 diagnostic with the same gamma/filler but
    message_space_size=24 and wrong_key_count=12 completed in 3.97 seconds:
      synthetic_construction_pass = true
      ambiguity_sequence_count = 1
      truth sequence exact/covered = true/true
      owner message/global owner recovered = true/true
      exact score margin = 0.1472903461
    This supports the interpretation that case 24 is not a seed-specific
    scientific failure; the unresolved full-tier issue is CPU exact-scoring
    scale at 64 messages, 64 wrong keys, and 24 public ambiguity sequences.

  gamma 500.0, public sequence-consistency check:
    high mismatch sequence alignment exact = 4 / 4
    random non-burst residual false positives = 1 / 64
    random non-burst sequence false positives = 0 / 64
```

Interpretation: at gamma 1.4, public AISB acquisition can still be recovered
with the diagnostic threshold 0.01 and random non-burst rejection remains clean
in the checked synthetic distribution. The active failure moves to owner /
wrong-key separability margin, with a fine synthetic transition between gamma
1.35 and 1.36 under the current 0.02 diagnostic pass boundary. The first local
repair signal is clear: increasing secret-state span restores margin across
the first 12 checked gamma 1.4 cases and the first 12 checked gamma 1.6 cases,
while increasing public AISB burst count alone does not. The pruned two-stage
screen is explicitly retained as a negative acceleration diagnostic because it
can miss the exact owner/message winner here. This is a CPU synthetic boundary
and mechanism-design finding, not a GPU or video-model boundary. The current
checked stronger-mismatch envelope reaches gamma 5.0 with secret-state
filler_multiplier 2 and a diagnostic residual threshold 0.0125 on cases 0-2.
Gamma 8.0 case 0 remains positive but close to the margin boundary; gamma 10.0
with filler_multiplier 2 recovers the owner message but fails the owner-vs-wrong
diagnostic margin. Increasing the secret-state filler span to 3 restores gamma
10.0 cases 0-11 in the full 64-message / 64-wrong-key exact tier. At gamma 15.0
and above the full exact tier is too slow for rapid CPU iteration in the current
Python diagnostic. A conservative bounded-DTW early-abandon scorer was added and
validated against exact scores; process-level parallelism then made gamma 15.0
full 64-message / 64-wrong-key cases 0-2 runnable locally with 4 workers, all
passing with minimum margin 0.0406. The still-faster boundary checks use a
smaller 24-message / 12-wrong-key exact tier. That smaller tier remains positive
at gamma 15.0, 20.0, and 30.0 with filler_multiplier 3, but gamma 30.0 already
requires a narrower public residual threshold 0.0105. Gamma 50.0 exposes the
current public AISB acquisition boundary for single-burst residual gating:
checked true-burst residuals and checked random non-burst residuals overlap, so
a single residual threshold cannot satisfy both zero false negatives and zero
false positives in this synthetic distribution. Adding a public template-sequence support
constraint changes the result: at gamma 50.0 and 100.0, requiring support for
the 12-burst public template cycle preserves checked true sequences while
filtering isolated residual false positives. When one public sequence-supported
alignment is frozen before affine calibration, the smaller exact owner/wrong
tier remains positive on cases 0-2, but gamma 50.0 cases 0-11 expose public
alignment non-uniqueness: owner-message recovery stays 12/12 while exact public
alignment drops to 9/12. Carrying a small public ambiguity set through affine
calibration/equalization and scoring owner/wrong keys over the same alignment
set closes the checked gamma 50.0 and 100.0 cases 0-23 in the 24-message /
12-wrong-key tier and cases 0-2 in the full 64-message / 64-wrong-key tier.
Expanding the full tier to cases 3-5 shows the next active issue is again
owner-vs-wrong evidence length/margin: gamma 50.0 is restored by increasing
secret-state filler span from 3 to 4, while gamma 100.0 still has one case just
below the diagnostic 0.02 line at filler 4. Pushing that specific case to
filler 5 is currently blocked by Python CPU runtime in the exact scorer rather
than by GPU/video needs. An ordered bounded scorer was added as a safe CPU
diagnostic: it sorts candidates by a cheap decimated score and then still uses
exact bounded DTW for every accepted score. On the checked 24-message /
12-wrong-key case it matches the exact winner, owner score, wrong score, and
margin while abandoning 40 candidates; on the full 64-message / 64-wrong-key
gamma 100.0 filler 5 case it still does not finish within 5 minutes. A
pure-Python DP hot-path optimization removes per-cell tuple construction and
preserves the original transition tie-break order; it cuts the checked
ordered-bounded equivalence regression from roughly 63 seconds to roughly
29 seconds, but still does not make the full filler-5 case finish within
5 minutes. A local C implementation of the same flattened exact bounded-DTW
recurrence then removes that Python wall-time blocker without changing the
scientific decision rule. With `ordered_bounded_c`, gamma 100.0 full-tier cases
0-11 pass at filler_multiplier 5 and near_tie_ratio 5.0. At gamma 500.0, cases
0-11 pass with filler_multiplier 5, while cases 12-23 require
filler_multiplier 6 to restore margin. Extending filler_multiplier 6 through a
full 64-message-index synthetic round gives 63/64 passes: all 64 cases recover
the owner message and global owner, but case 54 fails because public acquisition
does not cover the true sequence. Raising the span to filler_multiplier 7
restores checked cases 52-55, including exact coverage for case 54, and extends
to 63/64 completed cases across the full message-index round with all completed
cases passing. The one missing case, case 24, exceeded a 5-minute local CPU
single-case runtime window and remains a CPU runtime slow case, not an observed
scientific or GPU/video failure. This makes the active boundary public
acquisition support plus CPU runtime for dense ambiguity cases, not GPU/video
runtime and not owner/wrong-key scoring. The case 20 diagnostic distinguishes
exact public truth recovery from public truth coverage: the exact truth set is
not one returned sequence, but it is fully covered by a sequence with one extra
public false burst, and owner/wrong scoring over that same contaminated public
alignment still passes. The gamma 500.0 public
sequence-consistency diagnostic still rejects random non-burst sequences after
the public support constraint, even though isolated residual false positives
appear. This confirms that the current productive path remains CPU-only
synthetic mechanism testing, not GPU/video execution.
This is a synthetic mechanism/evidence-length and public-acquisition
diagnostic, not a GPU boundary or fixed-FPR result.


## 2026-08-01 pass 9: overlap tie-break and non-affine mismatch diagnostic

Command:

```bash
python3 experiments/run_aisb_channel_mismatch_probe.py
```

Implementation correction:

- The redundant 9-point AISB template intentionally uses exact public redundant
  anchor copies. Under deletion-aware scans, that creates a finite-noise
  ambiguity: a shifted `start+1` window or a same-start complete window can
  occasionally have slightly lower residual than the true deletion candidate.
- The candidate selector now groups overlapping low-residual windows. Inside a
  near-tie residual band, it applies a public deterministic tie-break that
  prefers the earliest start and then the explicit deletion-aware candidate over
  a same-start complete-window candidate. This closes the observed shifted-window
  ambiguity without estimating `A_x,b_x` during acquisition.

Result summary:

```text
quadratic channel mismatch, gamma = 0.5:
  pass_count = 12 / 12
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  owner_message_recovery_count = 12 / 12
  score_margin_mean = 0.105

random_non_burst_cases:
  pass_count = 64 / 64
  accepted_count_total = 0
  best_residual_min = 0.107

gamma_margin_diagnostic:
  gamma_0.0 pass_count = 6 / 6
  gamma_0.3 pass_count = 6 / 6
  gamma_0.5 pass_count = 6 / 6
  gamma_1.0 pass_count = 5 / 6
```

Interpretation:

- The complete synthetic mechanism loop remains viable under a deterministic
  quadratic observation-space perturbation at `gamma = 0.5`.
- The `gamma = 1.0` diagnostic exposes a remaining high-mismatch robustness
  boundary. This is not threshold calibration and not a fixed-FPR result.
- The correction is still a public acquisition rule. It does not use owner key,
  message, affine channel estimates, or output scoring to choose burst windows.

Evidence boundary remains unchanged: synthetic affine-channel diagnostics only;
no real video observations, no GPU, no fixed-FPR claim, and no paper claim.


## 2026-08-01 pass 8: AISB public alignment plus payload/message scoring

Command:

```bash
python3 experiments/run_aisb_payload_probe.py
```

Result summary:

```text
payload_cases at noise_std = 0.016:
  pass_count = 12 / 12
  alignment_accuracy_mean = 1.0
  false_positive = 0
  false_negative = 0
  owner_message_recovery_count = 12 / 12
  score_margin_mean = 0.143
  score_margin_min = 0.0796
  state_reconstruction_mse_mean = 0.000134

noise_margin_diagnostic:
  noise_0.012 pass_count = 6 / 6
  noise_0.016 pass_count = 6 / 6
  noise_0.020 pass_count = 5 / 6, false_negative = 1
```

Interpretation:

- After public AISB alignment is accepted and frozen, the synthetic affine
  channel is calibrated once from public burst pairs and reused for all keys.
- Owner and wrong keys are scored over the same fixed 8-message candidate set.
  The owner key recovers the true message in all current `noise_std = 0.016`
  payload cases and remains separated from the best wrong-key/message score.
- This closes the current CPU-only mechanism loop:

```text
AISB acquisition
-> freeze public alignment
-> estimate A_x,b_x
-> equalize relation observations
-> key/message-conditioned state synchronization
```

- The `noise_std = 0.020` margin diagnostic still shows a finite acquisition
  robustness boundary. It is not a threshold-tuning or fixed-FPR result.

Evidence boundary remains unchanged: synthetic affine-channel diagnostics only;
no real video observations, no GPU, no fixed-FPR claim, and no paper claim.


## 2026-08-02 pass 15: expanded ambiguity-set stress grid

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 0.5,0.8,1.0 \
  --noise-stds 0.012,0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 4 \
  --ambiguity-message-space-size 16 \
  --ambiguity-wrong-key-count 24 \
  --ambiguity-case-count 4
```

Result summary:

```text
single_path:
  gamma_0.5_noise_0.012: pass 4 / 4, alignment 4 / 4, owner 4 / 4, min_margin 0.03245
  gamma_0.5_noise_0.016: pass 4 / 4, alignment 4 / 4, owner 4 / 4, min_margin 0.03230
  gamma_0.8_noise_0.012: pass 4 / 4, alignment 4 / 4, owner 4 / 4, min_margin 0.02482
  gamma_0.8_noise_0.016: pass 3 / 4, alignment 3 / 4, owner 4 / 4, min_margin 0.02473
  gamma_1.0_noise_0.012: pass 2 / 4, alignment 3 / 4, owner 4 / 4, min_margin 0.01600
  gamma_1.0_noise_0.016: pass 2 / 4, alignment 3 / 4, owner 4 / 4, min_margin 0.01591

ambiguity_set, 16 messages / 24 wrong keys / 4 cases per cell:
  gamma_0.5_noise_0.012: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, min_margin 0.26845
  gamma_0.5_noise_0.016: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, min_margin 0.26651
  gamma_0.8_noise_0.012: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, min_margin 0.21481
  gamma_0.8_noise_0.016: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, min_margin 0.21500
  gamma_1.0_noise_0.012: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, min_margin 0.19274
  gamma_1.0_noise_0.016: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, min_margin 0.19356
```

Interpretation:

- The expanded grid confirms the previous reduced-grid diagnosis: forcing one
  public alignment fails first under stronger non-affine/noise stress, while
  owner-message recovery remains possible.
- Carrying the public ambiguity set through calibration/equalization/scoring
  restores every checked 16-message / 24-wrong-key cell through gamma 1.0 and
  noise 0.016, with margins far above the 0.02 diagnostic line.
- The next CPU-only question is scale and coverage, not GPU execution:
  increase ambiguity-set case count and candidate-space size until the local
  exact scorer reaches a practical CPU limit or exposes a scientific failure.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 25: full-tier gamma 15.0/20.0 hard-cell check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 15.0,20.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1
```

Result summary:

```text
single_path:
  gamma_15.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 7 / 8, min_margin -0.05097
  gamma_20.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 7 / 8, min_margin -0.05085

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_15.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.04351
  gamma_20.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.03753
```

Interpretation:

- The full-tier ambiguity-set route remains positive at gamma 15.0 and 20.0
  across the checked 8 cases.
- The margin is still above the 0.02 diagnostic line, but the downward trend is
  real and should be mapped with further CPU-only stress before any GPU/video
  work.
- Forced single-path public alignment continues to fail and remains only a
  negative diagnostic.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.







## 2026-08-04 pass 240: burst16 noise0.62 threshold0.013125 diagnostic-pruned full layer

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.013125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,16,24,32,40,48,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Layer result summary:

```text
cases 0-7:   pass 8 / 8, min_margin 0.08909555, mean_margin 0.12647397, max_exact 22174
cases 8-15:  pass 8 / 8, min_margin 0.09989343, mean_margin 0.12248513, max_exact 38391
cases 16-23: pass 8 / 8, min_margin 0.06739550, mean_margin 0.10103453, max_exact 14931
cases 24-31: pass 8 / 8, min_margin 0.08525248, mean_margin 0.11799040, max_exact 44356
cases 32-39: pass 8 / 8, min_margin 0.07058426, mean_margin 0.11925992, max_exact 9218
cases 40-47: pass 8 / 8, min_margin 0.04831724, mean_margin 0.11199554, max_exact 21705
cases 48-55: pass 8 / 8, min_margin 0.06156125, mean_margin 0.09380282, max_exact 32982
cases 56-63: pass 8 / 8, min_margin 0.06733981, mean_margin 0.11483182, max_exact 37450

total pass 64 / 64
truth covered 64 / 64
owner/global recovery 64 / 64
min_margin 0.04831724
mean_margin 0.11348427
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 44356
```

Interpretation:

- residual_threshold 0.013125 is also sufficient for the checked noise_std 0.62
  layer.
- This tightens the lower bound above the failing 0.0125 threshold.
- The worst-case exact-score count remains 44356, so this threshold does not
  remove the observed CPU burden, although some segments are cheaper than at
  0.01375 / 0.015.
- The next CPU-only boundary check is a still narrower threshold above 0.0125,
  or direct per-case diagnosis of the cases that drive exact-score count 44356.
- This remains rapid CPU-only synthetic mechanism triage, not exhaustive
  wrong-key, fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.

## 2026-08-04 pass 239: burst16 noise0.62 threshold0.01375 diagnostic-pruned full layer

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.01375 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,16,24,32,40,48,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Layer result summary:

```text
cases 0-7:   pass 8 / 8, min_margin 0.09964987, mean_margin 0.12655882, max_exact 22174
cases 8-15:  pass 8 / 8, min_margin 0.09989343, mean_margin 0.12183782, max_exact 38391
cases 16-23: pass 8 / 8, min_margin 0.06739550, mean_margin 0.10118665, max_exact 14931
cases 24-31: pass 8 / 8, min_margin 0.08525248, mean_margin 0.11799040, max_exact 44356
cases 32-39: pass 8 / 8, min_margin 0.07058426, mean_margin 0.12424039, max_exact 12870
cases 40-47: pass 8 / 8, min_margin 0.04831724, mean_margin 0.11399906, max_exact 33495
cases 48-55: pass 8 / 8, min_margin 0.06156125, mean_margin 0.09430532, max_exact 33681
cases 56-63: pass 8 / 8, min_margin 0.06733981, mean_margin 0.11478596, max_exact 37450

total pass 64 / 64
truth covered 64 / 64
owner/global recovery 64 / 64
min_margin 0.04831724
mean_margin 0.11436305
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 44356
```

Interpretation:

- residual_threshold 0.01375 is sufficient for the checked noise_std 0.62
  layer and restores the 0.0125 truth-coverage failure.
- It does not materially reduce the worst-case exact-score count relative to
  threshold 0.015 on this layer; both peak at 44356.
- The next useful CPU-only refinement is to test a narrower threshold just above
  0.0125, or to map whether the 44356 exact-score peak is tied to specific
  cases rather than the threshold alone.
- This remains rapid CPU-only synthetic mechanism triage, not exhaustive
  wrong-key, fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.

## 2026-08-04 pass 238: burst16 noise0.62 threshold0.015 diagnostic-pruned full layer

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.015 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,16,24,32,40,48,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Layer result summary:

```text
cases 0-7:   pass 8 / 8, min_margin 0.09964987, mean_margin 0.12578026, max_exact 22174
cases 8-15:  pass 8 / 8, min_margin 0.09765906, mean_margin 0.12227069, max_exact 38391
cases 16-23: pass 8 / 8, min_margin 0.06739550, mean_margin 0.10118665, max_exact 14931
cases 24-31: pass 8 / 8, min_margin 0.08525248, mean_margin 0.11804630, max_exact 44356
cases 32-39: pass 8 / 8, min_margin 0.07058426, mean_margin 0.12439181, max_exact 12870
cases 40-47: pass 8 / 8, min_margin 0.04831724, mean_margin 0.11398609, max_exact 33495
cases 48-55: pass 8 / 8, min_margin 0.06156125, mean_margin 0.09430532, max_exact 33681
cases 56-63: pass 8 / 8, min_margin 0.06733981, mean_margin 0.11477806, max_exact 37450

total pass 64 / 64
truth covered 64 / 64
owner/global recovery 64 / 64
min_margin 0.04831724
mean_margin 0.11434315
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 44356
```

Interpretation:

- Raising the residual threshold from 0.0125 to 0.015 restores the full
  checked noise_std 0.62 layer under the current diagnostic-pruned criterion.
- This supports the diagnosis that the 0.0125 failure was a public
  AISB/sequence-coverage threshold boundary, not an owner/wrong-key scoring
  collapse.
- The cost increased materially: max exact-score count rose to 44356. The next
  CPU-only question is whether a lower threshold between 0.0125 and 0.015 can
  keep truth coverage while reducing ambiguity/exact work.
- This remains rapid CPU-only synthetic mechanism triage, not exhaustive
  wrong-key, fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.

## 2026-08-04 pass 237: burst16 noise0.62 diagnostic-pruned boundary case

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std 0.62 cases 0-7:
  pass 7 / 8
  truth covered 7 / 8
  owner/global recovery 8 / 8
  min_margin 0.08909555
  mean_margin 0.12425739
  max_ambiguity_sequence_count 96
  max_exact_score_count_after_diagnostic_screen 14780

failed case:
  case_index 4
  truth_sequence_covered_by_ambiguity false
  truth_sequence_indices []
  global_best_role owner
  global_best_message message_4
  true_message message_4
  score_margin 0.11665732
  ambiguity_sequence_count 1
  exact_score_count 154
```

Diagnostic reruns for the failed case:

```text
residual_threshold 0.0125, diagnostic_top_k 256:
  pass false, truth_covered false, owner message recovered, margin 0.09298596

residual_threshold 0.015, diagnostic_top_k 256:
  pass true, truth_covered true, owner message recovered, margin 0.09689029, ambiguity 8

residual_threshold 0.02, diagnostic_top_k 256:
  pass true, truth_covered true, owner message recovered, margin 0.09689029, ambiguity 8

residual_threshold 0.03, diagnostic_top_k 256:
  pass true, truth_covered true, owner message recovered, margin 0.09689029, ambiguity 12
```

Interpretation:

- The active noise_std 0.62 boundary is a public AISB/sequence-coverage
  threshold boundary, not an owner/wrong-key scoring failure.
- The owner/global message still recovers correctly in the failed case, but the
  current pass criterion requires public truth-sequence coverage, so the tier is
  correctly marked failed.
- Raising the public residual threshold from 0.0125 to 0.015 restores this
  failed case in CPU-only diagnostics, but it also increases ambiguity/exact
  work. The next scientific step is therefore CPU-only threshold/ambiguity
  tradeoff mapping, not GPU.
- This is not a video, GPU, fixed-FPR, governance, reproducibility, or paper
  result.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.

## 2026-08-04 pass 236: burst16 noise0.60 diagnostic-pruned full layer

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.60 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,16,24,32,40,48,56> \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise060_burst16_top2_diag_owner64_cases<start>_<end>_workers1_cases.jsonl
```

Layer result summary:

```text
cases 0-7:   pass 8 / 8, min_margin 0.08737220, mean_margin 0.12666053, max_exact 14797
cases 8-15:  pass 8 / 8, min_margin 0.09648578, mean_margin 0.12347736, max_exact 22972
cases 16-23: pass 8 / 8, min_margin 0.07097559, mean_margin 0.10218365, max_exact 7449
cases 24-31: pass 8 / 8, min_margin 0.08964361, mean_margin 0.12084982, max_exact 22156
cases 32-39: pass 8 / 8, min_margin 0.06964755, mean_margin 0.12055351, max_exact 5551
cases 40-47: pass 8 / 8, min_margin 0.04807966, mean_margin 0.11439331, max_exact 16727
cases 48-55: pass 8 / 8, min_margin 0.06032519, mean_margin 0.09422267, max_exact 22075
cases 56-63: pass 8 / 8, min_margin 0.06802977, mean_margin 0.11343750, max_exact 29977

total pass 64 / 64
truth covered 64 / 64
owner/global recovery 64 / 64
min_margin 0.04807966
mean_margin 0.11447229
max_ambiguity_sequence_count 192
max_exact_score_count_after_diagnostic_screen 29977
```

Interpretation:

- The noise_std 0.60 diagnostic-pruned tier remains positive across the full
  checked 64-case layer.
- The lowest margin remains at cases 40-47, but it is still positive under the
  current diagnostic-pruned criterion.
- The maximum exact-score count rose to 29977; CPU-only runtime and local native
  helper throughput are becoming practical constraints, but no scientific
  failure or GPU/video boundary was reached.
- During this layer, multi-worker native helper compilation showed a local race;
  the completed evidence used single-worker reruns where needed. This is a CPU
  infrastructure issue, not a method failure.
- This remains rapid CPU-only synthetic mechanism triage, not exhaustive
  wrong-key, fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.

## 2026-08-04 pass 235: burst16 noise0.58 diagnostic-pruned full layer

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.58 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index <0,8,16,24,32,40,48,56> \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise058_burst16_top2_diag_owner64_cases<start>_<end>_workers4_cases.jsonl
```

Layer result summary:

```text
cases 0-7:   pass 8 / 8, min_margin 0.09066536, mean_margin 0.12749145, max_exact 14757
cases 8-15:  pass 8 / 8, min_margin 0.09670570, mean_margin 0.12543780, max_exact 22969
cases 16-23: pass 8 / 8, min_margin 0.07675474, mean_margin 0.10487223, max_exact 7479
cases 24-31: pass 8 / 8, min_margin 0.09429304, mean_margin 0.12306625, max_exact 22160
cases 32-39: pass 8 / 8, min_margin 0.07471587, mean_margin 0.12343519, max_exact 5539
cases 40-47: pass 8 / 8, min_margin 0.04732445, mean_margin 0.11718482, max_exact 16718
cases 48-55: pass 8 / 8, min_margin 0.06536369, mean_margin 0.09585849, max_exact 22072
cases 56-63: pass 8 / 8, min_margin 0.06759679, mean_margin 0.11340797, max_exact 15021

total pass 64 / 64
truth covered 64 / 64
owner/global recovery 64 / 64
min_margin 0.04732445
mean_margin 0.11634427
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22969
```

Interpretation:

- The noise_std 0.58 diagnostic-pruned tier remains positive across the full
  checked 64-case layer.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This layer was completed under the faster exploration cadence: per-segment
  pass/fail was checked immediately, while documentation and validation are
  consolidated at the complete noise-layer boundary.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 234: burst16 noise0.58 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.58 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise058_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07675474
mean_margin 0.10487223
max_ambiguity_sequence_count 48
max_exact_score_count_after_diagnostic_screen 7479
```

Interpretation:

- The noise_std 0.58 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 233: burst16 noise0.58 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.58 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise058_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09670570
mean_margin 0.12543780
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22969
```

Interpretation:

- The noise_std 0.58 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 232: burst16 noise0.58 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.58 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise058_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09066536
mean_margin 0.12749145
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14757
```

Interpretation:

- Raising to noise_std 0.58, the diagnostic-pruned tier is positive on the
  first checked segment, cases 0-7.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 231: burst16 noise0.56 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06647356
mean_margin 0.11441890
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15022
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 230: burst16 noise0.56 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07028426
mean_margin 0.09768298
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22063
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 229: burst16 noise0.56 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05104159
mean_margin 0.11930278
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16733
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 228: burst16 noise0.56 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07365479
mean_margin 0.12607518
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5532
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 227: burst16 noise0.56 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09902808
mean_margin 0.12300990
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22149
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 226: burst16 noise0.56 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07814625
mean_margin 0.10672219
max_ambiguity_sequence_count 48
max_exact_score_count_after_diagnostic_screen 7517
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 225: burst16 noise0.56 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09464360
mean_margin 0.12665552
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22996
```

Interpretation:

- The noise_std 0.56 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 224: burst16 noise0.56 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.56 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise056_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09051082
mean_margin 0.12831955
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14738
```

Interpretation:

- Raising to noise_std 0.56, the diagnostic-pruned tier is positive on the
  first checked segment, cases 0-7.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 223: burst16 noise0.54 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06999606
mean_margin 0.11579392
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15026
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 222: burst16 noise0.54 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06931735
mean_margin 0.09915820
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22050
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 221: burst16 noise0.54 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.04712975
mean_margin 0.11869580
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16746
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 220: burst16 noise0.54 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08281960
mean_margin 0.12867356
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5525
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 219: burst16 noise0.54 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10307755
mean_margin 0.12497781
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22130
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 218: burst16 noise0.54 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08079990
mean_margin 0.11020319
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15042
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 217: burst16 noise0.54 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09827562
mean_margin 0.12811809
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23011
```

Interpretation:

- The noise_std 0.54 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 216: burst16 noise0.54 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.54 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise054_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09154538
mean_margin 0.12818281
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14701
```

Interpretation:

- Raising to noise_std 0.54, the diagnostic-pruned tier is positive on the
  first checked segment, cases 0-7.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 215: burst16 noise0.52 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07680597
mean_margin 0.11679914
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15034
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 214: burst16 noise0.52 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07073198
mean_margin 0.10280239
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22036
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 213: burst16 noise0.52 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05007786
mean_margin 0.12004527
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16712
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 212: burst16 noise0.52 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08209579
mean_margin 0.12913824
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5526
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 211: burst16 noise0.52 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10843663
mean_margin 0.12861571
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22084
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 210: burst16 noise0.52 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08655462
mean_margin 0.11291160
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15078
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 209: burst16 noise0.52 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09866961
mean_margin 0.12819033
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23027
```

Interpretation:

- The noise_std 0.52 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 208: burst16 noise0.52 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.52 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise052_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09044599
mean_margin 0.12942522
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14630
```

Interpretation:

- Raising to noise_std 0.52, the diagnostic-pruned tier is positive on the
  first checked 8-case slice.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 207: burst16 noise0.50 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07806989
mean_margin 0.11648695
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15012
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 206: burst16 noise0.50 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06911470
mean_margin 0.10373729
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22026
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 205: burst16 noise0.50 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05578955
mean_margin 0.12100324
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16692
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 204: burst16 noise0.50 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08194289
mean_margin 0.12897807
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5516
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 203: burst16 noise0.50 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10415634
mean_margin 0.12827687
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22021
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 202: burst16 noise0.50 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08730083
mean_margin 0.11343862
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15079
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 201: burst16 noise0.50 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09635263
mean_margin 0.12984420
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23057
```

Interpretation:

- The noise_std 0.50 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 200: burst16 noise0.50 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.50 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise050_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09375566
mean_margin 0.13039071
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14544
```

Interpretation:

- Raising to noise_std 0.50, the diagnostic-pruned tier is positive on the
  first checked 8-case slice.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 199: burst16 noise0.48 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08137164
mean_margin 0.11673163
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14960
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 198: burst16 noise0.48 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06854657
mean_margin 0.10487484
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22014
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 197: burst16 noise0.48 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05655783
mean_margin 0.12149102
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16671
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 196: burst16 noise0.48 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08064904
mean_margin 0.13022385
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5507
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 195: burst16 noise0.48 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10637933
mean_margin 0.12832311
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21979
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 194: burst16 noise0.48 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08788011
mean_margin 0.11302014
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15083
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 193: burst16 noise0.48 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09699572
mean_margin 0.13141568
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23124
```

Interpretation:

- The noise_std 0.48 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 192: burst16 noise0.48 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.48 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise048_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08923162
mean_margin 0.13121660
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14513
```

Interpretation:

- Raising to noise_std 0.48, the diagnostic-pruned tier is positive on the
  first checked 8-case slice.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 191: burst16 noise0.46 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07997671
mean_margin 0.11716957
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14923
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 190: burst16 noise0.46 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06771293
mean_margin 0.10520413
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22006
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 189: burst16 noise0.46 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05920578
mean_margin 0.12232096
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16682
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 188: burst16 noise0.46 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08296188
mean_margin 0.13540991
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5505
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 187: burst16 noise0.46 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10937305
mean_margin 0.13016290
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21985
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 186: burst16 noise0.46 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09003097
mean_margin 0.11576149
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15105
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 185: burst16 noise0.46 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09899634
mean_margin 0.13361676
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23175
```

Interpretation:

- The noise_std 0.46 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 184: burst16 noise0.46 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.46 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise046_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08912916
mean_margin 0.13253681
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14479
```

Interpretation:

- Raising to noise_std 0.46, the diagnostic-pruned tier is positive on the
  first checked 8-case slice.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 183: burst16 noise0.44 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07963960
mean_margin 0.11895991
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14894
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 182: burst16 noise0.44 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07073149
mean_margin 0.10676073
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21992
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 181: burst16 noise0.44 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06277592
mean_margin 0.12289586
max_ambiguity_sequence_count 108
max_exact_score_count_after_diagnostic_screen 16688
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 180: burst16 noise0.44 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08761476
mean_margin 0.13501623
max_ambiguity_sequence_count 36
max_exact_score_count_after_diagnostic_screen 5498
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 179: burst16 noise0.44 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11341359
mean_margin 0.13338902
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22020
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 178: burst16 noise0.44 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09127598
mean_margin 0.11725961
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15130
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 177: burst16 noise0.44 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09980410
mean_margin 0.13755346
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23153
```

Interpretation:

- The noise_std 0.44 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 176: burst16 noise0.44 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.44 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise044_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08862351
mean_margin 0.13445427
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14428
```

Interpretation:

- Raising to noise_std 0.44, the diagnostic-pruned tier is positive on the
  first checked 8-case slice.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 175: burst16 noise0.42 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08430650
mean_margin 0.12115983
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14863
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 174: burst16 noise0.42 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06861692
mean_margin 0.10785955
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21972
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 173: burst16 noise0.42 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06141315
mean_margin 0.12552489
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21766
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 172: burst16 noise0.42 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08782796
mean_margin 0.13505514
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3677
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 171: burst16 noise0.42 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11412904
mean_margin 0.13448986
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22043
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 170: burst16 noise0.42 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09197900
mean_margin 0.11638722
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15147
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 169: burst16 noise0.42 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09784793
mean_margin 0.13843671
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23116
```

Interpretation:

- The noise_std 0.42 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 168: burst16 noise0.42 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.42 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise042_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08818240
mean_margin 0.13545645
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14405
```

Interpretation:

- Raising synthetic noise to noise_std 0.42 remains positive on checked cases
  0-7 under the current burst16/top2/diagnostic-pruned C tier.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 167: burst16 noise0.40 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08638474
mean_margin 0.12246477
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14835
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 166: burst16 noise0.40 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06940009
mean_margin 0.10873880
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21945
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 165: burst16 noise0.40 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06297970
mean_margin 0.12678059
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21780
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 164: burst16 noise0.40 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08839395
mean_margin 0.13671151
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3677
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 163: burst16 noise0.40 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11608262
mean_margin 0.13603371
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 44276
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 162: burst16 noise0.40 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08691652
mean_margin 0.11581620
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15162
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 161: burst16 noise0.40 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09863218
mean_margin 0.14257307
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23094
```

Interpretation:

- The noise_std 0.40 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 160: burst16 noise0.40 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.40 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise040_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08821583
mean_margin 0.13579885
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21806
```

Interpretation:

- Raising synthetic noise to noise_std 0.40 remains positive on checked cases
  0-7 under the current burst16/top2/diagnostic-pruned C tier.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 159: burst16 noise0.38 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08474072
mean_margin 0.12241280
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 14816
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 158: burst16 noise0.38 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07151290
mean_margin 0.11014365
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21921
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 157: burst16 noise0.38 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06385964
mean_margin 0.12776847
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21815
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 156: burst16 noise0.38 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08942796
mean_margin 0.13875238
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3772
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 155: burst16 noise0.38 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11695454
mean_margin 0.13680789
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 44211
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 154: burst16 noise0.38 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08678848
mean_margin 0.11670245
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15162
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 153: burst16 noise0.38 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10081056
mean_margin 0.14179541
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23052
```

Interpretation:

- The noise_std 0.38 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 152: burst16 noise0.38 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.38 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise038_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09210281
mean_margin 0.13758768
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21804
```

Interpretation:

- Raising synthetic noise to noise_std 0.38 remains positive on checked
  cases 0-7 under the current burst16/top2/diagnostic-pruned C tier.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 151: burst16 noise0.36 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08903584
mean_margin 0.12310556
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18463
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 150: burst16 noise0.36 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06968034
mean_margin 0.11088204
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21908
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 149: burst16 noise0.36 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06148871
mean_margin 0.12946268
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21869
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 148: burst16 noise0.36 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09238279
mean_margin 0.14068361
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3762
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 147: burst16 noise0.36 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11924334
mean_margin 0.13706302
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 44200
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 146: burst16 noise0.36 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08668786
mean_margin 0.11755274
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15183
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 145: burst16 noise0.36 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10295556
mean_margin 0.14341460
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23015
```

Interpretation:

- The noise_std 0.36 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 144: burst16 noise0.36 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.36 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise036_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09361803
mean_margin 0.13794792
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21792
```

Interpretation:

- Raising synthetic noise to noise_std 0.36 remains positive on checked
  cases 0-7 under the current burst16/top2/diagnostic-pruned C tier.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 143: burst16 noise0.34 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09374521
mean_margin 0.12557854
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18475
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 56-63, closing the checked range at 64 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 142: burst16 noise0.34 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06498330
mean_margin 0.11168867
max_ambiguity_sequence_count 216
max_exact_score_count_after_diagnostic_screen 32875
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 48-55, extending the checked range to 56 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 141: burst16 noise0.34 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06057691
mean_margin 0.12947855
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21906
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 40-47, extending the checked range to 48 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 140: burst16 noise0.34 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09147009
mean_margin 0.14480315
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3753
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 32-39, extending the checked range to 40 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 139: burst16 noise0.34 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.12069871
mean_margin 0.13720627
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 43994
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 24-31, extending the checked range to 32 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 138: burst16 noise0.34 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08209735
mean_margin 0.11794515
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15193
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 16-23, extending the checked range to 24 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 137: burst16 noise0.34 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10929955
mean_margin 0.14608259
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23046
```

Interpretation:

- The noise_std 0.34 diagnostic-pruned tier remains positive through checked
  cases 8-15, extending the checked range to 16 / 64.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 136: burst16 noise0.34 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.34 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise034_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09776
mean_margin 0.13952
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21720
```

Interpretation:

- Raising synthetic noise to noise_std 0.34 remains positive on checked
  cases 0-7 under the current burst16/top2/diagnostic-pruned C tier.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 135: burst16 noise0.32 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09623
mean_margin 0.12674
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18478
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 56-63.
- This closes the checked noise_std 0.32 range at 64 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Across the closed noise_std 0.32 range, the tightest checked margin is case47
  at 0.06073; no CPU-only scientific failure or GPU/video boundary was
  reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 134: burst16 noise0.32 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07016
mean_margin 0.11161
max_ambiguity_sequence_count 216
max_exact_score_count_after_diagnostic_screen 32865
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 48-55.
- This extends the checked noise_std 0.32 range to 56 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 133: burst16 noise0.32 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.06073
mean_margin 0.12990
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21936
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 40-47.
- This extends the checked noise_std 0.32 range to 48 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- The weakest checked noise_std 0.32 case so far is case47 with margin 0.06073.
  This is a tighter margin but still not a CPU-only scientific failure or
  GPU/video boundary.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 132: burst16 noise0.32 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08919
mean_margin 0.14436
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3752
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 32-39.
- This extends the checked noise_std 0.32 range to 40 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 131: burst16 noise0.32 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11960
mean_margin 0.13834
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 43919
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 24-31.
- This extends the checked noise_std 0.32 range to 32 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case29 remains a local CPU scoring hotspot, but workers=4 completes it; no
  CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 130: burst16 noise0.32 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08368
mean_margin 0.11901
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15188
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 16-23.
- This extends the checked noise_std 0.32 range to 24 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 129: burst16 noise0.32 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11095
mean_margin 0.14852
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 23089
```

Interpretation:

- The noise_std 0.32 diagnostic-pruned tier remains positive through checked
  cases 8-15.
- This extends the checked noise_std 0.32 range to 16 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 128: burst16 noise0.32 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.32 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise032_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10207
mean_margin 0.14104
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21629
```

Interpretation:

- Raising synthetic noise to noise_std 0.32 remains positive on checked
  cases 0-7 under the current burst16/top2/diagnostic-pruned C tier.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 127: burst16 noise0.30 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10058
mean_margin 0.12805
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18481
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 56-63.
- This closes the checked noise_std 0.30 range at 64 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Across the closed noise_std 0.30 range, the tightest checked margin is case47
  at 0.05886; no CPU-only scientific failure or GPU/video boundary was
  reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 126: burst16 noise0.30 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07119
mean_margin 0.11292
max_ambiguity_sequence_count 216
max_exact_score_count_after_diagnostic_screen 32883
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 48-55.
- This extends the checked noise_std 0.30 range to 56 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 125: burst16 noise0.30 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05886
mean_margin 0.12905
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21952
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 40-47.
- This extends the checked noise_std 0.30 range to 48 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- The weakest checked noise_std 0.30 case so far is case47 with margin 0.05886.
  This is a tighter margin but still not a CPU-only scientific failure or
  GPU/video boundary.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 124: burst16 noise0.30 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09366
mean_margin 0.14518
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3752
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 32-39.
- This extends the checked noise_std 0.30 range to 40 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 123: burst16 noise0.30 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11978
mean_margin 0.13902
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 43970
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 24-31.
- This extends the checked noise_std 0.30 range to 32 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case29 again presents the largest local CPU scoring load in this segment,
  but workers=4 completes it; no CPU-only scientific failure or GPU/video
  boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 122: burst16 noise0.30 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08514
mean_margin 0.12209
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15182
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 16-23.
- This extends the checked noise_std 0.30 range to 24 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 121: burst16 noise0.30 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11246
mean_margin 0.14859
max_ambiguity_sequence_count 240
max_exact_score_count_after_diagnostic_screen 38411
```

Interpretation:

- The noise_std 0.30 diagnostic-pruned tier remains positive through checked
  cases 8-15.
- This extends the checked noise_std 0.30 range to 16 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This remains diagnostic-pruned mechanism triage, not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 120: burst16 noise0.30 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.30 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise030_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09960
mean_margin 0.13982
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21585
```

Interpretation:

- Raising synthetic noise to noise_std 0.30 remains positive on checked
  cases 0-7 under the current burst16/top2/diagnostic-pruned C tier.
- The weakest checked case in this segment is case1 with margin 0.09960.
- This extends CPU-only noise/margin mapping beyond the closed noise_std 0.28
  range; no CPU-only scientific failure or GPU/video boundary was reached.
- This is still diagnostic-pruned mechanism triage: all owner messages are
  exact-scored, but wrong keys are exact-scored only after public cheap
  screening. It is not exhaustive wrong-key, fixed-FPR, governance,
  reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 119: burst16 noise0.28 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10453
mean_margin 0.13048
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18480
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 56-63.
- This closes the checked noise_std 0.28 range at 64 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Across the closed noise_std 0.28 range, the tightest checked margin is case47
  at 0.05336; no CPU-only scientific failure or GPU/video boundary was
  reached.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 118: burst16 noise0.28 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07161
mean_margin 0.11495
max_ambiguity_sequence_count 216
max_exact_score_count_after_diagnostic_screen 32911
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 48-55.
- This extends the checked noise_std 0.28 range to 56 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 117: burst16 noise0.28 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05336
mean_margin 0.12734
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21976
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 40-47.
- This extends the checked noise_std 0.28 range to 48 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case47 is now the tightest checked noise0.28 case, with margin 0.05336, but
  it still recovers the owner message and global owner winner.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 116: burst16 noise0.28 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09460
mean_margin 0.14736
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3744
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 32-39.
- This extends the checked noise_std 0.28 range to 40 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 115: burst16 noise0.28 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases24_31_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11995
mean_margin 0.14007
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 43939
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 24-31.
- This extends the checked noise_std 0.28 range to 32 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case29 remains a heavy local CPU scoring case, with ambiguity_sequence_count
  288 and total_exact_score_count 43939, but it passed with margin 0.13118
  under workers=4.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 114: burst16 noise0.28 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases16_23_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08554
mean_margin 0.12282
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15148
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 16-23.
- This extends the checked noise_std 0.28 range to 24 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case21 is the tightest checked noise0.28 case so far, with margin 0.08554,
  but it still recovers the owner message and global owner winner.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-04 pass 113: burst16 noise0.28 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases8_15_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11334
mean_margin 0.14850
max_ambiguity_sequence_count 240
max_exact_score_count_after_diagnostic_screen 38470
```

Interpretation:

- The noise_std 0.28 diagnostic-pruned tier remains positive through checked
  cases 8-15.
- This extends the checked noise_std 0.28 range to 16 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case11 is a heavy local CPU scoring case, with ambiguity_sequence_count 240
  and total_exact_score_count 38470, but it passed with margin 0.19324 under
  workers=4.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 112: burst16 noise0.28 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.28 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise028_burst16_top2_diag_owner64_cases0_7_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10748
mean_margin 0.14134
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21503
```

Interpretation:

- Raising synthetic noise to noise_std 0.28 remains positive on checked
  cases 0-7.
- No CPU-only scientific failure or GPU/video boundary was reached.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 111: burst16 noise0.26 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise026_burst16_top2_diag_owner64_cases56_63_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10519
mean_margin 0.13437
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18462
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 56-63.
- This closes the checked noise_std 0.26 range at 64 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Across the closed noise_std 0.26 range, the tightest checked margin remains
  case47 at 0.05891; no CPU-only scientific failure or GPU/video boundary was
  reached.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 110: burst16 noise0.26 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise026_burst16_top2_diag_owner64_cases48_55_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07383
mean_margin 0.11603
max_ambiguity_sequence_count 216
max_exact_score_count_after_diagnostic_screen 32924
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 48-55.
- This extends the checked noise_std 0.26 range to 56 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case51 is a heavy local CPU scoring case, with ambiguity_sequence_count 216
  and total_exact_score_count 32924, but it passed with margin 0.10179 under
  workers=4.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 109: burst16 noise0.26 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise026_burst16_top2_diag_owner64_cases40_47_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05891
mean_margin 0.12618
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22006
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 40-47.
- This extends the checked noise_std 0.26 range to 48 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- Case47 is the current tightest checked noise0.26 margin at 0.05891, but it
  still recovers the owner message and global owner winner.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 108: burst16 noise0.26 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1 \
  --case-jsonl /tmp/sc_sstw_noise026_burst16_top2_diag_owner64_cases32_39_workers4_cases.jsonl
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09435
mean_margin 0.14723
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3740
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 32-39.
- This extends the checked noise_std 0.26 range to 40 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- The batch completed with `workers=4`; no GPU or video-model boundary was
  reached.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 107: burst16 noise0.26 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

- Cases 24-28 completed in the original workers=1 segment command.
- Case29 was a workers=1 CPU throughput hotspot. After enabling
  sequence-level parallel diagnostic-pruned scoring, case29 passed with
  `workers=4`, ambiguity_sequence_count 288, total_exact_score_count 43841,
  and score_margin 0.13249.
- Cases 30-31 were then completed with `workers=4` and the same frozen
  diagnostic-pruned scoring parameters.

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.12042
mean_margin 0.14057
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 43841
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 24-31.
- This extends the checked noise_std 0.26 range to 32 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- The case29 hotspot was an execution throughput issue in the local CPU
  scorer, not a scientific failure and not a GPU requirement.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 106: burst16 noise0.26 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08616
mean_margin 0.12152
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15144
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 16-23.
- This extends the checked noise_std 0.26 range to 24 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 105: burst16 noise0.26 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11718
mean_margin 0.14933
max_ambiguity_sequence_count 240
max_exact_score_count_after_diagnostic_screen 38464
```

Interpretation:

- The noise_std 0.26 diagnostic-pruned tier remains positive through checked
  cases 8-15.
- This extends the checked noise_std 0.26 range to 16 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 104: burst16 noise0.26 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.26 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10408
mean_margin 0.14053
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21444
```

Interpretation:

- Raising synthetic noise to noise_std 0.26 remains positive on the first
  checked 8-case segment under top2 acquisition and diagnostic-pruned scoring.
- The weakest checked margin remains above the 0.02 diagnostic line in this
  segment.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 103: burst16 noise0.24 diagnostic-pruned cases 56-63

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10521
mean_margin 0.13386
max_ambiguity_sequence_count 120
max_exact_score_count_after_diagnostic_screen 18460
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 56-63.
- This closes the checked noise_std 0.24 range at 64 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 102: burst16 noise0.24 diagnostic-pruned cases 48-55

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.07433
mean_margin 0.11711
max_ambiguity_sequence_count 216
max_exact_score_count_after_diagnostic_screen 32936
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 48-55.
- This extends the checked noise_std 0.24 range to 56 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 101: burst16 noise0.24 diagnostic-pruned cases 40-47

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Execution note:

- Cases 40-46 were produced by the 8-case segment command. The segment was
  manually interrupted before case47 completed because case44 had already shown
  high local CPU scoring cost. Case47 was then run alone with the same frozen
  parameters.

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.05653
mean_margin 0.12709
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 22060
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 40-47.
- This extends the checked noise_std 0.24 range to 48 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- The manual interruption was an execution-management choice during CPU-only
  segmentation, not a failed case; case47 passed when run alone.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 100: burst16 noise0.24 diagnostic-pruned cases 32-39

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.09328
mean_margin 0.14806
max_ambiguity_sequence_count 24
max_exact_score_count_after_diagnostic_screen 3733
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 32-39.
- This extends the checked noise_std 0.24 range to 40 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 99: burst16 noise0.24 diagnostic-pruned cases 24-31

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.12069
mean_margin 0.14126
max_ambiguity_sequence_count 288
max_exact_score_count_after_diagnostic_screen 43749
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 24-31.
- This extends the checked noise_std 0.24 range to 32 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 98: burst16 noise0.24 diagnostic-pruned cases 16-23

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.08395
mean_margin 0.12132
max_ambiguity_sequence_count 96
max_exact_score_count_after_diagnostic_screen 15134
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 16-23.
- This extends the checked noise_std 0.24 range to 24 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 97: burst16 noise0.24 diagnostic-pruned cases 8-15

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.11475
mean_margin 0.14840
max_ambiguity_sequence_count 240
max_exact_score_count_after_diagnostic_screen 38367
```

Interpretation:

- The noise_std 0.24 diagnostic-pruned tier remains positive through checked
  cases 8-15.
- This extends the checked noise_std 0.24 range to 16 / 64 under top2
  acquisition and diagnostic-pruned scoring.
- This is still CPU-only synthetic mechanism triage: the wrong-key set is
  public-screened before exact scoring, so this is not exhaustive wrong-key,
  fixed-FPR, governance, reproducibility, or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 96: burst16 noise0.24 diagnostic-pruned cases 0-7

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.24 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
pass 8 / 8
truth covered 8 / 8
owner/global recovery 8 / 8
min_margin 0.10372
mean_margin 0.14013
max_ambiguity_sequence_count 144
max_exact_score_count_after_diagnostic_screen 21409
```

Interpretation:

- Raising synthetic noise to noise_std 0.24 remains positive on the first
  checked 8-case segment when using top2 acquisition and diagnostic-pruned
  owner/wrong scoring.
- The weakest margin is still well above the 0.02 diagnostic line in this
  checked segment.
- This does not convert the last-stage diagnostic-pruned wrong-key screen into
  exhaustive wrong-key evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 95: burst16 noise0.22 top2 diagnostic-pruned final segments

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
cases 48-55:
  pass 8 / 8
  truth covered 8 / 8
  owner/global recovery 8 / 8
  min_margin 0.07371
  max_ambiguity_sequence_count 216
  max_exact_score_count_after_diagnostic_screen 32930

cases 56-63:
  pass 8 / 8
  truth covered 8 / 8
  owner/global recovery 8 / 8
  min_margin 0.10467
  max_ambiguity_sequence_count 120
  max_exact_score_count_after_diagnostic_screen 18432
```

Interpretation:

- The noise_std 0.22, burst_count 16 checked message-index range now reaches
  64 / 64 synthetic diagnostic passes when the final 16 cases use
  `top_k_per_start=2` acquisition plus diagnostic pruned scoring.
- This is not a full exhaustive wrong-key proof for cases 48-63: the diagnostic
  screen exact-scores all owner messages but only a public cheap-screened
  subset of wrong candidates. It is therefore a CPU feasibility diagnostic,
  not fixed-FPR, governance, or paper evidence.
- The observed top1 failure at case 50 is specifically an acquisition
  beam-width issue: `top_k_per_start=2` restores truth coverage, and the
  owner/message chain then passes.
- The new bottleneck is CPU scoring scale under larger public ambiguity sets,
  not GPU/video execution. The next useful local work is CPU-side diagnostic
  screening and, if stronger evidence is needed, a principled exact-scoring
  pruning/proof layer.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 94: top2 acquisition repair of noise0.22 case50

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 50 \
  --case-count 1 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c
```

Result summary:

```text
case 50:
  pass true
  truth_sequence_covered_by_ambiguity true
  truth_sequence_in_ambiguity true
  ambiguity_sequence_count 24
  total_exact_score_count 99840
  owner/global message message_50
  score_margin 0.14948
```

Interpretation:

- The earlier top1 case50 miss was not an owner/wrong scoring failure. It
  occurred because the scan retained only one candidate per observed start and
  dropped a very close true deletion hypothesis.
- `top_k_per_start=2` restores truth coverage and the official exact/bounded
  CLI path recovers the owner message with a strong diagnostic margin.
- The cost increase is large: a single case reaches 99,840 exact scores, so
  top2 acquisition should be paired with CPU-side scoring diagnostics before
  larger synthetic sweeps.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 93: burst16 noise0.22 cases 40-47 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise022_burst16_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise022_burst16_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.22, 64 messages / 64 wrong keys,
burst_count 16, cases 40-47:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.05231614041101118
  score_margin_mean: 0.08852270366450563

case margins:
  case 40: 0.09743
  case 41: 0.13656
  case 42: 0.08285
  case 43: 0.08581
  case 44: 0.09636
  case 45: 0.07888
  case 46: 0.07797
  case 47: 0.05232
```

Interpretation:

- The sixth burst_count 16 noise_std 0.22 segment passes all checked cases,
  extending the checked range to 48 / 64.
- The weakest checked noise_std 0.22 margin is now case 47 at 0.05232,
  still above the 0.02 diagnostic line.
- Case 42 had a large ambiguity set with 27 sequences and 112,320 exact-score
  candidates; it still passed, so this segment mainly exposes CPU exact-scoring
  throughput cost rather than a GPU/video requirement.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 92: burst16 noise0.22 cases 32-39 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise022_burst16_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise022_burst16_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.22, 64 messages / 64 wrong keys,
burst_count 16, cases 32-39:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.088727875742182
  score_margin_mean: 0.11651861290443126

case margins:
  case 32: 0.11971
  case 33: 0.09752
  case 34: 0.16690
  case 35: 0.09129
  case 36: 0.08873
  case 37: 0.11634
  case 38: 0.13077
  case 39: 0.12090
```

Interpretation:

- The fifth burst_count 16 noise_std 0.22 segment passes all checked cases,
  extending the checked range to 40 / 64.
- The weakest checked noise_std 0.22 margin remains case 21 at 0.06563,
  still above the 0.02 diagnostic line.
- No GPU/video boundary has been reached; the active validation route remains
  CPU-only synthetic noise/evidence-span mapping.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 91: burst16 noise0.22 cases 24-31 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise022_burst16_cases24_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise022_burst16_cases24_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.22, 64 messages / 64 wrong keys,
burst_count 16, cases 24-31:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.06989032711208004
  score_margin_mean: 0.09442954252693636

case margins:
  case 24: 0.08581
  case 25: 0.10142
  case 26: 0.11177
  case 27: 0.09515
  case 28: 0.11427
  case 29: 0.09827
  case 30: 0.06989
  case 31: 0.07885
```

Interpretation:

- The fourth burst_count 16 noise_std 0.22 segment passes all checked cases,
  extending the checked range to 32 / 64.
- The weakest checked noise_std 0.22 margin remains case 21 at 0.06563,
  still above the 0.02 diagnostic line.
- CPU-only synthetic mapping remains the active validation path; there is
  still no GPU/video requirement from this evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 90: burst16 noise0.22 cases 16-23 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise022_burst16_cases16_23_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise022_burst16_cases16_23_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.22, 64 messages / 64 wrong keys,
burst_count 16, cases 16-23:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.06563366160031336
  score_margin_mean: 0.09417521326381177

case margins:
  case 16: 0.12338
  case 17: 0.09633
  case 18: 0.08712
  case 19: 0.09710
  case 20: 0.08328
  case 21: 0.06563
  case 22: 0.11551
  case 23: 0.08505
```

Interpretation:

- The third burst_count 16 noise_std 0.22 segment also passes all checked cases,
  extending the checked range to 24 / 64.
- The weakest checked noise_std 0.22 margin so far is now case 21 at 0.06563,
  still above the 0.02 diagnostic line.
- CPU-only synthetic mapping remains the active validation path.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 89: burst16 noise0.22 cases 8-15 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise022_burst16_cases8_15_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise022_burst16_cases8_15_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.22, 64 messages / 64 wrong keys,
burst_count 16, cases 8-15:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.08860158873712265
  score_margin_mean: 0.10761001934054856

case margins:
  case 8: 0.10204
  case 9: 0.08976
  case 10: 0.10919
  case 11: 0.14968
  case 12: 0.11574
  case 13: 0.08860
  case 14: 0.10523
  case 15: 0.10063
```

Interpretation:

- The second burst_count 16 noise_std 0.22 segment also passes all checked
  cases, extending the checked noise_std 0.22 range to 16 / 64.
- The weakest checked noise_std 0.22 margin so far remains case 1 from pass 88
  at 0.07796, still above the 0.02 diagnostic line.
- The CPU-only evidence-span/noise map remains positive; no GPU/video boundary
  has been reached.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 88: burst16 noise0.22 first segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.22 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise022_burst16_cases0_7_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise022_burst16_cases0_7_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.22, 64 messages / 64 wrong keys,
burst_count 16, cases 0-7:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.07796198428934065
  score_margin_mean: 0.11306734327070983

case margins:
  case 0: 0.11443
  case 1: 0.07796
  case 2: 0.09827
  case 3: 0.13609
  case 4: 0.09436
  case 5: 0.11027
  case 6: 0.11856
  case 7: 0.15460
```

Interpretation:

- After completing the full noise_std 0.20 checked range at burst_count 16,
  the first noise_std 0.22 segment also passes all 8 checked cases.
- The weakest margin in this first harder segment is 0.07796, still well above
  the 0.02 diagnostic line.
- The active CPU-only route remains synthetic noise/evidence-span mapping; no
  GPU/video step is required yet.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 87: burst16 completes noise0.20 full 64-case range

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_burst16_cases56_63_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_burst16_cases56_63_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.20, 64 messages / 64 wrong keys,
burst_count 16, cases 56-63:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.06400563729784581
  score_margin_mean: 0.09735494143546497

combined burst_count 16, noise_std 0.20 checked range:
  cases 32-39: pass 8 / 8, min margin 0.08868
  cases 40-47: pass 8 / 8, min margin 0.05190
  cases 48-55: pass 8 / 8, min margin 0.04821
  cases 56-63: pass 8 / 8, min margin 0.06401
  cases 0-31 were already positive at burst_count 12.
```

Interpretation:

- The full checked 64-message index range is now positive at gamma 100.0,
  noise_std 0.20, and 64 wrong keys when the public trajectory evidence is
  increased to 16 bursts.
- The original burst_count 12 failure at case 34 is therefore not a fundamental
  AISB acquisition/scoring failure. It is a trajectory evidence-span boundary:
  adding public synchronization evidence restores owner/wrong-key separation.
- The next CPU-only mechanism question is to map the next harder synthetic
  noise tier or the evidence-span/noise tradeoff, not to move to GPU/video.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 86: burst16 noise0.20 cases 48-55 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_burst16_cases48_55_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_burst16_cases48_55_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.20, 64 messages / 64 wrong keys,
burst_count 16, cases 48-55:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.04821129952667336
  score_margin_mean: 0.085806844955841

case margins:
  case 48: 0.10415
  case 49: 0.10553
  case 50: 0.14745
  case 51: 0.06230
  case 52: 0.06393
  case 53: 0.04821
  case 54: 0.09049
  case 55: 0.06439
```

Interpretation:

- The burst_count 16 trajectory-evidence setting continues to pass the third
  checked noise_std 0.20 segment after the original burst_count 12 boundary,
  extending the checked range to 56 / 64.
- Case 53 is the weakest point in this segment at margin 0.04821, still above
  the 0.02 diagnostic line and close to case 47 from pass 85.
- The current CPU-only evidence continues to support evidence-span expansion
  as the active mechanism route.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 85: burst16 noise0.20 cases 40-47 segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_burst16_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_burst16_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.20, 64 messages / 64 wrong keys,
burst_count 16, cases 40-47:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.05189661508119581
  score_margin_mean: 0.09063010987131831

case margins:
  case 40: 0.09897
  case 41: 0.13612
  case 42: 0.08589
  case 43: 0.08609
  case 44: 0.09674
  case 45: 0.08687
  case 46: 0.08246
  case 47: 0.05190
```

Interpretation:

- The burst_count 16 trajectory-evidence setting continues to pass the next
  checked noise_std 0.20 segment, extending the checked range from 40 / 64 to
  48 / 64.
- Case 47 becomes the weakest checked burst16 noise_std 0.20 point so far, but
  its margin 0.05190 remains above the 0.02 diagnostic line.
- This reinforces the pass 84 interpretation: the earlier burst_count 12
  boundary is evidence-span limited, not an AISB acquisition failure or a
  GPU/video requirement.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 84: increased AISB trajectory evidence restores noise0.20 boundary segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_burst16_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_burst16_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
gamma 100.0, noise_std 0.20, 64 messages / 64 wrong keys,
burst_count 16, cases 32-39:
  pass_count: 8 / 8
  owner_message_recovery_count: 8 / 8
  global_owner_recovery_count: 8 / 8
  truth_sequence_covered_by_ambiguity_count: 8 / 8
  score_margin_min: 0.08868035804343444
  score_margin_mean: 0.11679403759246887

case margins:
  case 32: 0.12396
  case 33: 0.09879
  case 34: 0.16218
  case 35: 0.09141
  case 36: 0.08868
  case 37: 0.11585
  case 38: 0.13306
  case 39: 0.12042
```

Interpretation:

- Increasing the public AISB trajectory evidence from 12 bursts to 16 bursts
  restores the previously failing noise_std 0.20 cases 32-39 segment.
- The specific case 34 boundary is no longer near the diagnostic line: its
  margin rises from 0.01754 at 12 bursts to 0.16218 at 16 bursts.
- This supports the current mechanism hypothesis: wrong-key separation is
  evidence-span limited in the synthetic construction, and can be improved by
  adding public trajectory evidence without changing the acquisition rule,
  owner/wrong search fairness, or moving to GPU/video.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 83: case 34 top-k competition diagnostic

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/diagnose_aisb_case_competition.py \
  --case-index 34 \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --top-k 12 \
  > /tmp/sc_sstw_case34_noise020_competition.json
```

Result summary:

```text
case 34, gamma 100.0, noise_std 0.200:
  ambiguity_sequence_count: 2
  truth_sequence_indices: [0]
  truth_covered_sequence_indices: [0]
  exact_score_count: 8320

top global:
  1. owner, message_34, sequence 0, score -0.5881813644526389
  2. wrong key wrong_sequence_ambiguity_12_64_34_30, message_41,
     sequence 0, score -0.6057185828630267

competition summary:
  owner_message_recovered: true
  owner_global_winner: true
  score_margin: 0.0175372184103878
  diagnostic_pass: false
```

Interpretation:

- The first checked noise_std 0.200 diagnostic failure is not caused by owner
  message confusion or public alignment failure: the exact truth sequence is
  sequence 0, and the owner true message remains the global top scorer.
- The failure is a wrong-key competition margin boundary. The nearest checked
  wrong-key candidate is close enough that the margin falls below the 0.02
  diagnostic line, even though the owner still wins.
- The active CPU-only question is therefore trajectory-evidence separation
  against larger wrong-key/message competition, not AISB acquisition and not a
  GPU/video-model boundary.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 82: case 34 noise-boundary single-case diagnostics

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 --noise-std 0.19 --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 \
  --wrong-key-count 64 --start-index 34 --case-count 1 --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise019_case34_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise019_case34_cases.jsonl \
  --progress-interval 4096

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 --noise-std 0.185 --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 \
  --wrong-key-count 64 --start-index 34 --case-count 1 --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise0185_case34_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise0185_case34_cases.jsonl \
  --progress-interval 4096

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 --noise-std 0.1825 --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 \
  --wrong-key-count 64 --start-index 34 --case-count 1 --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise01825_case34_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise01825_case34_cases.jsonl \
  --progress-interval 4096

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 --noise-std 0.181 --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 \
  --wrong-key-count 64 --start-index 34 --case-count 1 --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise0181_case34_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise0181_case34_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 34 single-case diagnostics:
  noise_std 0.1900: diagnostic fail, margin 0.01799, owner/global correct
  noise_std 0.1850: diagnostic fail, margin 0.01822, owner/global correct
  noise_std 0.1825: diagnostic fail, margin 0.01833, owner/global correct
  noise_std 0.1810: diagnostic fail, margin 0.01840, owner/global correct

reference from full segments:
  noise_std 0.1800, cases 32-39 segment: pass, margin 0.02120 at case 34
  noise_std 0.2000, cases 32-39 segment: fail, margin 0.01754 at case 34
```

Interpretation:

- Case 34 remains owner/global correct across the checked single-case
  diagnostics, so the boundary is still diagnostic margin, not message
  recovery.
- The single-case margins are not a smooth interpolation from the full 0.180
  segment to the 0.200 segment. The synthetic case generator changes with
  noise_std, so this probe should be read as distributional boundary evidence,
  not as a literal continuous noise scaling of one fixed observation.
- CPU-only evidence is sufficient to identify a near-boundary synthetic tier:
  at gamma 100.0, 64 messages, 64 wrong keys, and the current diagnostic line,
  noise_std around 0.18-0.20 is the active margin boundary.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 81: gamma 100.0, noise 0.200, cases 32-39 diagnostic boundary

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12171, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10786, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: diagnostic fail, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.01754, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04987, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05546, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06660, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12468, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07827, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.20 status:
  checked completed cases 0-39: diagnostic pass 39 / 40.
  segment 32-39 diagnostic pass 7 / 8.
  first diagnostic failure: case 34, margin 0.01754 below the 0.02 line.
```

Interpretation:

- This is the first checked diagnostic failure for the active AISB public
  ambiguity-set scoring route at gamma 100.0 and the 64-message / 64-wrong-key
  tier.
- The owner/global winner is still correct in case 34, so this is not a
  message-recovery failure. It is a diagnostic margin boundary: the separation
  margin falls below the predeclared 0.02 line at noise_std 0.200.
- This is not a GPU boundary. The next useful work is CPU-only boundary
  localization around case 34, not video/model execution.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 80: gamma 100.0, noise 0.200, cases 24-31 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_cases24_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_cases24_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 24: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09079, ambiguity_sequence_count 24, total_exact_score_count 99840
case 25: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.14229, ambiguity_sequence_count 1, total_exact_score_count 4160
case 26: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08990, ambiguity_sequence_count 1, total_exact_score_count 4160
case 27: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12342, ambiguity_sequence_count 1, total_exact_score_count 4160
case 28: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07860, ambiguity_sequence_count 1, total_exact_score_count 4160
case 29: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08508, ambiguity_sequence_count 1, total_exact_score_count 4160
case 30: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10589, ambiguity_sequence_count 3, total_exact_score_count 12480
case 31: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11973, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.20 status:
  checked completed cases 0-31: pass 32 / 32.
  segment 24-31 minimum margin 0.07860 at case 28.
  overall checked minimum remains 0.02329 at case 7.
```

Interpretation:

- The fourth noise_std 0.200 segment remains positive across all checked cases.
- The segment includes the large 24-sequence ambiguity case, with 99,840 exact
  candidate scores, and still recovers the owner message globally.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.200 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 79: gamma 100.0, noise 0.200, cases 16-23 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_cases16_23_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_cases16_23_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 16: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09151, ambiguity_sequence_count 1, total_exact_score_count 4160
case 17: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09740, ambiguity_sequence_count 3, total_exact_score_count 12480
case 18: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10134, ambiguity_sequence_count 2, total_exact_score_count 8320
case 19: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12345, ambiguity_sequence_count 1, total_exact_score_count 4160
case 20: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12268, ambiguity_sequence_count 1, total_exact_score_count 4160
case 21: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04049, ambiguity_sequence_count 3, total_exact_score_count 12480
case 22: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07066, ambiguity_sequence_count 6, total_exact_score_count 24960
case 23: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08379, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.20 status:
  checked completed cases 0-23: pass 24 / 24.
  segment 16-23 minimum margin 0.04049 at case 21.
  overall checked minimum remains 0.02329 at case 7.
```

Interpretation:

- The third noise_std 0.200 segment remains positive across all checked cases.
- The segment includes both a thin-margin case and a larger ambiguity-set case,
  and the owner/global scorer still separates the correct message under the
  shared public ambiguity-set contract.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.200 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 78: gamma 100.0, noise 0.200, cases 8-15 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_cases8_15_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_cases8_15_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 8: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04764, ambiguity_sequence_count 3, total_exact_score_count 12480
case 9: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05644, ambiguity_sequence_count 4, total_exact_score_count 16640
case 10: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13091, ambiguity_sequence_count 1, total_exact_score_count 4160
case 11: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09435, ambiguity_sequence_count 1, total_exact_score_count 4160
case 12: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.05482, ambiguity_sequence_count 1, total_exact_score_count 4160
case 13: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.14054, ambiguity_sequence_count 3, total_exact_score_count 12480
case 14: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04686, ambiguity_sequence_count 1, total_exact_score_count 4160
case 15: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.10237, ambiguity_sequence_count 9, total_exact_score_count 37440

combined noise0.20 status:
  checked completed cases 0-15: pass 16 / 16.
  segment 8-15 minimum margin 0.04686 at case 14.
  overall checked minimum remains 0.02329 at case 7.
```

Interpretation:

- The second noise_std 0.200 segment remains positive across all checked cases,
  including a larger ambiguity-set case with 37,440 exact candidate scores.
- The first segment's case 7 remains the thin-margin point for this tier.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.200 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 77: gamma 100.0, noise 0.200, cases 0-7 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.20 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise020_cases0_7_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise020_cases0_7_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 0: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09724, ambiguity_sequence_count 1, total_exact_score_count 4160
case 1: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11660, ambiguity_sequence_count 2, total_exact_score_count 8320
case 2: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08554, ambiguity_sequence_count 3, total_exact_score_count 12480
case 3: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07584, ambiguity_sequence_count 1, total_exact_score_count 4160
case 4: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06520, ambiguity_sequence_count 3, total_exact_score_count 12480
case 5: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07864, ambiguity_sequence_count 1, total_exact_score_count 4160
case 6: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11108, ambiguity_sequence_count 3, total_exact_score_count 12480
case 7: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02329, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.20 status:
  checked completed cases 0-7: pass 8 / 8.
  segment 0-7 minimum margin 0.02329 at case 7.
```

Interpretation:

- The first noise_std 0.200 segment remains positive, but the minimum margin is
  again close to the 0.02 diagnostic line.
- This is still a CPU-only synthetic pass, not a GPU/video boundary. The next
  useful check is segmented continuation of the same tier to determine whether
  the near-boundary margin persists or crosses the diagnostic line.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 76: gamma 100.0, noise 0.180, cases 56-63 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases56_63_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases56_63_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 56: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07888, ambiguity_sequence_count 1, total_exact_score_count 4160
case 57: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10131, ambiguity_sequence_count 1, total_exact_score_count 4160
case 58: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.08235, ambiguity_sequence_count 1, total_exact_score_count 4160
case 59: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13357, ambiguity_sequence_count 2, total_exact_score_count 8320
case 60: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08025, ambiguity_sequence_count 1, total_exact_score_count 4160
case 61: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09845, ambiguity_sequence_count 3, total_exact_score_count 12480
case 62: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07341, ambiguity_sequence_count 1, total_exact_score_count 4160
case 63: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12940, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.18 status:
  checked completed cases 0-63: pass 64 / 64.
  segment 56-63 minimum margin 0.07341 at case 62.
  overall checked minimum remains 0.02120 at case 34.
```

Interpretation:

- The full checked noise_std 0.180 message-index range now passes 64 / 64 at
  gamma 100.0 with the 64-message / 64-wrong-key tier.
- The weakest checked point, case 34 at margin 0.02120, is close to the 0.02
  diagnostic line. The method mechanism has not failed, but the next CPU-only
  stress tier should be treated as near-boundary rather than a routine margin
  expansion.
- No GPU boundary has been reached. The next useful work is still CPU-only:
  either probe the next noise tier in small segments or inspect the thin-margin
  cases before any video/GPU escalation.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 75: gamma 100.0, noise 0.180, cases 48-55 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases48_55_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases48_55_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 48: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07474, ambiguity_sequence_count 4, total_exact_score_count 16640
case 49: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06501, ambiguity_sequence_count 3, total_exact_score_count 12480
case 50: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09561, ambiguity_sequence_count 1, total_exact_score_count 4160
case 51: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11524, ambiguity_sequence_count 2, total_exact_score_count 8320
case 52: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09558, ambiguity_sequence_count 3, total_exact_score_count 12480
case 53: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03912, ambiguity_sequence_count 1, total_exact_score_count 4160
case 54: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07719, ambiguity_sequence_count 1, total_exact_score_count 4160
case 55: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.14360, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.18 status:
  checked completed cases 0-55: pass 56 / 56.
  segment 48-55 minimum margin 0.03912 at case 53.
  overall checked minimum remains 0.02120 at case 34.
```

Interpretation:

- The seventh noise_std 0.180 segment remains positive across all checked
  cases.
- Several cases again recover correctly with truth covered by the public
  ambiguity set even when the exact truth sequence is not the chosen public
  representative.
- No GPU boundary has been reached; the next useful work is the final CPU-only
  segment for cases 56-63 at the same tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 74: gamma 100.0, noise 0.180, cases 40-47 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 40: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06789, ambiguity_sequence_count 1, total_exact_score_count 4160
case 41: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11504, ambiguity_sequence_count 1, total_exact_score_count 4160
case 42: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10703, ambiguity_sequence_count 2, total_exact_score_count 8320
case 43: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12472, ambiguity_sequence_count 1, total_exact_score_count 4160
case 44: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10302, ambiguity_sequence_count 1, total_exact_score_count 4160
case 45: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03113, ambiguity_sequence_count 1, total_exact_score_count 4160
case 46: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06173, ambiguity_sequence_count 1, total_exact_score_count 4160
case 47: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12221, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.18 status:
  checked completed cases 0-47: pass 48 / 48.
  segment 40-47 minimum margin 0.03113 at case 45.
  overall checked minimum remains 0.02120 at case 34.
```

Interpretation:

- The sixth noise_std 0.180 segment remains positive across all checked cases.
- The segment minimum is comfortably above the diagnostic line, but the tier
  remains close-boundary because case 34 from the previous segment is only
  slightly above 0.02.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.180 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 73: gamma 100.0, noise 0.180, cases 32-39 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12826, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10966, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02120, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.05036, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06149, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06689, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12644, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07799, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.18 status:
  checked completed cases 0-39: pass 40 / 40.
  segment 32-39 minimum margin 0.02120 at case 34.
  overall checked minimum margin is now 0.02120 at case 34.
```

Interpretation:

- The fifth noise_std 0.180 segment remains positive across all checked cases.
- Case 34 is now the weakest checked point in this tier, only slightly above
  the 0.02 diagnostic line. This is not a failure, but it is the current local
  margin boundary to monitor before escalating to higher noise or larger
  candidate spaces.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.180 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 72: gamma 100.0, noise 0.180, cases 24-31 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases24_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases24_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 24: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09340, ambiguity_sequence_count 24, total_exact_score_count 99840
case 25: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.14220, ambiguity_sequence_count 1, total_exact_score_count 4160
case 26: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09061, ambiguity_sequence_count 1, total_exact_score_count 4160
case 27: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12108, ambiguity_sequence_count 1, total_exact_score_count 4160
case 28: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07770, ambiguity_sequence_count 1, total_exact_score_count 4160
case 29: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08416, ambiguity_sequence_count 1, total_exact_score_count 4160
case 30: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10539, ambiguity_sequence_count 3, total_exact_score_count 12480
case 31: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12202, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.18 status:
  checked completed cases 0-31: pass 32 / 32.
  segment 24-31 minimum margin 0.07770 at case 28.
  overall checked minimum margin remains 0.02868 at case 7.
```

Interpretation:

- The fourth noise_std 0.180 segment remains positive, including one large
  ambiguity-set case with 24 public candidate sequences and 99,840 exact
  candidate scores.
- Case 28 again shows why the active route freezes and scores the public
  ambiguity set rather than forcing a single exact public path: the truth is
  covered by the ambiguity set and owner/global scoring still recovers the
  correct message, even though the exact truth sequence is not the selected
  representative.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.180 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 71: gamma 100.0, noise 0.180, cases 16-23 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases16_23_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases16_23_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 16: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09188, ambiguity_sequence_count 1, total_exact_score_count 4160
case 17: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09288, ambiguity_sequence_count 3, total_exact_score_count 12480
case 18: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09853, ambiguity_sequence_count 2, total_exact_score_count 8320
case 19: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12300, ambiguity_sequence_count 1, total_exact_score_count 4160
case 20: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12408, ambiguity_sequence_count 1, total_exact_score_count 4160
case 21: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03936, ambiguity_sequence_count 3, total_exact_score_count 12480
case 22: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07066, ambiguity_sequence_count 6, total_exact_score_count 24960
case 23: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08676, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.18 status:
  checked completed cases 0-23: pass 24 / 24.
  segment 16-23 minimum margin 0.03936 at case 21.
  overall checked minimum margin remains 0.02868 at case 7.
```

Interpretation:

- The third noise_std 0.180 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Cases 21 and 22 reproduce the known pattern: one thin-margin case and one
  larger ambiguity-set case, both still correctly separated by shared
  ambiguity-set scoring.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.180 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 70: gamma 100.0, noise 0.180, cases 8-15 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases8_15_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases8_15_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 8: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.05047, ambiguity_sequence_count 3, total_exact_score_count 12480
case 9: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05785, ambiguity_sequence_count 4, total_exact_score_count 16640
case 10: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13365, ambiguity_sequence_count 1, total_exact_score_count 4160
case 11: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09538, ambiguity_sequence_count 1, total_exact_score_count 4160
case 12: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.05504, ambiguity_sequence_count 1, total_exact_score_count 4160
case 13: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13935, ambiguity_sequence_count 3, total_exact_score_count 12480
case 14: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04281, ambiguity_sequence_count 1, total_exact_score_count 4160
case 15: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.10067, ambiguity_sequence_count 9, total_exact_score_count 37440

combined noise0.18 status:
  checked completed cases 0-15: pass 16 / 16.
  segment 8-15 minimum margin 0.04281 at case 14.
  overall checked minimum margin remains 0.02868 at case 7.
```

Interpretation:

- The second noise_std 0.180 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 15 again produces a larger public ambiguity set, but ordered bounded
  scoring still recovers the owner/global winner and true message.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.180 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 69: gamma 100.0, noise 0.180, cases 0-7 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.18 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise018_cases0_7_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise018_cases0_7_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 0: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09715, ambiguity_sequence_count 1, total_exact_score_count 4160
case 1: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11911, ambiguity_sequence_count 2, total_exact_score_count 8320
case 2: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08507, ambiguity_sequence_count 3, total_exact_score_count 12480
case 3: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07376, ambiguity_sequence_count 1, total_exact_score_count 4160
case 4: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06991, ambiguity_sequence_count 3, total_exact_score_count 12480
case 5: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07338, ambiguity_sequence_count 1, total_exact_score_count 4160
case 6: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11049, ambiguity_sequence_count 3, total_exact_score_count 12480
case 7: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02868, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.18 status:
  checked completed cases 0-7: pass 8 / 8.
  segment 0-7 minimum margin 0.02868 at case 7.
```

Interpretation:

- The first noise_std 0.180 segment remains positive under the same gamma 100.0,
  64-message / 64-wrong-key ordered bounded global-c tier.
- Case 7 is again the weakest early case and is close enough to track, but it
  remains above the 0.02 diagnostic line.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation of the noise_std 0.180 tier.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 68: gamma 100.0, noise 0.160, cases 56-63 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases56_63_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases56_63_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 56: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08039, ambiguity_sequence_count 1, total_exact_score_count 4160
case 57: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09987, ambiguity_sequence_count 1, total_exact_score_count 4160
case 58: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.08338, ambiguity_sequence_count 1, total_exact_score_count 4160
case 59: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13552, ambiguity_sequence_count 2, total_exact_score_count 8320
case 60: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08357, ambiguity_sequence_count 1, total_exact_score_count 4160
case 61: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09979, ambiguity_sequence_count 3, total_exact_score_count 12480
case 62: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07393, ambiguity_sequence_count 1, total_exact_score_count 4160
case 63: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12722, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.16 status:
  checked completed cases 0-63: pass 64 / 64.
  segment 56-63 minimum margin 0.07393 at case 62.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The full checked noise_std 0.160, gamma 100.0, 64-message / 64-wrong-key
  range now passes 64 / 64 under ordered bounded global-c scoring.
- Owner/global recovery and truth coverage are 64 / 64 across the completed
  range. Some cases are covered by the public ambiguity set without the exact
  truth sequence being present, which remains an expected ambiguity-set
  behavior rather than a failure.
- No GPU boundary has been reached. The next useful scientific stress is a
  higher-noise CPU-only tier or margin-localization around the existing weakest
  synthetic cases.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 67: gamma 100.0, noise 0.160, cases 48-55 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases48_55_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases48_55_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 48: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07424, ambiguity_sequence_count 4, total_exact_score_count 16640
case 49: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06278, ambiguity_sequence_count 3, total_exact_score_count 12480
case 50: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09719, ambiguity_sequence_count 1, total_exact_score_count 4160
case 51: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11565, ambiguity_sequence_count 2, total_exact_score_count 8320
case 52: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09912, ambiguity_sequence_count 3, total_exact_score_count 12480
case 53: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03796, ambiguity_sequence_count 1, total_exact_score_count 4160
case 54: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07927, ambiguity_sequence_count 1, total_exact_score_count 4160
case 55: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.14663, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.16 status:
  checked completed cases 0-55: pass 56 / 56.
  segment 48-55 minimum margin 0.03796 at case 53.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The seventh noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Several cases have truth covered but not exact in the public ambiguity set;
  the shared ambiguity-set scorer still recovers the owner message.
- No GPU boundary has been reached; the remaining work in this tier is the
  final CPU-only segment for cases 56-63.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 66: gamma 100.0, noise 0.160, cases 40-47 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 40: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06922, ambiguity_sequence_count 1, total_exact_score_count 4160
case 41: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11657, ambiguity_sequence_count 1, total_exact_score_count 4160
case 42: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10820, ambiguity_sequence_count 2, total_exact_score_count 8320
case 43: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12275, ambiguity_sequence_count 1, total_exact_score_count 4160
case 44: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10546, ambiguity_sequence_count 1, total_exact_score_count 4160
case 45: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03000, ambiguity_sequence_count 1, total_exact_score_count 4160
case 46: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06327, ambiguity_sequence_count 1, total_exact_score_count 4160
case 47: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12167, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.16 status:
  checked completed cases 0-47: pass 48 / 48.
  segment 40-47 minimum margin 0.03000 at case 45.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The sixth noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 45 is another thin-margin point but remains positive under the same
  shared ambiguity-set and fair message-search contract.
- No GPU boundary has been reached; the remaining work in this tier is still
  CPU-only segmented continuation.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 65: gamma 100.0, noise 0.160, cases 32-39 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13131, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10897, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02742, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04844, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06163, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06827, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12678, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08308, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.16 status:
  checked completed cases 0-39: pass 40 / 40.
  segment 32-39 minimum margin 0.02742 at case 34.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The fifth noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 34 is a new thin-margin point in this segment, but it remains positive
  and still sits slightly above the current overall weakest case 7.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 64: gamma 100.0, noise 0.160, cases 24-31 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases24_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases24_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 24: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09295, ambiguity_sequence_count 24, total_exact_score_count 99840
case 25: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.14116, ambiguity_sequence_count 1, total_exact_score_count 4160
case 26: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09074, ambiguity_sequence_count 1, total_exact_score_count 4160
case 27: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11970, ambiguity_sequence_count 1, total_exact_score_count 4160
case 28: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07525, ambiguity_sequence_count 1, total_exact_score_count 4160
case 29: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08414, ambiguity_sequence_count 1, total_exact_score_count 4160
case 30: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10421, ambiguity_sequence_count 3, total_exact_score_count 12480
case 31: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12278, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.16 status:
  checked completed cases 0-31: pass 32 / 32.
  segment 24-31 minimum margin 0.07525 at case 28.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The fourth noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 24 creates a much larger public ambiguity set
  (24 sequences, 99840 exact scores), but shared ambiguity-set scoring still
  recovers the owner/global winner and true message.
- No GPU boundary has been reached; the immediate risk is CPU exhaustive
  scoring cost as ambiguity grows, not a need for video-model validation.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 63: gamma 100.0, noise 0.160, cases 16-23 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases16_23_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases16_23_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 16: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09333, ambiguity_sequence_count 1, total_exact_score_count 4160
case 17: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09570, ambiguity_sequence_count 3, total_exact_score_count 12480
case 18: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10114, ambiguity_sequence_count 2, total_exact_score_count 8320
case 19: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12154, ambiguity_sequence_count 1, total_exact_score_count 4160
case 20: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12489, ambiguity_sequence_count 1, total_exact_score_count 4160
case 21: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03949, ambiguity_sequence_count 3, total_exact_score_count 12480
case 22: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07103, ambiguity_sequence_count 6, total_exact_score_count 24960
case 23: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08680, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.16 status:
  checked completed cases 0-23: pass 24 / 24.
  segment 16-23 minimum margin 0.03949 at case 21.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The third noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 22 creates the largest public ambiguity set in this segment
  (6 sequences, 24960 exact scores), but shared ambiguity-set scoring still
  recovers the owner/global winner and true message.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 62: gamma 100.0, noise 0.160, cases 8-15 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases8_15_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases8_15_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 8: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.05054, ambiguity_sequence_count 3, total_exact_score_count 12480
case 9: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05543, ambiguity_sequence_count 4, total_exact_score_count 16640
case 10: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13831, ambiguity_sequence_count 1, total_exact_score_count 4160
case 11: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09675, ambiguity_sequence_count 1, total_exact_score_count 4160
case 12: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.05907, ambiguity_sequence_count 1, total_exact_score_count 4160
case 13: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13921, ambiguity_sequence_count 3, total_exact_score_count 12480
case 14: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04486, ambiguity_sequence_count 1, total_exact_score_count 4160
case 15: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09922, ambiguity_sequence_count 9, total_exact_score_count 37440

combined noise0.16 status:
  checked completed cases 0-15: pass 16 / 16.
  segment 8-15 minimum margin 0.04486 at case 14.
  overall checked minimum margin remains 0.02633 at case 7.
```

Interpretation:

- The second noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 15 again creates a larger public ambiguity set, but shared ambiguity-set
  scoring still recovers the owner/global winner and true message.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented continuation and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 61: gamma 100.0, noise 0.160, cases 0-7 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.16 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise016_cases0_7_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise016_cases0_7_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 0: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09534, ambiguity_sequence_count 1, total_exact_score_count 4160
case 1: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12050, ambiguity_sequence_count 2, total_exact_score_count 8320
case 2: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08608, ambiguity_sequence_count 3, total_exact_score_count 12480
case 3: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07407, ambiguity_sequence_count 1, total_exact_score_count 4160
case 4: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07159, ambiguity_sequence_count 3, total_exact_score_count 12480
case 5: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07264, ambiguity_sequence_count 1, total_exact_score_count 4160
case 6: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10842, ambiguity_sequence_count 3, total_exact_score_count 12480
case 7: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02633, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.16 status:
  checked completed cases 0-7: pass 8 / 8.
  segment 0-7 minimum margin 0.02633 at case 7.
```

Interpretation:

- The first noise_std 0.160 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- The weakest point is already close to the 0.02 diagnostic line, so the next
  CPU-only direction is segmented continuation plus margin localization, not
  GPU/video.
- This extends the synthetic noise boundary beyond the completed noise_std
  0.140 64-case round but does not create a fixed-FPR or paper claim.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 60: gamma 100.0, noise 0.140, cases 56-63 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases56_63_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases56_63_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 56: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08083, ambiguity_sequence_count 1, total_exact_score_count 4160
case 57: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09971, ambiguity_sequence_count 1, total_exact_score_count 4160
case 58: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.08554, ambiguity_sequence_count 1, total_exact_score_count 4160
case 59: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13506, ambiguity_sequence_count 2, total_exact_score_count 8320
case 60: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08031, ambiguity_sequence_count 1, total_exact_score_count 4160
case 61: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10357, ambiguity_sequence_count 3, total_exact_score_count 12480
case 62: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07577, ambiguity_sequence_count 1, total_exact_score_count 4160
case 63: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12933, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.14 status:
  checked completed cases 0-63: pass 64 / 64.
  segment 56-63 minimum margin 0.07577 at case 62.
  overall checked minimum margin remains 0.02480 at case 45.
```

Interpretation:

- The full noise_std 0.140 64-message-index synthetic round passes 64 / 64
  under gamma 100.0, filler_multiplier 7, message_space 64, wrong_key_count 64,
  and ordered_bounded_global_c exact scoring.
- Owner/global recovery, owner-message recovery, and truth coverage all remain
  64 / 64 across the segmented run.
- The weakest checked point is case 45 with margin 0.02480, above but close to
  the 0.02 diagnostic line. This identifies the next CPU-only direction as
  margin localization or a higher-noise/harder-edit boundary search, not GPU.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 59: gamma 100.0, noise 0.140, cases 48-55 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases48_55_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases48_55_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 48: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07565, ambiguity_sequence_count 4, total_exact_score_count 16640
case 49: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06410, ambiguity_sequence_count 3, total_exact_score_count 12480
case 50: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09622, ambiguity_sequence_count 1, total_exact_score_count 4160
case 51: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11706, ambiguity_sequence_count 2, total_exact_score_count 8320
case 52: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10475, ambiguity_sequence_count 3, total_exact_score_count 12480
case 53: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03679, ambiguity_sequence_count 1, total_exact_score_count 4160
case 54: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07425, ambiguity_sequence_count 1, total_exact_score_count 4160
case 55: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.15162, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.14 status:
  checked completed cases 0-55: pass 56 / 56.
  segment 48-55 minimum margin 0.03679 at case 53.
  overall checked minimum margin remains 0.02480 at case 45.
```

Interpretation:

- The seventh noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Several cases again have truth exact false but truth coverage true, consistent
  with the public ambiguity-set mechanism rather than forced-path acquisition.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented coverage and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 58: gamma 100.0, noise 0.140, cases 40-47 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 40: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06942, ambiguity_sequence_count 1, total_exact_score_count 4160
case 41: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11945, ambiguity_sequence_count 1, total_exact_score_count 4160
case 42: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10791, ambiguity_sequence_count 2, total_exact_score_count 8320
case 43: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12444, ambiguity_sequence_count 1, total_exact_score_count 4160
case 44: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10607, ambiguity_sequence_count 1, total_exact_score_count 4160
case 45: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02480, ambiguity_sequence_count 1, total_exact_score_count 4160
case 46: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06267, ambiguity_sequence_count 1, total_exact_score_count 4160
case 47: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12221, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.14 status:
  checked completed cases 0-47: pass 48 / 48.
  segment 40-47 minimum margin 0.02480 at case 45.
  overall checked minimum margin is now 0.02480 at case 45.
```

Interpretation:

- The sixth noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- The weakest checked noise_std 0.140 point moved from case 34 to case 45 and
  is still above, but close to, the 0.02 diagnostic line.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented coverage and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 57: gamma 100.0, noise 0.140, cases 32-39 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13229, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10884, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02546, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06241, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06183, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07025, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12714, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08141, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.14 status:
  checked completed cases 0-39: pass 40 / 40.
  segment 32-39 minimum margin 0.02546 at case 34.
  overall checked minimum margin is now 0.02546 at case 34.
```

Interpretation:

- The fifth noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- The weakest checked noise_std 0.140 point moved from case 7 to case 34 and is
  still above, but close to, the 0.02 diagnostic line.
- This is still a CPU-only margin-localization result. It does not trigger a
  GPU boundary or a video-model requirement.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 56: gamma 100.0, noise 0.140, cases 24-31 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases24_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases24_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 24: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10268, ambiguity_sequence_count 24, total_exact_score_count 99840
case 25: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.14005, ambiguity_sequence_count 1, total_exact_score_count 4160
case 26: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09196, ambiguity_sequence_count 1, total_exact_score_count 4160
case 27: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11972, ambiguity_sequence_count 1, total_exact_score_count 4160
case 28: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07765, ambiguity_sequence_count 1, total_exact_score_count 4160
case 29: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08695, ambiguity_sequence_count 1, total_exact_score_count 4160
case 30: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10205, ambiguity_sequence_count 3, total_exact_score_count 12480
case 31: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12004, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.14 status:
  checked completed cases 0-31: pass 32 / 32.
  segment 24-31 minimum margin 0.07765 at case 28.
  overall checked minimum margin remains 0.02774 at case 7.
```

Interpretation:

- The fourth noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 28 again separates exact truth membership from coverage: the truth
  sequence is not the exact listed ambiguity sequence, but the public ambiguity
  set still covers it and owner/global scoring recovers the true message.
- The checked noise_std 0.140 range is now 32 / 32. The weakest point remains
  case 7, above but close to the 0.02 diagnostic line.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented coverage and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 55: gamma 100.0, noise 0.140, cases 16-23 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases16_23_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases16_23_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 16: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09089, ambiguity_sequence_count 1, total_exact_score_count 4160
case 17: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09457, ambiguity_sequence_count 3, total_exact_score_count 12480
case 18: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10483, ambiguity_sequence_count 2, total_exact_score_count 8320
case 19: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12681, ambiguity_sequence_count 1, total_exact_score_count 4160
case 20: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12179, ambiguity_sequence_count 1, total_exact_score_count 4160
case 21: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04096, ambiguity_sequence_count 3, total_exact_score_count 12480
case 22: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07473, ambiguity_sequence_count 6, total_exact_score_count 24960
case 23: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08735, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.14 status:
  checked completed cases 0-23: pass 24 / 24.
  segment 16-23 minimum margin 0.04096 at case 21.
  overall checked minimum margin remains 0.02774 at case 7.
```

Interpretation:

- The third noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- The checked noise_std 0.140 range is now 24 / 24. The weakest point remains
  case 7, above but close to the 0.02 diagnostic line.
- No GPU boundary has been reached; the next useful work remains CPU-only
  segmented coverage and margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 54: gamma 100.0, noise 0.140, cases 8-15 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases8_15_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases8_15_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 8: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04340, ambiguity_sequence_count 3, total_exact_score_count 12480
case 9: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05688, ambiguity_sequence_count 4, total_exact_score_count 16640
case 10: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13758, ambiguity_sequence_count 1, total_exact_score_count 4160
case 11: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09891, ambiguity_sequence_count 1, total_exact_score_count 4160
case 12: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06145, ambiguity_sequence_count 1, total_exact_score_count 4160
case 13: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13842, ambiguity_sequence_count 3, total_exact_score_count 12480
case 14: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04524, ambiguity_sequence_count 1, total_exact_score_count 4160
case 15: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.10462, ambiguity_sequence_count 9, total_exact_score_count 37440

combined noise0.14 status:
  checked completed cases 0-15: pass 16 / 16.
  segment 8-15 minimum margin 0.04340 at case 8.
  overall checked minimum margin remains 0.02774 at case 7.
```

Interpretation:

- The second noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- The current checked noise_std 0.140 range is 16 / 16. The weakest point is
  still case 7 from pass 53, above but close to the 0.02 diagnostic line.
- This remains rapid CPU-only synthetic method-feasibility evidence, not video
  or paper evidence.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 53: gamma 100.0, noise 0.140, cases 0-7 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.14 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise014_cases0_7_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise014_cases0_7_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 0: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09880, ambiguity_sequence_count 1, total_exact_score_count 4160
case 1: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11936, ambiguity_sequence_count 2, total_exact_score_count 8320
case 2: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08591, ambiguity_sequence_count 3, total_exact_score_count 12480
case 3: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07179, ambiguity_sequence_count 1, total_exact_score_count 4160
case 4: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07182, ambiguity_sequence_count 3, total_exact_score_count 12480
case 5: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07706, ambiguity_sequence_count 1, total_exact_score_count 4160
case 6: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10739, ambiguity_sequence_count 3, total_exact_score_count 12480
case 7: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02774, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.14 status:
  checked completed cases 0-7: pass 8 / 8.
  segment 0-7 minimum margin 0.02774 at case 7.
```

Interpretation:

- The first noise_std 0.140 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- The weakest point is again near the diagnostic line but still above it. This
  suggests the current CPU-only margin boundary is not yet localized at
  noise_std 0.140 cases 0-7.
- No GPU, video model, saved-video observation, fixed-FPR calibration, or paper
  claim is involved.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 52: gamma 100.0, noise 0.120, cases 56-63 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases56_63_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases56_63_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 56: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08231, ambiguity_sequence_count 1, total_exact_score_count 4160
case 57: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09854, ambiguity_sequence_count 1, total_exact_score_count 4160
case 58: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.08653, ambiguity_sequence_count 1, total_exact_score_count 4160
case 59: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13413, ambiguity_sequence_count 2, total_exact_score_count 8320
case 60: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08082, ambiguity_sequence_count 1, total_exact_score_count 4160
case 61: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10442, ambiguity_sequence_count 3, total_exact_score_count 12480
case 62: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07624, ambiguity_sequence_count 1, total_exact_score_count 4160
case 63: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12656, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.12 status:
  checked completed cases 0-63: pass 64 / 64.
  segment 56-63 minimum margin 0.07624 at case 62.
  overall checked minimum margin remains 0.02572 at case 34.
```

Interpretation:

- The full checked message-index range at noise_std 0.120 passes 64 / 64, with
  owner/global recovery 64 / 64 and truth coverage 64 / 64.
- The weakest checked point is case 34 at margin 0.02572, still above the 0.02
  diagnostic line but close enough to mark the current CPU-only boundary as
  margin stress rather than public acquisition failure.
- This strengthens the current AISB route under synthetic affine-channel
  diagnostics. It is still not a governance result, fixed-FPR result, video
  result, GPU result, or paper claim.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 51: gamma 100.0, noise 0.120, cases 48-55 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases48_55_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases48_55_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 48: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07746, ambiguity_sequence_count 4, total_exact_score_count 16640
case 49: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06630, ambiguity_sequence_count 3, total_exact_score_count 12480
case 50: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09797, ambiguity_sequence_count 1, total_exact_score_count 4160
case 51: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11958, ambiguity_sequence_count 2, total_exact_score_count 8320
case 52: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10500, ambiguity_sequence_count 3, total_exact_score_count 12480
case 53: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03094, ambiguity_sequence_count 1, total_exact_score_count 4160
case 54: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07359, ambiguity_sequence_count 1, total_exact_score_count 4160
case 55: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.14931, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.12 status:
  checked completed cases 0-55: pass 56 / 56.
  segment 48-55 minimum margin 0.03094 at case 53.
  overall checked minimum margin remains 0.02572 at case 34.
```

Interpretation:

- The seventh noise_std 0.120 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Multiple cases have truth exact false but truth covered true; this remains
  expected under public ambiguity-set scoring and is not treated as public path
  identification evidence.
- CPU-only segmented coverage has not hit a failure or GPU boundary yet.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 50: gamma 100.0, noise 0.120, cases 40-47 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 40: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06828, ambiguity_sequence_count 1, total_exact_score_count 4160
case 41: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11853, ambiguity_sequence_count 1, total_exact_score_count 4160
case 42: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10667, ambiguity_sequence_count 2, total_exact_score_count 8320
case 43: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12393, ambiguity_sequence_count 1, total_exact_score_count 4160
case 44: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10675, ambiguity_sequence_count 1, total_exact_score_count 4160
case 45: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02737, ambiguity_sequence_count 1, total_exact_score_count 4160
case 46: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06424, ambiguity_sequence_count 1, total_exact_score_count 4160
case 47: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12000, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.12 status:
  checked completed cases 0-47: pass 48 / 48.
  segment 40-47 minimum margin 0.02737 at case 45.
  overall checked minimum margin remains 0.02572 at case 34.
```

Interpretation:

- The sixth noise_std 0.120 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 45 is another thin-margin pass but remains above the 0.02 diagnostic
  line.
- The CPU-only synthetic evidence continues to support the AISB public
  ambiguity-set plus owner/wrong-key scoring route; no GPU or video test is
  required yet.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 49: gamma 100.0, noise 0.120, cases 32-39 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13126, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10899, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02572, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06382, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06231, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07263, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12624, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08104, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.12 status:
  checked completed cases 0-39: pass 40 / 40.
  segment 32-39 minimum margin 0.02572 at case 34.
  overall checked minimum margin is now 0.02572 at case 34.
```

Interpretation:

- The fifth noise_std 0.120 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 34 is now the weakest checked noise_std 0.120 point. It remains above
  the 0.02 diagnostic line but is close enough to keep the active boundary as
  trajectory-evidence margin.
- This still does not require GPU: the limiting factor remains CPU exact-score
  throughput and synthetic margin localization.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 48: gamma 100.0, noise 0.120, cases 24-31 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases24_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases24_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 24: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10672, ambiguity_sequence_count 24, total_exact_score_count 99840
case 25: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13762, ambiguity_sequence_count 1, total_exact_score_count 4160
case 26: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09459, ambiguity_sequence_count 1, total_exact_score_count 4160
case 27: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12124, ambiguity_sequence_count 1, total_exact_score_count 4160
case 28: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07662, ambiguity_sequence_count 1, total_exact_score_count 4160
case 29: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08616, ambiguity_sequence_count 1, total_exact_score_count 4160
case 30: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10172, ambiguity_sequence_count 3, total_exact_score_count 12480
case 31: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11009, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.12 status:
  checked completed cases 0-31: pass 32 / 32.
  segment 24-31 minimum margin 0.07662 at case 28.
  overall checked minimum margin remains 0.02647 at case 7.
```

Interpretation:

- The fourth noise_std 0.120 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 24 is the current CPU-throughput stress point in this noise tier:
  ambiguity_sequence_count 24 and 99,840 exact ordered candidates, but it still
  passes with margin 0.10672 under the same bounded exact scorer.
- The checked range is now 32 / 32 at noise_std 0.120. No GPU or video boundary
  is reached; the active next step remains segmented CPU coverage.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 47: gamma 100.0, noise 0.120, cases 16-23 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 16 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases16_23_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases16_23_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 16: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09661, ambiguity_sequence_count 1, total_exact_score_count 4160
case 17: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09581, ambiguity_sequence_count 3, total_exact_score_count 12480
case 18: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10460, ambiguity_sequence_count 2, total_exact_score_count 8320
case 19: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12934, ambiguity_sequence_count 1, total_exact_score_count 4160
case 20: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12285, ambiguity_sequence_count 1, total_exact_score_count 4160
case 21: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04487, ambiguity_sequence_count 3, total_exact_score_count 12480
case 22: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07521, ambiguity_sequence_count 6, total_exact_score_count 24960
case 23: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07764, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.12 status:
  checked completed cases 0-23: pass 24 / 24.
  segment 16-23 minimum margin 0.04487 at case 21.
  overall checked minimum margin remains 0.02647 at case 7.
```

Interpretation:

- The third noise_std 0.120 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Case 20 again has truth exact false but truth covered true, consistent with
  the current ambiguity-set scoring interpretation rather than a forced-path
  acquisition claim.
- The checked range is now 24 / 24 at noise_std 0.120. The weakest checked
  point remains near the diagnostic boundary, so the right CPU-only next step is
  continuing segmented coverage rather than moving to GPU.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 46: gamma 100.0, noise 0.120, cases 8-15 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 8 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases8_15_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases8_15_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 8: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04285, ambiguity_sequence_count 3, total_exact_score_count 12480
case 9: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05609, ambiguity_sequence_count 4, total_exact_score_count 16640
case 10: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13779, ambiguity_sequence_count 1, total_exact_score_count 4160
case 11: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09946, ambiguity_sequence_count 1, total_exact_score_count 4160
case 12: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06086, ambiguity_sequence_count 1, total_exact_score_count 4160
case 13: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13285, ambiguity_sequence_count 3, total_exact_score_count 12480
case 14: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04456, ambiguity_sequence_count 1, total_exact_score_count 4160
case 15: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.11002, ambiguity_sequence_count 9, total_exact_score_count 37440

combined noise0.12 status:
  checked completed cases 0-15: pass 16 / 16.
  segment 8-15 minimum margin 0.04285 at case 8.
  overall checked minimum margin remains 0.02647 at case 7.
```

Interpretation:

- The second noise_std 0.120 segment remains positive, with owner/global
  recovery 8 / 8 and truth coverage 8 / 8.
- Cases 9 and 15 again show that the true path need not be the exact public
  ambiguity winner, but it remains covered by the public ambiguity set and the
  owner score still wins over the wrong-key/message search.
- The weakest checked noise_std 0.120 margin is still case 7 from pass 45. This
  keeps the active synthetic boundary at trajectory-evidence span/margin rather
  than public acquisition.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 45: gamma 100.0, noise 0.120, cases 0-7 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.12 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 0 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise012_cases0_7_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise012_cases0_7_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 0: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09272, ambiguity_sequence_count 1, total_exact_score_count 4160
case 1: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12100, ambiguity_sequence_count 2, total_exact_score_count 8320
case 2: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09241, ambiguity_sequence_count 3, total_exact_score_count 12480
case 3: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07167, ambiguity_sequence_count 1, total_exact_score_count 4160
case 4: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07719, ambiguity_sequence_count 3, total_exact_score_count 12480
case 5: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07877, ambiguity_sequence_count 1, total_exact_score_count 4160
case 6: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10778, ambiguity_sequence_count 3, total_exact_score_count 12480
case 7: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02647, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.12 status:
  checked completed cases 0-7: pass 8 / 8.
  minimum margin 0.02647 at case 7.
```

Interpretation:

- Raising noise_std from 0.100 to 0.120 remains positive in the first checked
  8-case segment, with owner/global recovery 8 / 8 and truth coverage 8 / 8.
- The weakest point, case 7 at margin 0.02647, is close to the 0.02 diagnostic
  line. This is still a pass, but it indicates the synthetic margin boundary is
  likely nearby.
- The next CPU-only step is to continue segmented noise0.12 coverage, while
  separately preparing to localize the failure mode if a margin drops below
  0.02 or owner/wrong recovery fails.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 44: gamma 100.0, noise 0.100, cases 56-63 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 56 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_cases56_63_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_cases56_63_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 56: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08248, ambiguity_sequence_count 1, total_exact_score_count 4160
case 57: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09843, ambiguity_sequence_count 1, total_exact_score_count 4160
case 58: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.08885, ambiguity_sequence_count 1, total_exact_score_count 4160
case 59: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13720, ambiguity_sequence_count 2, total_exact_score_count 8320
case 60: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08222, ambiguity_sequence_count 1, total_exact_score_count 4160
case 61: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10425, ambiguity_sequence_count 3, total_exact_score_count 12480
case 62: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07754, ambiguity_sequence_count 1, total_exact_score_count 4160
case 63: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12546, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.10 status:
  cases 0-55 were already checked: pass 56 / 56,
    minimum margin 0.02995 across the prior range.
  cases 56-63 add pass 8 / 8, minimum margin 0.07754.
  checked completed cases 0-63: pass 64 / 64.
```

Interpretation:

- The noise_std 0.100 tier now completes the full 64-case checked
  message-index range with owner/global recovery 64 / 64 and truth coverage
  64 / 64.
- The weakest checked noise0.10 point remains case 45 at margin 0.02995, still
  above the 0.02 diagnostic line.
- This supports the current CPU-only scientific mechanism claim at the
  synthetic-construction level: AISB public ambiguity acquisition plus
  key-conditioned state scoring can remain separable under the tested high
  affine mismatch and higher observation noise.
- The next useful CPU-only scientific step is not GPU. It is to probe the
  noise/margin boundary above 0.10 or reduce secret-state span to locate a
  synthetic failure threshold.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-03 pass 43: gamma 100.0, noise 0.100, cases 48-55 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 48 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_cases48_55_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_cases48_55_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 48: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07649, ambiguity_sequence_count 4, total_exact_score_count 16640
case 49: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06771, ambiguity_sequence_count 3, total_exact_score_count 12480
case 50: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09765, ambiguity_sequence_count 1, total_exact_score_count 4160
case 51: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12024, ambiguity_sequence_count 2, total_exact_score_count 8320
case 52: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10516, ambiguity_sequence_count 3, total_exact_score_count 12480
case 53: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03239, ambiguity_sequence_count 1, total_exact_score_count 4160
case 54: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07269, ambiguity_sequence_count 1, total_exact_score_count 4160
case 55: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.15209, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.10 status:
  cases 0-47 were already checked: pass 48 / 48,
    minimum margin 0.02995 across the prior range.
  cases 48-55 add pass 8 / 8, minimum margin 0.03239.
  checked completed cases 0-55: pass 56 / 56.
```

Interpretation:

- The noise_std 0.100 tier now has 56 checked cases with owner/global recovery
  56 / 56 and truth coverage 56 / 56.
- The weakest checked noise0.10 point remains case 45 at margin 0.02995.
- Several cases in this segment again have ambiguity coverage without exact
  truth-sequence identity, reinforcing that the method criterion is shared
  public ambiguity support plus key-conditioned owner/wrong separation.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 42: gamma 100.0, noise 0.100, cases 40-47 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 40 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_cases40_47_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_cases40_47_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 40: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06699, ambiguity_sequence_count 1, total_exact_score_count 4160
case 41: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11344, ambiguity_sequence_count 1, total_exact_score_count 4160
case 42: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10909, ambiguity_sequence_count 2, total_exact_score_count 8320
case 43: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12184, ambiguity_sequence_count 1, total_exact_score_count 4160
case 44: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10755, ambiguity_sequence_count 1, total_exact_score_count 4160
case 45: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02995, ambiguity_sequence_count 1, total_exact_score_count 4160
case 46: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06470, ambiguity_sequence_count 1, total_exact_score_count 4160
case 47: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11777, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.10 status:
  cases 0-39 were already checked: pass 40 / 40,
    minimum margin 0.03029 across the prior range.
  cases 40-47 add pass 8 / 8, minimum margin 0.02995.
  checked completed cases 0-47: pass 48 / 48.
```

Interpretation:

- The noise_std 0.100 tier now has 48 checked cases with owner/global recovery
  48 / 48 and truth coverage 48 / 48.
- Case 45 is now the weakest checked noise0.10 point at margin 0.02995, still
  above the 0.02 diagnostic line but thin enough to keep tracking.
- The evidence still supports the current CPU-only direction: public AISB
  ambiguity coverage plus key-conditioned owner/wrong scoring remains positive
  in the checked synthetic envelope, while exact public alignment identity is
  not required.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 41: gamma 100.0, noise 0.100, cases 32-39 CPU segment

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 32 \
  --case-count 8 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_cases32_39_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_cases32_39_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12902, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10812, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03635, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06779, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06243, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07253, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12260, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08394, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.10 status:
  cases 0-31 were already checked: pass 32 / 32,
    minimum margin 0.03029 across the prior range.
  cases 32-39 add pass 8 / 8, minimum margin 0.03635.
  checked completed cases 0-39: pass 40 / 40.
```

Interpretation:

- The noise_std 0.100 tier now has 40 checked cases with owner/global recovery
  40 / 40 and truth coverage 40 / 40.
- The weakest checked noise0.10 point remains case 7 at margin 0.03029; case 34
  is the weakest point in the new cases 32-39 segment at margin 0.03635.
- Case 36 required 24,960 exact scores and still passed, so the current
  bottleneck remains CPU throughput for exact ambiguity scoring, not a GPU or
  real-video boundary.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 40: gamma 100.0, noise 0.100, cases 24-31 CPU segments

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 1 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_case24_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_case24_cases.jsonl \
  --progress-interval 2048

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 25 \
  --case-count 1 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_case25_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_case25_cases.jsonl \
  --progress-interval 2048

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.10 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 26 \
  --case-count 6 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c \
  --progress-jsonl /tmp/sc_sstw_noise010_cases26_31_progress.jsonl \
  --case-jsonl /tmp/sc_sstw_noise010_cases26_31_cases.jsonl \
  --progress-interval 4096
```

Result summary:

```text
case 24: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10655, ambiguity_sequence_count 24, total_exact_score_count 99840
case 25: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13875, ambiguity_sequence_count 1, total_exact_score_count 4160
case 26: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09155, ambiguity_sequence_count 1, total_exact_score_count 4160
case 27: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12346, ambiguity_sequence_count 1, total_exact_score_count 4160
case 28: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07178, ambiguity_sequence_count 1, total_exact_score_count 4160
case 29: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08834, ambiguity_sequence_count 1, total_exact_score_count 4160
case 30: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10088, ambiguity_sequence_count 3, total_exact_score_count 12480
case 31: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11367, ambiguity_sequence_count 2, total_exact_score_count 8320

combined noise0.10 status:
  cases 0-23 were already checked: pass 24 / 24,
    minimum margin 0.03029 across the prior range.
  cases 24-31 add pass 8 / 8, minimum margin 0.07178.
  checked completed cases 0-31: pass 32 / 32.
```

Interpretation:

- The noise_std 0.100 tier now has 32 checked cases with owner/global recovery
  32 / 32 and truth coverage 32 / 32.
- The weakest checked noise0.10 point remains case 7 at margin 0.03029.
- Case 24 is the current heaviest checked point in this tier, with 24 ambiguity
  sequences and 99,840 exact scores; progress JSONL makes the CPU run
  diagnosable without changing the final stdout report.
- Cases 28 and 20 show the same intended behavior: exact public truth-sequence
  identity is not required when the shared ambiguity set covers the truth and
  the key-conditioned owner/wrong scoring separates.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 39: gamma 100.0, noise 0.100, cases 16-23 single-case segments

Commands:

```bash
for case_index in 16 17 18 19 20 21 22 23; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.10 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 16: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09630, ambiguity_sequence_count 1, total_exact_score_count 4160
case 17: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09655, ambiguity_sequence_count 3, total_exact_score_count 12480
case 18: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09566, ambiguity_sequence_count 2, total_exact_score_count 8320
case 19: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13197, ambiguity_sequence_count 1, total_exact_score_count 4160
case 20: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12764, ambiguity_sequence_count 1, total_exact_score_count 4160
case 21: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03862, ambiguity_sequence_count 3, total_exact_score_count 12480
case 22: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07464, ambiguity_sequence_count 6, total_exact_score_count 24960
case 23: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07565, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.10 status:
  cases 0-15 were already checked: pass 16 / 16,
    minimum margin 0.03029 across the prior range.
  cases 16-23 add pass 8 / 8, minimum margin 0.03862.
  checked completed cases 0-23: pass 24 / 24.
```

Interpretation:

- The noise_std 0.100 tier now has 24 checked cases with owner/global recovery
  24 / 24 and truth coverage 24 / 24.
- The weakest checked noise0.10 point remains case 7 at margin 0.03029.
- Case 20 again confirms that exact public truth-sequence identity is not the
  criterion; shared ambiguity-set coverage and owner/wrong scoring are.
- More CPU-only segmented coverage remains available before any GPU or real
  video boundary.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 38: gamma 100.0, noise 0.100, cases 8-15 single-case segments

Commands:

```bash
for case_index in 8 9 10 11 12 13 14 15; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.10 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 8: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03992, ambiguity_sequence_count 3, total_exact_score_count 12480
case 9: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.05393, ambiguity_sequence_count 4, total_exact_score_count 16640
case 10: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13936, ambiguity_sequence_count 1, total_exact_score_count 4160
case 11: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10013, ambiguity_sequence_count 1, total_exact_score_count 4160
case 12: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06500, ambiguity_sequence_count 1, total_exact_score_count 4160
case 13: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12676, ambiguity_sequence_count 3, total_exact_score_count 12480
case 14: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.04512, ambiguity_sequence_count 1, total_exact_score_count 4160
case 15: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.10805, ambiguity_sequence_count 9, total_exact_score_count 37440

combined noise0.10 status:
  cases 0-7 were already checked: pass 8 / 8, minimum margin 0.03029.
  cases 8-15 add pass 8 / 8, minimum margin 0.03992.
  checked completed cases 0-15: pass 16 / 16.
```

Interpretation:

- The noise_std 0.100 tier remains positive for all completed cases 0-15.
- The weakest completed noise0.10 point remains case 7 at margin 0.03029.
- Case 15 initially exposed a CPU throughput/no-output issue in the ordering
  heuristic before exact scoring. Adding diagnostic JSONL progress showed the
  hotspot was before the exact candidate loop; replacing only the ordering
  heuristic with a sparse aligned-distance score made the same exact/bounded
  scorer finish and pass.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 37: gamma 100.0, noise 0.100, cases 0-7 single-case segments

Commands:

```bash
for case_index in 0 1 2 3 4 5 6 7; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.10 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 0: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09137, ambiguity_sequence_count 1, total_exact_score_count 4160
case 1: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12706, ambiguity_sequence_count 2, total_exact_score_count 8320
case 2: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.09231, ambiguity_sequence_count 3, total_exact_score_count 12480
case 3: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07652, ambiguity_sequence_count 1, total_exact_score_count 4160
case 4: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07484, ambiguity_sequence_count 3, total_exact_score_count 12480
case 5: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08166, ambiguity_sequence_count 1, total_exact_score_count 4160
case 6: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11334, ambiguity_sequence_count 3, total_exact_score_count 12480
case 7: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03029, ambiguity_sequence_count 1, total_exact_score_count 4160

combined noise0.10 cases 0-7 status:
  pass 8 / 8, owner/global 8 / 8, truth covered 8 / 8,
    minimum margin 0.03029.
```

Interpretation:

- Raising noise_std from 0.080 to 0.100 remains positive in the first checked
  8 single-case segments at the full 64-message / 64-wrong-key tier.
- The weakest checked point is case 7 with margin 0.03029, so the margin is
  still above the 0.02 diagnostic line but remains thin enough to justify more
  segmented CPU-only coverage before considering any real-video boundary.
- Exact public truth-sequence identity remains unnecessary; case 4 is covered
  by the shared ambiguity set even though the exact truth sequence is not
  directly present.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 36: gamma 100.0, noise 0.080, cases 56-63 single-case segments

Commands:

```bash
for case_index in 56 57 58 59 60 61 62 63; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.08 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 56: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08561, ambiguity_sequence_count 1, total_exact_score_count 4160
case 57: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10042, ambiguity_sequence_count 1, total_exact_score_count 4160
case 58: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.08825, ambiguity_sequence_count 1, total_exact_score_count 4160
case 59: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.13815, ambiguity_sequence_count 2, total_exact_score_count 8320
case 60: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08357, ambiguity_sequence_count 1, total_exact_score_count 4160
case 61: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10408, ambiguity_sequence_count 3, total_exact_score_count 12480
case 62: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.07797, ambiguity_sequence_count 1, total_exact_score_count 4160
case 63: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12636, ambiguity_sequence_count 2, total_exact_score_count 8320

combined segmented 0-63 status:
  cases 0-55 were already checked at filler_multiplier 7: pass 56 / 56,
    minimum margin 0.02597 across the prior range.
  cases 56-63 add pass 8 / 8, minimum margin 0.07797.
  total checked segmented range: pass 64 / 64.
```

Interpretation:

- The full 64-message index range is now checked at gamma 100.0, noise_std
  0.080, 64 messages, and 64 wrong keys: 64 / 64 synthetic pass.
- The weakest checked point remains case 45 with margin 0.02597, still above
  the 0.02 diagnostic line. The active scientific boundary remains
  trajectory-evidence margin under synthetic affine channels.
- Exact public truth-sequence identity remains unnecessary; truth coverage by
  the public ambiguity set plus shared owner/wrong scoring is the relevant
  mechanism.
- The only practical issue observed in this tier is CPU batching throughput for
  large monolithic runs. GPU is still not required for the current synthetic
  mechanism question.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 35: gamma 100.0, noise 0.080, cases 48-55 single-case segments

Commands:

```bash
for case_index in 48 49 50 51 52 53 54 55; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.08 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 48: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07681, ambiguity_sequence_count 4, total_exact_score_count 16640
case 49: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06627, ambiguity_sequence_count 3, total_exact_score_count 12480
case 50: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.09976, ambiguity_sequence_count 1, total_exact_score_count 4160
case 51: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11487, ambiguity_sequence_count 2, total_exact_score_count 8320
case 52: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10533, ambiguity_sequence_count 3, total_exact_score_count 12480
case 53: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03381, ambiguity_sequence_count 1, total_exact_score_count 4160
case 54: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07654, ambiguity_sequence_count 1, total_exact_score_count 4160
case 55: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.15153, ambiguity_sequence_count 2, total_exact_score_count 8320

combined segmented 0-55 status:
  cases 0-47 were already checked at filler_multiplier 7: pass 48 / 48,
    minimum margin 0.02597 across the prior range.
  cases 48-55 add pass 8 / 8, minimum margin 0.03381.
  total checked segmented range: pass 56 / 56.
```

Interpretation:

- The checked gamma 100.0 / noise_std 0.080 / 64-message / 64-wrong-key tier
  now reaches 56 deterministic case indices without owner/global recovery
  failure.
- The current weakest checked point remains case 45 from the previous segment
  at margin 0.02597.
- Cases 49, 50, and 55 reinforce that exact public truth-sequence identity is
  not required for the shared ambiguity-set scoring mechanism.
- More CPU-only segmented coverage remains possible before any GPU or real
  video boundary is reached.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 34: gamma 100.0, noise 0.080, cases 40-47 single-case segments

Commands:

```bash
for case_index in 40 41 42 43 44 45 46 47; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.08 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 40: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06692, ambiguity_sequence_count 1, total_exact_score_count 4160
case 41: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11233, ambiguity_sequence_count 1, total_exact_score_count 4160
case 42: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10970, ambiguity_sequence_count 2, total_exact_score_count 8320
case 43: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.12614, ambiguity_sequence_count 1, total_exact_score_count 4160
case 44: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11132, ambiguity_sequence_count 1, total_exact_score_count 4160
case 45: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.02597, ambiguity_sequence_count 1, total_exact_score_count 4160
case 46: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.07165, ambiguity_sequence_count 1, total_exact_score_count 4160
case 47: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.11642, ambiguity_sequence_count 1, total_exact_score_count 4160

combined segmented 0-47 status:
  cases 0-39 were already checked at filler_multiplier 7: pass 40 / 40,
    minimum margin 0.02771 across the prior range.
  cases 40-47 add pass 8 / 8, minimum margin 0.02597.
  total checked segmented range: pass 48 / 48.
```

Interpretation:

- The checked synthetic envelope now covers 48 deterministic case indices at
  gamma 100.0, noise_std 0.080, 64 messages, and 64 wrong keys with no owner or
  global recovery failure.
- Case 45 is the current weakest checked point, with margin 0.02597. This is
  still above the 0.02 diagnostic line but confirms the active boundary remains
  trajectory-evidence margin, not AISB acquisition.
- Cases 43 and 36 both show that exact public truth-sequence identity is not
  required; truth coverage plus shared alignment-set owner/wrong scoring is the
  active synthetic mechanism.
- More CPU-only segmented coverage is still possible. GPU is not required for
  the current synthetic mechanism question.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 33: gamma 100.0, noise 0.080, cases 32-39 single-case segments

Commands:

```bash
for case_index in 32 33 34 35 36 37 38 39; do
  PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
    --gamma 100.0 \
    --noise-std 0.08 \
    --residual-threshold 0.0125 \
    --near-tie-ratio 5.0 \
    --filler-multiplier 7 \
    --message-space-size 64 \
    --wrong-key-count 64 \
    --start-index "${case_index}" \
    --case-count 1 \
    --workers 1 \
    --scoring-mode ordered_bounded_global_c
done
```

Result summary:

```text
case 32: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12895, ambiguity_sequence_count 2, total_exact_score_count 8320
case 33: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.10829, ambiguity_sequence_count 1, total_exact_score_count 4160
case 34: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.03403, ambiguity_sequence_count 2, total_exact_score_count 8320
case 35: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06691, ambiguity_sequence_count 1, total_exact_score_count 4160
case 36: pass, owner/global 1/1, truth covered 1/1, truth exact 0/1,
  margin 0.06251, ambiguity_sequence_count 6, total_exact_score_count 24960
case 37: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.06973, ambiguity_sequence_count 1, total_exact_score_count 4160
case 38: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.12338, ambiguity_sequence_count 1, total_exact_score_count 4160
case 39: pass, owner/global 1/1, truth covered 1/1, truth exact 1/1,
  margin 0.08554, ambiguity_sequence_count 2, total_exact_score_count 8320

combined segmented 0-39 status:
  cases 0-31 were already checked at filler_multiplier 7: pass 32 / 32,
    minimum margin 0.02771.
  cases 32-39 add pass 8 / 8, minimum margin 0.03403.
  total checked segmented range: pass 40 / 40.
```

Interpretation:

- The harder gamma 100.0 / noise_std 0.080 / 64-message / 64-wrong-key tier
  remains positive through the checked segmented 40-case range when the
  secret-state filler span is 7.
- The weakest new case is case 34 with margin 0.03403, still above the 0.02
  diagnostic line.
- Case 36 again shows why exact public truth-sequence identity is not the
  criterion: the exact truth path is not directly present, but the shared
  ambiguity set still covers truth and owner/wrong scoring recovers the owner
  message with margin 0.06251.
- The active limitation remains CPU exact-scorer throughput/batching, not a
  GPU/video requirement and not a method-mechanism failure.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 32: gamma 100.0, noise 0.080, segmented 32-case check

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.08 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 16 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode ordered_bounded_global_c

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.08 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 24 \
  --case-count 8 \
  --workers 4 \
  --scoring-mode ordered_bounded_global_c
```

Result summary:

```text
single 32-case batch:
  interrupted after >40 minutes without output; no scientific failure observed.

segmented 16-23:
  pass 8 / 8, truth covered 8 / 8, owner 8 / 8, global 8 / 8,
    min_margin 0.03817, max total_exact_score_count 24960

segmented 24-31:
  pass 8 / 8, truth covered 8 / 8, owner 8 / 8, global 8 / 8,
    min_margin 0.07430, max total_exact_score_count 99840

combined segmented 0-31 status:
  cases 0-15 were already checked at filler_multiplier 7: pass 16 / 16,
    min_margin 0.02771.
  cases 16-31 add pass 16 / 16.
  total checked segmented range: pass 32 / 32.
```

Interpretation:

- The full 32-case range is scientifically positive when run in CPU-manageable
  segments.
- The interrupted single 32-case batch is a local CPU throughput/batching issue,
  not a method failure and not a GPU/video boundary.
- Exact public truth sequence identity is still not required by the ambiguity
  set route; shared-set owner/wrong scoring remains the active evidence layer.
- The next useful CPU-only work is either more segmented coverage or profiling
  the exact scorer throughput. GPU is not required for the current synthetic
  mechanism question.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 31: gamma 100.0, noise 0.080, 16-case span check

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 100.0 \
  --noise-stds 0.080 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 16 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 16 \
  --ambiguity-workers 1

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.08 \
  --residual-threshold 0.0125 \
  --near-tie-ratio 5.0 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --start-index 0 \
  --case-count 16 \
  --workers 1 \
  --scoring-mode ordered_bounded_global_c
```

Result summary:

```text
filler_multiplier 5, ambiguity_set, 64 messages / 64 wrong keys / 16 cases:
  gamma_100.0_noise_0.080: pass 15 / 16, truth covered 16 / 16, truth exact 16 / 16,
    owner 16 / 16, global 16 / 16, min_margin 0.01957

case 12 targeted localization:
  filler_multiplier 6: pass 1 / 1, owner/global/truth 1 / 1, min_margin 0.06477

filler_multiplier 6, 16 cases:
  pass 15 / 16, truth covered 16 / 16, truth exact 16 / 16,
    owner 16 / 16, global 16 / 16, min_margin 0.01928

case 14 targeted localization:
  filler_multiplier 7: pass 1 / 1, owner/global/truth 1 / 1, min_margin 0.04816

filler_multiplier 7, 16 cases:
  pass 16 / 16, truth covered 16 / 16, truth exact 13 / 16,
    owner 16 / 16, global 16 / 16, min_margin 0.02771
```

Interpretation:

- The first 16-case hard-noise run exposes a thin diagnostic-margin boundary,
  not an owner/wrong or public-coverage failure.
- Increasing secret-state filler span restores the checked tier. This suggests
  the active synthetic mechanism is viable but margin-limited: longer
  trajectory evidence improves separation.
- Exact public truth sequence uniqueness is not required by the ambiguity-set
  route. At filler_multiplier 7, truth coverage remains 16 / 16 while exact
  truth sequence identity is 13 / 16; owner/wrong scoring over the shared public
  ambiguity set still recovers the owner message and global owner in all cases.
- The next CPU-only question is larger coverage under filler_multiplier 7 or a
  deliberately harder noise/edit distribution. No GPU is required for this
  synthetic mechanism question.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 30: full-tier gamma 100.0, noise 0.080 check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 100.0 \
  --noise-stds 0.080 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1
```

Result summary:

```text
single_path:
  gamma_100.0_noise_0.080: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.05035

ambiguity_set, 64 messages / 64 wrong keys / 8 cases:
  gamma_100.0_noise_0.080: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.02655
```

Interpretation:

- The full-tier ambiguity-set route remains positive at noise_std 0.080, but
  margin remains close to the 0.02 diagnostic line.
- Since raising noise from 0.040 to 0.080 did not collapse the margin, the next
  useful CPU-only check is wider case-count coverage under the same harder
  noise.
- Forced single-path public alignment remains a negative diagnostic.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 29: full-tier gamma 100.0, noise 0.040/0.050 checks

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 100.0 \
  --noise-stds 0.040 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1

PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 100.0 \
  --noise-stds 0.050 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1
```

Result summary:

```text
single_path:
  gamma_100.0_noise_0.040: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.05044
  gamma_100.0_noise_0.050: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.05044

ambiguity_set, 64 messages / 64 wrong keys / 8 cases:
  gamma_100.0_noise_0.040: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.02491
  gamma_100.0_noise_0.050: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.02645
```

Interpretation:

- The full-tier ambiguity-set route still passes at noise_std 0.040 and 0.050,
  but the minimum margin is now near the 0.02 diagnostic line.
- This is the first checked stress setting in this pass sequence that looks
  close to a synthetic robustness boundary. The next useful CPU-only step is to
  bracket the noise boundary more aggressively, not to move to GPU/video.
- Forced single-path public alignment remains a negative diagnostic.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 28: full-tier gamma 100.0, noise 0.020 check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 100.0 \
  --noise-stds 0.020 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1
```

Result summary:

```text
single_path:
  gamma_100.0_noise_0.020: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.05040

ambiguity_set, 64 messages / 64 wrong keys / 8 cases:
  gamma_100.0_noise_0.020: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.03601
```

Interpretation:

- Raising noise_std from 0.016 to 0.020 at gamma 100.0 does not break the
  checked full-tier ambiguity-set route.
- The minimum margin remains above the 0.02 diagnostic line and is essentially
  unchanged from the noise 0.016 run, so the next CPU-only stress should use a
  larger noise jump or harder edit distribution.
- This is still only rapid synthetic feasibility triage. It is not a
  governance, reproducibility, fixed-FPR, or paper-readiness result.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 27: full-tier gamma 100.0 hard-cell check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 100.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1
```

Result summary:

```text
single_path:
  gamma_100.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.05039

ambiguity_set, 64 messages / 64 wrong keys / 8 cases:
  gamma_100.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.03597
```

Interpretation:

- The full-tier ambiguity-set route remains positive at gamma 100.0 across the
  checked 8 cases.
- The minimum margin is similar to gamma 50.0 rather than collapsing, so in the
  current synthetic generator the next useful stress axis is noise/edit severity
  or case-count coverage, not GPU/video execution.
- Forced single-path public alignment remains only a negative diagnostic.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 26: full-tier gamma 30.0/50.0 hard-cell check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 30.0,50.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8 \
  --ambiguity-workers 1
```

Result summary:

```text
single_path:
  gamma_30.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 7 / 8, min_margin -0.05068
  gamma_50.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 7 / 8, min_margin -0.05052

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_30.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.03758
  gamma_50.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.03673
```

Interpretation:

- The full-tier ambiguity-set route remains positive at gamma 30.0 and 50.0
  across the checked 8 cases.
- The minimum margin remains above the 0.02 diagnostic line. In this checked
  range it no longer decreases sharply, so the active next CPU-only question is
  whether higher gamma, higher case count, or a harder noise/edit setting
  exposes a synthetic failure before CPU throughput becomes the blocker.
- Forced single-path public alignment remains a negative diagnostic and should
  not be revived as the active method path.

Evidence boundary remains unchanged: rapid CPU-only synthetic relation-channel
diagnostics for method-mechanism feasibility; no governance result, video, GPU,
fixed-FPR calibration, reproducibility claim, paper claim, or deployment claim.


## 2026-08-02 pass 16: full-tier gamma 8.0/10.0 hard-cell check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 8.0,10.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_8.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.05110
  gamma_10.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 7 / 8, min_margin -0.05111

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_8.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.04885
  gamma_10.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.04410
```

Interpretation:

- The ambiguity-set route remains positive through gamma 10.0 in the full
  64-message / 64-wrong-key tier.
- The margin is decreasing but still above the 0.02 diagnostic line.
- The next CPU-only boundary probe should test a larger gamma jump, such as
  gamma 15.0 and 20.0, before considering any real-video/GPU work.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 17: full-tier gamma 4.0/5.0 hard-cell check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 4.0,5.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_4.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.04956
  gamma_5.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 7 / 8, min_margin -0.05052

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_4.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.05059
  gamma_5.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.04987
```

Interpretation:

- The ambiguity-set route remains positive through gamma 5.0 in the full
  64-message / 64-wrong-key tier.
- The minimum margin is still safely above the 0.02 diagnostic line, but it is
  now much thinner than at gamma 1.0 to 3.0.
- The next CPU-only probe should continue the high-gamma search, for example
  gamma 8.0 and 10.0 at the same full tier.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 18: full-tier gamma 2.5/3.0 hard-cell check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 2.5,3.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_2.5_noise_0.016: pass 1 / 8, alignment 3 / 8, owner 6 / 8, min_margin -0.04037
  gamma_3.0_noise_0.016: pass 1 / 8, alignment 2 / 8, owner 6 / 8, min_margin -0.04337

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_2.5_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.06521
  gamma_3.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.05906
```

Interpretation:

- The ambiguity-set route remains positive through gamma 3.0 in the full
  64-message / 64-wrong-key tier, but the margin is trending downward.
- Forced single-path is no longer a useful comparator except as a negative
  control: it remains 1 / 8 pass and loses owner-message recovery.
- The next CPU-only probe should continue the gamma margin search above 3.0 or
  increase case count at gamma 3.0; this still does not require GPU/video.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 19: full-tier gamma 1.6/2.0 hard-cell check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 1.6,2.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_1.6_noise_0.016: pass 3 / 8, alignment 5 / 8, owner 7 / 8, min_margin -0.03260
  gamma_2.0_noise_0.016: pass 1 / 8, alignment 3 / 8, owner 6 / 8, min_margin -0.03531

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_1.6_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.08160
  gamma_2.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.07753
```

Interpretation:

- The ambiguity-set route remains positive through gamma 2.0 in the full
  64-message / 64-wrong-key tier, although the minimum margin is decreasing.
- Forced single-path acquisition is now clearly non-viable for this stress
  envelope: it falls to 1 / 8 pass and 6 / 8 owner recovery at gamma 2.0.
- The next CPU-only margin-boundary probe is gamma above 2.0, not GPU/video.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 20: full-tier high-gamma hard-cell check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 1.2,1.4 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_1.2_noise_0.016: pass 3 / 8, alignment 5 / 8, owner 7 / 8, min_margin -0.03457
  gamma_1.4_noise_0.016: pass 3 / 8, alignment 5 / 8, owner 7 / 8, min_margin -0.03756

ambiguity_set, 64 messages / 64 wrong keys / 8 cases per high-gamma cell:
  gamma_1.2_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.09883
  gamma_1.4_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.08829
```

Interpretation:

- The ambiguity-set route remains positive in the full candidate tier after
  increasing deterministic mismatch to gamma 1.2 and 1.4.
- Forced single-path acquisition degrades further, including owner-message
  misses, which reinforces that the public ambiguity set should be preserved
  through owner/wrong scoring.
- The next CPU-only boundary to map is the gamma margin transition or longer
  case-count coverage at full tier; this still does not require GPU/video.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 21: full-tier hard-cell 16-case scale check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 1.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 16 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 16
```

Result summary:

```text
single_path:
  gamma_1.0_noise_0.016: pass 6 / 16, alignment 11 / 16,
    owner 15 / 16, min_margin -0.01633

ambiguity_set, 64 messages / 64 wrong keys / 16 cases:
  gamma_1.0_noise_0.016: pass 16 / 16, truth covered 16 / 16,
    truth exact 16 / 16, owner 16 / 16, global 16 / 16,
    candidate_count 4160 per case, min_margin 0.11003
```

Interpretation:

- The forced single-path route now fails both alignment and one owner-message
  recovery case under the hard-cell 16-case check.
- The public ambiguity-set route remains positive at the full 64-message /
  64-wrong-key candidate tier across all 16 checked cases, with unchanged
  minimum margin relative to the 8-case check.
- This strengthens the method mechanism conclusion: carry public ambiguity
  through scoring; do not collapse acquisition to one public path.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 22: full-tier hard-cell ambiguity-set scale check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 1.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 64 \
  --ambiguity-wrong-key-count 64 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_1.0_noise_0.016: pass 4 / 8, alignment 5 / 8, owner 8 / 8, min_margin -0.01633

ambiguity_set, 64 messages / 64 wrong keys / 8 cases:
  gamma_1.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, truth exact 8 / 8,
    owner 8 / 8, global 8 / 8, candidate_count 4160 per case, min_margin 0.11003
```

Interpretation:

- The full 64-message / 64-wrong-key candidate tier remains positive on the
  hardest checked gamma 1.0/noise 0.016 cell across 8 cases.
- This is the strongest current local evidence that the public ambiguity-set
  method fixes the forced-alignment failure without needing GPU/video evidence.
- The active boundary remains CPU scale mapping: larger case counts and harder
  synthetic mismatch are still local exact-scoring workloads until they either
  expose a scientific failure or become a practical CPU-throughput boundary.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 23: hard-cell ambiguity-set candidate-space scale check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 0.8,1.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 4 \
  --ambiguity-message-space-size 24 \
  --ambiguity-wrong-key-count 24 \
  --ambiguity-case-count 4
```

Result summary:

```text
single_path:
  gamma_0.8_noise_0.016: pass 3 / 4, alignment 3 / 4, owner 4 / 4, min_margin 0.02473
  gamma_1.0_noise_0.016: pass 2 / 4, alignment 3 / 4, owner 4 / 4, min_margin 0.01591

ambiguity_set, 24 messages / 24 wrong keys / 4 cases per hard cell:
  gamma_0.8_noise_0.016: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, global 4 / 4, min_margin 0.21298
  gamma_1.0_noise_0.016: pass 4 / 4, truth covered 4 / 4, owner 4 / 4, global 4 / 4, min_margin 0.16866
```

Interpretation:

- Increasing the ambiguity-set owner/wrong candidate space from 16 messages to
  24 messages while retaining 24 wrong keys preserves all checked hard-cell
  passes.
- This strengthens the current CPU-only feasibility direction: the scientific
  mechanism is not blocked at this medium candidate tier, and the next useful
  local work is further ambiguity-set scale mapping.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-02 pass 24: hard-cell ambiguity-set scale check

Command:

```bash
python3 experiments/run_aisb_stress_grid_probe.py \
  --gammas 0.8,1.0 \
  --noise-stds 0.016 \
  --message-space-size 16 \
  --wrong-key-count 24 \
  --case-count 8 \
  --ambiguity-message-space-size 16 \
  --ambiguity-wrong-key-count 24 \
  --ambiguity-case-count 8
```

Result summary:

```text
single_path:
  gamma_0.8_noise_0.016: pass 5 / 8, alignment 5 / 8, owner 8 / 8, min_margin 0.00258
  gamma_1.0_noise_0.016: pass 4 / 8, alignment 5 / 8, owner 8 / 8, min_margin -0.01633

ambiguity_set, 16 messages / 24 wrong keys / 8 cases per hard cell:
  gamma_0.8_noise_0.016: pass 8 / 8, truth covered 8 / 8, owner 8 / 8, global 8 / 8, min_margin 0.18767
  gamma_1.0_noise_0.016: pass 8 / 8, truth covered 8 / 8, owner 8 / 8, global 8 / 8, min_margin 0.16001
```

Interpretation:

- The hardest previously checked cells confirm that single-path public
  alignment is the wrong active acquisition policy: owner-message recovery can
  remain 8 / 8 while forced alignment and diagnostic margin fail.
- The ambiguity-set route remains positive across 8 checked cases per hard
  cell and preserves strong margins, so the active CPU-only route is still to
  scale ambiguity-set evidence rather than move to GPU/video.

Evidence boundary remains unchanged: synthetic relation-channel diagnostics
only; no video, GPU, fixed-FPR calibration, or paper claim.


## 2026-08-05 pass 246: burst16 noise0.62 threshold0.02 diagnostic-pruned cost reference

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.62 \
  --residual-threshold 0.02 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std=0.62, residual_threshold=0.02, completed cases 0-32:
  pass: 33 / 33
  truth sequence covered: 33 / 33
  owner/global message recovery: 33 / 33
  min score margin: 0.07635967
  mean score margin: 0.11830551
  max ambiguity sequence count: 428
  max exact-score count after diagnostic screen: 64619

case 32-39 segment was stopped after case 32 completed because the next case
remained in exact C scoring for several minutes. This is recorded as a CPU cost
boundary, not as a scientific fail.
```

Interpretation:

- The wider residual threshold 0.02 preserves the checked acquisition and
  owner/wrong-key mechanism on the completed 33 cases, including the previously
  failing lower-threshold region.
- The same setting substantially increases exact-search cost: the observed
  max exact-score count rose to 64619 and the next case in the 32-39 segment
  became too slow for the current quick CPU-only mapping loop.
- The active operating choice should therefore not continue widening the public
  residual threshold. For the current CPU-only scientific triage, 0.015 remains
  the pragmatic diagnostic setting, while 0.01251953125 marks the narrowest
  checked full-layer pass above the failing 0.0125 threshold.

Evidence boundary remains unchanged: CPU-only synthetic mechanism triage only;
no GPU, real video, fixed-FPR calibration, governance evidence, reproducibility
claim, or paper claim.


## 2026-08-05 pass 247: burst16 noise0.64 threshold0.015 diagnostic-pruned CPU-boundary sample

Command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py \
  --gamma 100.0 \
  --noise-std 0.64 \
  --residual-threshold 0.015 \
  --near-tie-ratio 5.0 \
  --per-cluster-limit 3 \
  --top-k-per-start 2 \
  --filler-multiplier 7 \
  --message-space-size 64 \
  --wrong-key-count 64 \
  --burst-count 16 \
  --workers 1 \
  --scoring-mode diagnostic_pruned_c \
  --diagnostic-top-k-global 64 \
  --diagnostic-top-k-owner 64 \
  --diagnostic-top-k-per-wrong-key 1
```

Result summary:

```text
noise_std=0.64, residual_threshold=0.015, completed cases 0-10:
  pass: 11 / 11
  truth sequence covered: 11 / 11
  owner/global message recovery: 11 / 11
  min score margin: 0.09242166
  mean score margin: 0.11595991
  max ambiguity sequence count: 144
  max exact-score count after diagnostic screen: 22153

case 8-15 segment was stopped after case 10 completed because the next case
remained in exact C scoring for several minutes. This is recorded as a CPU
exact-search cost boundary, not as a scientific fail.
```

Interpretation:

- Raising the synthetic observation noise from 0.62 to 0.64 at the pragmatic
  residual threshold 0.015 preserves the checked mechanism on the completed
  11 cases: AISB truth coverage, owner/global recovery, and positive margin
  all remain intact.
- The local blocker is now exact CPU scoring latency on some cases, not a need
  for GPU/video and not a mechanism failure.
- The next productive CPU-only work should target exact scoring cost reduction
  or a smaller diagnostic candidate slice for boundary localization before
  spending full 64x64 exact layers at higher noise.

Evidence boundary remains unchanged: CPU-only synthetic mechanism triage only;
no GPU, real video, fixed-FPR calibration, governance evidence, reproducibility
claim, or paper claim.


## 2026-08-05 boundary 248: burst16 two-point burst-internal deletion acquisition check

Probe summary:

```text
noise_std=0.62, residual_threshold=0.015, gamma=100.0, burst_count=16,
filler_multiplier=7, current AISB scanner allow_single_deletion=True.

Two internal points were deleted from every retained public burst. The current
public candidate model can encode at most one missing template point.

8-case acquisition-only result:
  case_count: 8
  truth coverage under current single-deletion candidate model: 0 / 8
  truth_count per case: 16
  accepted_count range under current model: 0..6
```

Interpretation:

- This is a useful synthetic boundary, not a scoring or owner/wrong-key failure.
  The current AISB public acquisition model is explicitly single-deletion; with
  two missing points per burst, the true synchronization path is not representable
  in the current candidate key.
- Therefore Stage C2 should stop here unless the method is extended with a
  double-missing AISB template/candidate contract. Running 64-case owner/wrong
  scoring on top of an unrepresentable public path would be wasted CPU.
- C1 and single burst-internal deletion are already covered by the threshold
  runs because the exact sequence ambiguity probe uses crop, non-burst
  deletion/repeat, and one missing point per retained burst.

Evidence boundary remains unchanged: CPU-only synthetic mechanism triage only;
no GPU, real video, fixed-FPR calibration, governance evidence, reproducibility
claim, or paper claim.


## 2026-08-05 pass 249: temporal robustness C2/C3/C4 CPU-only synthetic layers

Implemented and checked a double-missing AISB acquisition extension and a
CPU-only temporal robustness probe. The double-missing extension adds a
12-point public AISB template with two redundant copies of each anchor class;
this is required because the 9-point single-deletion template cannot guarantee
anchor-class survival when two burst samples are missing.

Result summary:

```text
C2 burst-internal two-point deletion, 64 cases:
  acquisition pass: 64 / 64
  owner/wrong pass: 64 / 64
  false positives + false negatives: 0
  min owner/wrong margin: 0.30199808
  mean owner/wrong margin: 0.39259782

C3 piecewise synthetic clock distortion, 64 cases:
  pass: 64 / 64
  exact public alignment: 64 / 64
  truth covered by public candidates: 64 / 64
  false positives: 0
  false negatives: 0
  min owner/wrong margin: 0.30039331
  mean owner/wrong margin: 0.35337501

C4 combined perturbation, 64 cases:
  pass: 64 / 64
  exact single-path public alignment: 62 / 64
  truth covered by public candidates: 64 / 64
  forced single-path false positives: 2
  forced single-path false negatives: 2
  min owner/wrong margin: 0.29703240
  mean owner/wrong margin: 0.36661783
```

Interpretation:

- C2 is no longer blocked after switching from the 9-point single-deletion
  redundant template to a 12-point double-redundant template. The failure mode
  was public acquisition representability, not owner/wrong scoring.
- C3 passes under piecewise synthetic speed/clock edits with exact public path
  recovery and strong owner/wrong separation.
- C4 passes under the active ambiguity-set interpretation: the true public path
  is present in the candidate set for all cases and owner/wrong separation
  remains positive. Two cases show that a forced single public path can choose
  a shifted low-residual neighbor, which reinforces the earlier method switch
  away from single-path acquisition.

The CPU-only temporal robustness target is therefore complete for the checked
synthetic construction. The next scientific boundary is no longer synthetic
temporal edits; it is transfer from this toy relation-channel construction to a
real observation/injection setting, which requires GPU/real-model work and must
remain separate from any paper or fixed-FPR claim.

Evidence boundary remains unchanged: CPU-only synthetic mechanism triage only;
no GPU, real video, fixed-FPR calibration, governance evidence, reproducibility
claim, or paper claim.
