# SC-SSTW Feasibility

This repository is a lightweight CPU-only sandbox for testing whether the
self-calibrated relation-subspace SSTW route is scientifically plausible before
moving it into a governed project.

Current scope:

- synthetic relation channel only;
- no GPU, video model, Wan runtime, or saved-video observation;
- no formal experimental claim;
- no inherited SSTW governance harness.
- this flow is only for rapid scientific feasibility triage of method
  mechanisms; it does not attempt governance, reproducibility, fixed-FPR
  validation, or paper-readiness requirements.

Current result summary is tracked in [RESULTS.md](RESULTS.md).
The current acquisition-method switch is tracked in
[docs_method_switch.md](docs_method_switch.md).

Current GPU boundary:

- CPU-only synthetic synchronization feasibility is closed for the current
  AISB + self-calibrated state synchronization route.
- A minimal GPU/saved-video observation probe has run successfully and produced
  10 videos plus a result package. The probe returned `no_observable_signal`.
- The failed observation path is specific: text-prompt-only geometric control
  plus saved-video patch-brightness readout. It does not invalidate the CPU
  synthetic synchronization mechanism.
- The next route should inspect or build a motion-subject observation probe:
  frame difference, moving-blob centroid, foreground mask, or optical-flow
  centroid.
- This repository still contains diagnostic feasibility evidence only: no
  detector, no observer, no fixed-FPR calibration, no attack suite, and no paper
  claim.

Minimal GPU observation probe entrypoints remain available for diagnostics:

```bash
python3 experiments/run_gpu_observation_probe.py --dry-run
# In Colab GPU, use notebooks/sc_sstw_gpu_observation_probe.ipynb and run cells 1-5.
```

First feasibility question:

```text
q_i = A_x u_i + b_x + noise
-> pilot-based channel calibration
-> state equalization
-> state-constrained temporal synchronization
-> owner key score separates from wrong keys under deletion/crop/speed edits
```

Current first-pass boundary:

- the synthetic probe uses oracle `source_indices` after crop/deletion;
- this isolates channel calibration, state equalization, and key-conditioned
  temporal scoring;
- it does not yet solve pilot re-acquisition from observations alone.

Second-pass boundary:

- `periodic_beam_observation_only` attempts public-pilot re-acquisition without
  owner-key access;
- the original four-cardinal-direction pilot cycle is tested separately from an
  asymmetric five-direction public pilot code and a non-cyclic public sync code;
- this is still a synthetic relation-channel probe, not video evidence.

Observed initial result on the default synthetic probe:

```text
feasibility_pass = true
pilot_reacquisition = oracle_source_indices
owner_score > best_wrong_score
channel_condition_number < 10
pilot_reconstruction_mse < 0.01
```

Run:

```bash
python3 -m unittest discover -s tests -v
python3 experiments/run_synthetic_probe.py
python3 experiments/run_synthetic_batch.py
python3 experiments/run_aisb_probe.py
python3 experiments/run_aisb_stress_probe.py
python3 experiments/run_aisb_payload_probe.py
python3 experiments/run_aisb_channel_mismatch_probe.py
python3 experiments/run_aisb_ambiguity_probe.py
python3 experiments/run_aisb_capacity_probe.py
python3 experiments/run_aisb_capacity_mismatch_probe.py
python3 experiments/run_aisb_stress_mismatch_payload_probe.py
python3 experiments/run_aisb_stress_grid_probe.py
python3 experiments/run_aisb_stress_grid_probe.py --gammas 0.5,0.8,1.0 --noise-stds 0.012,0.016 --message-space-size 16 --wrong-key-count 24 --case-count 4 --ambiguity-message-space-size 16 --ambiguity-wrong-key-count 24 --ambiguity-case-count 4
python3 experiments/run_aisb_stress_grid_probe.py --gammas 0.8,1.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 16 --ambiguity-wrong-key-count 24 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 0.8,1.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 4 --ambiguity-message-space-size 24 --ambiguity-wrong-key-count 24 --ambiguity-case-count 4
python3 experiments/run_aisb_stress_grid_probe.py --gammas 1.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 1.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 16 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 16
python3 experiments/run_aisb_stress_grid_probe.py --gammas 1.2,1.4 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 1.6,2.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 2.5,3.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 4.0,5.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 8.0,10.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 15.0,20.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 30.0,50.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 100.0 --noise-stds 0.016 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 100.0 --noise-stds 0.020 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 100.0 --noise-stds 0.040,0.050 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_stress_grid_probe.py --gammas 100.0 --noise-stds 0.080 --message-space-size 16 --wrong-key-count 24 --case-count 8 --ambiguity-message-space-size 64 --ambiguity-wrong-key-count 64 --ambiguity-case-count 8
python3 experiments/run_aisb_long_sequence_probe.py
python3 experiments/run_aisb_long_ambiguity_probe.py
python3 experiments/run_aisb_multi_ambiguity_probe.py
python3 experiments/run_aisb_pruned_search_probe.py
python3 experiments/run_aisb_exact_search_scale_probe.py
python3 experiments/run_aisb_threshold_margin_probe.py
python3 experiments/run_aisb_sequence_consistency_probe.py
python3 experiments/run_aisb_sequence_supported_exact_probe.py
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py
python3 experiments/run_aisb_threshold_candidate_exact_probe.py
python3 experiments/run_aisb_threshold_margin_probe.py --gamma 1.0
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 1.0 --residual-threshold 0.0075
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 1.4 --residual-threshold 0.01 --filler-multiplier 2
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 1.6 --residual-threshold 0.01 --filler-multiplier 2
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 2.0 --residual-threshold 0.01 --filler-multiplier 2
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 5.0 --residual-threshold 0.0125 --filler-multiplier 2
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 10.0 --residual-threshold 0.0125 --filler-multiplier 3
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 15.0 --residual-threshold 0.0125 --filler-multiplier 3 --workers 4
python3 experiments/run_aisb_threshold_candidate_exact_probe.py --gamma 30.0 --residual-threshold 0.0105 --filler-multiplier 3 --message-space-size 24 --wrong-key-count 12
python3 experiments/run_aisb_sequence_consistency_probe.py --gamma 100.0 --thresholds 0.01,0.0125,0.02,0.03
python3 experiments/run_aisb_sequence_supported_exact_probe.py --gamma 100.0 --residual-threshold 0.0125 --filler-multiplier 3 --message-space-size 24 --wrong-key-count 12
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --residual-threshold 0.0125 --near-tie-ratio 3.0 --filler-multiplier 3 --message-space-size 24 --wrong-key-count 12 --case-count 24 --workers 4
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --residual-threshold 0.0125 --near-tie-ratio 3.0 --filler-multiplier 3 --message-space-size 64 --wrong-key-count 64 --case-count 3 --workers 4
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 50.0 --residual-threshold 0.0125 --near-tie-ratio 3.0 --filler-multiplier 4 --message-space-size 64 --wrong-key-count 64 --start-index 3 --case-count 3 --workers 4
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 5 --message-space-size 64 --wrong-key-count 64 --case-count 12 --scoring-mode ordered_bounded_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 0 --case-count 16 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 16 --case-count 8 --workers 4 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 24 --case-count 8 --workers 4 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 32 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 40 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 48 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.08 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 56 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 0 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 8 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 16 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 24 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_case24_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_case24_cases.jsonl --progress-interval 2048
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 25 --case-count 1 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_case25_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_case25_cases.jsonl --progress-interval 2048
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 26 --case-count 6 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_cases26_31_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_cases26_31_cases.jsonl --progress-interval 4096
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 32 --case-count 8 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_cases32_39_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_cases32_39_cases.jsonl --progress-interval 4096
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 40 --case-count 8 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_cases40_47_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_cases40_47_cases.jsonl --progress-interval 4096
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 48 --case-count 8 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_cases48_55_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_cases48_55_cases.jsonl --progress-interval 4096
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.10 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 56 --case-count 8 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise010_cases56_63_progress.jsonl --case-jsonl /tmp/sc_sstw_noise010_cases56_63_cases.jsonl --progress-interval 4096
python3 experiments/run_aisb_sequence_ambiguity_exact_probe.py --gamma 100.0 --noise-std 0.12 --residual-threshold 0.0125 --near-tie-ratio 5.0 --filler-multiplier 7 --message-space-size 64 --wrong-key-count 64 --start-index 0 --case-count 8 --workers 1 --scoring-mode ordered_bounded_global_c --progress-jsonl /tmp/sc_sstw_noise012_cases0_7_progress.jsonl --case-jsonl /tmp/sc_sstw_noise012_cases0_7_cases.jsonl --progress-interval 4096
python3 experiments/run_aisb_sequence_consistency_probe.py --gamma 500.0 --thresholds 0.0125 --high-case-count 4 --random-case-count 64 --min-sequence-support 12
```

Current CPU search boundary:

- the pruned two-stage search is retained only as a negative diagnostic because
  it can change or miss the exact winner in high-mismatch long-secret-span
  cases;
- the active CPU route uses exhaustive scoring with an exact score-only DTW
  implementation to reduce memory without changing the scorer;
- larger exhaustive tiers are still a CPU runtime question, not a GPU or video
  requirement.
- the checked stronger-mismatch envelope reaches gamma 100.0 under the full
  64-message / 64-wrong-key exact tier with filler_multiplier 3 across cases
  0-7 in the current stress-grid probe, with ambiguity-set owner/wrong recovery
  8/8 and minimum margins still above the 0.02 diagnostic line: 0.0435 at
  gamma 15.0, 0.0375 at gamma 20.0, 0.0376 at gamma 30.0, and 0.0367 at
  gamma 50.0, and 0.0360 at gamma 100.0. The same gamma 100.0 full tier also
  remains positive at noise_std 0.020 with minimum margin 0.0360;
  increasing noise_std to 0.040 and 0.050 still passes 8/8 but moves close to
  the diagnostic boundary, with minimum margins 0.0249 and 0.0264; noise_std
  0.080 still passes 8/8 with minimum margin 0.0266. Extending noise_std 0.080
  to 16 cases at filler_multiplier 5 exposes one thin diagnostic miss
  (15/16 pass, minimum margin 0.0196) while owner/global recovery and truth
  coverage remain 16/16. Increasing the secret-state filler span to 7 restores
  the same checked 16-case tier to 16/16 pass with minimum margin 0.0277.
  Continuing in two 8-case chunks covers cases 16-31 as well: both chunks pass
  8/8, so the checked 32-case range passes when run in CPU-manageable segments.
  Cases 32-39 were then checked as single-case CPU segments; all 8 pass, with
  minimum margin 0.0340, extending the checked range to 40/40 under this
  synthetic setting. Cases 40-47 were also checked as single-case CPU
  segments; all 8 pass, with minimum margin 0.0260, extending the checked
  range to 48/48. Cases 48-55 were also checked as single-case CPU segments;
  all 8 pass, with minimum margin 0.0338, extending the checked range to
  56/56. Cases 56-63 were then checked the same way; all 8 pass, with
  minimum margin 0.0780, completing the full 64-case checked message-index
  range at 64/64. The weakest checked point remains case 45 with margin 0.0260.
  Raising noise_std to 0.10 still passes the first checked 8 single-case
  segments, with minimum margin 0.0303 at case 7, but the margin remains thin.
  Continuing cases 8-15 also passes 8/8, with minimum margin 0.0399 at case 8.
  Case 15 originally exposed a CPU ordering-throughput hotspot before exact
  scoring; replacing the ordering-only Python DTW heuristic with a sparse
  aligned-distance heuristic made the same exact/bounded scorer finish and
  pass with margin 0.1080. Cases 16-23 also pass 8/8, with minimum margin
  0.0386 at case 21. Cases 24-31 add another 8/8 pass, with minimum margin
  0.0718 at case 28. Cases 32-39 also pass 8/8, with minimum margin 0.0363
  at case 34. Cases 40-47 also pass 8/8, with minimum margin 0.0299 at case
  45. Cases 48-55 also pass 8/8, with minimum margin 0.0324 at case 53,
  Cases 56-63 complete the checked message-index range at 64/64, with minimum
  margin 0.0775 in that final segment. The weakest noise_std 0.10 checked point
  remains case 45 at margin 0.0299.
  Raising noise_std to 0.12 completes the checked message-index range at
  64/64 pass. The weakest checked noise_std 0.12 point is case 34 with margin
  0.0257, positive but close to the 0.02 diagnostic line.
  Raising noise_std again to 0.14 completes the full checked 64-case
  message-index range at 64/64 pass; cases 56-63 add an 8/8 segment with
  minimum margin 0.0758 at case 62, while the weakest checked point remains
  case 45 with margin 0.0248, close to the 0.02 diagnostic line. The full
  checked noise_std 0.16 range now passes cases 0-63 at 64/64, with segment
  56-63 minimum margin 0.0739 at case 62 and overall checked minimum margin
  0.0263 at case 7. The full checked noise_std 0.18 range now passes cases
  0-63 at 64/64, with segment 56-63 minimum margin 0.0734 at case 62; case 34
  remains the overall checked minimum for the tier at 0.0212, close to the
  0.02 diagnostic line. Raising noise_std to 0.20 passes cases 0-31 at 32/32,
  then reaches the first checked diagnostic boundary at case 34 in segment
  32-39: owner/global recovery is still correct, but margin 0.0175 falls below
  the 0.02 diagnostic line. This is a synthetic margin boundary, not a GPU or
  video-model boundary. Single-case checks of case 34 at noise_std 0.1900,
  0.1850, 0.1825, and 0.1810 also remain owner/global correct but below the
  diagnostic line, indicating distributional near-boundary behavior rather than
  a simple smooth interpolation of one fixed observation. A top-k competition
  diagnostic for case 34 at noise_std 0.20 confirms the failure mode: the exact
  truth sequence is selected and owner message_34 remains the global top
  scorer, but the nearest wrong-key candidate is close enough to reduce the
  margin to 0.01754. The active synthetic boundary is wrong-key separation
  margin, not AISB acquisition. Increasing public trajectory evidence from
  12 bursts to 16 bursts restores the same noise_std 0.20 cases 32-39 segment
  to 8/8 pass with minimum margin 0.08868 and case 34 margin 0.16218, so the
  current CPU-only direction is evidence-span mapping rather than GPU/video.
  Continuing burst_count 16 through cases 40-47 also passes 8/8, with minimum
  margin 0.05190 at case 47, extending the checked noise_std 0.20 range to
  48/64. Cases 48-55 also pass 8/8, with minimum margin 0.04821 at case 53,
  extending the checked noise_std 0.20 range to 56/64. Cases 56-63 complete
  the checked message-index range at burst_count 16: 64/64 pass at
  gamma 100.0 / noise_std 0.20 / 64 messages / 64 wrong keys. The weakest
  checked burst16 noise_std 0.20 margin is 0.04821 at case 53. Raising to
  noise_std 0.22, the first burst16 segment cases 0-7 also passes 8/8 with
  minimum margin 0.07796; cases 8-15 also pass 8/8 with minimum margin
  0.08860; cases 16-23 also pass 8/8 with minimum margin 0.06563, extending
  the checked noise_std 0.22 range to 24/64. Cases 24-31 also pass 8/8 with
  minimum margin 0.06989, extending the checked noise_std 0.22 range to
  32/64. Cases 32-39 also pass 8/8 with minimum margin 0.08873, extending
  the checked noise_std 0.22 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.05232, extending the checked noise_std 0.22 range to
  48/64. Case50 then exposed a top1 acquisition beam-width miss: the true
  deletion candidate was second-best at one observed start. Retaining
  `top_k_per_start=2` restores truth coverage and the exact/bounded official
  CLI path passes case50 with margin 0.14948. The remaining cases 48-63 pass
  as a CPU diagnostic tier with top2 acquisition and pruned owner/wrong scoring
  that exact-scores all owner messages plus a public cheap-screened wrong-key
  subset; cases 48-55 pass 8/8 with minimum margin 0.07371, and cases 56-63
  pass 8/8 with minimum margin 0.10467. This extends the checked noise_std
  0.22 range to 64/64 for synthetic mechanism triage, but the last 16 cases
  are diagnostic-pruned evidence rather than exhaustive wrong-key proof.
  Raising again to noise_std 0.24, the same top2 + diagnostic-pruned tier
  passes cases 0-7 with minimum margin 0.10372, cases 8-15 with minimum
  margin 0.11475, cases 16-23 with minimum margin 0.08395, cases 24-31
  with minimum margin 0.12069, and cases 32-39 with minimum margin 0.09328,
  cases 40-47 with minimum margin 0.05653, and cases 48-55 with minimum
  margin 0.07433, and cases 56-63 with minimum margin 0.10521, closing the
  checked noise_std 0.24 range at 64/64. Raising to noise_std 0.26, cases
  0-7 pass 8/8 with minimum margin 0.10408, and cases 8-15 pass 8/8 with
  minimum margin 0.11718, and cases 16-23 pass 8/8 with minimum margin
  0.08616, cases 24-31 pass 8/8 with minimum margin 0.12042, and cases 32-39
  pass 8/8 with minimum margin 0.09435, and cases 40-47 pass 8/8 with
  minimum margin 0.05891, and cases 48-55 pass 8/8 with minimum margin
  0.07383, and cases 56-63 pass 8/8 with minimum margin 0.10519, closing
  the checked noise_std 0.26 range at 64/64. Case29 exposed a local CPU
  exact-scoring throughput hotspot under workers=1, but sequence-level
  parallel scoring completed it with workers=4; this was not a scientific
  failure or GPU boundary. Raising to noise_std 0.28, cases 0-7 pass 8/8
  with minimum margin 0.10748, and cases 8-15 pass 8/8 with minimum margin
  0.11334, cases 16-23 pass 8/8 with minimum margin 0.08554, and cases
  24-31 pass 8/8 with minimum margin 0.11995, and cases 32-39 pass 8/8
  with minimum margin 0.09460, and cases 40-47 pass 8/8 with minimum margin
  0.05336, cases 48-55 pass 8/8 with minimum margin 0.07161, and cases
  56-63 pass 8/8 with minimum margin 0.10453, closing the checked
  noise_std 0.28 range at 64/64. Raising to noise_std 0.30, cases 0-7 pass
  8/8 with minimum margin 0.09960, and cases 8-15 pass 8/8 with minimum
  margin 0.11246, and cases 16-23 pass 8/8 with minimum margin 0.08514,
  and cases 24-31 pass 8/8 with minimum margin 0.11978, extending the
  checked noise_std 0.30 range to 32/64. Cases 32-39 also pass 8/8 with
  minimum margin 0.09366, extending the checked noise_std 0.30 range to
  40/64. Cases 40-47 also pass 8/8 with minimum margin 0.05886, extending
  the checked noise_std 0.30 range to 48/64. Cases 48-55 also pass 8/8 with
  minimum margin 0.07119, and cases 56-63 pass 8/8 with minimum margin
  0.10058, closing the checked noise_std 0.30 range at 64/64. Raising to
  noise_std 0.32, cases 0-7 pass 8/8 with minimum margin 0.10207, and
  cases 8-15 pass 8/8 with minimum margin 0.11095, extending the checked
  noise_std 0.32 range to 16/64. Cases 16-23 also pass 8/8 with minimum
  margin 0.08368, and cases 24-31 pass 8/8 with minimum margin 0.11960,
  extending the checked noise_std 0.32 range to 32/64. Cases 32-39 also pass
  8/8 with minimum margin 0.08919, extending the checked noise_std 0.32 range
  to 40/64. Cases 40-47 also pass 8/8 with minimum margin 0.06073, extending
  the checked noise_std 0.32 range to 48/64. Cases 48-55 also pass 8/8 with
  minimum margin 0.07016, and cases 56-63 pass 8/8 with minimum margin
  0.09623, closing the checked noise_std 0.32 range at 64/64. Raising to
  noise_std 0.34, cases 0-7 pass 8/8 with minimum margin 0.09776, and cases
  8-15 pass 8/8 with minimum margin 0.10929955, extending the checked
  noise_std 0.34 range to 16/64. Cases 16-23 also pass 8/8 with minimum
  margin 0.08209735, extending the checked noise_std 0.34 range to 24/64.
  Cases 24-31 also pass 8/8 with minimum margin 0.12069871, extending the
  checked noise_std 0.34 range to 32/64. Cases 32-39 also pass 8/8 with
  minimum margin 0.09147009, extending the checked noise_std 0.34 range to
  40/64. Cases 40-47 also pass 8/8 with minimum margin 0.06057691,
  extending the checked noise_std 0.34 range to 48/64. Cases 48-55 also pass
  8/8 with minimum margin 0.06498330, extending the checked noise_std 0.34
  range to 56/64. Cases 56-63 also pass 8/8 with minimum margin 0.09374521,
  closing the checked noise_std 0.34 range at 64/64. Raising to noise_std
  0.36, cases 0-7 pass 8/8 with minimum margin 0.09361803, and cases 8-15
  pass 8/8 with minimum margin 0.10295556, extending the checked noise_std
  0.36 range to 16/64. Cases 16-23 also pass 8/8 with minimum margin
  0.08668786, extending the checked noise_std 0.36 range to 24/64. Cases
  24-31 also pass 8/8 with minimum margin 0.11924334, extending the checked
  noise_std 0.36 range to 32/64. Cases 32-39 also pass 8/8 with minimum
  margin 0.09238279, extending the checked noise_std 0.36 range to 40/64.
  Cases 40-47 also pass 8/8 with minimum margin 0.06148871, extending the
  checked noise_std 0.36 range to 48/64. Cases 48-55 also pass 8/8 with
  minimum margin 0.06968034, extending the checked noise_std 0.36 range to
  56/64. Cases 56-63 also pass 8/8 with minimum margin 0.08903584, closing
  the checked noise_std 0.36 range at 64/64. Raising to noise_std 0.38,
  cases 0-7 pass 8/8 with minimum margin 0.09210281, and cases 8-15 pass
  8/8 with minimum margin 0.10081056, extending the checked noise_std 0.38
  range to 16/64. Cases 16-23 also pass 8/8 with minimum margin 0.08678848,
  extending the checked noise_std 0.38 range to 24/64. Cases 24-31 also pass
  8/8 with minimum margin 0.11695454, extending the checked noise_std 0.38
  range to 32/64. Cases 32-39 also pass 8/8 with minimum margin 0.08942796,
  extending the checked noise_std 0.38 range to 40/64. Cases 40-47 also pass
  8/8 with minimum margin 0.06385964, extending the checked noise_std 0.38
  range to 48/64. Cases 48-55 also pass 8/8 with minimum margin 0.07151290,
  extending the checked noise_std 0.38 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.08474072, closing the checked noise_std 0.38
  range at 64/64. Raising to noise_std 0.40, cases 0-7 pass 8/8 with
  minimum margin 0.08821583. Cases 8-15 also pass 8/8 with minimum margin
  0.09863218, extending the checked noise_std 0.40 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.08691652, extending the checked
  noise_std 0.40 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.11608262, extending the checked noise_std 0.40 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08839395, extending the
  checked noise_std 0.40 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.06297970, extending the checked noise_std 0.40 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.06940009,
  extending the checked noise_std 0.40 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.08638474, closing the checked noise_std 0.40
  range at 64/64. Raising to noise_std 0.42, cases 0-7 pass 8/8 with
  minimum margin 0.08818240. Cases 8-15 also pass 8/8 with minimum margin
  0.09784793, extending the checked noise_std 0.42 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.09197900, extending the checked
  noise_std 0.42 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.11412904, extending the checked noise_std 0.42 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08782796, extending the
  checked noise_std 0.42 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.06141315, extending the checked noise_std 0.42 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.06861692,
  extending the checked noise_std 0.42 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.08430650, closing the checked noise_std 0.42
  range at 64/64. Raising to noise_std 0.44, cases 0-7 pass 8/8 with
  minimum margin 0.08862351. Cases 8-15 also pass 8/8 with minimum margin
  0.09980410, extending the checked noise_std 0.44 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.09127598, extending the checked
  noise_std 0.44 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.11341359, extending the checked noise_std 0.44 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08761476, extending the
  checked noise_std 0.44 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.06277592, extending the checked noise_std 0.44 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.07073149,
  extending the checked noise_std 0.44 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.07963960, closing the checked noise_std 0.44
  range at 64/64. Raising to noise_std 0.46, cases 0-7 pass 8/8 with
  minimum margin 0.08912916. Cases 8-15 also pass 8/8 with minimum margin
  0.09899634, extending the checked noise_std 0.46 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.09003097, extending the checked
  noise_std 0.46 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.10937305, extending the checked noise_std 0.46 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08296188, extending the
  checked noise_std 0.46 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.05920578, extending the checked noise_std 0.46 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.06771293,
  extending the checked noise_std 0.46 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.07997671, closing the checked noise_std 0.46
  range at 64/64. Raising to noise_std 0.48, cases 0-7 pass 8/8 with
  minimum margin 0.08923162. Cases 8-15 also pass 8/8 with minimum margin
  0.09699572, extending the checked noise_std 0.48 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.08788011, extending the checked
  noise_std 0.48 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.10637933, extending the checked noise_std 0.48 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08064904, extending the
  checked noise_std 0.48 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.05655783, extending the checked noise_std 0.48 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.06854657,
  extending the checked noise_std 0.48 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.08137164, closing the checked noise_std 0.48
  range at 64/64. Raising to noise_std 0.50, cases 0-7 pass 8/8 with
  minimum margin 0.09375566. Cases 8-15 also pass 8/8 with minimum margin
  0.09635263, extending the checked noise_std 0.50 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.08730083, extending the checked
  noise_std 0.50 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.10415634, extending the checked noise_std 0.50 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08194289, extending the
  checked noise_std 0.50 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.05578955, extending the checked noise_std 0.50 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.06911470,
  extending the checked noise_std 0.50 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.07806989, closing the checked noise_std 0.50
  range at 64/64. Raising to noise_std 0.52, cases 0-7 pass 8/8 with
  minimum margin 0.09044599. Cases 8-15 also pass 8/8 with minimum margin
  0.09866961, extending the checked noise_std 0.52 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.08655462, extending the checked
  noise_std 0.52 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.10843663, extending the checked noise_std 0.52 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08209579, extending the
  checked noise_std 0.52 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.05007786, extending the checked noise_std 0.52 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.07073198,
  extending the checked noise_std 0.52 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.07680597, closing the checked noise_std 0.52
  range at 64/64. Raising to noise_std 0.54, cases 0-7 pass 8/8 with
  minimum margin 0.09154538. Cases 8-15 also pass 8/8 with minimum margin
  0.09827562, extending the checked noise_std 0.54 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.08079990, extending the checked
  noise_std 0.54 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.10307755, extending the checked noise_std 0.54 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.08281960, extending the
  checked noise_std 0.54 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.04712975, extending the checked noise_std 0.54 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.06931735,
  extending the checked noise_std 0.54 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.06999606, closing the checked noise_std 0.54
  range at 64/64. Raising to noise_std 0.56, cases 0-7 pass 8/8 with
  minimum margin 0.09051082. Cases 8-15 also pass 8/8 with minimum margin
  0.09464360, extending the checked noise_std 0.56 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.07814625, extending the checked
  noise_std 0.56 range to 24/64. Cases 24-31 also pass 8/8 with minimum
  margin 0.09902808, extending the checked noise_std 0.56 range to 32/64.
  Cases 32-39 also pass 8/8 with minimum margin 0.07365479, extending the
  checked noise_std 0.56 range to 40/64. Cases 40-47 also pass 8/8 with
  minimum margin 0.05104159, extending the checked noise_std 0.56 range to
  48/64. Cases 48-55 also pass 8/8 with minimum margin 0.07028426,
  extending the checked noise_std 0.56 range to 56/64. Cases 56-63 also pass
  8/8 with minimum margin 0.06647356, closing the checked noise_std 0.56
  range at 64/64. Raising to noise_std 0.58, cases 0-7 pass 8/8 with
  minimum margin 0.09066536. Cases 8-15 also pass 8/8 with minimum margin
  0.09670570, extending the checked noise_std 0.58 range to 16/64. Cases
  16-23 also pass 8/8 with minimum margin 0.07675474, extending the checked
  noise_std 0.58 range to 24/64. The remaining cases 24-63 also pass,
  closing the checked noise_std 0.58 range at 64/64 with full-layer minimum
  margin 0.04732445. Raising to noise_std 0.60, the full 64-case layer also
  passes 64/64 with truth coverage 64/64, owner/global recovery 64/64,
  full-layer minimum margin 0.04807966, mean margin 0.11447229, maximum
  ambiguity sequence count 192, and maximum exact-score count 29977. CPU-only
  synthetic mapping can continue, but exact-score throughput and native helper
  compilation are now practical CPU infrastructure constraints.
  At noise_std 0.62, cases 0-7 produce the first checked boundary: 7/8 pass,
  owner/global recovery remains 8/8, but case 4 loses public truth-sequence
  coverage under residual_threshold 0.0125. A CPU-only diagnostic rerun recovers
  that case at residual_threshold 0.015, so the next scientific question is the
  residual-threshold versus ambiguity/exact-work tradeoff, not GPU.
  A full noise_std 0.62 layer at residual_threshold 0.015 then passes 64/64
  with truth coverage 64/64, owner/global recovery 64/64, full-layer minimum
  margin 0.04831724, mean margin 0.11434315, maximum ambiguity sequence count
  288, and maximum exact-score count 44356. The recovery confirms the threshold
  boundary diagnosis, while the exact-score increase makes the next CPU-only
  question a threshold/ambiguity tradeoff rather than a GPU requirement.
  A lower intermediate threshold, residual_threshold 0.01375, also passes the
  full noise_std 0.62 layer at 64/64 with truth coverage 64/64, owner/global
  recovery 64/64, full-layer minimum margin 0.04831724, mean margin
  0.11436305, maximum ambiguity sequence count 288, and maximum exact-score
  count 44356. This narrows the required threshold above 0.0125, but it does
  not reduce the observed worst-case exact-score burden relative to 0.015.
  residual_threshold 0.013125 also passes the full noise_std 0.62 layer at
  64/64 with truth coverage 64/64, owner/global recovery 64/64, full-layer
  minimum margin 0.04831724, mean margin 0.11348427, maximum ambiguity sequence
  count 288, and maximum exact-score count 44356. This tightens the viable
  threshold above the failing 0.0125 setting, but the worst-case exact-score
  burden remains unchanged.
  residual_threshold 0.0128125 also passes the full noise_std 0.62 layer at
  64/64 with truth coverage 64/64, owner/global recovery 64/64, full-layer
  minimum margin 0.04831724, mean margin 0.11294394, maximum ambiguity sequence
  count 288, and maximum exact-score count 44356. This further narrows the
  viable threshold interval above 0.0125 while preserving the same observed
  worst-case exact-score burden.
  residual_threshold 0.01265625 also passes the full noise_std 0.62 layer at
  64/64 with truth coverage 64/64, owner/global recovery 64/64, full-layer
  minimum margin 0.04831724, mean margin 0.11303899, maximum ambiguity sequence
  count 192, and maximum exact-score count 29957. This keeps narrowing the
  viable threshold interval above 0.0125; the active boundary remains CPU
  synthetic exact-search mapping, not GPU/video.
  residual_threshold 0.012578125 also passes the full noise_std 0.62 layer at
  64/64 with truth coverage 64/64, owner/global recovery 64/64, full-layer
  minimum margin 0.04831724, mean margin 0.11307260, maximum ambiguity sequence
  count 192, and maximum exact-score count 29957. The checked boundary is now
  narrowed to between failing 0.0125 and passing 0.012578125.
  residual_threshold 0.0125390625 also passes the full noise_std 0.62 layer
  at 64/64 with truth coverage 64/64, owner/global recovery 64/64, full-layer
  minimum margin 0.04831724, mean margin 0.11305243, maximum ambiguity sequence
  count 192, and maximum exact-score count 29957. The checked public residual
  boundary is now between failing 0.0125 and passing 0.0125390625.
  residual_threshold 0.01251953125 also passes the full noise_std 0.62 layer
  at 64/64 with truth coverage 64/64, owner/global recovery 64/64, full-layer
  minimum margin 0.04831724, mean margin 0.11320780, maximum ambiguity sequence
  count 192, and maximum exact-score count 29957. The checked boundary is now
  between failing 0.0125 and passing 0.01251953125.
  residual_threshold 0.02 was checked as a wider cost reference, not as a
  recommended operating point. Cases 0-32 all passed with truth coverage and
  owner/global recovery 33/33, minimum margin 0.07635967, mean margin
  0.11830551, maximum ambiguity sequence count 428, and maximum exact-score
  count 64619; the next case became too slow for the quick CPU mapping loop.
  This confirms the useful threshold window without justifying more bisection:
  0.0125 is too tight, 0.01251953125 is the narrowest checked full-layer pass,
  0.015 is the pragmatic diagnostic setting, and 0.02 is wider but costlier.
  A first higher-noise check at noise_std 0.64 with residual_threshold 0.015
  passed the completed cases 0-10 at 11/11 with truth coverage and owner/global
  recovery 11/11, minimum margin 0.09242166, mean margin 0.11595991, maximum
  ambiguity sequence count 144, and maximum exact-score count 22153. The next
  case entered long exact C scoring, so the immediate blocker is CPU exact-search
  latency rather than a synthetic mechanism failure or GPU/video requirement.
  A C2 two-point burst-internal deletion acquisition check failed at the public
  candidate representation layer: with two deleted points in every retained
  burst, the current single-deletion AISB scanner covered the true path in 0/8
  checked cases. This is not an owner/wrong-key scoring failure; it means the
  current public burst template/candidate contract only supports one missing
  burst point. Further C2 progress requires a deliberate double-missing AISB
  acquisition extension rather than more scoring.
  Temporal robustness is complete for the checked CPU-only synthetic
  construction. A 12-point double-redundant AISB template resolves the earlier
  two-point burst deletion representability boundary: C2 passes 64/64 for both
  acquisition and owner/wrong scoring with min margin 0.30199808. C3 piecewise
  synthetic clock distortion passes 64/64 with exact public alignment and min
  margin 0.30039331. C4 combined perturbation passes 64/64 under the active
  ambiguity-set interpretation: truth candidate coverage is 64/64 and min
  owner/wrong margin is 0.29703240, while forced single-path exact alignment is
  62/64 due shifted low-residual neighbors. The next boundary is real
  observation/injection transfer, not more CPU synthetic temporal stress.
  A single 32-case batch was interrupted after more than 40 minutes without
  output, which is a local CPU throughput issue rather than a scientific or
  GPU boundary;
  gamma 30.0 remains checked only in the smaller 24-message / 12-wrong-key
  diagnostic tier under a narrower public residual threshold 0.0105;
- gamma 50.0 reaches the current public AISB residual-overlap boundary in the
  checked synthetic distribution: true burst residuals and random non-burst
  residuals overlap, so a single residual threshold cannot simultaneously keep
  all checked true bursts and reject all checked random non-bursts;
- adding public template-sequence support changes that boundary: at gamma 50.0
  and 100.0, isolated residual false positives are rejected when a candidate
  must support the 12-burst public template cycle. This still uses no owner key,
  message, affine-channel estimate, GPU, video, or fixed-FPR calibration;
- after that public sequence-supported alignment is frozen, the smaller
  24-message / 12-wrong-key exact owner/wrong tier passes checked gamma 50.0
  and 100.0 cases 0-2 with owner-message recovery 3/3 and minimum margins
  0.0562 and 0.0585 respectively;
- extending the same unique-alignment chain to gamma 50.0 cases 0-11 exposes a
  public acquisition non-uniqueness boundary: owner-message recovery remains
  12/12, but exact public alignment is only 9/12;
- carrying a small public ambiguity set instead of forcing one public alignment
  closes that checked boundary. With near_tie_ratio 3.0, the 24-message /
  12-wrong-key ambiguity-set exact tier passes gamma 50.0 and 100.0 cases 0-23
  with owner-message recovery 24/24 and minimum margins 0.0240 and 0.0227;
- the same ambiguity-set route also passes the full 64-message / 64-wrong-key
  exact tier for gamma 50.0 and 100.0 cases 0-2 with 4 CPU workers and minimum
  margins 0.0465 and 0.0466;
- extending full-tier gamma 50.0 to cases 3-5 with filler_multiplier 3 exposes
  a thin owner-vs-wrong margin at case 5 (0.0194), while increasing the
  secret-state filler span to 4 restores cases 3-5 with minimum margin 0.0206;
- full-tier gamma 100.0 cases 3-5 with filler_multiplier 4 has one thin case
  just below the 0.02 diagnostic margin (case 3 margin 0.0195). Increasing the
  secret-state filler span to 5 and carrying a slightly wider public ambiguity
  set with near_tie_ratio 5.0 restores checked cases 0-11 in the full
  64-message / 64-wrong-key tier, with minimum margin 0.0359;
- under gamma 500.0, the same full tier passes checked cases 0-11 with
  filler_multiplier 5 and checked cases 12-23 with filler_multiplier 6. Case 20
  separates exact public-truth recovery from coverage: the truth set is covered
  by a public sequence containing one extra false burst, and owner/wrong scoring
  over that same contaminated alignment still passes;
- extending gamma 500.0 / filler_multiplier 6 across a full 64-message-index
  synthetic round gives 63/64 passes. The only checked failure is case 54:
  owner message and global owner still recover, but the public ambiguity set
  does not cover the true burst sequence. Raising only the public/secret filler
  span to 7 restores the checked boundary cases 52-55, including case 54, with
  minimum margin 0.0388; case 54 itself recovers exact public-truth coverage and
  margin 0.0756. Extending filler 7 across the full 64-message-index synthetic
  round has completed 63/64 cases, all passing, with minimum completed margin
  0.0284. The only uncompleted case is case 24, which exceeded a 5-minute local
  CPU single-case runtime window and remains a CPU runtime slow case rather
  than an observed scientific failure.
  This is a public acquisition support boundary, not a GPU/video or
  owner/wrong-key scoring failure;
- `--scoring-mode ordered_bounded` preserves the exact scoring rule by using a
  cheap decimated score only for candidate ordering and exact bounded DTW for
  accepted scores. It matches the exact winner in the checked 24-message /
  12-wrong-key ambiguity case, but still does not finish the full 64-message /
  64-wrong-key gamma 100.0 filler 5 case within 5 minutes. A pure-Python DP
  hot-path optimization improves medium-tier runtime but does not remove this
  full-tier bottleneck;
- `--scoring-mode ordered_bounded_c` uses a local CPU C implementation of the
  same exact bounded DTW recurrence. It changes implementation speed only, not
  candidate space, scoring rule, threshold, or evidence status. This makes the
  full 64-message / 64-wrong-key gamma 100.0 and 500.0 tiers runnable locally
  in the checked ranges above. When `--workers > 1`, this mode scores public
  ambiguity sequences in parallel and then takes the same exact owner/wrong/global
  maxima; native inputs are also prepared once per sequence/candidate instead
  of copied on every DTW call, and native DP workspaces are reused per sequence.
  These changes affect CPU scheduling/copy/allocation overhead only, not the
  scientific scoring rule;
- `--scoring-mode ordered_bounded_global_c` instead sorts all public
  sequence/candidate pairs by the same cheap diagnostic score before exact
  bounded native DTW. It also preserves the exact final rule and matches the
  exact winner in the checked regression, but still does not complete the
  gamma 500.0 / filler 7 case 24 slow example inside a 3-minute local CPU
  window. The same case index at message_space_size 24 and wrong_key_count 12
  completes in 3.97 seconds with exact margin 0.1473, so the observed issue is
  full-tier CPU scoring scale rather than a seed-specific mechanism failure;
- the bounded scorer now uses a stronger safe lower bound: it accounts for
  unavoidable future skip cost and the maximum possible final diagonal-match
  count. This preserves exact non-abandoned scores and safe abandonment, but it
  still does not make the full case 24 tier finish inside the short local CPU
  window;
- the full 64-message / 64-wrong-key exact tier is now a CPU runtime bottleneck
  in pure Python, but no longer a GPU/video boundary. The active local route is
  exact CPU scoring with the native C hot path for larger tiers. Current
  non-full-tier stress remains positive under the checked crop/delete/repeat,
  burst-internal deletion, and deterministic non-affine mismatch diagnostics.
  The current stress grid shows a finite synthetic robustness boundary:
  gamma 0.5 passes at noise 0.012/0.016, gamma 0.8 passes at noise 0.012 but
  has one alignment miss at noise 0.016, and gamma 1.0 retains owner-message
  recovery while margin/alignment no longer pass all checked cases. Replacing
  the single public alignment with the already-defined public ambiguity-set
  scoring restores the checked reduced grid through gamma 1.0 and noise 0.016
  with 3/3 pass and margins above 0.20, so the productive CPU route is now
  ambiguity-set alignment rather than forced single-path acquisition.
