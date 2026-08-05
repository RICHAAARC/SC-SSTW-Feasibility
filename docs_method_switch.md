# Method Switch: AISB Before Self-Calibration

Status: CPU-only feasibility design note. No video, GPU, Wan runtime, saved-video
observation, fixed-FPR result, or paper claim.

This sandbox is intentionally limited to rapid scientific feasibility triage of
method mechanisms. It is not a governance, reproducibility, fixed-FPR, or
paper-readiness workflow.

## Decision

Stop using observation-only public-pilot label search driven by per-candidate
affine least-squares fitting.

Switch the acquisition chain to:

```text
affine-invariant synchronization burst
-> public clock decoding
-> freeze pilot/source alignment
-> estimate A_x,b_x
-> equalize relation observations
-> key-conditioned state synchronization
```

The core SC-SSTW route remains unchanged:

```text
2D relation subspace
+ video-specific self-calibration
+ key-conditioned state trajectory scoring
```

Only the public synchronization/acquisition primitive is being replaced.

## Reason

Current synthetic results show:

```text
oracle pilot/source labels:
  calibration + equalization + owner/wrong-key scoring passes

unknown pilot/source labels:
  affine least-squares acquisition is not identifiable
```

The failure mode is gauge ambiguity:

```text
wrong pilot labels + refit affine channel
```

can produce low reconstruction error while inducing the wrong state-space
orientation or clock alignment. Therefore, public alignment must be recovered
before estimating the video-specific affine channel.

## New primitive

AISB uses local burst geometry whose affine dependencies are known publicly.
For a burst template with points:

```text
u_1, ..., u_m in R^2
```

choose three non-collinear anchors. For every checksum point:

```text
u_j = lambda_j1 u_1 + lambda_j2 u_2 + lambda_j3 u_3
lambda_j1 + lambda_j2 + lambda_j3 = 1
```

After an unknown affine channel:

```text
q_j = lambda_j1 q_1 + lambda_j2 q_2 + lambda_j3 q_3
```

This residual can be scored without fitting `A_x,b_x`.

## Immediate experiment

Add a synthetic affine-channel AISB probe:

```text
known public burst templates
-> random full-rank affine channels
-> crop/delete between bursts
-> affine-invariant burst scan
-> public sequence alignment
```

Minimum useful signal:

```text
AISB alignment accuracy >= 0.9 on synthetic affine observations
```

If this fails, the method needs a stronger public synchronization carrier or a
more constrained channel model before any GPU work.

Initial CPU-only AISB result:

```text
true burst acquisition pass_count = 8 / 8
random non-burst rejection pass_count = 8 / 8
```

This supports the method switch at the synthetic affine-channel level, with the
current limitation that deletion inside a burst has not yet been handled.


## Current extension: one checksum deletion inside AISB

The current CPU-only implementation adds a deletion-aware AISB scan. For each
window that is one sample shorter than the public template, the scanner
enumerates the missing public template index and evaluates the remaining affine
constraints. This is still an acquisition statistic over public burst geometry;
it does not estimate the video-specific affine channel `A_x,b_x`.

The validated tier is intentionally narrow:

```text
complete AISB burst
+ one missing checksum point inside the burst
-> public alignment
-> freeze AISB pilot pairs
-> estimate A_x,b_x
-> equalize
-> owner/wrong-key state synchronization
```

The current synthetic batch passes this tier, including a mixed sequence where
public AISB bursts are interleaved with key-conditioned secret-state windows.
This closes the immediate loop that failed for observation-only affine pilot
acquisition: public alignment is recovered before affine calibration, and owner
scoring is then compared against wrong keys using the same shared calibration.

Important limitation:

- Deletion of one of the three primary AISB anchor points is not claimed solved.
  In the present 6-point template, anchor deletion can be confused with adjacent
  filler because too few public constraints remain at the burst boundary. A
  future probe should use a stronger burst code, redundant anchors, or an
  explicit boundary statistic before claiming general single-point burst
  deletion tolerance.

Evidence boundary remains unchanged: synthetic affine-channel construction only;
no GPU, video model, saved-video observation, fixed-FPR calibration, or paper
claim.


## Redundant AISB for arbitrary one-point deletion

The checksum-deletion result does not imply arbitrary deletion tolerance for the
short 6-point burst. Anchor deletion is a separate identifiability case because
the affine coordinate frame itself is partially missing.

The current CPU-only extension therefore adds a second public burst family:

```text
3 primary anchors
+ 3 affine checksum points
+ 3 redundant public anchor copies
```

When one primary anchor is missing, the acquisition statistic uses its matching
redundant public anchor copy. When a checksum or redundant copy is missing, the
original primary anchors remain available. This keeps acquisition public and
affine-invariant; it still does not fit `A_x,b_x` before public alignment is
frozen.

Latest synthetic diagnostic:

```text
redundant exact-anchor arbitrary one-point deletion:
  pass_count = 17 / 18
  false_positive = 1
  false_negative = 1
```

The exact redundant-anchor construction is therefore **not** a validated
arbitrary-deletion solution. It exposes a shifted-window ambiguity: when public
anchor copies are exact duplicates, an adjacent window can assign a different
missing template index and still satisfy the same affine geometry constraints.
This is a synthetic identifiability failure of that public sync code, not a GPU
or runtime problem.

The associated random non-burst diagnostic still rejects non-burst sequences at
the current threshold, but rejection is insufficient: the active blocker is
unique public alignment among near-overlapping true-burst hypotheses.

The same AISB-frozen alignment idea has been connected to a mixed sequence
containing secret-state windows, and owner/wrong-key scoring remains separated
in the tested positive tiers. That does not repair the arbitrary-deletion
alignment ambiguity above.

Evidence boundary remains unchanged: CPU-only synthetic affine-channel evidence;
no real video observations, no GPU, no fixed-FPR claim, and no paper claim.


## Stress-edit diagnostic

The next CPU-only probe keeps the redundant AISB construction but makes the edit
model harder:

```text
start/end crop
+ deterministic non-burst deletions
+ occasional non-burst repeats
+ variable burst spacing
+ one arbitrary missing point inside every retained burst
```

This stress probe still uses public affine-invariant burst geometry for
acquisition. It does not estimate `A_x,b_x` until after the burst alignment is
accepted, and it uses one shared calibration for owner and wrong-key scoring.

Current result:

```text
noise_std = 0.016:
  redundant AISB stress cases pass 12 / 12
  random non-burst cases reject 64 / 64

noise_std = 0.020:
  stress cases pass 4 / 6
```

Method implication:

- The AISB switch survives a more realistic local edit mix in the current
  synthetic affine-channel construction.
- The observed `noise_std = 0.020` false negatives are a real margin boundary.
  They should be treated as a robustness risk for the observation design, not as
  permission to claim calibrated detection or tune thresholds post hoc.
- The exact redundant-anchor arbitrary-deletion code should not be promoted as
  solved; future CPU work should test a stronger public sync burst code or an
  explicitly bounded public ambiguity-set scorer.

The next CPU-only method-mechanism question is therefore not GPU execution. It
is whether the state trajectory payload can be made more discriminative under
the same AISB-frozen public alignment without giving wrong keys a larger search
space or introducing fixed-FPR claims.


## Bounded public ambiguity-set fallback

The shifted-window result means exact redundant-anchor AISB should not be
treated as a unique arbitrary-deletion acquisition mechanism. A narrower
fallback remains CPU-testable:

```text
public AISB scan
-> retain a bounded low-residual ambiguity set
-> owner and wrong keys search the same candidate alignments
-> owner/wrong-key trajectory score must still separate
```

The current ambiguity-set implementation is deliberately fair: the owner key
does not receive the true alignment or true message, and wrong keys receive the
same public alignment set and same fixed message space.

Current diagnostic:

```text
ambiguity payload cases pass 12 / 12
random non-burst cases reject 64 / 64
maximum ambiguity sequence count in this batch = 1

targeted shifted-window ambiguity cases pass 6 / 6
ambiguity sequence count = 3 for every targeted case
```

This validates the scoring boundary, the empty-candidate rejection path, and a
targeted multi-alignment case where the truth is one of several public
hypotheses. It does **not** solve unique acquisition for exact redundant
anchors; it changes the viable route to a public ambiguity-set detector. The
next CPU-only mechanism choice is:

- design a stronger public sync burst that removes the shifted-window ambiguity;
  or
- continue with bounded ambiguity-set scoring and test harder ambiguity sets,
  larger message spaces, and stronger non-affine channel mismatch.


## Payload/message scoring after frozen AISB alignment

The current CPU-only payload probe keeps the same public AISB acquisition rule
and tests the next mechanism link:

```text
accepted AISB bursts
-> freeze public alignment
-> estimate one shared A_x,b_x from public burst pairs
-> equalize all observations
-> score key/message-conditioned state trajectories
```

The detector side does not receive the true message. Instead, owner and wrong
keys are both scored against the same fixed message set. This is important: a
wrong key is not given a larger or freer search space than the owner key.

Current synthetic result:

```text
noise_std = 0.016:
  payload cases pass 12 / 12
  owner message recovery 12 / 12
  mean owner-vs-best-wrong margin = 0.143
  minimum margin = 0.0796
```

This is the first complete CPU-only mechanism loop for the switched method. It
supports feasibility of the synthetic construction, but it is still not a real
video observation result and not a detection claim.

Current limiting boundary:

```text
noise_std = 0.020:
  payload cases pass 5 / 6 because AISB acquisition has one false negative
```

So the immediate remaining scientific issue is robustness of the observation
layer and burst code under higher noise or less affine synthetic channels. GPU
is still not required until a concrete video-observation adapter is defined.


## Overlap ambiguity and channel-mismatch diagnostic

A later CPU-only stress case exposed an AISB selection issue rather than a
payload-scoring issue. With exact redundant anchor copies, a deletion-aware scan
can produce several overlapping low-residual candidates:

```text
true deletion window
shifted start+1 deletion window
same-start complete-window candidate that consumes adjacent filler
```

This should not be described as solved by a tie-break. The current result is:

```text
redundant exact-anchor arbitrary one-point deletion:
  pass_count = 17 / 18
  false_positive = 1
  false_negative = 1
```

The exact redundant-anchor construction is therefore a negative result for
unique arbitrary-deletion public alignment. The viable CPU-only fallback is to
retain a bounded public ambiguity set and score owner/wrong keys fairly over the
same candidate alignments.

The channel-mismatch probe then adds a deterministic quadratic perturbation
after the synthetic affine relation channel. Current result:

```text
gamma = 0.5:
  mismatch cases pass 12 / 12
  random non-burst cases reject 64 / 64

gamma = 1.0:
  diagnostic pass 5 / 6
```

This supports a limited robustness margin beyond a perfectly affine synthetic
channel, while preserving the same evidence boundary: CPU-only diagnostics, no
video observation, no GPU, no fixed-FPR, and no paper claim.


## Bounded ambiguity-set and payload capacity diagnostics

The ambiguity-set probe tests the fallback route:

```text
public low-residual AISB candidate set
-> bounded alignment hypotheses
-> owner and wrong keys search the same alignments
-> owner and wrong keys search the same fixed message space
```

Current result:

```text
random payload cases:
  pass 12 / 12
  max ambiguity sequence count = 1

targeted shifted-window ambiguity cases:
  pass 6 / 6
  ambiguity sequence count = 3 for every case
```

This does not restore unique public acquisition for exact redundant anchors. It
does show that a small public ambiguity set can be scored fairly in the current
synthetic construction.

The capacity probe then expands the shared message search:

```text
8 messages / 12 wrong keys:   pass 8 / 8, min margin 0.108
16 messages / 24 wrong keys:  pass 8 / 8, min margin 0.106
32 messages / 24 wrong keys:  pass 8 / 8, min margin 0.0734
```

This supports the synthetic state-trajectory mechanism beyond the original
8-message toy setting. The decreasing margin is the relevant CPU-side risk to
stress next; it is not yet a real-video/GPU or calibrated detection claim.


## Capacity under non-affine mismatch

The combined CPU diagnostic adds the deterministic quadratic observation-space
mismatch to the payload-capacity probe:

```text
AISB public alignment
-> one shared affine calibration
-> equalization under residual non-affine error
-> owner/wrong-key message search
```

Current result:

```text
gamma = 0.5:
  8 messages / 12 wrong keys:   pass 6 / 6, min margin 0.0950
  16 messages / 24 wrong keys:  pass 6 / 6, min margin 0.0779
  32 messages / 24 wrong keys:  pass 6 / 6, min margin 0.0606

gamma = 1.0:
  16 messages / 24 wrong keys:  pass 4 / 6
```

This keeps the positive feasibility route alive under moderate non-affine
mismatch and exposes a clear CPU-side margin boundary at stronger mismatch.
The next useful local test is to stress edit density and ambiguity-set size
under `gamma = 0.5`; GPU is still not required.


## Edit-stress payload under non-affine mismatch

The next CPU diagnostic combines the harder edit model with non-affine mismatch
and larger shared message search:

```text
crop
+ non-burst deletions
+ non-burst repeats
+ one missing point in every retained burst
+ quadratic observation-space mismatch
+ owner/wrong-key message scoring
```

Current result:

```text
gamma = 0.5:
  16 messages / 24 wrong keys: pass 6 / 6, min margin 0.0324
  32 messages / 24 wrong keys: pass 6 / 6, min margin 0.0452

gamma = 0.8:
  16 messages / 24 wrong keys: pass 6 / 6, min margin 0.0248
```

This keeps CPU-only feasibility positive under the strongest combined synthetic
stress so far, but the `gamma = 0.8` margin is close to the floor. The next
local question is not GPU; it is whether the synthetic relation observation can
survive longer sequences, more wrong keys, or stronger ambiguity sets without
margin collapse.


## Long-sequence diagnostic

The long-sequence probe increases public burst count and trajectory length while
retaining larger message search:

```text
10-12 AISB public bursts
+ one missing point in every retained burst
+ crop / non-burst deletion / repeat edits
+ 32-message search
+ 16 wrong keys
+ optional quadratic mismatch
```

Current result:

```text
10 bursts, gamma 0.0, 32 messages: pass 2 / 2, min margin 0.138
10 bursts, gamma 0.5, 32 messages: pass 2 / 2, min margin 0.119
12 bursts, gamma 0.5, 32 messages: pass 2 / 2, min margin 0.116
12 bursts, gamma 0.8, 16 messages: pass 2 / 2, min margin 0.101
```

This suggests longer public calibration and trajectory evidence can recover
margin that was thin in the shorter high-edit stress case. The main local cost
is exhaustive CPU scoring over keys/messages/sequence length; it is not yet a
GPU or real-video boundary.


## Pruned-search diagnostic

A cheap two-stage scorer was tested to reduce CPU cost:

```text
decimated dynamic-time-sync screening
-> full dynamic-time-sync on selected candidates
```

Current result:

```text
exhaustive check tier:
  pruned_matches_exhaustive = false
  pruned margin = 0.0069

larger pruned smoke tiers:
  48 messages / 48 wrong keys pass under the pruned rule
```

The larger smoke tiers are not accepted as scientific evidence because the
small exhaustive tier proves the pruning heuristic can change the decision. The
correct method boundary is:

- use exhaustive scoring for pass/fail CPU diagnostics; or
- develop a mathematically safe bound before using pruning for larger claims.

This is a CPU cost and scoring-algorithm issue, not a GPU requirement.


## Long-sequence targeted ambiguity diagnostic

A targeted long-sequence ambiguity probe keeps one shifted-window ambiguity
cluster active while scoring owner and wrong keys over the same alignment set:

```text
10 AISB bursts
+ one public shifted-window ambiguity cluster
+ 16-message search
+ 12 wrong keys
+ optional gamma = 0.5 quadratic mismatch
```

Current result:

```text
gamma = 0.0: pass 4 / 4, ambiguity set size 2-3, min margin 0.210
gamma = 0.5: pass 4 / 4, ambiguity set size 2-3, min margin 0.181
```

This directly supports the bounded public ambiguity-set fallback in a longer
sequence. It does not claim unique AISB acquisition; it shows that a small
public ambiguity set can be resolved by key/message trajectory evidence under a
fair shared search.


## Multi-ambiguity-set diagnostic

The multi-ambiguity probe attempts to create two shifted-window ambiguity
regions in the same longer sequence. The resulting public ambiguity set is
non-unique in every case:

```text
gamma = 0.0: pass 3 / 3, ambiguity set size 3-9, min margin 0.245
gamma = 0.5: pass 3 / 3, ambiguity set size 3-6, min margin 0.174
```

Not every case forms two independent ambiguity clusters, so this is best read
as a non-unique public-alignment stress test rather than a strict two-cluster
construction proof. The method signal remains positive: shared ambiguity-set
scoring keeps owner/wrong separation without giving the owner privileged
alignment information.


## Exact score-only exhaustive scoring

The immediate correction is not to tune the pruned scorer. The current active
CPU route keeps exact exhaustive key/message scoring and replaces only the DP
storage strategy:

```text
existing dynamic-time-sync score
-> score-only rolling DP with the same transition order and normalization
-> exhaustive owner/wrong/message search
```

This preserves the scientific decision rule while reducing memory and avoiding
the unsafe screening step. Current synthetic diagnostics pass at moderate
search sizes:

```text
10 bursts, gamma 0.5, 16 messages, 12 wrong keys: pass 2 / 2
12 bursts, gamma 0.5, 24 messages, 12 wrong keys: pass 1 / 1
12 bursts, gamma 0.8, 16 messages, 12 wrong keys: pass 1 / 1
minimum margin across these tiers: 0.0663
one-off 48 messages / 48 wrong keys checks:
  gamma 0.5: pass, margin 0.0944, elapsed 32.9s
  gamma 0.8: pass, margin 0.0751, elapsed 34.7s
one-off 64 messages / 64 wrong keys checks:
  gamma 0.5: pass, margin 0.0863, elapsed 59.3s
  gamma 0.8: pass, margin 0.0423, elapsed 58.4s
```

Boundary:

- This is still synthetic affine-channel evidence only.
- It does not solve fixed-FPR calibration, real saved-video observation, or
  embedding into a video model.
- It does not require GPU yet. At 64 messages / 64 wrong keys, exact CPU search
  is still feasible, but high-mismatch margin is thinning; the next local
  question is synthetic robustness, not video-model execution.

## AISB residual-threshold margin

The first 64-message / 64-wrong-key `gamma = 0.8` multi-case check exposed a
public-acquisition false negative rather than an owner/wrong scoring failure:

```text
current threshold 0.006:
  alignment exact 3 / 4
  one true AISB residual = 0.006108
  false positives = 0

diagnostic threshold 0.00625:
  alignment exact 4 / 4
  false positives = 0
  random non-burst accepted total = 0 / 64
  random non-burst best residual min = 0.08006
  64-message / 64-wrong-key exact scoring pass = 12 / 12
  minimum owner-vs-wrong margin = 0.0423
  exact scoring elapsed = 175.3s for cases 0-2
  exact scoring elapsed = 183.6s for cases 3-5
  cases 6-8 interactive run completed; elapsed is not recorded in JSON
  cases 9-11 interactive run completed; elapsed is not recorded in JSON
```

This is a useful feasibility signal but not a calibrated threshold result. It
only shows that the current synthetic high-mismatch false negative is narrow,
still separable from random non-burst residuals in this toy distribution, and
does not break exact owner/wrong/message scoring in the first 12 high-mismatch
64-message / 64-wrong-key cases. GPU remains unnecessary; the remaining local
issue is larger synthetic coverage and, eventually, calibrated false-positive
control.

The first gamma 1.0 exact-scoring diagnostic is a useful negative boundary:
cases 0-2 pass 2 / 3 on strict exact public alignment while owner/message
recovery remains 3 / 3 and the minimum owner-vs-wrong margin is 0.0289. This
points to public AISB acquisition/ambiguity under stronger mismatch, not to a
GPU-bound or video-model issue.

Follow-up CPU-only threshold-margin diagnosis showed the same point more
sharply: for gamma 1.0, threshold 0.00625 has 3 false negatives across 4
high-mismatch cases, while threshold 0.0075 restores alignment exact 4 / 4 and
keeps random non-burst accepted total at 0 / 64. Exact owner/wrong/message
scoring at gamma 1.0 with diagnostic threshold 0.0075 passes cases 0-11 with
minimum margin 0.0289. This is still only a diagnostic threshold candidate, not
fixed-FPR calibration.

At gamma 1.2, the same CPU-only pattern persists but with thinner margins:
threshold 0.0075 still leaves one false negative across 4 high-mismatch cases,
threshold 0.01 restores alignment exact 4 / 4 and keeps random non-burst
accepted total at 0 / 256, and exact owner/wrong/message scoring passes cases
0-11 with minimum margin 0.0241. This remains synthetic threshold-margin
diagnosis, not a formal threshold or paper claim.

Gamma 1.3 remains just above the current pass boundary: threshold 0.01 restores
alignment exact 4 / 4, random non-burst accepted total stays 0 / 256, and exact
owner/wrong/message scoring passes cases 0-5 with minimum margin 0.0229.
Finer exact-scoring probes place the current synthetic separability transition
between gamma 1.35 and gamma 1.36: at gamma 1.35 cases 0-2 still pass with
minimum margin 0.02006, while at gamma 1.36 cases 0 and 1 fail with minimum
margin 0.01902. Gamma 1.4 confirms the same boundary: public acquisition is
still recoverable at diagnostic threshold 0.01 and random non-burst accepted
total remains 0 / 256, but exact scoring passes only 1 / 3 on cases 0-2. This
transition remains a CPU diagnostic result, not a GPU boundary.

The first evidence-length diagnostic separates useful from unhelpful expansion.
Increasing public AISB burst count alone does not reliably fix the gamma 1.36
margin problem: burst_count 16 passes only 2 / 3, and burst_count 20 still
passes only 2 / 3 with one wrong-key global winner. In contrast, preserving
burst_count 12 while doubling the secret-state filler span restores cases 0-2:
gamma 1.36 passes 3 / 3 with minimum margin 0.0895, and gamma 1.4 passes 12 /
12 across cases 0-11 with minimum margin 0.0710. It also passes gamma 1.6 cases
0-11 with minimum margin 0.0593 after public acquisition is recovered at
threshold 0.01. A pruned two-stage screen fails 2 / 3 on gamma 1.6 cases 0-2,
so it is only a negative acceleration diagnostic and not the active evidence
path. With an explicitly diagnostic threshold extension to 0.0125, the doubled
secret-state span passes gamma 2.5, 3.0, and 5.0 exact scoring on cases 0-2,
with minimum margins 0.0465, 0.0448, and 0.0372 respectively. Gamma 8.0 case 0
remains positive but close to the margin boundary at 0.0298; gamma 10.0 case 0
still recovers the owner message but fails the owner-vs-wrong diagnostic margin
at 0.0187 with filler_multiplier 2. Increasing the secret-state filler span to
3 restores gamma 10.0 cases 0-11 with minimum margin 0.0407 in the full
64-message / 64-wrong-key exact tier. At gamma 15.0 and above, that full tier is
a CPU runtime bottleneck for rapid iteration in the current Python diagnostic,
not a GPU/video boundary. A conservative bounded-DTW early-abandon scorer was
added and validated; process-level parallelism then made gamma 15.0 full
64-message / 64-wrong-key cases 0-2 runnable locally with 4 workers, all passing
with minimum margin 0.0406. A smaller 24-message / 12-wrong-key exact diagnostic
tier remains positive at gamma 15.0, 20.0, and 30.0 with filler_multiplier 3,
with minimum margins 0.0384, 0.0293, and 0.0286. Gamma
30.0 already requires a narrow public residual threshold 0.0105; gamma 50.0
exposes public AISB acquisition overlap for single-burst residual gating, where
checked true-burst residuals and checked random non-burst residuals cannot be
separated by a single residual threshold. A minimal public sequence-consistency
diagnostic then restores the checked acquisition boundary: requiring support for
the 12-burst public template cycle rejects isolated random residual false
positives at gamma 50.0 and 100.0 while preserving checked true sequences. The
current clean CPU route is therefore sequence-supported public acquisition. The
next checked boundary is whether acquisition must freeze one alignment or may
carry a small public ambiguity set. In the smaller 24-message / 12-wrong-key
exact tier, freezing one sequence-supported alignment passes checked gamma 50.0
and 100.0 cases 0-2, but gamma 50.0 cases 0-11 show exact public alignment only
9 / 12 while owner-message recovery remains 12 / 12. The method-design
implication is to avoid forcing an arbitrary public tie-break: carry the public
ambiguity set forward, estimate/equalize per alignment, and score owner/wrong
keys over the same alignment and message hypotheses. That ambiguity-set chain
passes checked gamma 50.0 and 100.0 cases 0-23 in the 24-message /
12-wrong-key tier with owner-message recovery 24 / 24 and minimum margins
0.0240 and 0.0227. It also passes the full 64-message / 64-wrong-key tier for
gamma 50.0 and 100.0 cases 0-2 with 4 CPU workers and minimum margins 0.0465
and 0.0466. Extending the full tier to cases 3-5 shows the next boundary is
still local CPU evidence-length/margin: gamma 50.0 case 5 falls just below the
0.02 diagnostic line at filler_multiplier 3 and recovers when the secret-state
filler span is increased to 4; gamma 100.0 case 3 remains just below the line at
filler 4, and checking filler 5 currently hits the Python exact-scorer runtime
bottleneck. A safe ordered-bounded scorer now uses cheap decimated scores only
to order candidates, then still uses exact bounded DTW for accepted scores. It
matches the exact winner and margin in the checked 24-message / 12-wrong-key
ambiguity case, but still does not finish the full 64-message / 64-wrong-key
gamma 100.0 filler 5 case within 5 minutes. A pure-Python DP hot-path
optimization preserves the original transition tie-break order and improves
medium-tier runtime, but still does not make that full filler-5 case finish
within 5 minutes. A local C implementation of the same exact flattened bounded
DTW recurrence removes that implementation bottleneck without changing the
candidate space or scoring rule. With this native CPU hot path, the full
64-message / 64-wrong-key gamma 100.0 tier passes cases 0-11 when
filler_multiplier is 5 and near_tie_ratio is 5.0. At gamma 500.0, cases 0-11
pass with filler_multiplier 5, while cases 12-23 require filler_multiplier 6
to restore diagnostic margin. Case 20 separates exact public-truth recovery
from public-truth coverage: the public sequence includes one extra false burst,
but owner/wrong scoring over the same contaminated alignment still separates. A
larger gamma 500.0 pass over one full 64-message-index round shows the next
boundary more directly: filler_multiplier 6 passes 63/64 cases, with all 64
cases recovering the owner message and global owner, but case 54 fails because
the public ambiguity set does not cover the true burst sequence. Increasing the
span to filler_multiplier 7 restores checked cases 52-55, including exact
coverage for case 54, and 63/64 completed cases across the full
64-message-index synthetic round all pass. The only uncompleted case is case
24, which exceeded a 5-minute local CPU single-case runtime window without a
JSON result. A follow-up case 24 diagnostic shows public acquisition itself is
not the blocker: 1221 burst candidates reduce to 24 supported ambiguity
sequences, the exact truth sequence is present and covered, and the remaining
workload is 24 * 4160 exact DTW owner/wrong candidates. Sequence-level
`ordered_bounded_c` worker parallelism plus prepared native C arrays and
per-sequence reusable native DP workspaces preserves the exact scoring rule but
still did not finish case 24 inside a 3-minute local CPU window with 8 workers.
`ordered_bounded_global_c` globally orders all sequence/candidate pairs before
the same bounded native DTW and matches the exact winner in the checked
regression, but it also did not finish case 24 inside a 3-minute local CPU
window. A stronger safe lower bound using unavoidable future skip cost and the
maximum possible final diagonal-match count matches exact-scorer regressions
but still does not make full-tier case 24 finish inside the same window. A
reduced diagnostic for the same case index at message_space_size 24 and
wrong_key_count 12 completes in 3.97 seconds with exact margin 0.1473, truth
exact/covered, and owner/global recovery. The current active issue is therefore
public acquisition support under dense distractors plus CPU scoring throughput
for dense ambiguity cases, not GPU/video execution and not key-conditioned
owner/wrong scoring. Independent non-full-tier stress checks remain positive
under the checked crop/delete/repeat, burst-internal deletion, and deterministic
non-affine mismatch diagnostics.

A systematic CPU stress grid now separates the next method boundary more
cleanly. With the same edit-stress payload construction, gamma 0.5 passes at
noise 0.012 and 0.016, and gamma 0.8 passes at noise 0.012. At gamma 0.8 with
noise 0.016, owner-message recovery remains 4 / 4 but public alignment drops to
3 / 4. At gamma 1.0, owner-message recovery still remains 4 / 4, while
alignment remains 3 / 4 and margin falls below the diagnostic 0.02 threshold in
some cases. The active synthetic mechanism issue is therefore public
acquisition robustness and diagnostic margin under stronger non-affine/noise
stress, not GPU/video execution.

The ambiguity-set variant directly addresses this boundary in the checked CPU
grid. Instead of forcing a single public alignment before scoring, it freezes a
public ambiguity set and scores owner/wrong keys over the same set. In the
expanded stress grid this restores 4 / 4 pass, truth coverage, global owner
recovery, and owner-message recovery for every checked 16-message /
24-wrong-key cell through gamma 1.0 and noise 0.016, with minimum margin
0.1927. A focused hard-cell scale check then raises gamma 0.8/1.0 at noise
0.016 to 8 cases per cell: forced single-path alignment falls to 5 / 8 and
5 / 8 alignment with margins as low as -0.0163, while ambiguity-set scoring
passes 8 / 8 in both cells with truth coverage, owner/global recovery, and
minimum margins 0.1877 and 0.1600. A full-tier hard-cell check at gamma 1.0 and
noise 0.016 then raises the ambiguity-set candidate space to 64 messages /
64 wrong keys. Across 16 cases, forced single-path falls to 6 / 16 pass,
11 / 16 alignment, and 15 / 16 owner-message recovery. The ambiguity-set route
passes 16 / 16 with exact truth coverage, owner/global recovery, candidate_count
4160 per case, and minimum margin 0.1100. Increasing deterministic mismatch to
gamma 1.2 and 1.4 at the same full 64-message / 64-wrong-key tier still passes
8 / 8 per high-gamma cell with minimum margins 0.0988 and 0.0883, while forced
single-path remains 3 / 8 pass and 7 / 8 owner-message recovery. Extending the
same tier to gamma 1.6 and 2.0 still passes 8 / 8 ambiguity-set cases with
minimum margins 0.0816 and 0.0775; forced single-path drops to 1 / 8 pass and
6 / 8 owner-message recovery at gamma 2.0. Pushing the same full tier to
gamma 2.5 and 3.0 still passes 8 / 8 ambiguity-set cases with minimum margins
0.0652 and 0.0591, while forced single-path remains only 1 / 8 pass. The same
full-tier ambiguity-set route remains positive at gamma 4.0 and 5.0 with
8 / 8 pass and minimum margins 0.0506 and 0.0499, although the decreasing
margin shows that the synthetic robustness boundary is approaching. At gamma
8.0 and 10.0, the same route still passes 8 / 8 at the full 64-message /
64-wrong-key tier, with minimum margins 0.0489 and 0.0441. The next checked
gamma jump to 15.0 and 20.0 also remains positive at the same tier and case
count, with 8 / 8 pass, exact truth coverage, owner/global recovery, and
minimum margins 0.0435 and 0.0375. Pushing again to gamma 30.0 and 50.0 still
passes 8 / 8 in both cells with exact truth coverage, owner/global recovery,
and minimum margins 0.0376 and 0.0367. The margin is no longer dropping quickly
in this checked range; gamma 100.0 also passes 8 / 8 with minimum margin 0.0360.
The same gamma 100.0 full tier remains positive at noise_std 0.020 with minimum
margin 0.0360; noise_std 0.040 and 0.050 also pass 8 / 8 but move close to the
0.02 diagnostic line, with minimum margins 0.0249 and 0.0264. noise_std 0.080
still passes 8 / 8 with minimum margin 0.0266. Extending that harder setting to
16 cases at filler_multiplier 5 exposes one thin diagnostic miss: owner and
global recovery remain 16 / 16 and truth coverage remains 16 / 16, but minimum
margin is 0.0196, just under the 0.02 line. Increasing only the secret-state
filler span to 7 restores the checked 16-case tier to 16 / 16 pass with minimum
margin 0.0277. Continuing the same filler 7 tier in two CPU-manageable 8-case
chunks covers cases 16-31; both chunks pass 8 / 8, giving a checked 32-case
range when run segmented. Cases 32-39 were then checked as single-case CPU
segments and pass 8 / 8 with minimum margin 0.0340, giving a checked 40-case
range under this synthetic setting. Cases 40-47 also pass as single-case CPU
segments with minimum margin 0.0260, giving a checked 48-case range. Cases
48-55 also pass as single-case CPU segments with minimum margin 0.0338, giving
a checked 56-case range. Cases 56-63 then pass with minimum margin 0.0780,
completing the full checked 64-case message-index range at 64 / 64. The
weakest checked point remains case 45 with margin 0.0260. Raising noise_std to
0.10 still passes the first checked 8 single-case segments, with minimum margin
0.0303 at case 7, but the margin remains thin. Continuing cases 8-15 also
passes 8 / 8 with minimum margin 0.0399 at case 8. Case 15 originally exposed
a CPU ordering-throughput hotspot before exact scoring; replacing the
ordering-only Python DTW heuristic with a sparse aligned-distance heuristic made
the same exact/bounded scorer finish and pass with margin 0.1080. Cases 16-23
also pass 8 / 8 with minimum margin 0.0386 at case 21, extending the
noise_std 0.10 checked range to 24 / 24. Cases 24-31 then add another
8 / 8 pass, with minimum margin 0.0718 at case 28, extending the checked
noise_std 0.10 range to 32 / 32. Cases 32-39 also pass 8 / 8, with minimum
margin 0.0363 at case 34. Cases 40-47 also pass 8 / 8, with minimum margin
0.0299 at case 45. Cases 48-55 also pass 8 / 8, with minimum margin 0.0324
at case 53. Cases 56-63 complete the checked range at 64 / 64, with minimum
margin 0.0775 in that final segment. The weakest noise_std 0.10 checked point
remains case 45 at margin 0.0299. Raising noise_std to 0.12 completes the
checked message-index range at 64 / 64 pass. The weakest checked noise_std 0.12
point is case 34 with margin 0.0257, positive
but close to the 0.02 diagnostic line. Raising noise_std again to 0.14 still
completes the full checked 64-case message-index range at 64 / 64 pass; cases
56-63 add an 8 / 8 segment with minimum margin 0.0758 at case 62, while the
weakest checked point remains case 45 with margin 0.0248, close to the 0.02
diagnostic line. The full checked noise_std 0.16 range now passes cases 0-63
at 64 / 64; the latest cases 56-63 segment has minimum margin 0.0739 at
case 62, and the overall checked minimum margin remains 0.0263 at case 7.
The full checked noise_std 0.18 range now passes cases 0-63 at 64 / 64, with
segment 56-63 minimum margin 0.0734 at case 62; case 34 remains the overall
checked minimum for the tier at 0.0212, close to the 0.02 diagnostic line.
Raising noise_std to 0.20 passes cases 0-31 at 32 / 32, then reaches the first
checked diagnostic boundary at case 34 in segment 32-39: owner/global recovery
is still correct, but margin 0.0175 falls below the 0.02 diagnostic line. This
is a synthetic margin boundary, not a GPU or video-model boundary. Single-case
checks of case 34 at noise_std 0.1900, 0.1850, 0.1825, and 0.1810 also remain
owner/global correct but below the diagnostic line, indicating distributional
near-boundary behavior rather than a simple smooth interpolation of one fixed
observation. A top-k competition diagnostic for case 34 at noise_std 0.20
confirms that the exact truth sequence is selected and owner message_34 remains
the global top scorer; the margin failure is caused by the nearest wrong-key
candidate being too close. The active synthetic boundary is therefore
wrong-key separation margin under the current trajectory evidence span, not
AISB acquisition. Increasing the public AISB trajectory evidence from 12 bursts
to 16 bursts restores the same noise_std 0.20 cases 32-39 segment to 8 / 8
pass with minimum margin 0.08868 and case 34 margin 0.16218. The current
CPU-only mechanism direction is evidence-span mapping rather than GPU/video
execution. Continuing burst_count 16 through cases 40-47 also passes 8 / 8,
with minimum margin 0.05190 at case 47, extending the checked noise_std 0.20
range to 48 / 64. Cases 48-55 also pass 8 / 8, with minimum margin 0.04821
at case 53, extending the checked noise_std 0.20 range to 56 / 64. Cases
56-63 complete the burst_count 16 checked message-index range: 64 / 64 pass
at gamma 100.0 / noise_std 0.20 / 64 messages / 64 wrong keys, with weakest
checked margin 0.04821 at case 53.
Raising to noise_std 0.22, the first burst_count 16 segment cases 0-7 also
passes 8 / 8 with minimum margin 0.07796; cases 8-15 also pass 8 / 8 with
minimum margin 0.08860; cases 16-23 also pass 8 / 8 with minimum margin
0.06563, extending the checked noise_std 0.22 range to 24 / 64. Cases 24-31
also pass 8 / 8 with minimum margin 0.06989, extending the checked
noise_std 0.22 range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum
margin 0.08873, extending the checked noise_std 0.22 range to 40 / 64. Cases
40-47 also pass 8 / 8 with minimum margin 0.05232, extending the checked
noise_std 0.22 range to 48 / 64. Case50 then exposes a top1 acquisition
beam-width miss: the true deleted-burst hypothesis is retained by
`top_k_per_start=2`, not by top1. With top2 acquisition, exact/bounded official
scoring passes case50 with margin 0.14948. To continue CPU-only mechanism
triage without making an exhaustive wrong-key claim, cases 48-63 were then run
with top2 acquisition and diagnostic-pruned scoring that exact-scores all owner
messages and a public cheap-screened wrong-key subset. Cases 48-55 pass 8 / 8
with minimum margin 0.07371; cases 56-63 pass 8 / 8 with minimum margin
0.10467. This extends the checked noise_std 0.22 mechanism-triage range to
64 / 64, while explicitly marking the last 16 cases as diagnostic-pruned rather
than exhaustive wrong-key evidence. Raising again to noise_std 0.24, the same
top2 + diagnostic-pruned tier passes cases 0-7 with minimum margin 0.10372 and
cases 8-15 with minimum margin 0.11475, cases 16-23 with minimum margin
0.08395, cases 24-31 with minimum margin 0.12069, and cases 32-39 with
minimum margin 0.09328, cases 40-47 with minimum margin 0.05653, and cases
48-55 with minimum margin 0.07433, and cases 56-63 with minimum margin
0.10521, closing the checked noise_std 0.24 range at 64 / 64. Raising to
noise_std 0.26, cases 0-7 pass 8 / 8 with minimum margin 0.10408, and cases
8-15 pass 8 / 8 with minimum margin 0.11718, and cases 16-23 pass 8 / 8
with minimum margin 0.08616, and cases 24-31 pass 8 / 8 with minimum margin
0.12042, cases 32-39 pass 8 / 8 with minimum margin 0.09435, and cases
40-47 pass 8 / 8 with minimum margin 0.05891, and cases 48-55 pass 8 / 8
with minimum margin 0.07383, and cases 56-63 pass 8 / 8 with minimum margin
0.10519, closing the checked noise_std 0.26 range at 64 / 64. Case29 exposed
a local CPU exact-scoring throughput hotspot under workers=1, but
sequence-level parallel scoring completed it with workers=4; this was not a
scientific failure or GPU boundary. Raising to noise_std 0.28, cases 0-7 pass
8 / 8 with minimum margin 0.10748, and cases 8-15 pass 8 / 8 with minimum
margin 0.11334, cases 16-23 pass 8 / 8 with minimum margin 0.08554, and
cases 24-31 pass 8 / 8 with minimum margin 0.11995, and cases 32-39 pass
8 / 8 with minimum margin 0.09460, and cases 40-47 pass 8 / 8 with minimum
margin 0.05336, cases 48-55 pass 8 / 8 with minimum margin 0.07161, and
cases 56-63 pass 8 / 8 with minimum margin 0.10453, closing the checked
noise_std 0.28 range at 64 / 64. Raising to noise_std 0.30, cases 0-7 pass
8 / 8 with minimum margin 0.09960, and cases 8-15 pass 8 / 8 with minimum
margin 0.11246, and cases 16-23 pass 8 / 8 with minimum margin 0.08514,
and cases 24-31 pass 8 / 8 with minimum margin 0.11978, extending the checked
noise_std 0.30 range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum
margin 0.09366, extending the checked noise_std 0.30 range to 40 / 64. Cases
40-47 also pass 8 / 8 with minimum margin 0.05886, extending the checked
noise_std 0.30 range to 48 / 64. Cases 48-55 also pass 8 / 8 with minimum
margin 0.07119, and cases 56-63 pass 8 / 8 with minimum margin 0.10058,
closing the checked noise_std 0.30 range at 64 / 64. Raising to noise_std
0.32, cases 0-7 pass 8 / 8 with minimum margin 0.10207, and cases 8-15 pass
8 / 8 with minimum margin 0.11095, extending the checked noise_std 0.32 range
to 16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin 0.08368,
and cases 24-31 pass 8 / 8 with minimum margin 0.11960, extending the checked
noise_std 0.32 range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum
margin 0.08919, extending the checked noise_std 0.32 range to 40 / 64. Cases
40-47 also pass 8 / 8 with minimum margin 0.06073, extending the checked
noise_std 0.32 range to 48 / 64. Cases 48-55 also pass 8 / 8 with minimum
margin 0.07016, and cases 56-63 pass 8 / 8 with minimum margin 0.09623,
closing the checked noise_std 0.32 range at 64 / 64. Raising to noise_std
0.34, cases 0-7 pass 8 / 8 with minimum margin 0.09776, and cases 8-15
pass 8 / 8 with minimum margin 0.10929955, extending the checked noise_std
0.34 range to 16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin
0.08209735, extending the checked noise_std 0.34 range to 24 / 64. Cases
24-31 also pass 8 / 8 with minimum margin 0.12069871, extending the
checked noise_std 0.34 range to 32 / 64. Cases 32-39 also pass 8 / 8 with
minimum margin 0.09147009, extending the checked noise_std 0.34 range to
40 / 64. Cases 40-47 also pass 8 / 8 with minimum margin 0.06057691,
extending the checked noise_std 0.34 range to 48 / 64. Cases 48-55 also pass
8 / 8 with minimum margin 0.06498330, extending the checked noise_std 0.34
range to 56 / 64. Cases 56-63 also pass 8 / 8 with minimum margin 0.09374521,
closing the checked noise_std 0.34 range at 64 / 64. Raising to noise_std
0.36, cases 0-7 pass 8 / 8 with minimum margin 0.09361803, and cases 8-15
pass 8 / 8 with minimum margin 0.10295556, extending the checked noise_std
0.36 range to 16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin
0.08668786, extending the checked noise_std 0.36 range to 24 / 64. Cases
24-31 also pass 8 / 8 with minimum margin 0.11924334, extending the checked
noise_std 0.36 range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum
margin 0.09238279, extending the checked noise_std 0.36 range to 40 / 64.
Cases 40-47 also pass 8 / 8 with minimum margin 0.06148871, extending the
checked noise_std 0.36 range to 48 / 64. Cases 48-55 also pass 8 / 8 with
minimum margin 0.06968034, extending the checked noise_std 0.36 range to
56 / 64. Cases 56-63 also pass 8 / 8 with minimum margin 0.08903584, closing
the checked noise_std 0.36 range at 64 / 64. Raising to noise_std 0.38, cases
0-7 pass 8 / 8 with minimum margin 0.09210281, and cases 8-15 pass 8 / 8
with minimum margin 0.10081056, extending the checked noise_std 0.38 range to
16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin 0.08678848,
extending the checked noise_std 0.38 range to 24 / 64. Cases 24-31 also pass
8 / 8 with minimum margin 0.11695454, extending the checked noise_std 0.38
range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum margin 0.08942796,
extending the checked noise_std 0.38 range to 40 / 64. The active route remains
CPU-only synthetic noise/evidence-span and scoring-scale mapping. Cases 40-47
also pass 8 / 8 with minimum margin 0.06385964, extending the checked
noise_std 0.38 range to 48 / 64. Cases 48-55 also pass 8 / 8 with minimum
margin 0.07151290, extending the checked noise_std 0.38 range to 56 / 64.
Cases 56-63 also pass 8 / 8 with minimum margin 0.08474072, closing the
checked noise_std 0.38 range at 64 / 64. Raising to noise_std 0.40, cases
0-7 pass 8 / 8 with minimum margin 0.08821583. Cases 8-15 also pass 8 / 8
with minimum margin 0.09863218, extending the checked noise_std 0.40 range
to 16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin 0.08691652,
extending the checked noise_std 0.40 range to 24 / 64. Cases 24-31 also pass
8 / 8 with minimum margin 0.11608262, extending the checked noise_std 0.40
range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum margin 0.08839395,
extending the checked noise_std 0.40 range to 40 / 64. Cases 40-47 also pass
8 / 8 with minimum margin 0.06297970, extending the checked noise_std 0.40
range to 48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.06940009,
extending the checked noise_std 0.40 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.08638474, closing the checked noise_std 0.40 range
at 64 / 64. Raising to noise_std 0.42, cases 0-7 pass 8 / 8 with minimum
margin 0.08818240. Cases 8-15 also pass 8 / 8 with minimum margin 0.09784793,
extending the checked noise_std 0.42 range to 16 / 64. Cases 16-23 also pass
8 / 8 with minimum margin 0.09197900, extending the checked noise_std 0.42
range to 24 / 64. Cases 24-31 also pass 8 / 8 with minimum margin 0.11412904,
extending the checked noise_std 0.42 range to 32 / 64. Cases 32-39 also pass
8 / 8 with minimum margin 0.08782796, extending the checked noise_std 0.42
range to 40 / 64. Cases 40-47 also pass 8 / 8 with minimum margin 0.06141315,
extending the checked noise_std 0.42 range to 48 / 64. Cases 48-55 also pass
8 / 8 with minimum margin 0.06861692, extending the checked noise_std 0.42
range to 56 / 64. Cases 56-63 also pass 8 / 8 with minimum margin 0.08430650,
closing the checked noise_std 0.42 range at 64 / 64. Raising to noise_std
0.44, cases 0-7 pass 8 / 8 with minimum margin 0.08862351. Cases 8-15 also
pass 8 / 8 with minimum margin 0.09980410, extending the checked noise_std
0.44 range to 16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin
0.09127598, extending the checked noise_std 0.44 range to 24 / 64. Cases
24-31 also pass 8 / 8 with minimum margin 0.11341359, extending the checked
noise_std 0.44 range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum
margin 0.08761476, extending the checked noise_std 0.44 range to 40 / 64.
Cases 40-47 also pass 8 / 8 with minimum margin 0.06277592, extending the
checked noise_std 0.44 range to 48 / 64. Cases 48-55 also pass 8 / 8 with
minimum margin 0.07073149, extending the checked noise_std 0.44 range to
56 / 64. Cases 56-63 also pass 8 / 8 with minimum margin 0.07963960, closing
the checked noise_std 0.44 range at 64 / 64. Raising to noise_std 0.46, cases
0-7 pass 8 / 8 with minimum margin 0.08912916. Cases 8-15 also pass 8 / 8
with minimum margin 0.09899634, extending the checked noise_std 0.46 range to
16 / 64. Cases 16-23 also pass 8 / 8 with minimum margin 0.09003097,
extending the checked noise_std 0.46 range to 24 / 64. Cases 24-31 also pass
8 / 8 with minimum margin 0.10937305, extending the checked noise_std 0.46
range to 32 / 64. Cases 32-39 also pass 8 / 8 with minimum margin 0.08296188,
extending the checked noise_std 0.46 range to 40 / 64. Cases 40-47 also pass
8 / 8 with minimum margin 0.05920578, extending the checked noise_std 0.46
range to 48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.06771293,
extending the checked noise_std 0.46 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.07997671, closing the checked noise_std 0.46
range at 64 / 64. Raising to noise_std 0.48, cases 0-7 pass 8 / 8 with
minimum margin 0.08923162. Cases 8-15 also pass 8 / 8 with minimum margin
0.09699572, extending the checked noise_std 0.48 range to 16 / 64. Cases
16-23 also pass 8 / 8 with minimum margin 0.08788011, extending the checked
noise_std 0.48 range to 24 / 64. Cases 24-31 also pass 8 / 8 with minimum
margin 0.10637933, extending the checked noise_std 0.48 range to 32 / 64.
Cases 32-39 also pass 8 / 8 with minimum margin 0.08064904, extending the
checked noise_std 0.48 range to 40 / 64. Cases 40-47 also pass 8 / 8 with
minimum margin 0.05655783, extending the checked noise_std 0.48 range to
48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.06854657,
extending the checked noise_std 0.48 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.08137164, closing the checked noise_std 0.48
range at 64 / 64. Raising to noise_std 0.50, cases 0-7 pass 8 / 8 with
minimum margin 0.09375566. Cases 8-15 also pass 8 / 8 with minimum margin
0.09635263, extending the checked noise_std 0.50 range to 16 / 64. Cases
16-23 also pass 8 / 8 with minimum margin 0.08730083, extending the checked
noise_std 0.50 range to 24 / 64. Cases 24-31 also pass 8 / 8 with minimum
margin 0.10415634, extending the checked noise_std 0.50 range to 32 / 64.
Cases 32-39 also pass 8 / 8 with minimum margin 0.08194289, extending the
checked noise_std 0.50 range to 40 / 64. Cases 40-47 also pass 8 / 8 with
minimum margin 0.05578955, extending the checked noise_std 0.50 range to
48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.06911470,
extending the checked noise_std 0.50 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.07806989, closing the checked noise_std 0.50
range at 64 / 64. Raising to noise_std 0.52, cases 0-7 pass 8 / 8 with
minimum margin 0.09044599. Cases 8-15 also pass 8 / 8 with minimum margin
0.09866961, extending the checked noise_std 0.52 range to 16 / 64. Cases
16-23 also pass 8 / 8 with minimum margin 0.08655462, extending the checked
noise_std 0.52 range to 24 / 64. Cases 24-31 also pass 8 / 8 with minimum
margin 0.10843663, extending the checked noise_std 0.52 range to 32 / 64.
Cases 32-39 also pass 8 / 8 with minimum margin 0.08209579, extending the
checked noise_std 0.52 range to 40 / 64. Cases 40-47 also pass 8 / 8 with
minimum margin 0.05007786, extending the checked noise_std 0.52 range to
48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.07073198,
extending the checked noise_std 0.52 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.07680597, closing the checked noise_std 0.52
range at 64 / 64. Raising to noise_std 0.54, cases 0-7 pass 8 / 8 with
minimum margin 0.09154538. Cases 8-15 also pass 8 / 8 with minimum margin
0.09827562, extending the checked noise_std 0.54 range to 16 / 64. Cases
16-23 also pass 8 / 8 with minimum margin 0.08079990, extending the checked
noise_std 0.54 range to 24 / 64. Cases 24-31 also pass 8 / 8 with minimum
margin 0.10307755, extending the checked noise_std 0.54 range to 32 / 64.
Cases 32-39 also pass 8 / 8 with minimum margin 0.08281960, extending the
checked noise_std 0.54 range to 40 / 64. Cases 40-47 also pass 8 / 8 with
minimum margin 0.04712975, extending the checked noise_std 0.54 range to
48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.06931735,
extending the checked noise_std 0.54 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.06999606, closing the checked noise_std 0.54
range at 64 / 64. Raising to noise_std 0.56, cases 0-7 pass 8 / 8 with
minimum margin 0.09051082. Cases 8-15 also pass 8 / 8 with minimum margin
0.09464360, extending the checked noise_std 0.56 range to 16 / 64. Cases
16-23 also pass 8 / 8 with minimum margin 0.07814625, extending the checked
noise_std 0.56 range to 24 / 64. Cases 24-31 also pass 8 / 8 with minimum
margin 0.09902808, extending the checked noise_std 0.56 range to 32 / 64.
Cases 32-39 also pass 8 / 8 with minimum margin 0.07365479, extending the
checked noise_std 0.56 range to 40 / 64. Cases 40-47 also pass 8 / 8 with
minimum margin 0.05104159, extending the checked noise_std 0.56 range to
48 / 64. Cases 48-55 also pass 8 / 8 with minimum margin 0.07028426,
extending the checked noise_std 0.56 range to 56 / 64. Cases 56-63 also pass
8 / 8 with minimum margin 0.06647356, closing the checked noise_std 0.56
range at 64 / 64. Raising to noise_std 0.58, cases 0-7 pass 8 / 8 with
minimum margin 0.09066536. Cases 8-15 also pass 8 / 8 with minimum margin
0.09670570, extending the checked noise_std 0.58 range to 16 / 64. Cases
16-23 also pass 8 / 8 with minimum margin 0.07675474, extending the checked
noise_std 0.58 range to 24 / 64. The remaining cases 24-63 also pass,
closing the checked noise_std 0.58 range at 64 / 64 with full-layer minimum
margin 0.04732445. Raising to noise_std 0.60, the full 64-case layer also
passes 64 / 64 with truth coverage 64 / 64, owner/global recovery 64 / 64,
full-layer minimum margin 0.04807966, mean margin 0.11447229, maximum
ambiguity sequence count 192, and maximum exact-score count 29977. CPU-only
synthetic mapping can continue, but exact-score throughput and native helper
compilation are now practical CPU infrastructure constraints.
At noise_std 0.62, cases 0-7 produce the first checked boundary: 7/8 pass,
owner/global recovery remains 8/8, but case 4 loses public truth-sequence
coverage under residual_threshold 0.0125. A CPU-only diagnostic rerun recovers
that case at residual_threshold 0.015, so the next scientific question is the
residual-threshold versus ambiguity/exact-work tradeoff, not GPU.
A single 32-case batch was interrupted
after more than 40 minutes without output, so the practical issue is
exact-scoring CPU throughput/batching, not a GPU/video requirement. This
supports the current
interpretation that the active synthetic boundary is trajectory-evidence
span/margin, not public acquisition. The current CPU-only method direction is
therefore:

```text
AISB acquisition
-> freeze public ambiguity set, not one forced path
-> estimate/equalize per public candidate alignment
-> score owner and wrong keys over the same public alignment set
```

This remains synthetic diagnostics only; it is not fixed-FPR, video evidence,
or a paper claim.

A gamma 500.0 public sequence-consistency diagnostic shows 1 / 64 isolated
residual false positive among random non-burst cases, but 0 / 64 after the
public template-sequence support constraint. The current method-design
implication is that owner/wrong separability is improved by longer secret-state
trajectory evidence, and public AISB acquisition should use sequence-level
public structure plus shared ambiguity-set scoring rather than isolated
single-burst residuals or a single forced public alignment. This remains
CPU-only synthetic evidence.

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


## 2026-08-05 GPU observation probe boundary

After CPU-only threshold-window and temporal-robustness closure, a minimal
GPU/saved-video observation probe was executed from Colab. It generated 10 short
Wan videos and packaged result JSON, subprocess log, and MP4 files under:

```text
G:\我的云端硬盘\SSTW\diagnostic_tests\sc_sstw_gpu_observation_probe\20260805_151517
```

The run is a diagnostic observation probe only. It does not perform detection,
observer synchronization, wrong-key testing, fixed-FPR calibration, video attack
testing, or a paper claim.

The result is negative for the tested observation path:

```text
probe_decision = no_observable_signal
repeatability_floor_l2 = 0.651854794829629
public_aisb_residual = 1.0996303271706644
public_visibility_snr_over_repeat_floor = 0.04765786877152943
state_relation_delta_l2 = 0.0006008096833611169
state_relation_snr_over_repeat_floor = 0.0009216925120849139
```

Method implication:

- CPU-only AISB acquisition, self-calibration, and state synchronization remain
  feasible in the synthetic relation-channel construction.
- The prompt-only saved-video patch-brightness observation path is not usable in
  its current form. The readout floor dominates the public relation signal, and
  the state-window direction signal is essentially absent.
- The next observation hypothesis should move away from static patch brightness
  and toward motion-subject readout: frame difference, moving-blob centroid,
  foreground mask, bbox/center tracking, or optical-flow centroid. This should
  begin as another minimal observation probe, not as a detector, Gate, observer,
  or fixed-FPR experiment.
