# Learned saved-MP4 observation front-end contract

This protocol is the only remaining observation route after the direct
model-output and internal-attention carriers both failed with the frozen
structured readout. Those failures remain `NO_GO` for their exact routes. The
overall method remains `NOT_DETERMINED`.

The independent auditor admitted this design only for repository implementation
and CPU/static review. It does not admit GPU generation, training, held-out
inspection, private scoring, temporal editing, or a method claim.

## Frozen scientific boundary

The Wan model revision, block-29 `attn1` output carrier, two analytic injection
bases, relative RMS, denoising/CFG behavior, 13 state points, and 49-to-13 frame
groups are byte-for-byte values in
`configs/learned_observation_frontend.json`. Only new public prompts and seeds
change. All viewed failed-run material and every derivative are quarantined.

At inference the learned component accepts only a standardized `13 x 30`
feature matrix extracted from one saved MP4 and returns a continuous `13 x 2`
matrix. Its forward signature has no dataset ID, time index, public state,
candidate key, message, alignment, prompt, seed, truth, clean counterpart,
paired difference, activation, or injection record. The 530-parameter MLP is
shared independently across windows and has no positional or cross-window path.

Training is public-only. Per training video, the nuisance affine map is fitted
only on indices 0--3. Indices 4--5 are held out from that affine fit and enter
only the outer relation loss. They are held out within each video's calibration;
their public coordinates are not unseen by model training. Indices 6--12 enter
only the unlabeled rank term. No fitted nuisance map is saved or used at
inference.

## Acquisition and calibration order

AISB searches exactly the eight forward complete length-six windows using only
the frozen `burst_alpha` public template. It admits no deletion, reversal,
cyclic search, alternate template, or candidate key. The complete static AISB
protocol has SHA-256
`d71fb9e125363b1834ec1da34280b73e49be2678f4a149ae68f9ca9f3f7f7369`.

For every observation the accepted ambiguity artifact must be canonically
serialized, written, closed, read back, and hash-verified before calibration.
Calibration consumes only the read-back artifact and indices 0--3. Evaluation
on indices 4--5 and truth auditing happen afterward. Truth never selects a
candidate.

## Stage isolation

- GPU train/validation, if later admitted, may access only dataset IDs
  41001--41006. It must freeze the step-2000 weights and normalizer before any
  held-out access.
- Dataset IDs 41007--41016 are inaccessible until a separate independent audit
  admits the public held-out/null stage.
- All two validation videos or all two held-out videos must pass every frozen
  per-video threshold. The eight independent clean null videos must produce
  zero acquisitions.
- The null result is a minimal empirical clock/source control, not a fixed-FPR
  or generalization claim.
- Owner/wrong-key and temporal-edit stages remain separately locked.

Any threshold change, new carrier, alternate decoder, checkpoint selection,
held-out fit leakage, old-run reuse, or failure-driven hyperparameter change is
a new protocol and cannot repair this one.
