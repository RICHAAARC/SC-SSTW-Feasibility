# RC0 fresh exact4 base generation package

This is a `DIAGNOSTIC_ONLY` Phase-B delivery. It creates four fresh, carrier-free
base MP4s for a later CPU effectiveness diagnostic. It is not a Gate, manifest,
provenance system, paper validation, carrier execution, or detector execution.

## Exact four attempts

The sole order is `liquid_mosaic` (seed 53011), `clockwork_escapement`
(53012), `steam_fins` (53013), and `magnetic_filings` (53014). Their prompts and
SHA-256 values are frozen in the config and differ from the earlier
`orbital_glass`/52001 and `articulated_paper`/52002 groups. The four grammars are
respectively a diffusive field, rigid mechanism, fluid advection, and collective
particle reorganization.

Each attempt calls the exact Wan 1.3B revision
`0fad780a534b6463e45facd96134c9f345acfa5b` with the existing frozen negative
prompt, BF16, default scheduler/sampler, 8 steps, guidance 5, 49 frames, and
512x320 geometry. It uses the official pipeline's default decoded-video output.
There is no latent output request, manual VAE decode, carrier schedule, residual,
hook, A/B condition, or GPU detector. Each base is encoded once as H264/yuv420p,
8 fps, quality 5, macroblock 16. Starting an attempt consumes it. Failure stops;
there is no retry, replacement, prompt/seed change, or additional sample.

The notebook only clones an exact ref, obtains and verifies the exact model
snapshot, invokes this base runner, and archives logs, identities, checksums, and
the four base MP4s to a new Drive target. The user alone runs it on Colab GPU.

## Frozen Phase C (registered, not executed here)

CPU Phase C will decode each base once and use that same frame array for
`OFF_R1`, `OFF_R2`, `A`, and `B`. All four conditions independently traverse the
same frozen H264 encoder. OFF residual is zero; A/B use the already frozen
integer-coordinate zero-sum RGB cosine carrier at 6/255 and schedules A/B.

Each resulting saved condition MP4 is then decoded, index-selected, and encoded
again. This makes all timing cases real saved MP4s, including identity. Symbol 0
is frame 0; symbols 1..12 are consecutive four-frame blocks. Exact symbol and
expanded 49-frame source-index arrays are in the config:

- identity: symbols 0..12;
- delete6_duplicate12: omit block 6 and append a second block 12;
- local_phase_plus1: omit block 6 and duplicate block 7 in its place;
- local_phase_minus1: duplicate block 5 in place of block 6.

No interpolation, ffmpeg filter, temporal search, added frame, or result-driven
mapping is allowed. The frozen target-only detector from Phase A evaluates each
transformed MP4. Only all fresh groups, OFF controls, A/B conditions, and all
four real-MP4 transforms passing may be reported as
`EFFECTIVE_ON_FRESH_SMALL_SAMPLE`. Phase B does not execute or claim Phase C.

All outputs retain `formal_result=false` and
`stage_progression_allowed=false`.
