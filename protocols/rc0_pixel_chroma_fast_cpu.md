# RC0 pixel-chroma fast CPU diagnostic

## Scope

This one-run experiment is `DIAGNOSTIC_ONLY`. It asks whether one frozen,
post-decode/pre-H264 pixel-chroma carrier survives saved MP4 (Step 1), then,
only if all four pixel cells pass, whether the already-frozen paired public 30D
readout retains the A/B schedule relation (Step 2). It is not a Gate, a formal
result, or a method claim.

The prior final-VAE-latent carrier is stopped. Its A/B videos and measurements
are not inputs to this design. The only image inputs are the two exact OFF_R1
members identified in `configs/rc0_pixel_chroma_fast_cpu.json` from source run
`dfa1c782d409fa02` and ZIP SHA-256
`76ddc314132dfc208ebc6dbc04bdbe77ca5da731b16c0345eb139a661c378a4f`.

## Frozen carrier

Each source MP4 is independently decoded once to the exact uint8 RGB array
`[49,320,512,3]`. That same in-memory array is the input to four separate H264
encoding calls in the order `OFF_R1, OFF_R2, A, B`. OFF has no residual.

For normalized RGB, let

- `phi_x(x) = cos(2*pi*x/512)`;
- `phi_y(y) = cos(2*pi*y/320)`;
- `u = (1,-1,0)/sqrt(2)`;
- `v = (1,1,-2)/sqrt(6)`.

Schedule point 0 maps to frame 0. Each point 1 through 12 maps to four
successive frames, covering frames 1 through 48. For A or B the raw residual is

`q_x(t) phi_x(x) u + q_y(t) phi_y(y) v`.

One scalar mean over the entire 49-frame residual is subtracted, then the full
clip is divided by its single global RMS. The resulting unit-RMS field is
multiplied by exactly `6/255`, added in float32, and clipped once to `[0,1]`.
The pre-clip target and post-clip actual residual RMS are recorded. There is no
alternative basis, strength, color projection, schedule, or selectable option.

## Budget and encoding

The only order is two groups (`orbital_glass`, `articulated_paper`) times four
conditions (`OFF_R1`, `OFF_R2`, `A`, `B`), exactly eight started and completed
attempts, retry index zero. A started failure stops the run; there is no retry,
replacement, additional sample, parameter adjustment, or result-dependent
selection. Every condition uses the same `imageio.v3.imwrite` FFMPEG call with
libx264, quality 5, macroblock 16, 8 fps, and requires saved H264/yuv420p,
512x320, 49-frame output.

## Frozen decisions

Step 1 is the existing RC0 Level P formula: OFF repeat noise must be at most
`1/255`; each A/B effect must be at least `2/255` and at least three times its
group's OFF floor. Both groups times both conditions must pass without
averaging or majority vote.

Only then is the frozen 30D extractor recomputed from all eight newly saved
MP4s and passed to the existing Level R formula. Each group and A/B condition
must accept its own template at start 0 while rejecting its own wrong windows
and every cross-template window under the unchanged residual boundary `0.25`.

- P fail: Step 1 `NOT_FEASIBLE`, Step 2 `INSUFFICIENT_TO_DECIDE`, route
  `CURRENT_CARRIER_NOT_FEASIBLE`; stop this pixel carrier.
- P pass and R pass: both steps `FEASIBLE`, route
  `KEEP_CARRIER_BUILD_BLIND_READOUT`; the next question is `STEP3_AISB_CPU`.
- P pass and R fail: Step 1 `FEASIBLE`, Step 2 `NOT_FEASIBLE`, route
  `KEEP_CARRIER_SWITCH_TO_VAE_READOUT`; the next minimum experiment is a VAE
  re-encode diagnostic prepared separately.
- Missing, invalid, undecodable, or excessive OFF-repeat evidence makes the
  affected question `INSUFFICIENT_TO_DECIDE`, route `DIAGNOSTIC_INSUFFICIENT`.

All outputs retain `formal_result=false` and
`stage_progression_allowed=false`. No result changes these frozen parameters.
