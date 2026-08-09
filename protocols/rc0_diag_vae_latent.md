# RC0-DIAG-CARRIER-SWITCH-VAE-LATENT

This `DIAGNOSTIC_ONLY` experiment asks Step 1: whether one frozen carrier
survives into saved MP4 pixels. It does not retune or reuse the failed block29
attention-output carrier. The existing paired public 30D Step 2 runs only if
all four Step 1 cells pass.

## Single frozen carrier

Diffusers `WanPipeline` v0.35.2 is called once per attempt with
`output_type="latent"`. This returns the final denoised latent after the last
scheduler step and before the pipeline's latent de-standardization and VAE
decode. No transformer or attention hook is registered.

The required final tensor is `[B,C,T,H,W] = [1,16,13,40,64]`. Schedule point
`t` maps one-to-one to latent time `t`. For spatial integer coordinates
`x=0..63`, `y=0..39`, define `phi_x=cos(2*pi*x/64)` and
`phi_y=cos(2*pi*y/40)`. At time `t`, channels 0..7 receive
`schedule[t].x * phi_x`; channels 8..15 receive
`schedule[t].y * phi_y`. The complete tensor is centered by its one global
mean and divided by its one global RMS. The injected delta is that unit-RMS
tensor times `0.03 * RMS(final_latent)`. It is constructed on the final
latent's device in float32, cast to its exact dtype, added once, then passed
through the frozen Wan latent de-standardization and `vae.decode` path.

OFF_R1 and OFF_R2 receive no residual. A uses the existing frozen A schedule;
B uses the existing frozen B schedule. The accepted absolute relative-RMS
error is `0.00005`; zero-mean and unit-RMS construction tolerances are each
`1e-6`. Shape, dtype, device, schedule identity, pre/post tensor hashes, and
the single injection event are recorded.

## Matched experiment and decision

The prompts, seeds, Wan revision, scheduler, sampler, eight steps, geometry,
FPS, H264/yuv420p encoding, two content groups, same-initial-latent clone
policy, condition order, and exact eight-attempt budget are unchanged. A
started attempt consumes the budget. There is no retry, replacement, added
sample, carrier-strength scan, content selection, threshold change, or
result-dependent tuning.

Step 1 uses the unchanged saved-MP4 RGB metric: absolute effect at least
`2/255`, effect at least `3*N`, and OFF repeat noise `N <= 1/255`, with both
groups and both A/B cells required. Step 1 is `FEASIBLE`, `NOT_FEASIBLE`, or
`INSUFFICIENT_TO_DECIDE`. If it is not `FEASIBLE`, Step 2 is short-circuited.
All artifacts and route suggestions remain `DIAGNOSTIC_ONLY`.
