# RC0 Fresh12 Method-Stability Base Extension

Status: `DIAGNOSTIC_ONLY` / `METHOD_ONLY`.

This phase adds exactly eight carrier-free Wan base videos to the four already frozen fresh bases. It does not regenerate the prior four, inject a carrier, run a detector, execute Phase C, establish a Gate, or support a paper/statistical claim.

## Frozen identities

The eight labels and categories are one-to-one and ordered: `rigid_blocks/rigid`, `elastic_membrane/nonrigid`, `laminar_plumes/fluid`, `granular_avalanche/particles`, `matte_balloon/low_texture`, `engraved_drum/high_texture`, `pendulum_bars/periodic`, and `soap_bubbles/aperiodic`. Prompts, prompt SHA-256 values, and seeds 54011--54018 are fixed in the config before any new GPU result is observed. They were selected from textual content constraints only, avoid text, cuts, and camera motion, and are distinct from orbital glass, articulated paper, liquid mosaic, clockwork escapement, steam fins, and magnetic filings.

Each group has exactly one BASE attempt. The model revision, default Wan scheduler/sampler, 512x320 geometry, 49 frames, 8 fps, 8 inference steps, guidance 5.0, negative prompt, BF16 dtype, official default pipeline decode, and H264/yuv420p quality-5 encoding are copied unchanged from the fresh4 base protocol. Carrier, hooks, manual latent decode, detection, replacement, retry, and sample expansion are forbidden. An attempt start consumes its sole budget position; a failure stops the run.

## Frozen later CPU phase (not executed here)

For each new base, a future CPU-only invocation will derive four clean conditions (`OFF_R1`, `OFF_R2`, `A`, `B`) and four real MP4 time transforms per condition, yielding 32 clean plus 128 transformed products, exactly 160. It directly references the already-passed 6/255 pixel-chroma carrier, A/B schedules, common encoder, target-only observation, AISB K=8, candidate-wise K2=1 calibration, dynamic-time-sync parameters, and the four fixed transforms. None may be redefined here.

Corrected evaluation semantics are immutable: each clean condition is scored only under identity; each transformed MP4 is decoded and scored only against its own transform truth. Success requires all 16 clean OFF controls to reject, all 16 clean A/B targets to accept with own/start0 identity, all 64 transformed OFF controls to reject, and all 64 transformed A/B targets to pass. There is no averaging or majority vote. This future CPU phase is one run plus an independent replay and is not part of the GPU notebook.

All outputs retain `formal_result=false` and `stage_progression_allowed=false`.
