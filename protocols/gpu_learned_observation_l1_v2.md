## Observation L1-v2 bounded development protocol

This protocol preserves the original L1 `Contradicted` result. IDs 41001–41006 are development-only. IDs 41007–41008 are fresh and forbidden to the CPU development runner.

Exactly two candidates are admitted. A1 applies per-video, per-feature median/MAD normalization with scale `1.4826`, absolute MAD floor `1e-6`, and clipping to `[-6, 6]`. A2 applies A1 and then a length-preserving high-pass: the interior is `x[t]-(x[t-1]+x[t+1])/2`; the left boundary is `x[0]-x[1]`; the right boundary is `x[N]-x[N-1]`.

Each candidate uses one affine linear readout with intercept and ridge `1e-6`. It is fitted only on 41001–41004 against the public 13-point target. The eight fixed six-point windows start at indices 0–7; start 0 is correct and starts 1–7 are wrong. No deletion variants are permitted.

Thresholds are computed before 41005–41006 evaluation from four leave-one-video-out fits over 41001–41004. Upper-bound metrics use the maximum correct-window LOO value (residual, condition, held-out MSE); lower-bound metrics use the minimum (global second singular value, affine second singular value). A window is finally accepted only when all five bounds pass.

The development gate requires the correct window of both 41005 and 41006 to pass and every wrong window to fail. A1 has priority over A2. If neither passes, the terminal decision is `STOP_NOT_GPU_READY`; there is no third candidate, GPU CLI, or Colab notebook. L2 remains closed.

Only if the development gate passes may a later commit preregister fresh carrier-on/off pairs and freeze the GPU gate before reading any fresh observation.
