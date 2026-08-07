# GPU learned observation L1

This Gate generates only dataset IDs 41001 through 41006 from the frozen
`configs/learned_observation_frontend.json`: four train and two validation.

The formal repository CLI must:

1. use the fixed Wan revision and block-29 `attn1` residual on every one of the
   16 cond/uncond calls;
2. encode H.264/yuv420p MP4 and decode exactly 49 frames;
3. hash and cross-check each actual injection record against its generation
   manifest;
4. obtain the 30D feature only from each single saved MP4;
5. fit the normalizer and 530-parameter front-end only on IDs 41001--41004;
6. freeze and hash the weights before creating or reading IDs 41005--41006;
7. bind the frozen weights to the actual four training MP4 digests;
8. evaluate IDs 41005--41006 independently through acquisition, frozen
   ambiguity SHA, public-only C calibration, and H validation;
9. persist per-video execution integrity plus exact CLI stdout/stderr before
   checksums and archive creation.

Both validation cases must satisfy every unchanged threshold to become an L2
candidate. While `auditor_decision` is pending, `gpu_l2_admission` remains
false; only the independent auditor may admit L2. This is not held-out, null,
owner/wrong-key, temporal-edit, fixed-FPR, generalization, or method evidence.
