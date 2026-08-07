# GPU internal-output challenger v1

## Historical boundary

This is an independent protocol. It does not modify or reinterpret direct-carrier
v2 at commit `9c1be010353367f0ee1a95afc372999dafbaf812`. The preserved v2 Drive
archive has SHA256
`5b4e8bf820dc7e12f0eed2520e88098a3d3a0c1b7b3e01ec178337e7f4d08b26`.
That protocol and its frozen direct final model-output residual plus current
structured saved-video readout are `NO_GO`; the overall method is still
`NOT_DETERMINED`.

## Unique challenger

The sole candidate is `WanTransformer3DModel.blocks[29].attn1` output under
Diffusers 0.35.2. The hook replaces the tensor after attention `to_out` and
dropout but before `gate_msa`, the block residual, cross-attention, FFN, final
norm/projection, CFG, scheduler, VAE, and MP4. No other block may be searched.
Processor changes, QK bias, value/QKV changes, scheduler mutation, guidance
carriers, and final transformer-output carriers are forbidden.

The model must have 30 `WanTransformerBlock` blocks, patch size `[1,2,2]`, and
inner dimension 1536. The patch embedding must be `[1,1536,13,20,32]`; the
attention output must be `[1,8320,1536]` BF16. Conv3D patch output is flattened
in `(T,H,W)` C-order with width fastest:

`index = ((t * 20) + h) * 32 + w`.

Every real transformer call must verify full value equality between
`patch_output.flatten(2).transpose(1,2)` and the actual block-0 input. Only the
boolean, shapes, and a fixed-sample digest are saved; the tensor is not exposed
as final evidence.

## Frozen carrier

For each of the same 13 v2 points `q_t=(q_t1,q_t2)`, all 1536 channels receive

`Bx(t,h,w,d) = sqrt(2) cos(2 pi (w+0.5) / 32)`

`By(t,h,w,d) = sqrt(2) cos(2 pi (h+0.5) / 20)`.

The candidate is `q_t1 Bx + q_t2 By`. The effective BF16 delta is measured only
after `modified = output + candidate` and `effective_delta = modified - output`.
Its relative RMS must be `0.03 +/- 0.00005`. The hook returns a distinct tensor,
does not mutate the original tensor version, and is removed in `finally`.

The exact eight timesteps are `999,954,899,832,749,642,499,299`. Each step calls
`cond` then `uncond`, producing the frozen 16-call sequence in the config. Branch
names must be read from the Pipeline's actual `transformer.cache_context(name)`;
they may not be inferred from call index. The outer transformer wrapper may
observe calls but may not alter output. The target attention processor object and
every `to_q`, `to_k`, `to_v`, and `to_out` parameter object/version must remain
unchanged across both generations.

## Evidence and decision

Internal effective delta, paired final transformer output, paired latent, paired
RGB-before-MP4, and paired saved MP4 are diagnostic only. They cannot satisfy
the carrier Gate.

The only eligible relation observation is the key-independent continuous
`13 x 2` readout from one saved watermarked MP4. Both public AISB residual
`<= 0.25` and centered two-dimensional condition number `<= 10` are required,
in addition to all execution-integrity conditions. A pass establishes only a
public saved-video relation-propagation bridge. It does not establish public-only
calibration, held-out performance, owner/wrong-key separation, edit robustness,
or method PASS.

Failure makes this single challenger protocol `NO_GO`. It does not authorize a
different block, basis, threshold, candidate search, or a method-level NO_GO.

GPU execution is allowed only after independent static implementation
`GATE_PASS`, from a clean exact commit through the thin Colab Notebook and the
repository CLI. Each run must preserve the complete Drive evidence package.
