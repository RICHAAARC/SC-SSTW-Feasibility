# S1 local single-pair dictionary screen

This is a `DIAGNOSTIC_ONLY` S0 construction-selection tool. It does not change the SSTW method identity, does not freeze a replacement B1/B2 before the screen result, and does not enter S2.

The unique method authority for this delivery is commit `b53ed1c23a91f0798e1955e55f53daea2298ed75`, tree `91bb9c99b1d4a49faf46a5eb92bd41573719e7f5`, and raw `SSTW_METHOD_AUTHORITY.md` SHA-256 `4013eb61c8b8d3165729e4f89ecc0936e05a734511227a44a818ea0ae850688f`. It fixes the current node as `S1_LOCAL_PAIR_DICTIONARY_EXACT10`.

## Registered prior result

Run `0d3223669982a6c3`, archive SHA-256 `07a4c137382ce67aaf16d1775ea244287568a7c2d7d3fedcc5fa104a75a31cdb`, source `e8923e5de8b752ed3211af98c4120340acdea100`, ended in valid `S1_NO_GO_THIS_CONSTRUCTION` with exact20 and structural checks passing.

The real block-14 native sparse-SDPA interface succeeded. The relation Jacobian diagonals were positive, cross leakage was approximately 0.004–0.006, axis cosine was low, and a real two-axis local-relation odd response existed. It was too weak or unstable: relation odd/ULP was only about 1.20–5.89, block response about 0.0045–0.032, velocity response about 0.0666–0.491; common-mode and temporal-TV checks failed, and guidance velocity global relative RMS was about 0.024, above 0.01.

Lambda-only repair is excluded. Reaching 8× velocity separation would require lambda about 16–120, while the quality budget implies lambda no more than about 0.406–0.413 and even/odd about no more than 0.15–0.16. No strength scan is allowed. The narrow failure hypothesis is the original single Patch-pair basis choice. S2 remains HOLD.

## Frozen dictionary

Block 14, true normalized-and-RoPE Q/K relation logits, lambda 1, scheduler index 4/timestep 749, prompt, seed, prefix, 13 temporal queries, and cond/uncond branches are unchanged.

Each basis is still exactly one antisymmetric Patch-token pair with coefficients `(+1/sqrt(2),-1/sqrt(2))`. Horizontal candidates use `(h,w-r),(h,w+r)` and vertical candidates use `(h-r,w),(h+r)` for radii 1 through 8. All 13 queries share the same radius and every pair is in bounds. Multi-pair stencils and non-relation carriers are excluded.

The exact GPU budget is eight prefix calls plus one step-4 OFF capture for each CFG branch: ten transformer calls total. The sixteen candidate `+/-lambda` responses are analytic counterfactuals from those same OFF probabilities; they do not run additional transformer calls and do not modify block or velocity.

## Formula and selection

For baseline pair probabilities `p_a,p_b`, injecting deltas `(+d,-d)` gives normalization

`D = 1 + p_a(exp(d)-1) + p_b(exp(-d)-1)`.

Any observed local-pair probability is divided by `D`, with the corresponding exponential factor applied only when its token is one of the injected pair. The relation observation is the observed pair difference. Plus/minus, odd, even, temporal-TV and BF16-ULP separation use the existing S1 relation-layer definitions and thresholds.

An axis/radius is eligible only when cond and uncond both pass odd/ULP, even/odd and temporal-TV. Axis ranking is uniquely: maximize the worst-branch separation, minimize the worst-branch even ratio, minimize the worst-branch TV, then minimize radius.

Every eligible horizontal/vertical combination must also pass both branches' axis cosine, gain ratio, positive Jacobian diagonal and cross leakage. Combination ranking is uniquely: maximize the worst axis/branch separation, minimize the worst axis/branch even ratio, minimize the worst axis/branch TV, then horizontal radius, then vertical radius.

No eligible combination yields `PAIR_DICTIONARY_NO_GO`; exact20 is not run and lambda/Flow are not scanned. A unique combination yields `PAIR_DICTIONARY_READY`, which only permits freezing that pair and running one new exact20. `PAIR_DICTIONARY_NO_GO` permits one finite propagation-sensitivity construction screen over a predeclared multi-pair zero-sum, CFG-aware, single/three-Flow-step family. If that finite screen still has no solution, the untrained Wan Patch-relation carrier route stops. S2 remains HOLD throughout this dictionary screen.
