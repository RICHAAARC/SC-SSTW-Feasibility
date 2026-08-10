# S1 finite propagation-sensitivity multi-pair screen

This `METHOD_ONLY / DIAGNOSTIC_ONLY` construction screen is bound to authority commit `0f85bd70de6f042c68560ed348722fc4fbd112e9`, tree `d791cff7b5cbebb6ec690f16196bde9ee6d50bd1`, raw authority SHA-256 `85c32f49897a6b7c6248d5fd6fe4c7a307b4b841dc7388551d095854b6189292`.

Run `79f92e73ef3a6223` (ZIP SHA-256 `fbedc7cccdb31889621dfc6e228888062a1d22978601c273d753674777bab229`) was a valid `PAIR_DICTIONARY_NO_GO`: all 16 radius-1..8 single-pair candidates were ineligible because cond/uncond temporal variation was approximately 0.75--1.30 rather than below 0.10. It forbids another radius, single-pair, or lambda scan. The remaining finite question is whether sparse two-pair zero-sum bases plus finite Flow support and one analytic CFG weighting can improve true relation-to-block-to-velocity propagation.

## Frozen bases

For a query row, an atomic pair is

`P_r=(e_(q-r)-e_(q+r))/sqrt(2)` horizontally, or with offsets `32r` vertically.

Horizontal atoms are exactly radii 2 and 5; vertical atoms exactly 2 and 8. The only two-pair directions are

`B_coherent=(P_a+P_b)/sqrt(2)` and `B_contrast=(P_a-P_b)/sqrt(2)`.

Their four token coefficients are respectively `(1/2,-1/2,1/2,-1/2)` and `(1/2,-1/2,-1/2,1/2)`: sum zero and L2 norm one. Blocks are exactly 7, 14, 22. Lambda total energy is one. Flow supports are step 4 with weight one, or steps 3/4/5 with weights `(1,1,1)/sqrt(3)`.

## Stage 1 and call budget

Stage 1 evaluates 3 blocks times 2 horizontal signs times 2 vertical signs = 12 constructions. Every construction receives real step-4 plus/minus probes for both axes and both CFG branches using the fixed equal L2-normalized fit weight `(1,1)/sqrt(2)`, and records relation, block, and velocity. It is a finite construction selection, not a claim that every candidate ran a complete S1.

The shared OFF prefix costs eight calls. Two step-4 OFF repeats on two branches cost four. Each construction costs `2 axes * 2 signs * 2 branches = 8` active calls, so Stage 1 is `8+4+12*8=108` calls. A construction is rankable only when all numeric cells are finite and defined, every layer/axis/branch odd RMS is positive, and relation diagonals have the correct orientation. Ranking is worst velocity separation descending, worst velocity even ratio ascending, worst velocity TV ascending, worst guidance velocity global relative RMS ascending, then block and sign order.

## Analytic CFG solution

For the selected construction and one support, concatenate the two-axis relation odd vectors for cond and uncond. With guidance coefficients `(5,-4)`, these form columns of `X`; the corresponding even/common-mode vectors form `Y`. Let `S=X^T X` and `N=Y^T Y`. Spectrally compute the Moore--Penrose inverse square root `N^(+1/2)` and the maximum eigenvector of `N^(+1/2) S N^(+1/2)`. Map it back, normalize its L2 norm to one, and flip only its global sign so both aggregate guided own-axis odd orientations are positive.

The solution has no ridge, grid, or learned parameter. Rank below two, a repeated maximum eigenvalue, non-finite values, zero branch weight, or either axis orientation flip invalidates that support. One weight is used for both axes and every plus/minus probe of that support.

## Stage 2 and total calls

The single-step fit reuses Stage 1 equal-weight responses, but those outputs cannot be reused as final evidence: applying its derived branch weight costs eight new calls. The three-step support advances real changed latents/QK at steps 3, 4, 5. Shared OFF continuation and terminal repeats cost six calls; its equal L2-normalized weight-fit trajectories cost 24 calls; its independently weighted application trajectories cost 24 calls. The complete non-degenerate path is exactly 170 transformer calls, retry zero. Fail-closed early stops do not spend an invalid weighted application: the only possible terminal counts are 108 (no usable Stage1 construction), or 138/146/162/170 depending on which analytic support weights are valid.

UniPC is stateful. Immediately before the real step-3 forward, the runner retains both `latent_at_step3` and a complete deep snapshot of the scheduler including its step index, order, lower-order count, model-output history, timestep history, and any last sample. Immediately after the real step-3 scheduler update it similarly retains `latent_at_step4` and a complete step-4 scheduler snapshot. Every three-step fit/apply condition starts from its own deep fork of the unchanged step-3 snapshot and its own clone of the same step-3 latent, then performs its real step-3 and step-4 scheduler updates. The OFF continuation alone starts from an independent fork of the step-4 snapshot and the step-4 latent. Single-step probes never invoke `scheduler.step`. No condition may share or mutate another condition's scheduler, and neither retained snapshot may change while its forks run. Reconstructing an empty scheduler from config is forbidden because it loses UniPC history.

Both supports are evaluated with the existing S1 relation/block/velocity odd, common/even, temporal-TV, axis cosine/gain, Jacobian/cross, and global-quality thresholds. Eligible supports use the frozen worst-metric ranking, with single-step winning the final exact tie. `PROPAGATION_CONSTRUCTION_READY` only freezes one construction for one later exact20. `PROPAGATION_CONSTRUCTION_NO_GO` yields `UNTRAINED_WAN_PATCH_RELATION_CARRIER_NOT_FEASIBLE`. Neither status enters S2.
