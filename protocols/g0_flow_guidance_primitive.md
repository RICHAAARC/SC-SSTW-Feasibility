# G0 Flow-guidance primitive

This is a `DIAGNOSTIC_ONLY` method-feasibility experiment.  It asks one
question: can the frozen two-coordinate observer provide two independent,
directionally correct and quality-bounded control axes on the real Wan Flow
state?

The scientific object is fully specified by
`configs/g0_flow_guidance_primitive.json` and `SSTW_METHOD_AUTHORITY.md`.
All conditions fork the same latent and complete UniPC scheduler state after
index 5.  OFF conditions receive no update.  Active conditions receive the
same two gradients computed at the common base point, with only their signs
changed, and then continue normal scheduler indices 6 and 7.

G0 decodes final RGB for measurement but writes no MP4 and does not execute
AISB, affine calibration or Viterbi.  OOM, autograd disconnection, VAE scaling
or shape errors, scheduler-fork errors and serialization errors are
`INSTRUMENTATION_INSUFFICIENT`, never scientific no-go.

No result-dependent observer, strength, boundary, prompt, seed or threshold
change is permitted inside this experiment.  A valid failure ends SSTW-v2
Flow-guidance and permits, but does not automatically start, a separately
versioned structured-initial-noise experiment.
