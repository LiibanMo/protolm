# ProtoLM tests

Run from the repository root:

```sh
uv run pytest -q -rxX
```

`conftest.py` enables local imports without modifying package configuration,
seeds and restores the CPU RNG for each test, and uses one CPU thread for tiny
tensor workloads. Tests require no downloads or accelerator hardware.

Coverage includes categorical entropy, exact integer hashing, functional
attention output/gradient equivalence with PyTorch, multi-head concatenation,
RoPE rotations and relative positions, padding, transformer causality and
residuals, entropy-model masking and gradients, state-dict serialization,
tiny next-byte overfitting, and device selection.

The loss and optimizer in model tests are test-side harnesses. Production
training, entropy patching, and BLT are not implemented and are not claimed as
covered. Dataset downloading is excluded to keep tests offline.

## Masking contracts

Explicit byte masks must exactly match the byte-ID tensor's `(B,T)` shape.
Regression tests reject masks that would broadcast across batch or sequence
dimensions, as well as masks with incorrect rank or sequence length. Correctly
shaped and automatically derived masks are covered by the model contract tests.

Fully masked attention rows are excluded from the valid low-level attention
contract used here. Rejection of empty examples is tested in both padding and
the entropy model. Padded query logits need not be zero; only valid predictions
contribute to the test loss. Current coverage assumes float32/float64 CPU
execution, not low-precision or accelerator correctness.
