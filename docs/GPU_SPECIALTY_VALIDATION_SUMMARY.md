# GPU Specialty Validation Summary

## Environment

- Branch: gpu-specialty-models
- Framework: Retail Forecast Intelligence & Variance Decomposition Framework
- GPU backend: PyTorch CUDA
- Device: NVIDIA GeForce RTX 4090 Laptop GPU

## GPU experiment modules

1. GPU MLP residual correction
2. GPU residual shrinkage gate
3. GPU LSTM residual correction
4. GPU attention residual correction
5. GPU attention quantile residual model
6. GPU quantile asymmetric buffer policy selector

## Key validated results

### Residual correction

The validation-selected conservative GPU residual policy improved test WAPE from 0.0719307 to 0.0711335 and reduced high-bias slice absolute bias from 37.8167 to 36.1642.

### LSTM and attention sequence residual correction

The best sequence policy was `gpu_lstm_residual__conservative_wape_gate` with alpha 0.20.

- Baseline test WAPE: 0.0719307
- LSTM conservative test WAPE: 0.0717125
- Baseline test absolute bias: 5.1701
- LSTM conservative test absolute bias: 3.8347

Attention residual correction was strongest as a high-bias diagnostic model, improving high-bias slice reliability under conservative gating.

### Quantile asymmetric buffer

The global validation-selected policies favored fixed safety buffers rather than GPU quantile buffers.

However, the GPU q50 quantile policy showed strong high-bias diagnostic improvement:

- High-bias absolute-bias reduction versus baseline: 92.48%
- High-bias lost-sales proxy reduction versus baseline: 44.76%
- High-bias coverage-F1 change versus baseline: +0.04%

## Claim boundary

These are optional offline GPU experiments using public/sample-compatible retail panels.

Safe claims:
- GPU neural residual correction improved selected point-forecast and bias metrics under validation-selected gates.
- GPU quantile modeling supported asymmetric planning diagnostics.
- Quantile models should be interpreted through planning tradeoffs, not only point-forecast WAPE.

Unsafe claims:
- Do not claim the GPU quantile model was the global winner.
- Do not claim production deployment.
- Do not imply Amazon internal data or Amazon internal systems were used.

## Guarded paper-method upgrade

Added a paper-method-inspired guarded correction experiment combining:
- PHT-inspired demand-state features
- expected-path fallback selection
- GPU residual bias correction
- conservative alpha gating

The aggressive validation-WAPE policy over-corrected test forecasts, but the conservative/sign-safe alpha policy selected alpha 0.1 and improved test WAPE from 0.0719307 to 0.0717160 while reducing test absolute bias from 5.1701 to 0.4568 versus the rolling baseline.

Claim boundary:
- Use conservative/sign-safe policy results only.
- Do not claim aggressive alpha results.
- Describe this as a paper-method-inspired adaptation, not a direct reproduction of the original PHT or degradation papers.
