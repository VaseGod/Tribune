# TRIBUNE Eval Note #1: Does abstention calibration survive quantization?

*Generated 2026-07-02 by `tribune quant-eval`.*

## Frozen seed set

- cases: **50**, weighted toward Medicaid and SNAP (weights: {'medicaid': 15, 'snap': 15, 'unemployment': 8, 'housing': 6, 'appeals': 6})
- content hash: `6ea565bbd3f2a0cb0f8b811ebeb3e0ccc39db3777f279c45da21df772f0dc722`
- generator seed: `20260701`

Runs are only comparable across time when this hash matches.

## Summary

Reference rung: **fp16**. Thresholds: kappa drop < 0.1, |ECE drift| < 0.05, abstention-rate shift < 0.1.

Calibration **does not survive** at: iq1. At these levels the verifier's assert/abstain behavior has drifted off its full-precision calibration; do not deploy the verifier at these quantization levels without recalibrating the abstention threshold.

Calibration survives (within thresholds) at: fp16, q8, q4, q2.

Note: a *higher* abstention rate is not a failure by itself — abstention is a success outcome. The hazard measured here is *drift*: the same case set getting materially different assert/abstain decisions after quantization, and any rise in confidently-wrong assertions.

## Agreement & calibration vs. gold labels

| rung | quant | n | kappa (gold) | FCR | abstention | over-refusal | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| fp16 | fp16 | 50 | 1.000 | 0.000 | 0.400 | 0.000 | 0.101 | 0.011 |
| q8 | gguf-q8_0 | 50 | 1.000 | 0.000 | 0.400 | 0.000 | 0.101 | 0.011 |
| q4 | gguf-q4_k_m | 50 | 1.000 | 0.000 | 0.480 | 0.080 | 0.100 | 0.011 |
| q2 | gguf-q2_k | 50 | 1.000 | 0.000 | 0.480 | 0.080 | 0.102 | 0.011 |
| iq1 | gguf-iq1_s | 50 | 1.000 | 0.000 | 0.740 | 0.340 | 0.100 | 0.010 |

## Drift vs. the full-precision reference run

| rung | kappa (vs ref) | label agreement | abstain-decision agreement | ECE drift |
|---|---|---|---|---|
| fp16 | 1.000 | 1.000 | 1.000 | 0.000 |
| q8 | 1.000 | 1.000 | 1.000 | 0.000 |
| q4 | 1.000 | 1.000 | 0.920 | -0.001 |
| q2 | 1.000 | 1.000 | 0.920 | 0.000 |
| iq1 | 1.000 | 1.000 | 0.660 | -0.001 |

## Honesty suite

| rung | abstention recall (of ambig.) | abstention precision | utility | verifier agreement |
|---|---|---|---|---|
| fp16 | 1.000 | 1.000 | 1.000 | 1.000 |
| q8 | 1.000 | 1.000 | 1.000 | 1.000 |
| q4 | 1.000 | 0.833 | 0.960 | 1.000 |
| q2 | 1.000 | 0.833 | 0.960 | 1.000 |
| iq1 | 1.000 | 0.541 | 0.830 | 1.000 |

## Cost (Phase-1 metrics)

| rung | mean $/task | mean $/success | turns | tokens in | tokens out |
|---|---|---|---|---|---|
| fp16 | 0.00000 | 0.00000 | 142 | 34138 | 5502 |
| q8 | 0.00000 | 0.00000 | 142 | 34138 | 5502 |
| q4 | 0.00000 | 0.00000 | 142 | 34138 | 5550 |
| q2 | 0.00000 | 0.00000 | 142 | 34138 | 5614 |
| iq1 | 0.00000 | 0.00000 | 142 | 34138 | 5758 |

## Hardware notes

Full-ladder runs against real weights need a llama.cpp or vLLM server per rung (e.g. `llama-server -m model-Q4_K_M.gguf --port 8081`). CPU-only boxes can run small GGUF models; Q2/IQ1 rungs of large models need significant RAM/VRAM. See docs/quant_sensitivity.md.
