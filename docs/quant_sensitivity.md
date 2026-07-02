# Quantization-sensitivity harness (Eval Note #1)

Nobody deploys a 753B verifier at FP16 on commodity hardware — aggressive
quantization (Q4, Q2, even ~1.6-bit IQ1-class) is how self-hosters run big
models. But **does the verifier's abstention calibration survive quantization?**
A verifier that still gets the easy cases right while its assert/abstain
judgement drifts is a live hazard for a benefits tool. This harness measures
that drift.

## What it runs

`tribune quant-eval` runs the **verifier** across a ladder of quantization
levels over a frozen seed set of 50 determination cases (weighted toward
Medicaid — state-dependent eligibility complexity — and SNAP), and reports per
rung, against both gold labels and the full-precision reference run:

- Cohen's kappa (and the harness's Krippendorff's alpha in the standard report),
- abstention rate, over-refusal rate, and abstain-decision agreement vs. the
  reference (calibration drift),
- ECE and Brier score on the calibrated confidence behind each assert/abstain
  decision (logged per record as `EvalRecord.confidence`),
- the honesty suite (false-confidence rate, abstention precision/recall,
  abstention-aware utility),
- all Phase-1 cost metrics (cost/task, turns, tokens).

The proposer configuration is never touched: each rung swaps only the verifier
backend, preserving proposer/verifier separation.

Abstention-rate movement is reported as **drift**, never failure — abstaining
is a success outcome. What gets flagged is the verifier behaving *differently*
than at full precision, and any rise in confidently-wrong assertions.

## The frozen seed set

The set is deterministic (generator seed `20260701`) and frozen by
`tribune/eval/quant_sensitivity/data/seed_manifest.json`, which records the case
ids and a content hash over the stable case payloads. Every eval note embeds the
hash; **runs are only comparable when hashes match**. If the synthetic generator
or rule corpus changes, `quant-eval` refuses to run against the stale manifest —
re-freeze deliberately with `make freeze-seed` (and treat prior notes as a
separate series).

## Running it

| command | what it does |
| --- | --- |
| `make quant-smoke` | CI smoke: 2 rungs × ~5 cases, deterministic mock backend, free |
| `make quant-full` | full ladder (fp16/q8/q4/q2/iq1) × 50 cases, mock backend, free |
| `make quant-real CONFIG=ladder.json` | full ladder against real served endpoints |

The mock backend simulates quantization noise as a seeded, deterministic flip of
the verifier's model-side review (structural checks stay production-strict), so
the whole pipeline — including CI — runs offline at zero cost.

### Real-endpoint ladder config

Both llama.cpp's server and vLLM speak the OpenAI chat-completions API, so a
rung is just a label + endpoint. Example `ladder.json`:

```json
{
  "rungs": [
    {"label": "fp16", "quant_format": "fp16", "provider_kind": "openai_compat",
     "backend": "vllm", "model": "zai-org/GLM-5.2", "base_url": "http://gpu-a:8000/v1",
     "reference": true},
    {"label": "q4", "quant_format": "gguf-q4_k_m", "provider_kind": "openai_compat",
     "backend": "llama.cpp", "model": "glm-5.2-q4_k_m", "base_url": "http://gpu-b:8080/v1"},
    {"label": "q2", "quant_format": "gguf-q2_k", "provider_kind": "openai_compat",
     "backend": "llama.cpp", "model": "glm-5.2-q2_k", "base_url": "http://gpu-b:8081/v1"}
  ]
}
```

Start one server per rung, e.g.:

```bash
# llama.cpp (GGUF quants)
llama-server -m glm-5.2-Q4_K_M.gguf --port 8080
# vLLM (fp16/fp8/nvfp4/awq/gptq)
vllm serve zai-org/GLM-5.2 --quantization fp8 --port 8000
```

### Hardware notes

- The mock ladder needs nothing — CPU, no downloads, no keys.
- Real small-model rungs (e.g. a 7B at Q4) run on a laptop CPU via llama.cpp.
- Real large-model rungs (e.g. GLM-5.2-class at Q2) need a machine with enough
  RAM/VRAM to hold the quantized weights; budget roughly
  `params × bits / 8` bytes plus context. Run rungs sequentially on one box by
  restarting the server per rung if memory is tight — the harness only needs one
  rung alive at a time when you point all rungs at the same port between runs.

## Output

A markdown eval note at `docs/eval_notes/eval_note_1_quant_sensitivity.md` with
all metric tables, the embedded seed-set hash, and an auto-generated summary
that names the rungs where calibration does/doesn't survive (thresholds are
documented in the note).
