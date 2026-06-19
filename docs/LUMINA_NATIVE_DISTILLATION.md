# Lumina-Native Semantic Evaluator Distillation

This is the current Stage-2 research target for ASCR.

The intended student is not an external RGB-feature localizer. The intended
student is the Lumina-DiMOO model path itself acting as a semantic
evaluator/MMU-like module:

```text
prompt + current image/image tokens
  -> Lumina-native evaluator
  -> SemanticEvaluation JSON
  -> GridSemanticReopeningSelector
  -> Lumina token reopening
```

Qwen3.7-plus remains the offline/API teacher and judge. It should not run inside
the compute-node ASCR loop.

Current server audit status: the Lumina-DiMOO MMU path is available through
`LuminaNativeEngine.answer_image()`, and it can return image-conditioned text.
However, the full evaluator still abstains because the raw output is natural
language rather than strict `SemanticEvaluation` JSON. The next gate is JSON
compliance, not image-understanding availability.

## What Is Official Stage 2 Now

Official Stage 2 means distilling Qwen3.7-style semantic evaluation into a
Lumina-native evaluator path. The target output is the existing ASCR
`SemanticEvaluation` contract:

```json
{
  "has_error": true,
  "summary": "short diagnosis",
  "regions": [
    {
      "cells": [{"label": "B2"}],
      "reason": "semantic mismatch",
      "confidence": 0.8,
      "error_type": "semantic",
      "action": "reopen"
    }
  ],
  "correction_instruction": "fix the selected region"
}
```

The selector remains fixed. It maps the JSON-selected grid cells into token
reopen masks. The generator remains Lumina-DiMOO.

## What The Old Grid Localizers Are

`grid-localizer-v0` and `grid-localizer-v1` are scaffold baselines. They helped
validate the data contracts, ASCR loop integration, before/after image manifest,
and Qwen3.7 image judge. They are not the formal distilled student and should
not be used as a main benchmark arm for Stage-2 claims.

Keep their outputs for debugging and historical comparison, but do not present
them as the final student model.

## Feasibility Audit

Before training, verify whether the current Lumina-DiMOO checkout and ASCR
wrapper expose an image-conditioned text generation hook.

Local/lightweight audit:

```bash
python -m ascr.cli.lumina_native_audit \
  --repo-path third_party/Lumina-DiMOO \
  --checkpoint-path models/lumina-dimoo \
  --scan-repo \
  --output outputs/stage2_lumina_native/audit/audit.json
```

Server/GPU audit:

```bash
source .venv-lumina/bin/activate

LUMINA_REPO=third_party/Lumina-DiMOO \
LUMINA_MODEL_PATH=models/lumina-dimoo \
bash scripts/training/run_lumina_native_audit.sh
```

Optional Slurm audit:

```bash
sbatch jobs/training/lumina_native_evaluator_audit.sbatch
```

The audit now checks both wrapper support and optional model/image smoke. It
should pass the hook check on the server. A smoke call that returns malformed
JSON is still treated as an abstention, not as a usable evaluator.

## JSON Compliance Probe

Before formal benchmark runs, probe whether Lumina can follow strict JSON
instructions on existing teacher grid images:

```bash
source .venv-lumina/bin/activate

DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/json_probe \
bash scripts/training/run_lumina_native_json_probe.sh
```

Optional Slurm probe:

```bash
sbatch jobs/training/lumina_native_json_probe.sbatch
```

The probe writes:

```text
outputs/stage2_lumina_native/json_probe/
  probe_rows.jsonl
  summary.json
```

If `parse_rate` is poor, do not run formal before/after benchmarks. Move to
Lumina-native SFT/LoRA smoke using Qwen teacher labels.

## Prepare Qwen Teacher Data For Lumina SFT

This step only prepares supervised examples. It does not launch LoRA/SFT until
the native evaluator hook and JSON-compliance gap are understood.

```bash
DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/sft_smoke \
LIMIT=10 \
bash scripts/training/prepare_lumina_native_sft.sh
```

Outputs:

```text
outputs/stage2_lumina_native/sft_smoke/
  sft_examples.jsonl
  manifest.json
```

Each SFT row contains the image path, compact evaluator prompt, and target
`SemanticEvaluation` JSON. The manifest is the starting point for the server
branch that implements LoRA/SFT smoke if prompt-only JSON compliance is poor.

## ASCR Integration Contract

The new evaluator backend is:

```bash
python -m ascr.cli.run_stage1 \
  --config configs/stage2/lumina/lumina_native_evaluator_smoke.yaml \
  --prompt "a red cube left of a blue sphere" \
  --evaluator lumina_native_evaluator
```

If the Lumina wrapper does not expose an answer method such as `answer_image`,
the evaluator returns `SemanticEvaluation.abstain(...)`. Abstention produces no
actionable reopen regions, so the loop does not perform unsafe semantic edits.

Once the hook exists, the backend expects one compact JSON object and parses it
through the same strict ASCR schema used by Qwen and other evaluators.

`run_stage1` shares the Lumina generator engine with the Lumina-native evaluator
when both are present, so one process does not load two copies of Lumina.

## Benchmark Policy

Formal Stage-2 benchmark arms are:

```text
before: Lumina direct generation
after:  Lumina-native evaluator + GridSemanticReopeningSelector + ASCR loop
```

Do not include the external grid localizer scaffold as a main benchmark arm.
Benchmark on:

- in-domain Hard64 holdout;
- out-of-domain Geneval smoke16 first, then larger Geneval if smoke passes.

Judge before/after quality on the login node with Qwen3.7-plus. Record winner
counts, mean score delta, changed-image count, fallback/abstention count, and
failed samples.

Only run this after JSON compliance is acceptable or an SFT smoke adapter is
loaded:

```bash
PROMPTS=configs/benchmarks/prompts/t2i_compbench_hard64.txt \
DOMAIN=in_domain_hard64_smoke16 \
LIMIT=16 \
OUTPUT_DIR=outputs/image_bench/lumina_native/in_domain_hard64_smoke16 \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_lumina_native_image_benchmark.sh

PROMPTS=configs/benchmarks/prompts/geneval_553.txt \
DOMAIN=geneval_smoke16 \
LIMIT=16 \
OUTPUT_DIR=outputs/image_bench/lumina_native/geneval_smoke16 \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_lumina_native_image_benchmark.sh
```
