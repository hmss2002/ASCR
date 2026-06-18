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

The audit passes as an environment check even when native evaluation is not yet
supported. Use `--require-supported` only when the server AI is explicitly
validating that the native evaluator hook has been implemented.

## Prepare Qwen Teacher Data For Lumina SFT

This step only prepares supervised examples. It does not launch LoRA/SFT until
the native evaluator hook is confirmed.

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
`SemanticEvaluation` JSON. The manifest records that actual LoRA/SFT is blocked
until the Lumina-native MMU/text hook is available.

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
