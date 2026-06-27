# Stage 4 Prompt Scaling Guide

Use this guide when expanding self-corruption prompts beyond Hard64.

## Goal

Stage 4 needs prompts that expose local, compositional visual structure so MMU
localization learns where corruption occurs rather than memorizing a narrow
dataset prior.

The first scalable target is not every available benchmark. Use staged pools:

1. Hard64 sanity set: current 64 prompts, 128 corrupted rows.
2. Hard256: 256 prompts sampled from the same distribution for LoRA stability.
3. Bench3-512: mixed prompts from DPG-Bench, GenAI-Bench, and DSG-1k.
4. Transfer holdout: GenEval and selected compositional prompts not used for
   LoRA.

## Source Pools

Useful local prompt files:

```text
configs/benchmarks/prompts/t2i_compbench_hard64.txt
configs/benchmarks/prompts/dpg_bench_1065.txt
configs/benchmarks/prompts/genai_bench_1600.txt
configs/benchmarks/prompts/dsg1k_1060.txt
configs/benchmarks/prompts/bench3_combined.txt
```

Selection rules:

- Prefer prompts with multiple objects, attributes, spatial relations, counts,
  or text rendering.
- Keep a mix of short literal prompts and long caption-style prompts.
- Avoid near-duplicates inside a training split.
- Keep benchmark transfer prompts out of LoRA training if they will be used as
  final evidence.
- Preserve source file and line index in generated prompt manifests.

## Suggested Scaling

Start with:

```text
Hard64 train/eval: 96/32 corrupted rows after split
Hard256: 256 prompts x 2 corruption types = 512 rows
Bench3-512: 512 prompts x 2 corruption types = 1024 rows
```

Use the same two stable corruption operators first:

```text
block_4x4_random_replace
local_shuffle_4x4
```

Add harder perturbations only after MMU/LoRA parse rate is stable.

## Operational Notes

- Build prompt manifests before generation so shards are reproducible.
- Use Slurm arrays or manual shards; keep one output directory per prompt shard.
- Merge manifests into one dataset only after verifying referenced image/token
  paths.
- Do not commit `outputs/`, generated datasets, adapters, checkpoints, or
  decoded images.

## Evaluation Discipline

Report three levels separately:

- self-corruption holdout localization;
- self-corruption ASCR reopen before/after;
- transfer to real prompt-following errors.

Do not present self-corruption metrics as real prompt-following repair until the
transfer holdout is run.
