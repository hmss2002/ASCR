# Stage 2: VLM-distilled semantic self-understanding for Lumina-DiMOO ASCR

## What Stage 1 already completed

Stage 1 is already a runnable ASCR prototype:

- Lumina-DiMOO can generate, decode, and natively reopen masked token regions.
- The main semantic control path is still **4x4 coarse localization -> project to 64x64 token grid -> dilation=1 -> native reopen**.
- The repo already has the right control interfaces: `ASCRLoop`, `SemanticEvaluation`, `GridSemanticReopeningSelector`, JSON traces, and Slurm-oriented execution patterns.
- Stage 1 traces already preserve the key supervision ingredients that Stage 2 needs: prompt, decoded/grid image, semantic diagnosis, selected cells, projected mask, and correction prompt.

## Why Stage 2 should not be framed as only a generic learned selector

If Stage 2 is written only as "train a better selector," it underspecifies the real research question.

The bottleneck is not merely mask prediction. The bottleneck is that UMM / Lumina-DiMOO itself has weak semantic self-understanding for compositional prompt-image mismatch. A stronger external VLM teacher can:

- diagnose what is semantically wrong,
- localize the wrong region,
- explain why it is wrong,
- and provide a correction instruction.

That means Stage 2 is better framed as **distilling semantic understanding into a reopening policy**, not just fitting a classifier on token masks.

## Core hypothesis

1. A strong external multimodal teacher can detect prompt-image semantic errors on Lumina intermediate outputs more reliably than Lumina's own native confidence dynamics.
2. Distilling that teacher's semantic diagnosis into a learned reopening selector / self-check module will improve not only selector diagnostics, but also the **final generated image quality** after reopening and continuation.
3. The safest first step is to distill into the **semantic reopening selector** before changing or fine-tuning Lumina-DiMOO itself.

## Teacher-only ASCR vs distilled-selector ASCR

| Variant | What makes decisions | Runtime cost | Scientific role |
|---|---|---:|---|
| **Teacher-only ASCR** | External VLM teacher directly diagnoses each image and chooses 4x4 cells | Highest API cost | Upper-bound control arm for Stage 2 |
| **Replay selector** | Replays teacher-labeled masks from a built dataset | Low | Minimum runnable Stage 2 training closure |
| **Distilled selector** | Learned student predicts reopen scores / masks without calling the external teacher at inference time | Medium once trained | Main Stage 2 target |

## Stage 2 default data flow

```text
prompt
-> Lumina-DiMOO generation
-> decoded image
-> 4x4 grid overlay
-> external VLM teacher diagnosis
-> selected 4x4 cells
-> projected 64x64 token mask with dilation
-> Lumina native reopening
-> before/after image
-> judge score / revision gain
-> training record
```

This keeps the Stage 2 teacher-only path aligned with the existing repo architecture and avoids prematurely rewriting the ASCR loop.

## Experimental arms

1. **Lumina baseline, no ASCR**
2. **Stage 1 Lumina + current Qwen3.5-9B local coarse selector**
3. **Stage 2 teacher-only Lumina + bailian/qwen3.7-plus via oFox**
4. **Stage 2 distilled selector** via the current lightweight learned coarse selector baseline

## Evaluation metrics

- clean accuracy
- pairwise regression rate
- correction success rate
- teacher/student agreement
- selected cell count
- API call count and cost proxy
- failure categories

The main Stage 2 claim must be made on **final images**, not on teacher or selector diagnostics alone.

## First implementation step

Stage 2 should **not** start by fine-tuning Lumina-DiMOO itself.

The first implementation step is:

1. make the external teacher path runnable,
2. collect traces at scale,
3. build a Stage 2 dataset,
4. train or materialize a semantic reopening selector baseline,
5. compare final-image outcomes against Stage 1 and the no-ASCR baseline.

This is the most architecture-consistent and lowest-risk path in the current repo.

## Current implementation status

The repo now contains all three layers:

1. **teacher-only selector** via `ascr/evaluators/ofox_vlm.py`
2. **replay selector baselines** via `teacher_replay` and `dataset_replay`
3. **lightweight learned coarse selector baseline** that trains from Stage 2 teacher traces and can be loaded as a runtime evaluator
