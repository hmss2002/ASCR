# Server AI Task: Lumina LoRA JSON v3 With Expanded Teacher Data

## Current State

You are continuing ASCR Stage 2 on the university server.

Latest local Codex conclusion:

- `feat/lumina-lora-json-v2-server` was reviewed and merged into `main`.
- LoRA v2 training ran successfully on a 45GB GPU.
- Loss converged from roughly `5.0` to `0.085`.
- JSON probe had `call_error_count=0`, `malformed_count=3`, `parse_rate=0.0`.
- Raw previews showed severely malformed JSON-like text: nested quotes, repeated keys, missing delimiters.
- Therefore the problem is not API failure. The current student learned JSON-like tokens but not valid `SemanticEvaluation JSON`.

Do not run formal before/after image benchmark until Lumina-native JSON probe produces parseable `SemanticEvaluation` outputs.

## What Changed In Main

Local Codex has now changed the mainline code before this handoff:

- SFT targets use canonical evaluator JSON only:
  - `has_error`
  - `summary`
  - `regions`
  - `correction_instruction`
- Runtime/debug fields are removed from training targets:
  - `raw`
  - `parser_error`
  - `should_abstain`
- `answer_image()` now aligns mask length before constructing the answer mask.
- `train_lumina_lora_smoke` now supports:
  - `--answer-mask-mode random|all`
  - `--ignore-pad-labels / --no-ignore-pad-labels`
- The recommended v3 training mode is:
  - `--answer-mask-mode all`
  - `--ignore-pad-labels`

The intent is to make training closer to Lumina masked answer generation at inference time.

## Scientific Direction

Formal Stage 2 remains:

```text
prompt -> Lumina generate image
       -> Lumina-native evaluator emits SemanticEvaluation JSON
       -> GridSemanticReopeningSelector maps cells to reopen mask
       -> Lumina token reopen loop
       -> final image
```

The target being distilled is Lumina's native semantic evaluator ability, not an external shallow localizer.

Qwen3.7-plus is the teacher and judge. It is not part of the final ASCR compute-node loop.

## Main Hypothesis To Test

The v2 LoRA failure was probably caused by both:

1. too little data: only 16 SFT examples;
2. noisy or mismatched training format: runtime fields in targets and random answer masking.

The next test should change both:

- use clean canonical targets;
- train with all answer tokens masked;
- expand teacher localization data before training.

## Required Branch

Create a fresh server branch from latest `main`:

```bash
cd ASCR
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b feat/lumina-lora-json-v3-data-server
```

## Environment

Use the Lumina environment for local model work:

```bash
source .venv-lumina/bin/activate
python -m pip install -r requirements/lumina.txt

export LUMINA_REPO=third_party/Lumina-DiMOO
export LUMINA_MODEL_PATH=models/lumina-dimoo
export HF_HOME=.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

Use the Qwen/API environment only on the login node:

```bash
source .venv-qwen36/bin/activate
export OFOX_API_KEY='<provided-by-user-shell>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048
export ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=2048
```

Do not write API keys into repo files, scripts, logs, or docs.

## Phase 1: Expand Qwen3.7 Teacher Localization Data

Run OFOX/Qwen3.7-plus only on the login node.

Start with a moderate in-domain expansion. The current canonical small dataset is:

```text
outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl
```

If more Stage-1 records already exist under a larger Lumina output root, use that root. Otherwise keep using the existing Hard64 root first.

Recommended first expansion:

```bash
source .venv-qwen36/bin/activate

python scripts/distill/api_probe.py

LIMIT=256 \
OUT_ROOT=outputs/lumina_qwen_hard64 \
DISTILL_OUT=outputs/teacher_distill/hard256_lumina_qwen_qwen37_compact \
bash scripts/distill/run_teacher_distill.sh

python -m ascr.distill.audit \
  --distill-dir outputs/teacher_distill/hard256_lumina_qwen_qwen37_compact

python -m ascr.distill.export_dataset \
  --distill-dir outputs/teacher_distill/hard256_lumina_qwen_qwen37_compact \
  --output outputs/teacher_distill/hard256_lumina_qwen_qwen37_compact/dataset.jsonl
```

If `LIMIT=256` cannot produce 256 records because the source root has fewer Stage-1 outputs, record the actual count and continue. Do not fabricate counts.

If API cost or runtime is a concern, run `LIMIT=128` first.

## Phase 2: Prepare Clean Lumina SFT Examples

Switch to Lumina environment:

```bash
source .venv-lumina/bin/activate
```

Prepare canonical evaluator SFT examples:

```bash
DATASET=outputs/teacher_distill/hard256_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/sft_v3_clean \
LIMIT=256 \
bash scripts/training/prepare_lumina_native_sft.sh
```

Inspect the generated targets:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/stage2_lumina_native/sft_v3_clean/sft_examples.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
print("rows", len(rows))
for row in rows[:3]:
    print(json.dumps(row["target_json"], ensure_ascii=False, sort_keys=True))
PY
```

Confirm targets do not contain `raw`, `parser_error`, or `should_abstain`.

Convert to Lumina training format:

```bash
DATASET=outputs/stage2_lumina_native/sft_v3_clean/sft_examples.jsonl \
OUTPUT_DIR=outputs/stage2_lumina_native/lumina_sft_data_v3 \
python -m ascr.training.prepare_lumina_sft_data --limit 256
```

## Phase 3: Train LoRA v3

Run on a GPU node. Single GPU is enough for this smoke.

```bash
source .venv-lumina/bin/activate

python -m ascr.training.train_lumina_lora_smoke \
  --data-jsonl outputs/stage2_lumina_native/lumina_sft_data_v3/train.jsonl \
  --output-dir outputs/stage2_lumina_native/lora_v3_clean_allmask \
  --epochs 10 \
  --lr 2e-5 \
  --image-size 512 \
  --max-seq-len 2048 \
  --answer-mask-mode all \
  --ignore-pad-labels \
  --lora-r 8 \
  --lora-alpha 16 \
  --seed 0
```

If convergence is too slow, try `--lr 5e-5`. If memory fails, lower `--max-seq-len` only after recording the OOM details.

## Phase 4: Probe JSON Compliance

Pick several teacher grid images from the same SFT set:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("outputs/stage2_lumina_native/sft_v3_clean/sft_examples.jsonl").read_text().splitlines() if line.strip()]
for row in rows[:5]:
    print(row["image_path"])
    print(row["prompt"])
PY
```

Run probe. Use multiple `--image/--prompt` pairs if practical.

```bash
python -m ascr.cli.lumina_native_json_probe \
  --lora-path outputs/stage2_lumina_native/lora_v3_clean_allmask \
  --image <grid-image-1> \
  --prompt "<matching-prompt-1>" \
  --image <grid-image-2> \
  --prompt "<matching-prompt-2>" \
  --image <grid-image-3> \
  --prompt "<matching-prompt-3>" \
  --output-dir outputs/stage2_lumina_native/json_probe_lora_v3_clean_allmask \
  --max-new-tokens 384
```

Inspect:

```bash
cat outputs/stage2_lumina_native/json_probe_lora_v3_clean_allmask/summary.json
head -n 20 outputs/stage2_lumina_native/json_probe_lora_v3_clean_allmask/probe_rows.jsonl
```

## Decision Rule

Do not run formal benchmark if `parse_rate == 0.0`.

Use this rule:

- `parse_rate == 0.0`: stop, record raw outputs, do not benchmark.
- `0.0 < parse_rate < 0.3`: record outputs and propose next formatting/training change.
- `0.3 <= parse_rate < 0.8`: run only a tiny ASCR loop smoke on 1-3 prompts.
- `parse_rate >= 0.8`: run in-domain holdout smoke and Geneval smoke16 before/after benchmark.

Malformed output must remain abstention in ASCR. Do not add unsafe repair that can reopen cells from invalid JSON unless the repaired payload passes `safe_parse_semantic_evaluation`.

## If v3 Still Fails

If parse rate remains zero, report raw examples and test one of these next options:

1. Increase data to 512 or 1000 localization examples if API budget allows.
2. Train with `--lr 5e-5` and `--epochs 20`.
3. Try a simpler target distribution with mostly no-error JSON examples mixed with error examples.
4. Inspect whether Lumina's masked text generation needs a different answer delimiter or prefill strategy.
5. Consider constrained post-processing only as a diagnostic, not as a formal evaluator, unless it is conservative and schema-validated.

## Required Log Entry

Append a detailed entry to:

```text
docs/AI_COLLAB_LOG.md
```

Include:

- branch name and final commit hash;
- exact commands;
- login node/API commands and counts;
- GPU node/job ids;
- dataset counts;
- teacher label audit summary;
- SFT example count and missing image count;
- LoRA hyperparameters;
- final loss and loss curve summary;
- JSON probe raw previews;
- parsed/malformed/call-error counts;
- parse rate;
- whether benchmark was skipped or run;
- exact next blocker.

## Git Rules

Commit and push code/docs plus small JSON summaries only.

Allowed small results:

```bash
git add docs/AI_COLLAB_LOG.md
git add -f outputs/teacher_distill/hard256_lumina_qwen_qwen37_compact/*.json*
git add -f outputs/stage2_lumina_native/sft_v3_clean/*.json*
git add -f outputs/stage2_lumina_native/lumina_sft_data_v3/manifest.json
git add -f outputs/stage2_lumina_native/lora_v3_clean_allmask/training_manifest.json
git add -f outputs/stage2_lumina_native/json_probe_lora_v3_clean_allmask/*.json*
```

Do not commit:

- API keys;
- `.env`;
- images;
- image token caches;
- LoRA adapter weights;
- checkpoints;
- model weights;
- logs;
- large outputs;
- `.venv*`;
- `.hf_home`;
- any `/grp01/...` absolute path in new source code.

Before pushing:

```bash
git status --short --branch
git diff --stat
rg -n "sk-|api_key|token|secret|password|OPENAI_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|GOOGLE_API_KEY|ANTHROPIC_API_KEY" docs ascr scripts tests
git diff --check
git commit -m "Run Lumina LoRA JSON v3 data expansion pass"
git push -u origin HEAD
```
