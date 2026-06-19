# Server AI Task: Lumina LoRA JSON v2

Read this before running the next server pass. This is the successor to
`docs/SERVER_AI_TASK_LUMINA_NATIVE_JSON_SFT.md`.

## 1. Current State

Expected repository:

```bash
https://github.com/hmss2002/ASCR.git
```

Expected branch:

```bash
main
```

The local Codex has reviewed these server branches:

- `feat/lumina-native-json-sft-server`
- `feat/lumina-sft-smoke-20260619`
- `feat/lumina-lora-smoke-20260619`

Do not re-merge those branches directly. Their useful findings were integrated
into main as cleaner code.

Accepted findings:

- Base Lumina MMU can read image + prompt but does not produce parseable JSON.
- Prompt-only JSON compliance failed: `parse_rate=0.0`.
- Full-parameter single-GPU SFT is not feasible on the observed 45GB GPU: OOM.
- LoRA SFT is feasible on one GPU.
- The first LoRA smoke moved output from natural language to JSON-like text.
- The first LoRA smoke still had `parse_rate=0.0`; output was JSON-like but
  malformed.

Formal benchmark is still blocked until Lumina-native output parses as valid
ASCR `SemanticEvaluation` JSON.

## 2. What Main Now Provides

Main now includes:

- `LuminaNativeEngine(lora_path=...)`
  - Loads PEFT LoRA adapters through `PeftModel.from_pretrained`.
  - Does not require probe scripts to patch private model fields.
- `align_answer_generation_lengths(...)`
  - Aligns `max_new_tokens`, `block_length`, and `steps` so Lumina's
    `generate_text_understanding` block constraints are satisfied.
- `ascr.training.prepare_lumina_sft_data`
  - Converts ASCR `sft_examples.jsonl` to Lumina-format training JSONL.
  - Writes image-token pickle cache and `manifest.json`.
- `ascr.training.train_lumina_lora_smoke`
  - Single-GPU LoRA smoke trainer.
  - Defaults are based on the successful server smoke:
    - `image_size=512`
    - `max_seq_len=2048`
    - `lora_r=8`
    - `lora_alpha=16`
    - `batch_size=1` implicit by row-wise training loop
- `ascr.cli.lumina_native_json_probe --lora-path`
  - Probes base or LoRA-adapted Lumina with the same parser used by ASCR.

Do not commit adapter weights, checkpoints, logs, generated images, or image
token caches.

## 3. Start A Fresh Server Branch

```bash
cd ASCR
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
git checkout -b feat/lumina-lora-json-v2-server
```

Record the commit hash in `docs/AI_COLLAB_LOG.md`.

## 4. Environment

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

If paths differ, use the real server paths and record them. Do not record
secrets.

## 5. Prepare Lumina-Format SFT Data

Run:

```bash
DATASET=outputs/stage2_lumina_native/sft_smoke/sft_examples.jsonl \
OUTPUT_DIR=outputs/stage2_lumina_native/lumina_sft_data_v2 \
python -m ascr.training.prepare_lumina_sft_data --limit 16
```

Expected outputs:

```text
outputs/stage2_lumina_native/lumina_sft_data_v2/train.jsonl
outputs/stage2_lumina_native/lumina_sft_data_v2/manifest.json
outputs/stage2_lumina_native/lumina_sft_data_v2/image_tokens/*.pkl
```

Inspect:

```bash
cat outputs/stage2_lumina_native/lumina_sft_data_v2/manifest.json
head -n 2 outputs/stage2_lumina_native/lumina_sft_data_v2/train.jsonl
```

Required log fields:

- `example_count`
- `skipped_count`
- whether all `answer_text` values are compact JSON strings

## 6. Train LoRA v2

First reproduce a controlled smoke:

```bash
python -m ascr.training.train_lumina_lora_smoke \
  --data-jsonl outputs/stage2_lumina_native/lumina_sft_data_v2/train.jsonl \
  --output-dir outputs/stage2_lumina_native/lora_v2 \
  --epochs 10 \
  --lr 5e-5 \
  --image-size 512 \
  --max-seq-len 2048 \
  --lora-r 8 \
  --lora-alpha 16 \
  --seed 0
```

If it OOMs, try exactly one lower-memory variant:

```bash
python -m ascr.training.train_lumina_lora_smoke \
  --data-jsonl outputs/stage2_lumina_native/lumina_sft_data_v2/train.jsonl \
  --output-dir outputs/stage2_lumina_native/lora_v2_lowmem \
  --epochs 10 \
  --lr 2e-5 \
  --image-size 384 \
  --max-seq-len 1536 \
  --lora-r 4 \
  --lora-alpha 8 \
  --seed 0
```

Do not run a broad hyperparameter sweep. The goal is to get parseable JSON or a
clear blocker.

Expected output:

```text
outputs/stage2_lumina_native/lora_v2/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
  training_manifest.json
```

Do not commit adapter files.

## 7. Probe LoRA JSON Compliance

Use an existing teacher grid image and matching prompt:

```bash
SMOKE_IMAGE=$(python - <<'PY'
import json
from pathlib import Path
dataset = Path("outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl")
row = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
loc = row["localizations"][0]
print(Path("outputs/lumina_qwen_hard64") / loc["grid_image"])
PY
)

SMOKE_PROMPT=$(python - <<'PY'
import json
from pathlib import Path
dataset = Path("outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl")
row = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
loc = row["localizations"][0]
print(loc.get("prompt") or row.get("prompt") or "")
PY
)

python -m ascr.cli.lumina_native_json_probe \
  --lora-path outputs/stage2_lumina_native/lora_v2 \
  --image "$SMOKE_IMAGE" \
  --prompt "$SMOKE_PROMPT" \
  --output-dir outputs/stage2_lumina_native/json_probe_lora_v2 \
  --max-new-tokens 384
```

Inspect:

```bash
cat outputs/stage2_lumina_native/json_probe_lora_v2/summary.json
head -n 5 outputs/stage2_lumina_native/json_probe_lora_v2/probe_rows.jsonl
```

Required log fields:

- `row_count`
- `parsed_count`
- `malformed_count`
- `call_error_count`
- `parse_rate`
- at least three `raw_preview` examples
- exact parse errors

## 8. Decision Rule

If `parse_rate == 0`:

- Do not run formal image benchmark.
- Record raw examples and exact parser errors.
- Suggest one next change:
  - more epochs / lower LR;
  - stricter target formatting;
  - deterministic JSON syntax repair;
  - architecture/training-format fix.

If `parse_rate > 0` and parsed rows are valid `SemanticEvaluation`:

- Run only a tiny benchmark smoke, not full benchmark:

```bash
PROMPTS=configs/benchmarks/prompts/t2i_compbench_hard_smoke8.txt \
DOMAIN=in_domain_hard8_lora_v2 \
LIMIT=8 \
OUTPUT_DIR=outputs/image_bench/lumina_native_lora_v2/in_domain_hard8 \
MAX_ITERATIONS=3 \
CONFIG=configs/stage2/lumina/lumina_native_evaluator_smoke.yaml \
bash scripts/benchmark/run_lumina_native_image_benchmark.sh
```

Before running that benchmark, set `lora_path` in a temporary config or pass a
config override only if the current CLI supports it. If it does not, stop and
ask local Codex to add a benchmark-time `LUMINA_LORA_PATH` override.

## 9. Collaboration Log Required Format

Append to:

```text
docs/AI_COLLAB_LOG.md
```

Use this structure:

```markdown
---

## 2026-06-19: Lumina LoRA JSON v2 server pass (Server AI)

### Git
- Branch:
- Base commit:
- Final commit:

### Environment
- Host:
- GPU node/job id:
- Python env:
- LUMINA_REPO:
- LUMINA_MODEL_PATH:
- peft version:

### Data Conversion
- command:
- example_count:
- skipped_count:
- output path:

### LoRA Training
- command:
- hyperparameters:
- loss curve:
- final_loss:
- memory/OOM notes:
- adapter output path:

### JSON Probe
- command:
- row_count:
- parsed_count:
- malformed_count:
- call_error_count:
- parse_rate:
- raw_preview examples:
- parser errors:

### Benchmark
- ran: yes/no
- reason:
- output path:
- summary if run:

### Files Committed
- docs:
- code:
- small JSON summaries:

### Next Action For Local Codex
- one concrete next step:
```

## 10. Commit Rules

Safe to commit:

- `docs/AI_COLLAB_LOG.md`
- code changes, if any
- small JSON summaries/manifests only

Do not commit:

- `outputs/stage2_lumina_native/lora_v2/adapter_model.safetensors`
- `outputs/stage2_lumina_native/lumina_sft_data_v2/image_tokens/*.pkl`
- generated images
- logs
- model weights
- checkpoints
- `.env`
- API keys
- caches

Push your branch:

```bash
git status --short --branch
git diff --stat
git add docs/AI_COLLAB_LOG.md
git add <safe code/doc changes only>
git add -f outputs/stage2_lumina_native/json_probe_lora_v2/*.json*
git add -f outputs/stage2_lumina_native/lumina_sft_data_v2/manifest.json
git commit -m "Run Lumina LoRA JSON v2 server pass"
git push -u origin HEAD
```
