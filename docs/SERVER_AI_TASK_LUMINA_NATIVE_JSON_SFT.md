# Server AI Task: Lumina-Native JSON Compliance And SFT Smoke

This document is the exact handoff for the university server AI. It should be
read together with:

- `docs/LUMINA_NATIVE_DISTILLATION.md`
- `docs/SERVER_AI_HANDOFF.md`
- `docs/AI_COLLAB_LOG.md`

## 1. Current State

Repository:

```bash
https://github.com/hmss2002/ASCR.git
```

Expected base branch:

```bash
main
```

Expected minimum commit:

```bash
357590e2542db4c581bf65707407fdd1f1ac9fbf
```

The local Codex has merged the previous server branch
`feat/lumina-native-audit-20260619` into `main`.

Important result from that branch:

- `LuminaNativeEngine.answer_image()` exists.
- Server GPU tests showed Lumina-DiMOO can read an image and output natural
  language text through its native MMU path.
- The blocker is not image understanding anymore.
- The blocker is **structured JSON compliance**: Lumina currently returns
  natural language instead of valid ASCR `SemanticEvaluation` JSON.

Formal Stage 2 target:

```text
prompt + current image
  -> Lumina-native semantic evaluator
  -> SemanticEvaluation JSON
  -> existing GridSemanticReopeningSelector
  -> Lumina token reopening
  -> repeat until no_error / abstain / no_actionable_region / max_iterations
```

Do not treat `grid-localizer-v0` or `grid-localizer-v1` as the formal student.
They are historical scaffold baselines only.

## 2. What Local Codex Added

The local Codex added or hardened the following:

- `LuminaNativeEngine.answer_image()` now has configurable answer generation
  settings:
  - `answer_steps`
  - `answer_block_length`
  - `answer_temperature`
  - `answer_cfg_scale`
- Default answer settings preserve the server-tested values:
  - `steps=64`
  - `block_length=128`
  - `temperature=0.0`
  - `cfg_scale=0.0`
- `run_stage1` can share one Lumina engine between:
  - `LuminaAdapter`
  - `LuminaNativeEvaluator`
- This avoids loading two copies of Lumina in one process.
- New JSON compliance probe:
  - `python -m ascr.cli.lumina_native_json_probe`
  - `scripts/training/run_lumina_native_json_probe.sh`
  - `jobs/training/lumina_native_json_probe.sbatch`
- New formal Lumina-native image benchmark runner:
  - `python -m ascr.benchmarks.lumina_native_benchmark`
  - `scripts/benchmark/run_lumina_native_image_benchmark.sh`
  - `jobs/benchmarks/lumina_native_image_benchmark.sbatch`
- API image judge wording is now generic ASCR before/after, not
  `student_localizer`.

Local validation already passed:

```text
python -m unittest discover -s tests
139 tests OK

python scripts/smoke_test.py
OK

python -m ascr.cli.preflight --mode local \
  --config configs/stage2/lumina/lumina_native_evaluator_smoke.yaml \
  --scan-secrets
OK, with expected local warnings because Windows lacks torch/Lumina weights
```

## 3. Your Main Goal On The Server

Your goal is to decide whether the current Lumina-native evaluator can produce
usable ASCR `SemanticEvaluation` JSON by prompt alone, and if not, start the
minimal Lumina-native SFT/LoRA route.

You must not run a formal before/after benchmark until there is a viable
structured evaluator.

Decision rule:

```text
If JSON probe parse_rate is poor:
  do not run formal benchmark;
  inspect Lumina-DiMOO MMU/training code;
  implement or plan a minimal LoRA/SFT smoke.

If JSON probe is reliable or SFT smoke produces valid SemanticEvaluation JSON:
  run formal Lumina-native before/after image benchmark;
  run login-node Qwen3.7 API judge.
```

Malformed JSON must remain an abstention. Never convert an invalid natural
language answer into a reopen decision unless it is parsed or trained to the
ASCR schema.

## 4. Git Workflow

Start from latest main and create a new branch:

```bash
cd ASCR
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
git checkout -b feat/lumina-native-json-sft-server
```

Confirm that `git rev-parse HEAD` is at least:

```bash
357590e2542db4c581bf65707407fdd1f1ac9fbf
```

At the end, commit and push your server branch:

```bash
git status --short --branch
git diff --stat
git add docs/AI_COLLAB_LOG.md
git add <safe code/doc files you changed>
git add -f outputs/stage2_lumina_native/**/*.json*  # only small JSON summaries/manifests
git commit -m "Run Lumina-native JSON probe and SFT smoke"
git push -u origin HEAD
```

Do not commit:

- API keys
- `.env`
- generated images
- logs
- model weights
- checkpoints
- Hugging Face caches
- large outputs
- private data

## 5. Environment Setup

Use the Lumina environment for GPU/MMU work:

```bash
source .venv-lumina/bin/activate

export LUMINA_REPO=third_party/Lumina-DiMOO
export LUMINA_MODEL_PATH=models/lumina-dimoo
export HF_HOME=.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

If paths differ on the server, use the real server paths and record them in
`docs/AI_COLLAB_LOG.md`. Do not record secrets.

Use the Qwen/OFOX environment only on the login node for API judging:

```bash
source .venv-qwen36/bin/activate
export OFOX_API_KEY='<provided-by-user-shell>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048
export ASCR_TEACHER_JSON_REPAIR_RETRIES=1
```

Do not send `OFOX_API_KEY` into compute-node jobs.

## 6. Step A: Run Lumina Native Audit

Run:

```bash
source .venv-lumina/bin/activate

bash scripts/training/run_lumina_native_audit.sh --load-model
```

Expected output:

```text
outputs/stage2_lumina_native/audit/audit.json
```

Record in `docs/AI_COLLAB_LOG.md`:

- command;
- GPU node/job id if Slurm was used;
- whether model loaded;
- whether `wrapper_supported_methods` includes `answer_image`;
- whether smoke output was parsed or abstained;
- output path.

Optional Slurm:

```bash
sbatch jobs/training/lumina_native_evaluator_audit.sbatch --load-model
```

## 7. Step B: Run JSON Compliance Probe

Run the default probe. It selects the first canonical teacher grid image from
the Qwen3.7 teacher dataset:

```bash
source .venv-lumina/bin/activate

DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/json_probe \
bash scripts/training/run_lumina_native_json_probe.sh
```

Expected outputs:

```text
outputs/stage2_lumina_native/json_probe/probe_rows.jsonl
outputs/stage2_lumina_native/json_probe/summary.json
```

Inspect:

```bash
cat outputs/stage2_lumina_native/json_probe/summary.json
head -n 3 outputs/stage2_lumina_native/json_probe/probe_rows.jsonl
```

Important fields:

- `row_count`
- `parsed_count`
- `malformed_count`
- `call_error_count`
- `parse_rate`
- `answer_steps`
- `answer_block_length`
- `answer_temperature`
- `answer_cfg_scale`

If parse rate is low, try a small number of controlled variants:

```bash
IMAGE='<path-to-existing-grid.ppm>'
PROMPT='<matching prompt>'

python -m ascr.cli.lumina_native_json_probe \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --output-dir outputs/stage2_lumina_native/json_probe_steps128 \
  --answer-steps 128 \
  --answer-block-length 128 \
  --max-new-tokens 512
```

Do not brute-force a large grid of settings. The goal is to understand whether
prompt-only JSON compliance is plausible.

## 8. Step C: Prepare SFT Examples

Prepare a small supervised dataset:

```bash
source .venv-lumina/bin/activate

DATASET=outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
IMAGE_ROOT=outputs/lumina_qwen_hard64 \
OUTPUT_DIR=outputs/stage2_lumina_native/sft_smoke \
LIMIT=16 \
bash scripts/training/prepare_lumina_native_sft.sh
```

Expected outputs:

```text
outputs/stage2_lumina_native/sft_smoke/sft_examples.jsonl
outputs/stage2_lumina_native/sft_smoke/manifest.json
```

Inspect:

```bash
cat outputs/stage2_lumina_native/sft_smoke/manifest.json
head -n 2 outputs/stage2_lumina_native/sft_smoke/sft_examples.jsonl
```

Record:

- example count;
- missing image count;
- path examples;
- whether targets contain valid `SemanticEvaluation` JSON.

## 9. Step D: Decide Whether To Implement SFT/LoRA Smoke

If JSON probe is poor, the next engineering task is a minimal Lumina-native
SFT/LoRA smoke.

Do not fake training.

Inspect the Lumina-DiMOO checkout for:

- MMU inference entrypoint;
- MMU training or SFT scripts;
- tokenizer/image-token formatting used by `generate_text_understanding`;
- LoRA or PEFT compatibility;
- whether the model exposes trainable language/MMU layers.

Useful searches:

```bash
rg -n "generate_text_understanding|inference_mmu|multimodal_understanding|LoRA|lora|peft|train|SFT|dataset" "$LUMINA_REPO"
```

If feasible, implement a **small smoke only**:

```text
input:  sft_examples.jsonl
output: adapter/checkpoint or explicit blocker report
scale:  5-16 examples
goal:   after reload, one held-out grid image produces parseable SemanticEvaluation JSON
```

If not feasible in one server pass, document the blocker precisely:

- missing train script;
- missing loss path;
- incompatible model wrapper;
- memory failure;
- dependency failure;
- unclear target token alignment.

## 10. Step E: Formal Benchmark Only If JSON Gate Clears

Run this only if prompt-only JSON compliance is good or an SFT/LoRA smoke
adapter can produce valid `SemanticEvaluation` JSON.

Compute/GPU node, no API key:

```bash
source .venv-lumina/bin/activate

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

Slurm version:

```bash
sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,PROMPTS=configs/benchmarks/prompts/t2i_compbench_hard64.txt,DOMAIN=in_domain_hard64_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/lumina_native/in_domain_hard64_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/lumina_native_image_benchmark.sbatch

sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,PROMPTS=configs/benchmarks/prompts/geneval_553.txt,DOMAIN=geneval_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/lumina_native/geneval_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/lumina_native_image_benchmark.sbatch
```

Expected outputs:

```text
outputs/image_bench/lumina_native/in_domain_hard64_smoke16/manifest.jsonl
outputs/image_bench/lumina_native/in_domain_hard64_smoke16/summary.json
outputs/image_bench/lumina_native/geneval_smoke16/manifest.jsonl
outputs/image_bench/lumina_native/geneval_smoke16/summary.json
```

Inspect:

- row count;
- error count;
- stop reasons;
- evaluator calls;
- selected token counts;
- abstention count;
- changed image count.

## 11. Step F: Login-Node Qwen3.7 Judge Only If Benchmark Ran

Run on login node only:

```bash
source .venv-qwen36/bin/activate

export OFOX_API_KEY='<provided-by-user-shell>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/lumina_native/in_domain_hard64_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/lumina_native/in_domain_hard64_smoke16 \
  --keep-going

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/lumina_native/geneval_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/lumina_native/geneval_smoke16 \
  --keep-going
```

Record:

- winner counts;
- mean before score;
- mean after score;
- mean delta;
- errors;
- examples where after wins or before wins.

## 12. Required Collaboration Log Entry

Append a detailed entry to:

```text
docs/AI_COLLAB_LOG.md
```

Use this structure:

```markdown
---

## 2026-06-19: Lumina-native JSON/SFT server pass (Server AI)

### Git
- Branch:
- Base commit:
- Final commit:

### Environment
- Host/login node:
- GPU node(s):
- Python env:
- LUMINA_REPO:
- LUMINA_MODEL_PATH:
- Offline/cache env:
- API env names set/unset, without values:

### Commands
- command:
  Result:
  Output path:
  Notes:

### Audit Result
- model_loaded:
- wrapper_supported_methods:
- smoke status:
- raw output type:

### JSON Probe Result
- row_count:
- parsed_count:
- malformed_count:
- call_error_count:
- parse_rate:
- best prompt variant:
- answer settings:

### SFT Prep Result
- example_count:
- missing_images:
- output paths:

### SFT/LoRA Feasibility
- feasible now:
- files inspected:
- blocker, if any:
- next code change needed:

### Benchmark/Judge Result
- ran benchmark: yes/no
- reason if not run:
- row counts:
- stop reasons:
- judge winners:
- mean score delta:

### Files committed
- docs:
- small JSON outputs:
- code:

### Next action for local Codex
- one concrete next step:
```

## 13. Safety Rules

- Never print or commit real API keys.
- Never commit `.env`.
- Never commit images unless explicitly asked.
- Never commit logs.
- Never commit model weights, checkpoints, or caches.
- Never force-push.
- If a command fails, record the exact command, error, and path in
  `docs/AI_COLLAB_LOG.md`.
- If JSON remains malformed, do not run formal benchmark just to produce a
  number. The correct next step is SFT/LoRA smoke or a precise blocker report.
