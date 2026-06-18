# Student Localizer Image Benchmark

This workflow evaluates the real distilled-student path:

```text
before distill: prompt -> generator -> initial image
after distill:  prompt -> generator -> student localizer -> GridSemanticReopeningSelector -> ASCR loop -> final image
```

The student learns the semantic localizer/evaluator role. The existing selector
still maps selected grid cells to token reopen masks.

## Train The Student Localizer

```bash
source .venv-qwen36/bin/activate

python -m ascr.training.train_localizer \
  --task grid-localizer-v0 \
  --dataset outputs/teacher_distill/hard64_lumina_qwen_qwen37_compact/dataset.jsonl \
  --image-root outputs/lumina_qwen_hard64 \
  --output-dir outputs/stage2_students/grid_localizer_v0 \
  --eval-mode holdout \
  --train-ratio 0.8 \
  --seed 0
```

Outputs:

```text
outputs/stage2_students/grid_localizer_v0/
  student_model.json
  metrics.json
  predictions.jsonl
  split_manifest.json
  holdout_prompts.txt
```

`grid-localizer-v0` is a lightweight, reproducible baseline that uses prompt
hash features and image grid-cell RGB features. It is more meaningful than
`cell-prior` because it reads the current image and prompt, but it is not the
final neural/DDP Stage-2 model.

## Run Before/After Image Generation On GPUs

In-domain holdout:

```bash
source .venv-lumina/bin/activate

STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json \
PROMPTS=outputs/stage2_students/grid_localizer_v0/holdout_prompts.txt \
DOMAIN=in_domain_hard64_holdout \
OUTPUT_DIR=outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_student_image_benchmark.sh
```

Geneval smoke:

```bash
STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json \
PROMPTS=configs/benchmarks/prompts/geneval_553.txt \
DOMAIN=geneval_smoke16 \
LIMIT=16 \
OUTPUT_DIR=outputs/image_bench/student_localizer_v0/geneval_smoke16 \
MAX_ITERATIONS=3 \
bash scripts/benchmark/run_student_image_benchmark.sh
```

Slurm:

```bash
sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json,PROMPTS=outputs/stage2_students/grid_localizer_v0/holdout_prompts.txt,DOMAIN=in_domain_hard64_holdout,OUTPUT_DIR=outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch

sbatch --export=ALL,OFOX_API_KEY=,OFOX_BASE_URL=,ASCR_TEACHER_MODEL=,ASCR_TEACHER_QUALITY_MAX_TOKENS=,ASCR_TEACHER_LOCALIZATION_MAX_TOKENS=,ASCR_TEACHER_JSON_REPAIR_RETRIES=,STUDENT_MODEL=outputs/stage2_students/grid_localizer_v0/student_model.json,PROMPTS=configs/benchmarks/prompts/geneval_553.txt,DOMAIN=geneval_smoke16,LIMIT=16,OUTPUT_DIR=outputs/image_bench/student_localizer_v0/geneval_smoke16,MAX_ITERATIONS=3 \
  jobs/benchmarks/student_image_benchmark_lumina.sbatch
```

The shared Slurm wrapper now works for both in-domain and Geneval paths. It
also strips any leaked OFOX/API judge environment variables as a defensive
backstop, but submissions should still blank them explicitly as shown above.

## Judge Before/After Quality On The Login Node

```bash
export OFOX_API_KEY='<real-key-from-user-shell>'
export OFOX_BASE_URL='https://api.ofox.ai/v1'
export ASCR_TEACHER_MODEL='bailian/qwen3.7-plus'
export ASCR_TEACHER_QUALITY_MAX_TOKENS=2048

source .venv-qwen36/bin/activate

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/student_localizer_v0/in_domain_hard64_holdout/manifest.jsonl \
  --output-dir outputs/api_judges/student_localizer_v0/in_domain_hard64_holdout \
  --keep-going

python -m ascr.benchmarks.api_image_judge \
  --manifest outputs/image_bench/student_localizer_v0/geneval_smoke16/manifest.jsonl \
  --output-dir outputs/api_judges/student_localizer_v0/geneval_smoke16 \
  --keep-going
```

The judge writes `judgments.jsonl`, `summary.json`, and `errors.jsonl`. It does
not print or persist API keys. It stores raw model text only if
`--include-raw-text` is explicitly set.

Important benchmark semantics: when the ASCR loop hits `max_iterations`, the
stage-1 config can conservatively fall back to the initial image for the loop's
selected final output. The benchmark manifest now records the actual last
candidate image as `after_image`, and keeps the fallback-selected image in
`selected_after_image` together with `fallback_applied` metadata so before/after
judging does not collapse real revisions back to the initial image.
