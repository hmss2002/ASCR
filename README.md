# ASCR: Alternating Semantic-Confidence Revision

ASCR is a research prototype for **selective semantic reopening** in masked / discrete
image-token generation. The project studies a failure mode where an image region becomes stable
under a model's confidence dynamics while still being semantically wrong with respect to the
prompt. ASCR addresses that gap by alternating native confidence refinement with semantic
localization and token reopening.

This `README.md` is the canonical project document. It replaces the previously split README,
master guide, architecture guide, stage design note, and blueprint files as the current source of
truth.

## 1. Current project position

**Mainline model:** Lumina-DiMOO
**Mainline selector:** Qwen3.5-9B coarse semantic selector
**Mainline task:** Stage-1 zero-training ASCR on Hard64-style compositional prompts
**Preserved comparison lines:** Show-o, MMaDA-8B, BAGEL-7B reference comparisons

The current project direction is no longer Show-o-first. Show-o remains important as the original
ASCR baseline and as a historical comparison line, but future work should treat **Lumina-DiMOO** as
the primary base model.

## 2. Core ASCR logic

### 2.1 Failure mode

ASCR is centered on **confidence-semantic inconsistency**:

> A token region can become confidence-stable under masked denoising while the decoded image still
> violates the text prompt in a meaningful semantic way.

Typical errors include:

- wrong count;
- wrong left/right, above/below, front/behind relation;
- wrong color or attribute binding;
- missing or extra objects;
- OCR or symbolic text mismatch;
- visually plausible but prompt-inconsistent local regions.

Native confidence remasking primarily asks:

> Where is the model uncertain?

ASCR asks a different question:

> Where is the model confident enough to stop changing, but still semantically wrong?

These two signals are not equivalent.

### 2.2 Method

ASCR alternates two operators:

| Operator | Question | Action |
| --- | --- | --- |
| Confidence block | Where is the model uncertain? | Continue native denoising / remasking |
| Semantic block | Where is the image semantically wrong? | Reopen selected token regions |

The zero-training implementation uses grid overlays, structured evaluator outputs, projection, and
fixed dilation as an engineering interface. The method itself is not the grid; the method is
**selective semantic reopening of confidence-stable but semantically wrong regions**.

### 2.3 Stage roadmap

| Stage | Goal | Current status |
| --- | --- | --- |
| Stage 1 | Zero-training semantic reopening with external or self evaluators | Implemented and evaluated across Show-o, MMaDA, Lumina |
| Stage 2 | Learned token-level semantic reopening selector | Scaffold exists under `ascr/training/` |
| Stage 3 | Cross-model ASCR framework across stronger discrete / masked generators | Direction established by current Lumina migration |

## 3. Current empirical summary

Under the same clean Gemini-style Hard64 rubric used in the latest cross-model comparison:

| Base model | Default generation | Baseline clean | + Qwen coarse ASCR clean | ASCR effect |
| --- | --- | ---: | ---: | --- |
| Lumina-DiMOO | 1024², 64 steps, cfg 4.0 | 82.8% | 84.4% | +1.6pp, no pairwise regressions |
| Show-o | 512², 50 steps, cfg 4.0 | 73.4% | 78.1% | +4.7pp, no pairwise regressions |
| MMaDA-8B | 512², HQ settings | 57.8% | 56.2% | single-image judge noise; pairwise still no regressions |

Key interpretation:

1. Lumina-DiMOO is the strongest current base model in this repository.
2. ASCR + Qwen coarse selector does not damage final quality in the tested settings.
3. Stronger base models leave less room for semantic repair, so ASCR triggers less often and gains
   are smaller.
4. Show-o and MMaDA results remain useful for mechanism analysis and ablations, but they are no
   longer the mainline.

Machine-readable result summaries live under `docs/results/summaries/`. Judge JSON files live
under `docs/results/judges/`.

## 4. Repository structure

```text
ASCR/
├── README.md                         # this canonical project document
├── ascr/                             # Python package
├── configs/                          # configs grouped by model/use case
│   ├── stage1/
│   │   ├── lumina/
│   │   ├── showo/
│   │   └── mmada/
│   ├── benchmarks/
│   │   ├── prompts/
│   │   └── data/
│   └── cluster/
├── jobs/                             # Slurm jobs grouped by use case
│   ├── stage1/
│   │   ├── lumina/
│   │   ├── showo/
│   │   ├── mmada/
│   │   └── variants/
│   ├── benchmarks/
│   ├── judges/
│   └── training/
├── scripts/                          # executable helpers grouped by role
│   ├── run/
│   ├── judge/
│   ├── benchmark/
│   ├── setup/
│   └── maintenance/
├── docs/
│   ├── results/
│   │   ├── summaries/
│   │   └── judges/
│   ├── examples/
│   └── archive/
├── requirements/
├── tests/
├── data/
└── external/
```

The repository intentionally keeps historical comparison assets, but current entrypoints should
prefer the Lumina-DiMOO line.

## 5. Python package map

| Path | Purpose |
| --- | --- |
| `ascr/core/` | ASCR loop, state, schemas, config loading, artifact writing |
| `ascr/generators/` | Base generator interface and Lumina / Show-o / MMaDA adapters |
| `ascr/evaluators/` | Qwen evaluator, MMaDA self-evaluator, remote evaluator, mock/local evaluators |
| `ascr/revision/` | Semantic-to-token reopening selectors and correction prompt composition |
| `ascr/grids/` | Overlay rendering and coarse-grid-to-token projection |
| `ascr/cli/` | Python module CLIs for Stage-1 and comparison variants |
| `ascr/benchmarks/` | Benchmark reporting helpers |
| `ascr/traces/` | Trace schemas and writers |
| `ascr/training/` | Stage-2 learned-selector scaffolding |

## 6. Mainline workflow

### 6.1 Lumina-DiMOO + Qwen coarse ASCR

Primary files:

| Role | Path |
| --- | --- |
| Config | `configs/stage1/lumina/stage1_lumina_qwen9b_coarse_hq.yaml` |
| Runner | `scripts/run/run_lumina_qwen_coarse_hard64.py` |
| Slurm job | `jobs/stage1/lumina/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch` |
| Generator adapter | `ascr/generators/lumina_dimoo.py` |
| Native engine | `ascr/generators/lumina_native.py` |
| Core loop | `ascr/core/loop.py` |
| Qwen evaluator | `ascr/evaluators/qwen_vl.py` |

### 6.2 Preserved comparison lines

| Track | Purpose | Key paths |
| --- | --- | --- |
| Show-o | Original ASCR baseline and direct-token experiments | `configs/stage1/showo/`, `ascr/generators/showo*.py`, `scripts/run/run_showo_qwen_coarse_hard64.py` |
| MMaDA-8B | Self-eval and Qwen-selector transfer experiments | `configs/stage1/mmada/`, `ascr/generators/mmada*.py`, `scripts/run/run_mmada_*` |
| BAGEL | External benchmark reference | `scripts/run/run_bagel_text2image.py`, benchmark jobs under `jobs/benchmarks/` |

## 7. Stage-1 loop

The practical ASCR loop is:

1. Start from the original prompt `P_orig`.
2. Run a native confidence / denoising block.
3. Decode the current token state to an intermediate image.
4. Render a localization interface, usually a coarse grid.
5. Evaluate prompt-image consistency against `P_orig`.
6. If the image matches, stop.
7. Otherwise localize semantic error regions.
8. Project localized regions into token space.
9. Apply reopening and optional dilation.
10. Compose a correction prompt with a preserve clause.
11. Force-mask selected tokens and continue the next confidence block.

Implementation rules:

- Always evaluate semantic correctness against the original prompt, not a drifting correction
  prompt.
- Treat evaluator outputs as strict contracts; malformed outputs should behave like abstentions.
- Preserve raw evaluator text, parsed JSON, selected cells, projected masks, and final artifacts.

## 8. Configs, prompts, and benchmark data

| Path | Purpose |
| --- | --- |
| `configs/stage1/lumina/` | Current Lumina-DiMOO Stage-1 configs |
| `configs/stage1/showo/` | Show-o historical and direct-token configs |
| `configs/stage1/mmada/` | MMaDA self-eval and Qwen-transfer configs |
| `configs/benchmarks/prompts/` | Prompt lists including Hard64, smoke sets, DrawBench, GenEval, DPG/DSG/GenAI |
| `configs/benchmarks/data/` | Benchmark metadata and CSV/JSONL payloads small enough for Git |
| `configs/cluster/` | Cluster partition templates |

## 9. Jobs and scripts

### Jobs

| Path | Purpose |
| --- | --- |
| `jobs/stage1/lumina/` | Lumina-DiMOO Stage-1 jobs |
| `jobs/stage1/showo/` | Show-o Stage-1 jobs |
| `jobs/stage1/mmada/` | MMaDA Stage-1 jobs |
| `jobs/stage1/variants/` | Direct-token / cap-sweep / variant generation jobs |
| `jobs/benchmarks/` | GenEval, BAGEL, and benchmark generation/scoring jobs |
| `jobs/judges/` | Judge-only Slurm jobs |
| `jobs/training/` | Stage-2 training jobs |

### Scripts

| Path | Purpose |
| --- | --- |
| `scripts/run/` | Main generation / experiment runners |
| `scripts/judge/` | Gemini, GPT, Qwen, OWL-ViT, and related judging/evaluation scripts |
| `scripts/benchmark/` | Prompt prep, conversion, pairing, merging, and summarization utilities |
| `scripts/setup/` | Environment and model download helpers |
| `scripts/maintenance/` | Submission helpers, diagnostics, cleanup, GitHub sync helper |

## 10. Results and examples

| Path | Purpose |
| --- | --- |
| `docs/results/summaries/` | Curated result JSON summaries |
| `docs/results/judges/phase11/` | Phase 11 judge outputs |
| `docs/results/judges/phase12/` | Phase 12 judge outputs |
| `docs/examples/` | Qualitative images and comparison figures |

Historical docs and old blueprint files are archived under `docs/archive/`. They are preserved for
traceability but should not be treated as current guidance.

## 11. Environment and data policy

Use dedicated virtual environments rather than the base server environment. Different model
families may require incompatible dependency stacks, especially where MMaDA and Qwen remote code
or transformer versions conflict.

### Runtime setup paths

Several paths referenced by configs and adapters are **runtime setup outputs**, not files expected
to be tracked by Git:

| Path | Meaning | How it is created |
| --- | --- | --- |
| `external/Show-o/` | Local Show-o source checkout for legacy/comparison paths | `bash scripts/setup/download_showo.sh` |
| `external/MMaDA/` | Local MMaDA source checkout for MMaDA experiments | manual/local cluster checkout; override with `repo_path` in MMaDA configs |
| `models/lumina-dimoo` | Lumina-DiMOO local model snapshot/cache | cluster model download/cache setup |
| `models/qwen3.5-9b` | Qwen evaluator local model snapshot/cache | cluster model download/cache setup |
| `models/show-o-512x512`, `models/phi-1_5`, `models/magvitv2` | Show-o legacy stack models | `bash scripts/setup/download_showo.sh` or local model cache |
| `models/mmada-8b-mixcot` | MMaDA local model snapshot/cache | local model download/cache setup |

These directories are intentionally ignored by Git. If a workflow fails because one of them is
missing, create the required local checkout/model cache instead of committing it.

Do not commit:

- model weights;
- generated images outside curated docs examples;
- large downloaded datasets;
- checkpoints;
- local virtual environments;
- local secrets or API keys;
- transient logs and Slurm outputs.

Small prompt lists, benchmark metadata, result summaries, code, configs, and job definitions should
be tracked.

## 12. Development and validation

Useful checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
python -m ascr.cli.run_stage1 --help
python -m ascr.cli.compare_stage1_variants --help
```

Reference checks used after repository reorganizations:

```bash
rg 'configs/(prompts|benchmark_data|stage1_lumina|stage1_showo|stage1_mmada8b|cluster_gpu|showo_local_512x512)' \
  README.md ascr configs jobs scripts data external
rg 'jobs/(geneval_gen_shard|geneval_merge_eval|stage1_t2i_compbench|stage1_hard64_variant_gen|stage1_geneval_score_single|bench_bagel_gen_shard)\.sbatch' \
  README.md ascr configs jobs scripts data external
```

Before committing structural changes:

1. search for stale paths with `rg`;
2. run available unit tests or import/smoke checks;
3. keep generated outputs and large artifacts out of Git;
4. commit related file moves and path updates together so history stays understandable.

## 13. Maintenance rule

Going forward:

1. Update this `README.md` for current project logic and current run paths.
2. Put machine-readable outputs under `docs/results/`.
3. Put qualitative examples under `docs/examples/`.
4. Put old or superseded narrative material under `docs/archive/`.
5. Do not create another competing high-level project guide unless this README is intentionally
   replaced.
