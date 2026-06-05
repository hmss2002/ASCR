# ASCR Master Guide

This document is the integrated project guide for ASCR. It combines the roles that were
previously split across the root README, the archived project-control README, and the two
English blueprint documents:

- `ASCR_Paper_Blueprint_EN.docx`
- `ASCR_Workflow_Playbook_EN.docx`

The goal is to keep **one long-form canonical document** that explains the project at three
levels at once:

1. **Research framing** — what problem ASCR claims to solve and why it matters.
2. **Implementation workflow** — how the Stage-1 loop actually runs in this repository.
3. **Repository reality** — which model path is current, which tracks are legacy, and where the
   code and operational assets live today.

## 1. Executive summary

ASCR (Alternating Semantic-Confidence Revision) is a test-time revision framework for masked
image-token generation. Its central thesis is that discrete denoising models can become
**confidence-stable while remaining semantically wrong** with respect to the prompt. ASCR treats
this as a distinct failure mode — **confidence-semantic inconsistency** — and addresses it by
alternating:

- a **confidence operator** that follows the model's native denoising/remasking dynamics, and
- a **semantic operator** that selectively reopens regions judged semantically wrong.

In the current repository, the **zero-training Stage-1 implementation** uses structured semantic
feedback, grid-based localization, and token reopening as an engineering interface for this idea.
The long-term method is **not** the grid itself; the long-term method is **selective semantic
reopening of confidence-stable but semantically wrong regions**.

## 2. Current repository status

### Mainline

The current recommended mainline is:

- **Generator:** Lumina-DiMOO
- **Selector / evaluator:** Qwen3.5-9B coarse semantic selector
- **Primary config:** `configs/stage1_lumina_qwen9b_coarse_hq.yaml`
- **Primary runner:** `scripts/run_lumina_qwen_coarse_hard64.py`
- **Primary job:** `jobs/stage1_lumina_qwen_coarse_hard64_8gpu.sbatch`

### Preserved comparison lines

- **Show-o:** original Stage-1 baseline line and direct-token experiments
- **MMaDA-8B:** self-eval and transferred-selector experiments
- **BAGEL-7B:** external benchmark comparison reference

### What changed relative to the original blueprints

The early blueprints were written around a Show-o-centered 16×16 formulation. The repository has
since evolved into a **multi-model discrete-diffusion codebase**, including 32×32 token-grid
paths and cross-model evaluation. The core method claim still transfers:

- native confidence dynamics and semantic correctness are different signals;
- semantic repair should reopen local token regions rather than retry the whole image;
- grid overlays and JSON outputs are **prototype interfaces**, not the final scientific object.

## 3. Problem definition

### Confidence-semantic inconsistency

ASCR is built around the following failure mode:

> A token region can become stable under the model's native confidence dynamics while the decoded
> image still violates the prompt in a semantically meaningful way.

Typical manifestations include:

- wrong count
- wrong left/right or front/behind relation
- wrong color or attribute binding
- missing or extra objects
- OCR / text mismatch
- visually plausible but semantically incorrect local regions

This matters because native remasking primarily answers:

- **Where is the model uncertain?**

while ASCR asks:

- **Where is the model confident enough to stop changing, but still wrong?**

Those are not equivalent.

## 4. Method principle

### Core method

ASCR alternates two operators:

| Operator | Question answered | Action |
| --- | --- | --- |
| Confidence block | Where is the model uncertain? | Native low-confidence remask / denoising |
| Semantic block | Where is the model semantically wrong despite local stability? | Forced reopening of selected regions |

The alternation is important. If confidence and semantic reopening are merged into one opaque
action, the mechanism becomes harder to interpret, ablate, and justify.

### Zero-training instantiation vs. method identity

| Layer | What it is | Status |
| --- | --- | --- |
| Method principle | Selective semantic reopening of confidence-stable but semantically wrong regions | Core ASCR claim |
| Zero-training interface | Grid overlays, structured JSON, fixed projection and dilation, correction prompts | Implemented in this repo |
| Learnable extension | Trainable selector / reopening head on token states | Planned Stage 2+ direction |

The grid is therefore a **means**, not the end.

## 5. Stage-1 workflow in this repository

The practical Stage-1 loop is:

1. Initialize the token grid and set `P_cur = P_orig`.
2. Run a short native denoising block under the current prompt.
3. Decode to an intermediate image.
4. Render an overlay (coarse grid or token grid, depending on variant).
5. Run semantic evaluation against the **original prompt**.
6. If the image already matches, stop.
7. Otherwise localize the minimal region set to reopen.
8. Project localized regions back to token space.
9. Apply reopening / dilation rules.
10. Compose the next correction prompt with a preserve clause.
11. Force-mask the selected tokens and return to the next confidence block.

Two implementation rules from the workflow playbook remain important:

- **Keep `P_orig` constant** for evaluation; only `P_cur` changes.
- **Treat prompt outputs as API contracts**, not casual free-form text.

## 6. State, interfaces, and artifacts

### Main state variables

| Symbol / artifact | Meaning |
| --- | --- |
| `P_orig` | Original generation prompt |
| `P_cur` | Current correction-conditioned prompt |
| `u` | Current image-token state |
| `I_mid` | Decoded intermediate image |
| `I_grid` | Intermediate image with overlay |
| `A_eval` | Semantic evaluation payload |
| `A_loc` | Localization payload |
| `M_conf` | Native confidence remask set |
| `M_sem` | Semantic reopening mask |
| `log[k]` | Per-cycle trace / artifact record |

### Main Python interfaces

| Interface | Role | Current location |
| --- | --- | --- |
| `GeneratorAdapter` | Generation / reopen-and-continue abstraction | `ascr/generators/base.py` |
| `SemanticEvaluator` | Semantic judgement interface | `ascr/evaluators/base.py` |
| `SemanticReopeningSelector` | Semantic payload → token mask | `ascr/revision/selector.py` |
| `TraceWriter` | Artifact and trace persistence | `ascr/traces/writer.py` |
| `ASCRLoop` | Core alternating loop | `ascr/core/loop.py` |

### Current model implementations

| Track | Main files |
| --- | --- |
| Lumina-DiMOO | `ascr/generators/lumina_dimoo.py`, `ascr/generators/lumina_native.py` |
| Show-o | `ascr/generators/showo.py`, `ascr/generators/showo_native.py` |
| MMaDA-8B | `ascr/generators/mmada.py`, `ascr/generators/mmada_native.py` |

## 7. Prompt and selector contracts

The workflow blueprint separates the semantic side into three contracts:

| Prompt / contract | Purpose | Required behavior |
| --- | --- | --- |
| Semantic evaluation | Decide whether the current image matches the prompt and describe main errors | Structured, machine-parseable, conservative |
| Localization | Return the minimal region set to reopen | Structured and allowed to abstain |
| Correction composition | Convert semantic findings into the next prompt | Fix the target error while preserving everything else |

Repository design rule:

- malformed evaluator output should behave like an **abstention**, not a silent success;
- traces should preserve raw intermediate artifacts for later analysis and training;
- logging should keep enough information to reproduce each outer revision cycle.

## 8. Benchmark and evidence strategy

The paper blueprint and workflow playbook agree on the same evidence chain:

1. Show that native masked denoising produces high-confidence semantic errors.
2. Show that confidence-only refinement does not reliably fix them.
3. Show that local semantic reopening can target them.
4. Show that alternating confidence and semantic revision is better than confidence-only,
   semantic-only, or strong global-retry baselines.

### Target benchmark categories

- relation
- count
- negation
- color constraints
- attribute binding
- missing / extra object
- OCR / text fidelity

### Strong baseline families

- confidence-only
- semantic-only
- alternating confidence ↔ semantic revision
- whole-image retry
- Best-of-N + rerank
- verifier-only correction
- inpainting-style local editing
- longer native denoising without semantic revision

The point is not just to get a higher score, but to show **why** local semantic reopening is the
right action for this failure mode.

## 9. Current empirical position

The repository has already moved beyond the earliest Show-o-only framing and now supports
cross-model comparisons under a shared judging setup.

Current high-level findings:

- **Base-model ordering under the current clean Hard64 rubric:** Lumina-DiMOO > Show-o > MMaDA-8B
- **ASCR + Qwen coarse selector does not harm final quality on the tested base models**
- **Stronger base models leave less semantic headroom**, so revision triggers less often
- **Lumina-DiMOO is therefore the correct mainline path for the repository today**

Machine-readable results remain in:

- `docs/fair_4way_hard64_results.json`
- `docs/lumina_qwen_coarse_hard64_results.json`
- `docs/mmada_qwen_coarse_hard64_results.json`
- `docs/mmada_self_coarse_hard64_results.json`
- `docs/mmada_self_hard64_results.json`

## 10. Operational workflow and reproducibility

### Repository navigation

| Path | Role |
| --- | --- |
| `ascr/` | core package |
| `configs/` | experiment configs and benchmark metadata |
| `jobs/` | Slurm entrypoints |
| `scripts/` | orchestration, judging, setup, data prep |
| `docs/` | integrated docs, architecture notes, results, history |
| `tests/` | wiring and loop regression tests |

### Environment policy

The project uses dedicated environments rather than the base server environment. Runtime and
download assumptions may differ between login nodes and offline compute nodes, so configuration,
artifacts, and prompt versions must remain explicit and reproducible.

### Artifact policy

Track source code, configs, prompts, docs, and small metadata in Git. Do **not** track:

- model weights
- generated image outputs
- checkpoints
- large downloaded datasets
- local secrets
- transient logs

## 11. Roadmap

### Stage 1: zero-training proof of concept

- semantic evaluation and localization with constrained interfaces
- local token reopening
- targeted benchmark design
- strong baseline comparison

### Stage 2: learned selector

- replace or augment grid-driven reopening with a trainable token-level reopening head
- supervise from traces, localization labels, or revision outcomes

### Stage 3: cross-model revision framework

- carry the ASCR principle across multiple unified or discrete masked generators
- separate model-specific engineering from the revision mechanism itself

## 12. How this document relates to the other docs

Use this file as the **single integrated long-form guide**.

Use the other documents as focused references:

| Document | Use it for |
| --- | --- |
| `README.md` | short front page and fast entry |
| `docs/architecture/repository_guide.md` | quick repository navigation |
| `docs/architecture/stage1_design.md` | concise design constraints |
| `docs/results_overview.md` | machine-readable result artifact index |
| `docs/history/experiment_changelog.md` | dated experiment history |
| `docs/history/project_control_legacy.md` | archived old control document |

## 13. Recommended maintenance rule

Going forward, update documents with this order of responsibility:

1. **`README.md`** — short and stable front page only
2. **`docs/ascr_master_guide.md`** — canonical integrated project narrative
3. **focused docs under `docs/architecture/`, `docs/results_overview.md`, and `docs/history/`** —
   detailed supporting references

That keeps one clear main document without losing the historical and operational detail that the
project has accumulated.
