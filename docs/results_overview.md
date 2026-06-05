# Results Overview

This file collects the key machine-readable result artifacts so the root `docs/` directory is
easier to navigate.

## Cross-model summary artifacts

| Artifact | Scope |
| --- | --- |
| `docs/fair_4way_hard64_results.json` | Same-rubric Hard64 comparison across Show-o and MMaDA paths |
| `docs/lumina_qwen_coarse_hard64_results.json` | Lumina-DiMOO + Qwen coarse Hard64 run |
| `docs/mmada_qwen_coarse_hard64_results.json` | MMaDA-8B + Qwen coarse Hard64 run |
| `docs/mmada_self_coarse_hard64_results.json` | MMaDA-8B self-eval coarse Hard64 run |
| `docs/mmada_self_hard64_results.json` | MMaDA-8B self-eval direct-token Hard64 run |

## Judge outputs

The previous ad-hoc phase folders (`docs/p11_judge/`, `docs/p12_judge/`) have been consolidated
under `docs/results/judges/`.

| Directory | Scope |
| --- | --- |
| `docs/results/judges/phase11/` | Phase 11 clean and pairwise judge JSON outputs |
| `docs/results/judges/phase12/` | Phase 12 Lumina-related clean and pairwise judge JSON outputs |

## Examples

`docs/examples/` stores qualitative examples and benchmark comparison images. The subdirectories
there are organized by comparison set rather than by model implementation.
