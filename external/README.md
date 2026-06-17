# External Local Checkouts

This directory is reserved for local third-party source trees that are needed on the cluster but should not be vendored into the ASCR Git repository.

Expected local checkouts:

- `external/Show-o/`: shallow clone of `https://github.com/showlab/Show-o.git` for local Stage 1 generator integration.
- `external/MMaDA/`: local MMaDA source checkout for MMaDA self-eval and transferred-selector experiments.

Use `scripts/setup/download_showo.sh` to recreate the checkout and download model snapshots into `models/`.

The MMaDA checkout is intentionally cluster-local and may come from a private/manual setup. If it
lives elsewhere, set `repo_path` in the relevant `configs/stage1/mmada/*.yaml` file.
