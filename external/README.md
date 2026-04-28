# External Local Checkouts

This directory is reserved for local third-party source trees that are needed on the cluster but should not be vendored into the ASCR Git repository.

Current local checkout:

- `external/Show-o/`: shallow clone of `https://github.com/showlab/Show-o.git` for local Stage 1 generator integration.

Use `scripts/download_showo.sh` to recreate the checkout and download model snapshots into `models/`.
