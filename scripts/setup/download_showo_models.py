from huggingface_hub import snapshot_download
import os

REPOS = [
    ("showlab/show-o-512x512", "show-o-512x512"),
    ("showlab/magvitv2", "magvitv2"),
    ("microsoft/phi-1_5", "phi-1_5"),
]


def main():
    model_root = os.environ.get("SHOWO_MODEL_ROOT", "models")
    for repo_id, local_name in REPOS:
        local_dir = os.path.join(model_root, local_name)
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=local_dir, max_workers=1)
    print("Show-o model snapshots are available under", model_root)


if __name__ == "__main__":
    main()
