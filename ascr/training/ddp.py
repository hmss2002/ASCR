import os


def get_distributed_context():
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
