import argparse
import json
from ascr.training.ddp import get_distributed_context


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reserved Stage 2 selector training entry point.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    payload = {"status": "reserved_for_stage2", "config": args.config, "distributed": get_distributed_context()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
