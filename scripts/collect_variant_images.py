import argparse
import json
from pathlib import Path


def find_comparison_files(run_root):
    return sorted(Path(run_root).rglob("comparison.json"))


def collect(run_root, arm_filter=None):
    records = []
    arms_seen = set()
    for path in find_comparison_files(run_root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        prompt = data.get("prompt")
        baseline_image = data.get("baseline_image")
        for arm in data.get("arms", []):
            arm_name = arm.get("arm")
            arms_seen.add(arm_name)
            if arm_filter and arm_name != arm_filter:
                continue
            records.append(
                {
                    "prompt": prompt,
                    "arm": arm_name,
                    "baseline_image": baseline_image,
                    "final_image": arm.get("final_image"),
                    "score": arm.get("score"),
                    "comparison_json": str(path),
                }
            )
    return records, sorted(a for a in arms_seen if a)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate per-prompt comparison.json files into a flat manifest for Gemini judging.")
    parser.add_argument("--run-root", required=True, help="Root directory containing comparison.json files (recursively).")
    parser.add_argument("--arm", default=None, help="Keep only this arm (e.g. direct_token or coarse_grid). Default: keep all.")
    parser.add_argument("--output", required=True, help="Path to write manifest.json.")
    args = parser.parse_args(argv)

    records, arms_seen = collect(args.run_root, args.arm)
    records.sort(key=lambda r: (r["arm"] or "", r["prompt"] or ""))
    manifest = {
        "run_root": str(args.run_root),
        "arm_filter": args.arm,
        "arms_seen": arms_seen,
        "count": len(records),
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "count": len(records), "arms_seen": arms_seen}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
