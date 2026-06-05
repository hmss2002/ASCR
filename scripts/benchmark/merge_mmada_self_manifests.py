#!/usr/bin/env python3
"""Merge per-prompt MMaDA-self Hard64 records into Gemini-judge manifests.

Reads OUT_ROOT/records/*.json and writes two manifests compatible with
scripts/judge/judge_variants_gemini.py (which indexes by ``final_image``):

  baseline_manifest.json : final_image = MMaDA's initial generation (no revision)
  self_manifest.json     : final_image = MMaDA self-revised closed-loop output

Each also carries baseline_image/final_image for clean-mode --image-field use.
"""

import json
import sys
from pathlib import Path


def main():
    out_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/mmada_self_hard64")
    records = []
    for rec_path in sorted((out_root / "records").glob("p*.json")):
        rec = json.loads(rec_path.read_text())
        if "baseline_image" in rec and "final_image" in rec:
            records.append(rec)

    baseline_records = [{"prompt": r["prompt"], "baseline_image": r["baseline_image"],
                         "final_image": r["baseline_image"]} for r in records]
    self_records = [{"prompt": r["prompt"], "baseline_image": r["baseline_image"],
                     "final_image": r["final_image"]} for r in records]

    (out_root / "baseline_manifest.json").write_text(json.dumps(
        {"arm": "baseline", "count": len(baseline_records), "records": baseline_records}, indent=2))
    (out_root / "self_manifest.json").write_text(json.dumps(
        {"arm": "self", "count": len(self_records), "records": self_records}, indent=2))

    revised = sum(1 for r in records if r.get("revisions", 0) > 0)
    print(f"records={len(records)} revised={revised} "
          f"baseline_manifest+self_manifest written under {out_root}")


if __name__ == "__main__":
    main()
