#!/usr/bin/env python3
"""Pair BAGEL hard64 images with a ShowO/ASCR hard64 aggregated suite by prompt.

Emits a suite.json with fields {baseline_image, ascr_final_image, prompt} so the
existing scripts/judge/judge_showo_ascr_pairwise_qwen.py can consume it.

Usage:
    python scripts/benchmark/pair_bagel_vs_hard64_run.py \
        --bagel-suite outputs/bagel_t2i_compbench_hard64_.../bagel_vs_ascr_suite.json \
        --hard64-suite outputs/benchmarks_t2i_compbench_qwen35_hard64_.../suite.json \
        --side {baseline,ascr} \
        --output OUT.json

`--side baseline` => left=ShowO baseline, right=BAGEL
`--side ascr`     => left=ASCR final,    right=BAGEL
"""
import argparse, json
from datetime import datetime
from pathlib import Path

def load_suite(p):
    d = json.loads(Path(p).read_text(encoding='utf-8'))
    results = d.get('results', d)
    if isinstance(results, dict):
        results = list(results.values())
    return results

def bagel_image(r):
    return r.get('bagel_image') or r.get('image') or r.get('baseline_image')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bagel-suite', required=True)
    ap.add_argument('--hard64-suite', required=True)
    ap.add_argument('--side', choices=['baseline','ascr'], required=True)
    ap.add_argument('--swap', action='store_true', help='Swap LEFT/RIGHT (BAGEL on LEFT, ShowO/ASCR on RIGHT) to debias Qwen position preference.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    bagel = {r['prompt']: bagel_image(r) for r in load_suite(args.bagel_suite) if bagel_image(r)}
    hard = {r['prompt']: r for r in load_suite(args.hard64_suite)}

    field = 'baseline_image' if args.side == 'baseline' else 'ascr_final_image'
    label = 'ShowO baseline' if args.side == 'baseline' else 'ASCR final'

    paired, missing = [], []
    for prompt, b_img in sorted(bagel.items()):
        r = hard.get(prompt)
        if not r or not r.get(field):
            missing.append(prompt)
            continue
        if args.swap:
            left_img, right_img = b_img, r[field]   # BAGEL LEFT, ShowO/ASCR RIGHT
        else:
            left_img, right_img = r[field], b_img   # ShowO/ASCR LEFT, BAGEL RIGHT
        paired.append({
            'prompt': prompt,
            'baseline_image': left_img,
            'ascr_final_image': right_img,
        })

    print(f"Paired {len(paired)} prompts; {len(missing)} unmatched ({label} vs BAGEL)")
    if missing[:5]:
        print(' missing examples:', missing[:5])

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': f'bagel_vs_hard64_{args.side}{"_swap" if args.swap else ""}_pairwise_v1',
        'created_at_utc': datetime.utcnow().isoformat()+'Z',
        'bagel_suite': args.bagel_suite,
        'hard64_suite': args.hard64_suite,
        'side': args.side,
        'swap': bool(args.swap),
        'prompt_count': len(paired),
        'results': paired,
    }, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('Wrote', out)

if __name__ == '__main__':
    main()
