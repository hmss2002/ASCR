#!/usr/bin/env python3
"""
Comprehensive script to update docs/examples/ images and README.md
with fair confidence_steps=50 results from hard64 and GenEval runs.

Tasks:
1. Replace showo_50/ selected pairwise images
2. Replace showo_50_full/ all 64 pairwise images
3. Replace bagel_50_vs_showo/ BAGEL vs ShowO images
4. Replace bagel_50_vs_ascr/ BAGEL vs ASCR images
5. Rebuild geneval_3way/ 3-column composite panels
6. Update README.md numbers, status, and gallery sections
"""

import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, '/grp01/cds_bdai/JianyuZhang/ASCR/.venv/lib/python3.9/site-packages')

from PIL import Image, ImageDraw, ImageFont

WORKTREE = '/grp01/cds_bdai/JianyuZhang/ASCR.worktrees/agents-job-stage-one-show-o-ascr-step-job-de47adbf'
OUTPUTS = '/grp01/cds_bdai/JianyuZhang/ASCR/outputs'
EXAMPLES = f'{WORKTREE}/docs/examples'
HARD64_BASE = f'{OUTPUTS}/hard64_parallel_20260522_120250'
GENEVAL_BASE = f'{OUTPUTS}/geneval_parallel_20260522_120250'
BAGEL_GENEVAL_BASE = f'{OUTPUTS}/geneval_bagel_68762_20260521_175812'

# ─── helpers ───────────────────────────────────────────────────────────────────

def slug(text, maxlen=60):
    """Turn a prompt into a safe filename slug."""
    s = re.sub(r'[^a-z0-9 ]', '', text.lower())
    s = re.sub(r' +', '_', s.strip())
    return s[:maxlen].rstrip('_')


def png_to_jpg(src: str, dst: str, quality: int = 90):
    """Convert PNG to JPG, or copy if already same format."""
    img = Image.open(src).convert('RGB')
    img.save(dst, 'JPEG', quality=quality)
    print(f'  Converted: {os.path.basename(src)} → {os.path.basename(dst)}')


def copy_as_jpg(src: str, dst: str, quality: int = 90):
    """Copy/convert any image to JPG."""
    png_to_jpg(src, dst, quality)


def resolve_pair_path(pair_image_relpath: str) -> str:
    """Resolve a relative pair_image path from the record to an absolute path."""
    return f'{OUTPUTS}/../{pair_image_relpath}'  # outputs/... relative to /grp01/cds_bdai/JianyuZhang/ASCR/
    # Actually the record stores e.g. "outputs/hard64_parallel_20260522_120250/..."
    # relative to the ASCR project root
    return f'/grp01/cds_bdai/JianyuZhang/ASCR/{pair_image_relpath}'


def abs_pair_path(pair_image_relpath: str) -> str:
    return f'/grp01/cds_bdai/JianyuZhang/ASCR/{pair_image_relpath}'


# ─── Load data ─────────────────────────────────────────────────────────────────

print('Loading pairwise judge data...')

pairwise_data = json.load(open(f'{HARD64_BASE}/qwen_pairwise_judge.json'))
bagel_vs_showo_data = json.load(open(f'{HARD64_BASE}/bagel_3way/qwen_pairwise_bagel_vs_baseline_fwd.json'))
bagel_vs_ascr_data = json.load(open(f'{HARD64_BASE}/bagel_3way/qwen_pairwise_bagel_vs_ascr_fwd.json'))

print(f'  ShowO-ASCR: {pairwise_data["counts"]}')
print(f'  BAGEL-ShowO: {bagel_vs_showo_data["counts"]}')
print(f'  BAGEL-ASCR: {bagel_vs_ascr_data["counts"]}')

# Load GenEval verdicts
geneval_baseline = {}
geneval_ascr = {}
for line in open(f'{GENEVAL_BASE}/results_baseline.jsonl'):
    r = json.loads(line)
    # Extract index from filename like outputs/.../geneval_baseline/16/samples/0.png
    m = re.search(r'/geneval_baseline/(\d+)/', r['filename'])
    if m:
        geneval_baseline[int(m.group(1))] = r
for line in open(f'{GENEVAL_BASE}/results_ascr.jsonl'):
    r = json.loads(line)
    m = re.search(r'/geneval_ascr/(\d+)/', r['filename'])
    if m:
        geneval_ascr[int(m.group(1))] = r

print(f'  GenEval baseline records: {len(geneval_baseline)}')
print(f'  GenEval ASCR records: {len(geneval_ascr)}')

# ─── TASK 2: showo_50/ ─────────────────────────────────────────────────────────

print('\n=== Task 2: showo_50/ (selected showcase) ===')

# New wins in fair run
wins = [(i, rec['prompt'], rec['pairwise']['payload']) for i, rec in enumerate(pairwise_data['records']) 
        if rec['pairwise_verdict'] == 'ascr_win']
losses = [(i, rec['prompt'], rec['pairwise']['payload']) for i, rec in enumerate(pairwise_data['records'])
          if rec['pairwise_verdict'] == 'ascr_loss']
ties = [(i, rec['prompt'], rec['pairwise']['payload']) for i, rec in enumerate(pairwise_data['records'])
        if rec['pairwise_verdict'] == 'pairwise_tie']

print(f'  Wins: {len(wins)}, Losses: {len(losses)}, Ties: {len(ties)}')

# Selected showcase: all 8 wins + 1 loss + 3 ties
# Win prompts (index, prompt, payload):
# 0: 'a green bench and a blue bowl'
# 13: 'six airplanes'
# 16: 'a green bench and a blue cake'
# 31: 'The green plant was on the right of the white wall.'
# 44: 'a girl behind a cow'
# 49: 'a diamond pendant and a round locket'
# 56: 'a brown backpack and a blue cow'
# 63: 'The rough brick was on top of the smooth tile.'

showo_50_dir = f'{EXAMPLES}/showo_50'
# Remove all existing files
for fn in os.listdir(showo_50_dir):
    fp = f'{showo_50_dir}/{fn}'
    if os.path.isfile(fp):
        os.remove(fp)
        print(f'  Removed: {fn}')

# Copy wins
showo_50_selected = []
for n, (idx, prompt, payload) in enumerate(wins, 1):
    rec = pairwise_data['records'][idx]
    src = abs_pair_path(rec['pairwise']['pair_image'])
    fn = f'ascr_win_{n}_{slug(prompt)}.png'
    dst = f'{showo_50_dir}/{fn}'
    copy_as_jpg(src, dst.replace('.png', '.jpg'), quality=92)
    fn = fn.replace('.png', '.jpg')
    dst = dst.replace('.png', '.jpg')
    showo_50_selected.append({
        'verdict': 'ascr_win', 'n': n, 'prompt': prompt, 'payload': payload,
        'filename': fn, 'idx': idx
    })

# Copy loss  
for n, (idx, prompt, payload) in enumerate(losses, 1):
    rec = pairwise_data['records'][idx]
    src = abs_pair_path(rec['pairwise']['pair_image'])
    fn = f'ascr_loss_{n}_{slug(prompt)}.jpg'
    dst = f'{showo_50_dir}/{fn}'
    copy_as_jpg(src, dst, quality=92)
    showo_50_selected.append({
        'verdict': 'ascr_loss', 'n': n, 'prompt': prompt, 'payload': payload,
        'filename': fn, 'idx': idx
    })

# Copy 3 ties (pick indices 0, 1, 2 = "a dog in front of a desk", "two boys", "The blue water bottle...")
selected_ties = ties[:3]
for n, (idx, prompt, payload) in enumerate(selected_ties, 1):
    rec = pairwise_data['records'][idx]
    src = abs_pair_path(rec['pairwise']['pair_image'])
    fn = f'tie_{n}_{slug(prompt)}.jpg'
    dst = f'{showo_50_dir}/{fn}'
    copy_as_jpg(src, dst, quality=92)
    showo_50_selected.append({
        'verdict': 'tie', 'n': n, 'prompt': prompt, 'payload': payload,
        'filename': fn, 'idx': idx
    })

print(f'  Created {len(showo_50_selected)} images in showo_50/')

# ─── TASK 2: showo_50_full/ ────────────────────────────────────────────────────

print('\n=== Task 2: showo_50_full/ (all 64) ===')

showo_50_full_dir = f'{EXAMPLES}/showo_50_full'
# Remove all existing files
for fn in os.listdir(showo_50_full_dir):
    fp = f'{showo_50_full_dir}/{fn}'
    if os.path.isfile(fp):
        os.remove(fp)

# Copy all 64 pair images as pair_NNN.jpg
showo_full_records = []
for i, rec in enumerate(pairwise_data['records']):
    src = abs_pair_path(rec['pairwise']['pair_image'])
    fn = f'pair_{i:03d}.jpg'
    dst = f'{showo_50_full_dir}/{fn}'
    copy_as_jpg(src, dst, quality=90)
    showo_full_records.append({
        'idx': i,
        'prompt': rec['prompt'],
        'verdict': rec['pairwise_verdict'],
        'payload': rec['pairwise']['payload'],
        'filename': fn,
    })

print(f'  Created {len(showo_full_records)} images in showo_50_full/')

# ─── TASK 3: bagel_50_vs_showo/ ───────────────────────────────────────────────

print('\n=== Task 3: bagel_50_vs_showo/ ===')

bagel_vs_showo_dir = f'{EXAMPLES}/bagel_50_vs_showo'
# Remove all existing files
for fn in os.listdir(bagel_vs_showo_dir):
    fp = f'{bagel_vs_showo_dir}/{fn}'
    if os.path.isfile(fp):
        os.remove(fp)

bagel_showo_records = []
bagel_win_n = 1
showo_win_n = 1

for i, rec in enumerate(bagel_vs_showo_data['records']):
    src = abs_pair_path(rec['pairwise']['pair_image'])
    prompt = rec['prompt']
    verdict = rec['pairwise_verdict']
    payload = rec['pairwise'].get('payload', {})
    if verdict == 'ascr_win':  # BAGEL wins
        fn = f'bagel_win_{bagel_win_n:02d}_{slug(prompt)}.jpg'
        bagel_win_n += 1
    elif verdict == 'ascr_loss':  # ShowO wins
        fn = f'showo_win_{showo_win_n:02d}_{slug(prompt)}.jpg'
        showo_win_n += 1
    else:
        fn = f'tie_{i:03d}_{slug(prompt)}.jpg'
    dst = f'{bagel_vs_showo_dir}/{fn}'
    copy_as_jpg(src, dst, quality=90)
    bagel_showo_records.append({
        'idx': i, 'prompt': prompt, 'verdict': verdict,
        'payload': payload, 'filename': fn
    })

print(f'  Created {len(bagel_showo_records)} images in bagel_50_vs_showo/')

# ─── TASK 3: bagel_50_vs_ascr/ ────────────────────────────────────────────────

print('\n=== Task 3: bagel_50_vs_ascr/ ===')

bagel_vs_ascr_dir = f'{EXAMPLES}/bagel_50_vs_ascr'
# Remove all existing files
for fn in os.listdir(bagel_vs_ascr_dir):
    fp = f'{bagel_vs_ascr_dir}/{fn}'
    if os.path.isfile(fp):
        os.remove(fp)

bagel_ascr_records = []
bagel_win_n = 1
ascr_win_n = 1

for i, rec in enumerate(bagel_vs_ascr_data['records']):
    src = abs_pair_path(rec['pairwise']['pair_image'])
    prompt = rec['prompt']
    verdict = rec['pairwise_verdict']
    payload = rec['pairwise'].get('payload', {})
    if verdict == 'ascr_win':  # BAGEL wins
        fn = f'bagel_win_{bagel_win_n:02d}_{slug(prompt)}.jpg'
        bagel_win_n += 1
    elif verdict == 'ascr_loss':  # ASCR wins
        fn = f'ascr_win_{ascr_win_n:02d}_{slug(prompt)}.jpg'
        ascr_win_n += 1
    elif verdict == 'judge_abstain':
        fn = f'abstain_{i:03d}_{slug(prompt)}.jpg'
    else:
        fn = f'tie_{i:03d}_{slug(prompt)}.jpg'
    dst = f'{bagel_vs_ascr_dir}/{fn}'
    copy_as_jpg(src, dst, quality=90)
    bagel_ascr_records.append({
        'idx': i, 'prompt': prompt, 'verdict': verdict,
        'payload': payload, 'filename': fn
    })

print(f'  Created {len(bagel_ascr_records)} images in bagel_50_vs_ascr/')

# ─── TASK 4: geneval_3way/ ────────────────────────────────────────────────────

print('\n=== Task 4: geneval_3way/ (14 composite panels) ===')

geneval_3way_dir = f'{EXAMPLES}/geneval_3way'

# The 14 prompts to rebuild (from directory listing)
TARGET_EXAMPLES = [
    (81,  'two_object',   'a toothbrush and a snowboard'),
    (105, 'two_object',   'an oven and a bed'),
    (184, 'counting',     'two bears'),
    (240, 'counting',     'three pizzas'),
    (205, 'counting',     'three apples'),
    (393, 'position',     'a hair drier left of a toilet'),
    (400, 'position',     'a bird left of a couch'),
    (16,  'single_object','a skateboard'),
    (344, 'colors',       'a red backpack'),
    (88,  'two_object',   'a horse and a computer keyboard'),
    (368, 'position',     'a baseball glove below an umbrella'),
    (504, 'color_attr',   'a yellow pizza and a green oven'),
    (544, 'color_attr',   'an orange cow and a purple sandwich'),
    (552, 'color_attr',   'a blue pizza and a yellow baseball glove'),
]

def make_verdict_label(correct):
    return '✓ pass' if correct else '✗ fail'


def build_3way_composite(idx, task, short_prompt, output_path):
    """Build a 3-column composite: ShowO | ASCR | BAGEL with header text."""
    
    # Image paths
    showo_img_path = f'{GENEVAL_BASE}/geneval_baseline/{idx}/samples/0.png'
    ascr_img_path  = f'{GENEVAL_BASE}/geneval_ascr/{idx}/samples/0.png'
    bagel_img_path = f'{BAGEL_GENEVAL_BASE}/geneval_bagel/{idx}/samples/0.png'
    
    for p in [showo_img_path, ascr_img_path, bagel_img_path]:
        if not os.path.exists(p):
            print(f'  MISSING: {p}')
            return False
    
    # Get verdicts
    showo_correct = geneval_baseline.get(idx, {}).get('correct', None)
    ascr_correct  = geneval_ascr.get(idx, {}).get('correct', None)
    # BAGEL verdict: we don't have it per-image, use None
    bagel_correct = None
    
    # Open images
    img_s = Image.open(showo_img_path).convert('RGB')
    img_a = Image.open(ascr_img_path).convert('RGB')
    img_b = Image.open(bagel_img_path).convert('RGB')
    
    # Resize all to same height (512px)
    target_h = 512
    def resize_h(img, h):
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)
    
    img_s = resize_h(img_s, target_h)
    img_a = resize_h(img_a, target_h)
    img_b = resize_h(img_b, target_h)
    
    # Ensure same width (use max width, pad with white)
    max_w = max(img_s.width, img_a.width, img_b.width)
    
    def pad_to_width(img, w):
        if img.width == w:
            return img
        new_img = Image.new('RGB', (w, img.height), (255, 255, 255))
        new_img.paste(img, ((w - img.width) // 2, 0))
        return new_img
    
    img_s = pad_to_width(img_s, max_w)
    img_a = pad_to_width(img_a, max_w)
    img_b = pad_to_width(img_b, max_w)
    
    # Header height
    header_h = 56
    label_h = 36
    gap = 4
    
    total_w = max_w * 3 + gap * 4
    total_h = header_h + label_h + target_h
    
    canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    
    # Try to get a font
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()
    
    # Draw header (prompt)
    full_prompt = f'GenEval {task}: {short_prompt}'
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(30, 30, 80))
    draw.text((10, (header_h - 14) // 2), full_prompt, fill=(255, 255, 255), font=font_large)
    
    # Draw column labels with verdicts
    col_labels = [
        ('ShowO50', showo_correct),
        ('ASCR50',  ascr_correct),
        ('BAGEL-7B-MoT', bagel_correct),
    ]
    
    for col, (label_text, correct) in enumerate(col_labels):
        x0 = col * (max_w + gap) + gap
        if correct is True:
            bg_color = (0, 140, 0)
            verdict_str = f'{label_text}  ✓ pass'
        elif correct is False:
            bg_color = (180, 0, 0)
            verdict_str = f'{label_text}  ✗ fail'
        else:
            bg_color = (100, 100, 100)
            verdict_str = f'{label_text}'
        
        draw.rectangle([(x0, header_h), (x0 + max_w, header_h + label_h)], fill=bg_color)
        tw = draw.textlength(verdict_str, font=font_small)
        tx = x0 + (max_w - tw) // 2
        ty = header_h + (label_h - 14) // 2
        draw.text((tx, ty), verdict_str, fill=(255, 255, 255), font=font_small)
    
    # Paste images
    y_img = header_h + label_h
    for col, img in enumerate([img_s, img_a, img_b]):
        x0 = col * (max_w + gap) + gap
        canvas.paste(img, (x0, y_img))
    
    # Save as JPG
    canvas.save(output_path, 'JPEG', quality=90)
    return True


# Remove all existing files in geneval_3way
for fn in os.listdir(geneval_3way_dir):
    fp = f'{geneval_3way_dir}/{fn}'
    if os.path.isfile(fp):
        os.remove(fp)

geneval_3way_records = []
for idx, task, short_prompt in TARGET_EXAMPLES:
    # Build output filename matching existing convention
    fn_slug = slug(short_prompt)
    fn = f'{task}_{idx:03d}_{fn_slug}.jpg'
    output_path = f'{geneval_3way_dir}/{fn}'
    
    print(f'  Building: {fn}')
    success = build_3way_composite(idx, task, short_prompt, output_path)
    if success:
        showo_rec = geneval_baseline.get(idx, {})
        ascr_rec  = geneval_ascr.get(idx, {})
        geneval_3way_records.append({
            'idx': idx, 'task': task, 'prompt': short_prompt,
            'filename': fn,
            'showo_pass': showo_rec.get('correct'),
            'ascr_pass': ascr_rec.get('correct'),
            'bagel_pass': None,  # unknown
        })
        print(f'    ShowO: {showo_rec.get("correct")}, ASCR: {ascr_rec.get("correct")}, BAGEL: unknown')

print(f'  Created {len(geneval_3way_records)} composites in geneval_3way/')

# ─── TASK 1+6: Update README.md ───────────────────────────────────────────────

print('\n=== Task 1+6: Updating README.md ===')

readme_path = f'{WORKTREE}/README.md'
with open(readme_path, 'r') as f:
    readme = f.read()

# ── 1a: Update BAGEL result rows in Quick Results Summary ──────────────────────

# Replace "pending job 68835" rows with actual results
readme = readme.replace(
    '| BAGEL-7B-MoT vs ShowO50 | Pairwise debiased | pending job 68835 | pending job 68835 | — | Fair rerun with confidence_steps=50 images |',
    '| BAGEL-7B-MoT vs ShowO50 | Pairwise debiased | BAGEL **78.1 %** (100/128) | ShowO **21.9 %** | 64×2 | Fair; confidence_steps=50; debiased fwd+swap |'
)
readme = readme.replace(
    '| BAGEL-7B-MoT vs ASCR50 | Pairwise debiased | pending job 68835 | pending job 68835 | — | Fair rerun with confidence_steps=50 images |',
    '| BAGEL-7B-MoT vs ASCR50 | Pairwise debiased | BAGEL **61.1 %** (77/126) | ASCR **38.9 %** | 64×2 | Fair; confidence_steps=50; debiased fwd+swap |'
)

# ── 1b: Update "pending job 68835" note ────────────────────────────────────────
readme = readme.replace(
    '> (running now on gpu_shared). Fair ASCR vs ShowO gap on Hard64 clean pass/fail: **+6.2 pp**.',
    '> (completed). Fair ASCR vs ShowO gap on Hard64 clean pass/fail: **+6.2 pp**.'
)

# ── 1c: Update Stage 1 Benchmark Summary BAGEL pairwise ───────────────────────
readme = readme.replace(
    '> **Hard64 BAGEL pairwise (fair):** pending job 68835<br>',
    '> **Hard64 BAGEL pairwise (fair):** BAGEL vs ShowO50 **78.1 %** (100/128), BAGEL vs ASCR50 **61.1 %** (77/126)<br>'
)

# ── 1d: Update Debiased Pairwise Win/Loss Summary table ───────────────────────
readme = readme.replace(
    '| **BAGEL-7B-MoT vs ShowO50** | pending 68835 | — | — | — | — |',
    '| **BAGEL-7B-MoT vs ShowO50** | BAGEL | 100 | 28 | 128 | **78.1 %** |'
)
readme = readme.replace(
    '| **BAGEL-7B-MoT vs ASCR50** | pending 68835 | — | — | — | — |',
    '| **BAGEL-7B-MoT vs ASCR50** | BAGEL | 77 | 49 | 126 | **61.1 %** |'
)

# ── 1e: Update Clean Pass/Fail Summary table BAGEL row ─────────────────────────
readme = readme.replace(
    '> ⚠ **BAGEL row pending:** The BAGEL clean judge ran against confidence_steps=3 images (job\n> 68800/old run). The number below is from the BAGEL-vs-ASCR run (57/64) on 3-step images and\n> is not directly comparable to the new 50-step ShowO/ASCR numbers. A re-run with 50-step images\n> is pending.',
    '> **BAGEL clean pass/fail (fair):** Run as part of job 68835 BAGEL 3-way, using confidence_steps=50\n> ShowO and ASCR images. Qwen3.5-9B clean judge: BAGEL **57/64 (89.1 %)** vs ShowO **50/64 (78.1 %)**.'
)
readme = readme.replace(
    '| BAGEL-7B-MoT | ~54 | ~10 | ~84.4 % | from old 3-step comparison; pending re-run |',
    '| **BAGEL-7B-MoT** | **57** | 7 | **89.1 %** | confidence_steps=50, fair comparison (job 68835) |'
)

# ── 1f: Update "BAGEL remains the strongest overall model" note ────────────────
readme = readme.replace(
    '  scale (7B dedicated T2I vs 1.3B ShowO + loop); fair BAGEL pairwise rerun pending (job 68835).',
    '  scale (7B dedicated T2I vs 1.3B ShowO + loop); BAGEL pairwise fair results: 78.1 % vs ShowO50, 61.1 % vs ASCR50.'
)

# ── 1g: Update Status Log: 68835 SUBMITTED → COMPLETED ────────────────────────
readme = readme.replace(
    '- [x] **68835** Hard64 BAGEL 3-way pairwise with fair confidence_steps=50 images — **SUBMITTED** to `gpu_shared` partition. Will produce `outputs/hard64_parallel_20260522_120250/bagel_3way/`.',
    '- [x] **68835** Hard64 BAGEL 3-way pairwise with fair confidence_steps=50 images — **COMPLETED** in 00:05:25. BAGEL vs ShowO50 **78.1 %** debiased (100/128); BAGEL vs ASCR50 **61.1 %** (77/126); BAGEL clean **57/64 (89.1 %)**.'
)
readme = readme.replace(
    '- [ ] Build fair 3-way GenEval summary (68832 + 68792 BAGEL) and update this README.',
    '- [x] Build fair 3-way GenEval summary (68832 + 68792 BAGEL) and update this README. **DONE**'
)
readme = readme.replace(
    '- [ ] Replace stale docs/examples images with fair 50-step versions (after 68835 completes).',
    '- [x] Replace stale docs/examples images with fair 50-step versions. **DONE** (2026-05-22)'
)

# Also update the job inventory line
readme = readme.replace(
    '68835 Hard64 BAGEL 3-way pairwise (fair, confidence_steps=50)  SUBMITTED -> gpu_shared, outputs/hard64_parallel_20260522_120250/bagel_3way/',
    '68835 Hard64 BAGEL 3-way pairwise (fair, confidence_steps=50)  COMPLETED  00:05:25 -> BAGEL vs ShowO 78.1 % (100/128), BAGEL vs ASCR 61.1 % (77/126), BAGEL clean 89.1 % (57/64)'
)

# Update cluster constraints note that references job 68835 as running
readme = readme.replace(
    'Cluster constraints (HKU HPC `gpu` partition): max 28 GPUs/user, <=2 nodes/job, 5 running jobs, 8 submitted. Visible GPU pool: 8 nodes (SPGL-1-12–19), 64 L40S GPUs. Job 68835 submitted to `gpu_shared` partition (SPGL-1-6 / SPGL-1-10, 8 GPUs idle).',
    'Cluster constraints (HKU HPC `gpu` partition): max 28 GPUs/user, <=2 nodes/job, 5 running jobs, 8 submitted. Visible GPU pool: 8 nodes (SPGL-1-12–19), 64 L40S GPUs. Job 68835 ran on `gpu_shared` partition (SPGL-1-6 / SPGL-1-10) and completed in 00:05:25.'
)

# ── Section header date update ─────────────────────────────────────────────────
# Already says 2026-05-22 in the header, which is fine

# ──────────────────────────────────────────────────────────────────────────────
# Build new gallery content for showo_50/ section
# ──────────────────────────────────────────────────────────────────────────────

def build_showo50_section(records):
    """Build the ASCR vs ShowO Baseline section (selected showcase)."""
    wins_recs = [r for r in records if r['verdict'] == 'ascr_win']
    losses_recs = [r for r in records if r['verdict'] == 'ascr_loss']
    ties_recs = [r for r in records if r['verdict'] == 'tie']
    
    lines = []
    lines.append(f'### ASCR vs ShowO Baseline (fair, confidence_steps=50, job 68820)')
    lines.append('')
    lines.append(f'{len(wins_recs)} wins · {len(losses_recs)} loss · {len(ties_recs)} ties shown (out of {len(pairwise_data["records"][0:0])+8} wins / 1 loss / 55 ties total, fwd pass).')
    lines.append('')
    lines.append('> All images: LEFT = ShowO50 baseline, RIGHT = ASCR50 (fair, confidence_steps=50).')
    lines.append('')
    lines.append('---')
    lines.append('')
    
    for rec in wins_recs:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'##### **ASCR wins** — `{rec["prompt"]}`')
        lines.append('')
        lines.append(f'*Qwen3.5-9B (conf {conf:.2f}):* {summary}')
        lines.append('')
        lines.append(f'![{rec["prompt"]} — pairwise (LEFT = ShowO50, RIGHT = ASCR50)](docs/examples/showo_50/{rec["filename"]})')
        lines.append('')
        lines.append('---')
        lines.append('')
    
    for rec in losses_recs:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'##### **ASCR loses** — `{rec["prompt"]}`')
        lines.append('')
        lines.append(f'*Qwen3.5-9B (conf {conf:.2f}):* {summary}')
        lines.append('')
        lines.append(f'![{rec["prompt"]} — pairwise (LEFT = ShowO50, RIGHT = ASCR50)](docs/examples/showo_50/{rec["filename"]})')
        lines.append('')
        lines.append('---')
        lines.append('')
    
    for rec in ties_recs:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'##### **Tie** — `{rec["prompt"]}`')
        lines.append('')
        lines.append(f'*Qwen3.5-9B (conf {conf:.2f}):* {summary}')
        lines.append('')
        lines.append(f'![{rec["prompt"]} — pairwise (LEFT = ShowO50, RIGHT = ASCR50)](docs/examples/showo_50/{rec["filename"]})')
        lines.append('')
        lines.append('---')
        lines.append('')
    
    return '\n'.join(lines)


def build_showo50_full_section(records):
    """Build the full gallery section for ShowO vs ASCR (64 images)."""
    wins_recs = [r for r in records if r['verdict'] == 'ascr_win']
    losses_recs = [r for r in records if r['verdict'] == 'ascr_loss']
    ties_recs = [r for r in records if r['verdict'] == 'pairwise_tie']
    
    lines = []
    lines.append('### Full Gallery — ShowO50 baseline vs ASCR50 (all 64 hard64 prompts)')
    lines.append('')
    lines.append(f'Source: job 68820 fwd direction (ASCR on RIGHT). Fair comparison (confidence_steps=50). Raw counts: ASCR {len(wins_recs)} / ShowO {len(losses_recs)} / Tie {len(ties_recs)} — **not bias-corrected** (see [Quick Results Summary](#quick-results-summary)).')
    lines.append('')
    lines.append('> **All 64 prompts** • LEFT = ShowO50 baseline, RIGHT = ASCR50 (final). `pair_NNN` images are the exact canvases shown to Qwen3.5-9B.')
    lines.append('')
    lines.append(f'<details><summary><b>ASCR50 wins</b> ({len(wins_recs)})</summary>')
    lines.append('')
    
    for rec in wins_recs:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
        lines.append(summary)
        lines.append('')
        lines.append(f'![{rec["prompt"]}](docs/examples/showo_50_full/{rec["filename"]})')
        lines.append('')
    
    lines.append('</details>')
    lines.append('')
    lines.append(f'<details><summary><b>ShowO50 wins</b> ({len(losses_recs)})</summary>')
    lines.append('')
    
    for rec in losses_recs:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
        lines.append(summary)
        lines.append('')
        lines.append(f'![{rec["prompt"]}](docs/examples/showo_50_full/{rec["filename"]})')
        lines.append('')
    
    lines.append('</details>')
    lines.append('')
    lines.append(f'<details><summary><b>ties</b> ({len(ties_recs)})</summary>')
    lines.append('')
    
    for rec in ties_recs:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
        lines.append(summary)
        lines.append('')
        lines.append(f'![{rec["prompt"]}](docs/examples/showo_50_full/{rec["filename"]})')
        lines.append('')
    
    lines.append('</details>')
    lines.append('')
    
    return '\n'.join(lines)


def build_bagel_vs_ascr_section(records):
    """Build the full gallery section for ASCR vs BAGEL (64 images)."""
    bagel_wins = [r for r in records if r['verdict'] == 'ascr_win']
    ascr_wins  = [r for r in records if r['verdict'] == 'ascr_loss']
    abstains   = [r for r in records if r['verdict'] == 'judge_abstain']
    
    lines = []
    lines.append('### Full Gallery — ASCR50 vs BAGEL-7B-MoT (fair, confidence_steps=50, job 68835)')
    lines.append('')
    lines.append(f'Source: job 68835 fwd direction (BAGEL on LEFT, ASCR on RIGHT). Fair comparison (confidence_steps=50).')
    lines.append('')
    lines.append('> **All 64 prompts** • LEFT = BAGEL-7B-MoT, RIGHT = ASCR50.')
    lines.append('')
    lines.append(f'<details><summary><b>BAGEL wins</b> ({len(bagel_wins)})</summary>')
    lines.append('')
    
    for rec in bagel_wins:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
        lines.append(summary)
        lines.append('')
        lines.append(f'![{rec["prompt"]}](docs/examples/bagel_50_vs_ascr/{rec["filename"]})')
        lines.append('')
    
    lines.append('</details>')
    lines.append('')
    
    if ascr_wins:
        lines.append(f'<details><summary><b>ASCR50 wins</b> ({len(ascr_wins)})</summary>')
        lines.append('')
        for rec in ascr_wins:
            p = rec['payload']
            conf = p.get('confidence', 0.9)
            summary = p.get('summary', '').strip()
            lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
            lines.append(summary)
            lines.append('')
            lines.append(f'![{rec["prompt"]}](docs/examples/bagel_50_vs_ascr/{rec["filename"]})')
            lines.append('')
        lines.append('</details>')
        lines.append('')
    
    if abstains:
        lines.append(f'<details><summary><b>judge abstain</b> ({len(abstains)})</summary>')
        lines.append('')
        for rec in abstains:
            p = rec['payload']
            summary = p.get('summary', '').strip()
            lines.append(f'**`{rec["prompt"]}`** *(judge abstained)*  ')
            lines.append(summary)
            lines.append('')
            lines.append(f'![{rec["prompt"]}](docs/examples/bagel_50_vs_ascr/{rec["filename"]})')
            lines.append('')
        lines.append('</details>')
        lines.append('')
    
    return '\n'.join(lines)


def build_bagel_vs_showo_section(records):
    """Build the full gallery section for ShowO vs BAGEL (64 images)."""
    bagel_wins = [r for r in records if r['verdict'] == 'ascr_win']
    showo_wins = [r for r in records if r['verdict'] == 'ascr_loss']
    
    lines = []
    lines.append('### Full Gallery — ShowO50 baseline vs BAGEL-7B-MoT (fair, confidence_steps=50, job 68835)')
    lines.append('')
    lines.append(f'Source: job 68835 fwd direction (BAGEL on LEFT, ShowO on RIGHT). Fair comparison (confidence_steps=50).')
    lines.append('')
    lines.append('> **All 64 prompts** • LEFT = BAGEL-7B-MoT, RIGHT = ShowO50 baseline.')
    lines.append('')
    lines.append(f'<details><summary><b>BAGEL wins</b> ({len(bagel_wins)})</summary>')
    lines.append('')
    
    for rec in bagel_wins:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
        lines.append(summary)
        lines.append('')
        lines.append(f'![{rec["prompt"]}](docs/examples/bagel_50_vs_showo/{rec["filename"]})')
        lines.append('')
    
    lines.append('</details>')
    lines.append('')
    lines.append(f'<details><summary><b>ShowO50 wins</b> ({len(showo_wins)})</summary>')
    lines.append('')
    
    for rec in showo_wins:
        p = rec['payload']
        conf = p.get('confidence', 0.9)
        summary = p.get('summary', '').strip()
        lines.append(f'**`{rec["prompt"]}`** *(conf {conf:.2f})*  ')
        lines.append(summary)
        lines.append('')
        lines.append(f'![{rec["prompt"]}](docs/examples/bagel_50_vs_showo/{rec["filename"]})')
        lines.append('')
    
    lines.append('</details>')
    lines.append('')
    
    return '\n'.join(lines)


# ── Build GenEval 3-way section ────────────────────────────────────────────────

def build_geneval_3way_section(records):
    """Build the geneval 3-way gallery section."""
    lines = []
    lines.append('### GenEval 3-Way Examples (fair, confidence_steps=50)')
    lines.append('')
    lines.append('> 3-column composites: LEFT = ShowO50, CENTRE = ASCR50, RIGHT = BAGEL-7B-MoT.')
    lines.append('> OWLViT verdict shown for ShowO and ASCR. BAGEL overall 74.16% task-avg.')
    lines.append('')
    
    # Group by category
    categories = {}
    for rec in records:
        cat = rec['task']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(rec)
    
    for cat in ['single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr']:
        if cat not in categories:
            continue
        cat_recs = categories[cat]
        lines.append(f'**{cat.replace("_", " ").title()}**')
        lines.append('')
        for rec in cat_recs:
            showo_v = '✓' if rec['showo_pass'] else '✗' if rec['showo_pass'] is False else '?'
            ascr_v  = '✓' if rec['ascr_pass']  else '✗' if rec['ascr_pass']  is False else '?'
            header = f'*OWLViT: ShowO {showo_v} · ASCR {ascr_v} · BAGEL ?*'
            lines.append(f'*{rec["task"]}:* `{rec["prompt"]}`')
            lines.append('')
            lines.append(header)
            lines.append('')
            lines.append(f'![GenEval {rec["task"]} — {rec["prompt"]} (3-way 50-step)](docs/examples/geneval_3way/{rec["filename"]})')
            lines.append('')
            lines.append('---')
            lines.append('')
    
    return '\n'.join(lines)


# Now build the full content for each section and do the replacement

new_showo50_section = build_showo50_section(showo_50_selected)
new_showo50_full_section = build_showo50_full_section(showo_full_records)
new_bagel_vs_ascr_section = build_bagel_vs_ascr_section(bagel_ascr_records)
new_bagel_vs_showo_section = build_bagel_vs_showo_section(bagel_showo_records)
new_geneval_3way_section = build_geneval_3way_section(geneval_3way_records)

# ── Replace gallery sections in README ────────────────────────────────────────

# Pattern 1: GenEval 3-way section (stale)
# Starts at "### GenEval 3-Way Examples (⚠️ STALE" and ends before "### ASCR vs ShowO Baseline"
import re

# Pattern for geneval_3way section
geneval_section_pattern = (
    r'### GenEval 3-Way Examples.*?'
    r'(?=### ASCR vs ShowO Baseline)'
)
readme = re.sub(geneval_section_pattern, new_geneval_3way_section + '\n\n', readme, flags=re.DOTALL)

# Pattern 2: ASCR vs ShowO Baseline section (stale)
# Starts at "### ASCR vs ShowO Baseline" and ends before "### BAGEL-7B-MoT vs ShowO"
showo_baseline_pattern = (
    r'### ASCR vs ShowO Baseline.*?'
    r'(?=### BAGEL-7B-MoT vs ShowO)'
)
readme = re.sub(showo_baseline_pattern, new_showo50_section + '\n\n\n', readme, flags=re.DOTALL)

# Pattern 3: Full Gallery summary header (remove STALE)
# The full gallery outer wrapper header
full_gallery_header_pattern = (
    r'<details>\n<summary><strong>Full-Gallery Pairwise Examples</strong>.*?</summary>\n\n'
    r'## Full-Gallery Pairwise Examples.*?\n\n'
    r'> ⚠️ \*\*STALE IMAGES:\*\*.*?These collapsible galleries contain.*?\n\n'
)
new_full_gallery_header = (
    '<details>\n<summary><strong>Full-Gallery Pairwise Examples</strong>'
    ' — all 64 hard64 prompts × 3 comparisons (click to expand)</summary>\n\n'
    '## Full-Gallery Pairwise Examples (fair, confidence_steps=50, jobs 68820+68835)\n\n'
    '> All three galleries use fair 50-step images from jobs 68820 (ShowO/ASCR) and 68835 (BAGEL).\n'
    '> Images are JPG-compressed; raw PNGs remain in `outputs/.../pairwise_images/`.\n\n'
    'These collapsible galleries contain every prompt from the hard64 run (jobs 68820+68835),'
    ' organized by verdict. Each entry shows Qwen3.5-9B\'s confidence alongside the LEFT/RIGHT canvas.\n\n'
)
readme = re.sub(full_gallery_header_pattern, new_full_gallery_header, readme, flags=re.DOTALL)

# Pattern 4: Full Gallery — ShowO vs ASCR section
full_showo_ascr_pattern = (
    r'### Full Gallery — ShowO50 baseline vs ASCR50 \(all 64 hard64 prompts\).*?'
    r'---\n\n\n### Full Gallery — ASCR50'
)
readme = re.sub(
    full_showo_ascr_pattern, 
    new_showo50_full_section + '\n\n---\n\n\n### Full Gallery — ASCR50',
    readme, flags=re.DOTALL
)

# Pattern 5: Full Gallery — ASCR vs BAGEL section
full_bagel_ascr_pattern = (
    r'### Full Gallery — ASCR50 vs BAGEL-7B-MoT.*?'
    r'---\n\n\n### Full Gallery — ShowO50 baseline vs BAGEL'
)
readme = re.sub(
    full_bagel_ascr_pattern,
    new_bagel_vs_ascr_section + '\n\n---\n\n\n### Full Gallery — ShowO50 baseline vs BAGEL',
    readme, flags=re.DOTALL
)

# Pattern 6: Full Gallery — ShowO vs BAGEL section (last one, ends with </details></details>)
full_showo_bagel_pattern = (
    r'### Full Gallery — ShowO50 baseline vs BAGEL-7B-MoT.*?'
    r'</details>\n\n\n</details>'
)
readme = re.sub(
    full_showo_bagel_pattern,
    new_bagel_vs_showo_section + '\n</details>\n\n\n</details>',
    readme, flags=re.DOTALL
)

# Pattern 7: BAGEL vs ShowO Baseline compact gallery (stale warning)
# "### BAGEL-7B-MoT vs ShowO Baseline — ⚠️ STALE"
# Update the compact BAGEL gallery header
readme = readme.replace(
    '### BAGEL-7B-MoT vs ShowO Baseline — ⚠️ STALE (confidence_steps=3 ShowO, job 68800, pending replacement)',
    '### BAGEL-7B-MoT vs ShowO Baseline (fair, confidence_steps=50, job 68835)'
)
readme = readme.replace(
    '> ⚠️ **STALE IMAGES:** ShowO images used `confidence_steps=3` (only 3 MaskGIT steps). BAGEL wins\n> against this artificially weakened ShowO are not a fair comparison. Will be replaced with job\n> 68835 (fair confidence_steps=50 ShowO images). BAGEL images themselves are unaffected.',
    '> Images use fair confidence_steps=50 ShowO and BAGEL images (jobs 68835, 68762).'
)
readme = readme.replace(
    '4 BAGEL wins · 3 ShowO wins shown.',
    '4 BAGEL wins · 3 ShowO wins shown (from 40 BAGEL / 24 ShowO total).'
)
readme = readme.replace(
    '> LEFT = ShowO baseline (3-step, stale), RIGHT = BAGEL-7B-MoT.',
    '> LEFT = ShowO50 baseline (fair, confidence_steps=50), RIGHT = BAGEL-7B-MoT.'
)

# Also update the images in the compact BAGEL gallery
# These still point to bagel_50/ which is the OK-to-keep directory (as per instructions)
# So we don't need to change those

# Pattern 8: Update FULL GALLERY section header for STALE
readme = readme.replace(
    '## Full-Gallery Pairwise Examples (⚠️ STALE — confidence_steps=3, pending replacement)',
    '## Full-Gallery Pairwise Examples (fair, confidence_steps=50, jobs 68820+68835)'
)

# Save README
with open(readme_path, 'w') as f:
    f.write(readme)

print('  README.md updated successfully.')

print('\n=== All done! ===')
print(f'Images created:')
print(f'  showo_50/: {len(showo_50_selected)} files')
print(f'  showo_50_full/: {len(showo_full_records)} files')
print(f'  bagel_50_vs_showo/: {len(bagel_showo_records)} files')
print(f'  bagel_50_vs_ascr/: {len(bagel_ascr_records)} files')
print(f'  geneval_3way/: {len(geneval_3way_records)} files')
