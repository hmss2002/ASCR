from ascr.evaluators.local_vlm import score_prompt_alignment


def semantic_improvement(before_score, after_score):
    return after_score - before_score


def collateral_damage(before_preservation, after_preservation):
    return before_preservation - after_preservation


def score_image(prompt, image_path, grid_size=4, image_size=512):
    return score_prompt_alignment(prompt, image_path, grid_size=grid_size, image_size=image_size)


def compare_scores(baseline, candidate, tolerance=0.03):
    delta = semantic_improvement(float(baseline['score']), float(candidate['score']))
    if delta > tolerance:
        verdict = 'ascr_improved'
    elif delta < -tolerance:
        verdict = 'ascr_regressed'
    else:
        verdict = 'tie_or_unclear'
    return {'baseline_score': float(baseline['score']), 'ascr_score': float(candidate['score']), 'delta': delta, 'verdict': verdict}
