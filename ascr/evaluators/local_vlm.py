from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ascr.core.schemas import GridCell, RegionSelection, SemanticEvaluation
from ascr.evaluators.base import SemanticEvaluator


@dataclass
class ColorEvidence:
    color: str
    score: float
    cell: GridCell
    center_x: float
    center_y: float

    def to_dict(self):
        return {'color': self.color, 'score': self.score, 'cell': self.cell.to_dict(), 'center_x': self.center_x, 'center_y': self.center_y}


COLOR_WORDS = {'red': 'red', 'blue': 'blue', 'green': 'green', 'yellow': 'yellow'}


def _image_array(image_path, image_size=512):
    image = Image.open(image_path).convert('RGB').resize((image_size, image_size))
    return np.asarray(image).astype(np.float32) / 255.0


def _dominance(array, color):
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    if color == 'red':
        return np.clip(red - np.maximum(green, blue), 0.0, 1.0)
    if color == 'blue':
        return np.clip(blue - np.maximum(red, green), 0.0, 1.0)
    if color == 'green':
        return np.clip(green - np.maximum(red, blue), 0.0, 1.0)
    if color == 'yellow':
        return np.clip(np.minimum(red, green) - blue, 0.0, 1.0)
    return np.zeros(array.shape[:2], dtype=np.float32)


def color_evidence(image_path, color, grid_size=4, image_size=512):
    scores = _dominance(_image_array(image_path, image_size=image_size), color)
    height, width = scores.shape
    cell_height = height // grid_size
    cell_width = width // grid_size
    best = None
    for row in range(grid_size):
        for col in range(grid_size):
            patch = scores[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width]
            score = float(patch.mean())
            if best is None or score > best.score:
                best = ColorEvidence(color, score, GridCell(row, col), (col + 0.5) / grid_size, (row + 0.5) / grid_size)
    weighted = np.where(scores > max(0.02, best.score * 0.35), scores, 0.0)
    total = float(weighted.sum())
    if total > 0.0:
        ys, xs = np.indices(scores.shape)
        best.center_x = float((weighted * xs).sum() / total / max(1, width - 1))
        best.center_y = float((weighted * ys).sum() / total / max(1, height - 1))
    return best


def _presence_score(evidence):
    return max(0.0, min(1.0, evidence.score / 0.12))


def score_prompt_alignment(prompt, image_path, grid_size=4, image_size=512):
    prompt_lower = prompt.lower()
    colors = [name for word, name in COLOR_WORDS.items() if word in prompt_lower]
    evidences = {color: color_evidence(image_path, color, grid_size=grid_size, image_size=image_size) for color in colors}
    checks = {}
    if colors:
        checks['color_presence'] = float(sum(_presence_score(evidence) for evidence in evidences.values()) / len(evidences))
    if 'red' in evidences and 'blue' in evidences and 'left' in prompt_lower:
        margin = evidences['blue'].center_x - evidences['red'].center_x
        checks['red_left_of_blue'] = 1.0 if margin > 0.08 else max(0.0, min(1.0, (margin + 0.08) / 0.16))
    if not checks:
        return {'score': 0.5, 'supported': False, 'checks': {}, 'evidence': {color: evidence.to_dict() for color, evidence in evidences.items()}}
    return {'score': float(sum(checks.values()) / len(checks)), 'supported': True, 'checks': checks, 'evidence': {color: evidence.to_dict() for color, evidence in evidences.items()}}


class LocalVLMEvaluator(SemanticEvaluator):
    def __init__(self, model_path=None, device='cuda', strict_json=True, backend='heuristic', grid_size=4, image_size=512, pass_threshold=0.62):
        self.model_path = model_path
        self.device = device
        self.strict_json = strict_json
        self.backend = backend
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.pass_threshold = float(pass_threshold)

    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        if self.backend not in {'heuristic', 'image_heuristic'}:
            return SemanticEvaluation.abstain(f'Unsupported local_vlm backend: {self.backend}')
        if not Path(grid_image_path).exists():
            return SemanticEvaluation.abstain(f'Missing image for evaluation: {grid_image_path}')
        alignment = score_prompt_alignment(original_prompt, grid_image_path, grid_size=self.grid_size, image_size=self.image_size)
        if not alignment['supported']:
            return SemanticEvaluation(False, summary='No heuristic evaluator rule matched this prompt.', raw=alignment)
        if alignment['score'] >= self.pass_threshold:
            return SemanticEvaluation(False, summary=f'Heuristic alignment passed with score {alignment["score"]:.3f}.', raw=alignment)
        cells = self._select_cells(original_prompt, alignment)
        regions = [RegionSelection(cells=cells, reason='heuristic semantic mismatch', confidence=max(0.0, 1.0 - alignment['score']), error_type='semantic', action='reopen')]
        return SemanticEvaluation(True, summary=f'Heuristic alignment score {alignment["score"]:.3f} is below threshold {self.pass_threshold:.3f}.', regions=regions, correction_instruction=self._instruction(original_prompt, alignment), raw=alignment)

    def _select_cells(self, prompt, alignment):
        prompt_lower = prompt.lower()
        evidence = alignment.get('evidence', {})
        cells = []
        for color in ('red', 'blue', 'green', 'yellow'):
            item = evidence.get(color)
            if item and item['score'] < 0.084:
                cells.append(GridCell.from_any(item['cell']))
        if 'red' in evidence and 'blue' in evidence and 'left' in prompt_lower and alignment.get('checks', {}).get('red_left_of_blue', 1.0) < 0.9:
            cells.append(GridCell.from_any(evidence['red']['cell']))
            cells.append(GridCell.from_any(evidence['blue']['cell']))
        if not cells:
            middle = self.grid_size // 2
            cells = [GridCell(max(0, middle - 1), max(0, middle - 1)), GridCell(max(0, middle - 1), min(self.grid_size - 1, middle)), GridCell(min(self.grid_size - 1, middle), max(0, middle - 1)), GridCell(min(self.grid_size - 1, middle), min(self.grid_size - 1, middle))]
        unique = []
        seen = set()
        for cell in cells:
            key = (cell.row, cell.col)
            if key not in seen:
                seen.add(key)
                unique.append(cell)
        return unique[:6]

    def _instruction(self, prompt, alignment):
        if 'red_left_of_blue' in alignment.get('checks', {}):
            return 'Regenerate the selected regions so the red object is clearly on the left and the blue object is clearly on the right while preserving the rest of the image.'
        return f'Regenerate the selected regions to better satisfy this prompt: {prompt}'
