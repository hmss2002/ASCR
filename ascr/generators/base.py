from abc import ABC, abstractmethod
from pathlib import Path
from ascr.core.state import GenerationState


class GeneratorAdapter(ABC):
    @abstractmethod
    def initialize(self, prompt, artifacts):
        raise NotImplementedError

    @abstractmethod
    def decode(self, state, output_path):
        raise NotImplementedError

    @abstractmethod
    def reopen_and_continue(self, state, mask, correction_prompt, artifacts):
        raise NotImplementedError


def _write_mock_ppm(path, token_grid, image_size=256):
    path = Path(path)
    lines = ["P3", f"{image_size} {image_size}", "255"]
    token_size = len(token_grid)
    step = max(1, image_size // token_size)
    for row in range(image_size):
        pixels = []
        for col in range(image_size):
            token_row = min(token_size - 1, row // step)
            token_col = min(token_size - 1, col // step)
            value = token_grid[token_row][token_col] % 255
            pixels.append(f"{40 + value % 120} {70 + (value * 3) % 120} {110 + (value * 7) % 120}")
        lines.append(" ".join(pixels))
    path.write_text(chr(10).join(lines) + chr(10), encoding="ascii")
    return path


class MockGeneratorAdapter(GeneratorAdapter):
    def __init__(self, token_grid_size=16, image_size=256):
        self.token_grid_size = token_grid_size
        self.image_size = image_size

    def initialize(self, prompt, artifacts):
        grid = [[(row * self.token_grid_size + col) % 127 for col in range(self.token_grid_size)] for row in range(self.token_grid_size)]
        return GenerationState(prompt=prompt, iteration=0, token_grid=grid, metadata={"generator": "mock"})

    def decode(self, state, output_path):
        path = _write_mock_ppm(output_path, state.token_grid, image_size=self.image_size)
        state.image_path = str(path)
        return state

    def reopen_and_continue(self, state, mask, correction_prompt, artifacts):
        next_grid = [row[:] for row in state.token_grid]
        for row, col in mask.selected_indices():
            next_grid[row][col] = (next_grid[row][col] + 37) % 255
        return GenerationState(prompt=correction_prompt, iteration=state.iteration + 1, token_grid=next_grid, metadata={"generator": "mock", "reopened_tokens": mask.count()})
