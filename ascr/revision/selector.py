from abc import ABC, abstractmethod
from ascr.core.schemas import GridCell, TokenReopenMask
from ascr.grids.projection import project_cells_to_token_mask


class SemanticReopeningSelector(ABC):
    @abstractmethod
    def select(self, evaluation):
        raise NotImplementedError


class GridSemanticReopeningSelector(SemanticReopeningSelector):
    def __init__(self, coarse_grid_size=4, token_grid_size=16, dilation=1):
        self.coarse_grid_size = coarse_grid_size
        self.token_grid_size = token_grid_size
        self.dilation = dilation

    def select(self, evaluation):
        cells = []
        for region in evaluation.actionable_regions():
            cells.extend(region.cells)
        if not cells:
            return TokenReopenMask.empty(self.token_grid_size)
        return project_cells_to_token_mask(cells, self.coarse_grid_size, self.token_grid_size, self.dilation)


def _dilate_indices(indices, token_grid_size, dilation):
    if dilation <= 0:
        return set(indices)
    dilated = set(indices)
    for row, col in list(indices):
        for delta_row in range(-dilation, dilation + 1):
            for delta_col in range(-dilation, dilation + 1):
                new_row = row + delta_row
                new_col = col + delta_col
                if 0 <= new_row < token_grid_size and 0 <= new_col < token_grid_size:
                    dilated.add((new_row, new_col))
    return dilated


class DirectTokenReopeningSelector(SemanticReopeningSelector):
    """Map evaluator-selected cells straight onto the token grid.

    Unlike :class:`GridSemanticReopeningSelector`, this selector treats each
    evaluation cell as a discrete image-token coordinate at ``select_grid_size``
    resolution (e.g. 0-31 for a 32x32 token grid). No coarse-to-fine projection
    or upsampling is performed; ``dilation`` defaults to 0 so exactly the chosen
    tokens are reopened. When ``select_grid_size`` is smaller than
    ``token_grid_size`` the cells are scaled by the integer factor so an
    intermediate granularity can still be experimented with.
    """

    def __init__(self, token_grid_size=32, select_grid_size=None, dilation=0):
        self.token_grid_size = int(token_grid_size)
        self.select_grid_size = int(select_grid_size) if select_grid_size else int(token_grid_size)
        self.dilation = int(dilation)
        if self.token_grid_size % self.select_grid_size != 0:
            raise ValueError("token_grid_size must be divisible by select_grid_size")

    def select(self, evaluation):
        cells = []
        for region in evaluation.actionable_regions():
            cells.extend(region.cells)
        if not cells:
            return TokenReopenMask.empty(self.token_grid_size)
        factor = self.token_grid_size // self.select_grid_size
        selected = set()
        for raw_cell in cells:
            cell = GridCell.from_any(raw_cell, self.select_grid_size)
            start_row = cell.row * factor
            start_col = cell.col * factor
            for row in range(start_row, start_row + factor):
                for col in range(start_col, start_col + factor):
                    selected.add((row, col))
        selected = _dilate_indices(selected, self.token_grid_size, self.dilation)
        return TokenReopenMask.from_indices(sorted(selected), token_grid_size=self.token_grid_size)
