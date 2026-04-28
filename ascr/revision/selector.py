from abc import ABC, abstractmethod
from ascr.core.schemas import TokenReopenMask
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
