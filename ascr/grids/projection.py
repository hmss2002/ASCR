from ascr.core.schemas import GridCell, TokenReopenMask


def project_cells_to_token_mask(cells, coarse_grid_size=4, token_grid_size=16, dilation=1):
    if token_grid_size % coarse_grid_size != 0:
        raise ValueError("token_grid_size must be divisible by coarse_grid_size")
    factor = token_grid_size // coarse_grid_size
    selected = set()
    for raw_cell in cells:
        cell = GridCell.from_any(raw_cell, coarse_grid_size)
        start_row = cell.row * factor
        start_col = cell.col * factor
        for row in range(start_row, start_row + factor):
            for col in range(start_col, start_col + factor):
                selected.add((row, col))
    if dilation > 0:
        dilated = set(selected)
        for row, col in selected:
            for delta_row in range(-dilation, dilation + 1):
                for delta_col in range(-dilation, dilation + 1):
                    new_row = row + delta_row
                    new_col = col + delta_col
                    if 0 <= new_row < token_grid_size and 0 <= new_col < token_grid_size:
                        dilated.add((new_row, new_col))
        selected = dilated
    return TokenReopenMask.from_indices(sorted(selected), token_grid_size=token_grid_size)
