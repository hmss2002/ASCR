def summarize_mask(mask):
    return {
        "token_grid_size": mask.token_grid_size,
        "selected_count": mask.count(),
        "selected_indices": mask.selected_indices(),
    }
