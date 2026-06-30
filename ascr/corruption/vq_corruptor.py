"""Controlled corruption operators for Lumina VQ image-token grids.

These helpers are intentionally model-light: they operate on flat ``vq_ids``
lists and do not load Lumina weights. GPU scripts can use them after generation
to create clean/corrupted pairs for Stage-3 self-supervised repair studies.
"""

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Sequence, Tuple

from ascr.core.schemas import GridCell
from ascr.generators.lumina_native import CODEBOOK_SIZE, IMAGE_TOKEN_OFFSET, MASK_TOKEN_ID, NEWLINE_TOKEN_ID


RAW_TOKEN_SPACE = "raw_codebook"
OFFSET_TOKEN_SPACE = "offset_codebook"


@dataclass(frozen=True)
class CorruptionResult:
    clean_vq_ids: List[int]
    corrupted_vq_ids: List[int]
    selected_indices: List[Tuple[int, int]]
    corruption_type: str
    token_grid_size: int
    token_id_space: str
    changed_count: int
    mask_size: Optional[int] = None
    operator: Optional[str] = None
    source_indices: Optional[List[Tuple[int, int]]] = None
    source_mode: Optional[str] = None

    def to_metadata(self):
        payload = {
            "corruption_type": self.corruption_type,
            "token_grid_size": self.token_grid_size,
            "token_id_space": self.token_id_space,
            "selected_indices": list(self.selected_indices),
            "selected_count": len(self.selected_indices),
            "changed_count": self.changed_count,
        }
        if self.mask_size is not None:
            payload["mask_size"] = int(self.mask_size)
        if self.operator:
            payload["operator"] = self.operator
        if self.source_indices is not None:
            payload["source_indices"] = list(self.source_indices)
        if self.source_mode:
            payload["source_mode"] = self.source_mode
        return payload


def validate_vq_ids(vq_ids: Sequence[int], token_grid_size: int):
    grid_size = int(token_grid_size)
    expected = grid_size * grid_size
    if len(vq_ids) != expected:
        raise ValueError(f"Expected {expected} vq ids for {grid_size}x{grid_size}, got {len(vq_ids)}")
    return [int(value) for value in vq_ids]


def infer_token_id_space(vq_ids: Sequence[int]):
    """Infer whether VQ ids are raw codebook ids or Lumina offset-space ids."""
    specials = {MASK_TOKEN_ID, NEWLINE_TOKEN_ID}
    values = [int(value) for value in vq_ids if int(value) not in specials]
    if not values:
        return OFFSET_TOKEN_SPACE
    offset_count = sum(IMAGE_TOKEN_OFFSET <= value < IMAGE_TOKEN_OFFSET + CODEBOOK_SIZE for value in values)
    raw_count = sum(0 <= value < CODEBOOK_SIZE for value in values)
    if offset_count >= raw_count and offset_count > 0:
        return OFFSET_TOKEN_SPACE
    if raw_count > 0:
        return RAW_TOKEN_SPACE
    if any(value >= IMAGE_TOKEN_OFFSET for value in values):
        return OFFSET_TOKEN_SPACE
    return RAW_TOKEN_SPACE


def _random_codebook_token(original: int, rng: random.Random, token_id_space: str):
    offset = IMAGE_TOKEN_OFFSET if token_id_space == OFFSET_TOKEN_SPACE else 0
    replacement = offset + rng.randrange(CODEBOOK_SIZE)
    if replacement == int(original):
        replacement = offset + ((replacement - offset + 1) % CODEBOOK_SIZE)
    return replacement


def _block_indices(token_grid_size: int, block_size: int, rng: random.Random):
    grid_size = int(token_grid_size)
    size = int(block_size)
    if size < 1 or size > grid_size:
        raise ValueError(f"Invalid block_size={block_size} for token_grid_size={token_grid_size}")
    start_row = rng.randrange(0, grid_size - size + 1)
    start_col = rng.randrange(0, grid_size - size + 1)
    return [(row, col) for row in range(start_row, start_row + size) for col in range(start_col, start_col + size)]


def block_indices_from_origin(row: int, col: int, block_size: int):
    size = int(block_size)
    return [(r, c) for r in range(int(row), int(row) + size) for c in range(int(col), int(col) + size)]


def choose_token_mask(token_grid_size: int, mask_size: int, rng: random.Random):
    return _block_indices(token_grid_size, mask_size, rng)


def choose_corruption_indices(corruption_type: str, token_grid_size: int, rng: random.Random):
    kind = str(corruption_type)
    if kind == "single_random_replace":
        return _block_indices(token_grid_size, 1, rng)
    if kind == "block_2x2_random_replace":
        return _block_indices(token_grid_size, 2, rng)
    if kind == "block_4x4_random_replace":
        return _block_indices(token_grid_size, 4, rng)
    if kind == "local_shuffle_2x2":
        return _block_indices(token_grid_size, 2, rng)
    if kind == "local_shuffle_4x4":
        return _block_indices(token_grid_size, 4, rng)
    raise ValueError(f"Unsupported corruption_type: {corruption_type}")


def _normalise_indices(indices: Iterable[Tuple[int, int]], token_grid_size: int):
    grid_size = int(token_grid_size)
    normalised = []
    seen = set()
    for row, col in indices:
        row = int(row)
        col = int(col)
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError(f"Corruption index out of range: {(row, col)} for grid {grid_size}")
        if (row, col) not in seen:
            normalised.append((row, col))
            seen.add((row, col))
    if not normalised:
        raise ValueError("Corruption requires at least one selected token")
    return normalised


def _flat_index(row: int, col: int, token_grid_size: int):
    return int(row) * int(token_grid_size) + int(col)


def _block_origin(indices: Iterable[Tuple[int, int]]):
    values = list(indices)
    if not values:
        raise ValueError("Cannot infer block origin from empty indices")
    return min(row for row, _col in values), min(col for _row, col in values)


def _blocks_overlap(origin_a, origin_b, block_size: int):
    arow, acol = origin_a
    brow, bcol = origin_b
    size = int(block_size)
    return not (
        arow + size <= brow
        or brow + size <= arow
        or acol + size <= bcol
        or bcol + size <= acol
    )


def _source_block_indices(token_grid_size: int, mask_size: int, selected, rng: random.Random, mode: str):
    grid_size = int(token_grid_size)
    size = int(mask_size)
    selected_origin = _block_origin(selected)
    max_origin = grid_size - size
    candidates = []
    if mode == "neighbor":
        row0, col0 = selected_origin
        offsets = [
            (-size, 0),
            (size, 0),
            (0, -size),
            (0, size),
            (-size, -size),
            (-size, size),
            (size, -size),
            (size, size),
        ]
        for drow, dcol in offsets:
            origin = row0 + drow, col0 + dcol
            if 0 <= origin[0] <= max_origin and 0 <= origin[1] <= max_origin:
                candidates.append(origin)
    else:
        for row in range(0, max_origin + 1):
            for col in range(0, max_origin + 1):
                origin = (row, col)
                if _blocks_overlap(selected_origin, origin, size):
                    continue
                if mode == "far" and max(abs(row - selected_origin[0]), abs(col - selected_origin[1])) < size:
                    continue
                candidates.append(origin)
    if not candidates:
        for row in range(0, max_origin + 1):
            for col in range(0, max_origin + 1):
                origin = (row, col)
                if not _blocks_overlap(selected_origin, origin, size):
                    candidates.append(origin)
    if not candidates:
        raise ValueError(f"Could not choose non-overlapping source block for mask_size={mask_size}")
    origin = rng.choice(candidates)
    return block_indices_from_origin(origin[0], origin[1], size)


def _normalise_operator(value):
    operator = str(value or "random_replace").strip().lower().replace("-", "_")
    aliases = {
        "replace": "random_replace",
        "random": "random_replace",
        "shuffle": "local_shuffle",
        "copy_neighbor": "neighbor_copy",
        "patch_copy": "neighbor_copy",
        "copy_patch": "neighbor_copy",
        "same_image_transplant": "transplant",
        "patch_transplant": "transplant",
    }
    operator = aliases.get(operator, operator)
    if operator not in {"random_replace", "local_shuffle", "neighbor_copy", "transplant"}:
        raise ValueError(f"Unsupported corruption operator: {value}")
    return operator


def corrupt_vq_ids_with_operator(
    vq_ids: Sequence[int],
    token_grid_size: int,
    mask_size: int,
    operator: str,
    seed: Optional[int] = None,
    selected_indices: Optional[Iterable[Tuple[int, int]]] = None,
    token_id_space: Optional[str] = None,
):
    """Corrupt a token block using the canonical Stage-3 repair operator API."""
    clean = validate_vq_ids(vq_ids, token_grid_size)
    rng = random.Random(seed)
    size = int(mask_size)
    op = _normalise_operator(operator)
    if op == "local_shuffle" and size <= 1:
        raise ValueError("local_shuffle requires mask_size >= 2")
    selected = (
        _normalise_indices(selected_indices, token_grid_size)
        if selected_indices is not None
        else choose_token_mask(token_grid_size, size, rng)
    )
    id_space = token_id_space or infer_token_id_space(clean)
    corrupted = list(clean)
    source_indices = None
    source_mode = None
    if op == "local_shuffle":
        positions = [_flat_index(row, col, token_grid_size) for row, col in selected]
        values = [corrupted[position] for position in positions]
        shuffled = list(values)
        rng.shuffle(shuffled)
        if shuffled == values and len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]
        for position, value in zip(positions, shuffled):
            corrupted[position] = value
    elif op == "random_replace":
        for row, col in selected:
            position = _flat_index(row, col, token_grid_size)
            corrupted[position] = _random_codebook_token(corrupted[position], rng, id_space)
    else:
        source_mode = "same_image_neighbor" if op == "neighbor_copy" else "same_image_far"
        source_indices = _source_block_indices(
            token_grid_size,
            size,
            selected,
            rng,
            mode="neighbor" if op == "neighbor_copy" else "far",
        )
        for (dst_row, dst_col), (src_row, src_col) in zip(selected, source_indices):
            corrupted[_flat_index(dst_row, dst_col, token_grid_size)] = clean[_flat_index(src_row, src_col, token_grid_size)]
    changed_count = sum(1 for before, after in zip(clean, corrupted) if before != after)
    return CorruptionResult(
        clean_vq_ids=clean,
        corrupted_vq_ids=corrupted,
        selected_indices=selected,
        corruption_type=f"{op}_{size}x{size}",
        token_grid_size=int(token_grid_size),
        token_id_space=id_space,
        changed_count=changed_count,
        mask_size=size,
        operator=op,
        source_indices=source_indices,
        source_mode=source_mode,
    )


def corrupt_vq_ids(
    vq_ids: Sequence[int],
    token_grid_size: int,
    corruption_type: str,
    seed: Optional[int] = None,
    selected_indices: Optional[Iterable[Tuple[int, int]]] = None,
    token_id_space: Optional[str] = None,
):
    """Return a controlled corrupted copy of ``vq_ids``.

    ``random_replace`` operators replace selected token ids with valid codebook
    ids in the same raw/offset id space as the input. ``local_shuffle`` operators
    permute the selected token values inside one local block.
    """
    clean = validate_vq_ids(vq_ids, token_grid_size)
    rng = random.Random(seed)
    selected = (
        _normalise_indices(selected_indices, token_grid_size)
        if selected_indices is not None
        else choose_corruption_indices(corruption_type, token_grid_size, rng)
    )
    id_space = token_id_space or infer_token_id_space(clean)
    corrupted = list(clean)
    if str(corruption_type).startswith("local_shuffle"):
        positions = [_flat_index(row, col, token_grid_size) for row, col in selected]
        values = [corrupted[position] for position in positions]
        shuffled = list(values)
        rng.shuffle(shuffled)
        if shuffled == values and len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]
        for position, value in zip(positions, shuffled):
            corrupted[position] = value
    else:
        for row, col in selected:
            position = _flat_index(row, col, token_grid_size)
            corrupted[position] = _random_codebook_token(corrupted[position], rng, id_space)
    changed_count = sum(1 for before, after in zip(clean, corrupted) if before != after)
    return CorruptionResult(
        clean_vq_ids=clean,
        corrupted_vq_ids=corrupted,
        selected_indices=selected,
        corruption_type=str(corruption_type),
        token_grid_size=int(token_grid_size),
        token_id_space=id_space,
        changed_count=changed_count,
    )


def token_indices_to_cell_labels(indices: Iterable[Tuple[int, int]], token_grid_size: int, grid_size: int):
    """Project token indices to selector-grid labels.

    For grids up to 26 rows this returns familiar labels such as ``B2``. For
    larger grids it returns the numeric ``R{row}C{col}`` form used by direct-token
    evaluators.
    """
    token_grid_size = int(token_grid_size)
    grid_size = int(grid_size)
    if token_grid_size % grid_size != 0:
        raise ValueError("token_grid_size must be divisible by grid_size")
    factor = token_grid_size // grid_size
    cells = sorted({(int(row) // factor, int(col) // factor) for row, col in indices})
    if grid_size <= 26:
        return [GridCell(row, col).to_label() for row, col in cells]
    return [f"R{row}C{col}" for row, col in cells]
