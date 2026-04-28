from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationState:
    prompt: str
    iteration: int
    token_grid: List[List[int]]
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    prompt: str
    decoded_image: str
    grid_image: str
    evaluation_path: str
    mask_path: str
    correction_prompt_path: Optional[str]
    selected_token_count: int
    stop_reason: Optional[str] = None

    def to_dict(self):
        return self.__dict__.copy()
