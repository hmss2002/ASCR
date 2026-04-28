from abc import ABC, abstractmethod


class SemanticEvaluator(ABC):
    @abstractmethod
    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        raise NotImplementedError
