from ascr.evaluators.mock import MockSemanticEvaluator
from ascr.evaluators.local_vlm import LocalVLMEvaluator


def build_evaluator(name, config):
    name = (name or 'mock').lower()
    config = config or {}
    if name == 'mock':
        return MockSemanticEvaluator()
    if name in {'local_vlm', 'local-vlm'}:
        evaluator_config = config.get('evaluator', config)
        return LocalVLMEvaluator(
            model_path=evaluator_config.get('model_path'),
            device=evaluator_config.get('device', 'cuda'),
            strict_json=bool(evaluator_config.get('strict_json', True)),
            backend=evaluator_config.get('backend', 'heuristic'),
            grid_size=int(config.get('coarse_grid_size', evaluator_config.get('grid_size', 4))),
            image_size=int(config.get('image_size', evaluator_config.get('image_size', 512))),
            pass_threshold=float(evaluator_config.get('pass_threshold', 0.62)),
        )
    raise ValueError(f'Unknown evaluator: {name}')
