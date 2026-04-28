import tempfile
import unittest

from ascr.core.loop import ASCRLoop, ASCRRunConfig
from ascr.core.schemas import SemanticEvaluation
from ascr.core.state import GenerationState
from ascr.generators.base import MockGeneratorAdapter
from ascr.revision.selector import GridSemanticReopeningSelector


class NoInitializeMockGenerator(MockGeneratorAdapter):
    def initialize(self, prompt, artifacts):
        raise AssertionError('initialize should not be called when initial_state is supplied')


class PassingEvaluator:
    def evaluate(self, original_prompt, grid_image_path, iteration, current_prompt=None):
        return SemanticEvaluation(False, summary='ok')


class LoopInitialStateTests(unittest.TestCase):
    def test_run_uses_supplied_initial_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = NoInitializeMockGenerator(token_grid_size=4, image_size=64)
            loop = ASCRLoop(
                generator,
                PassingEvaluator(),
                GridSemanticReopeningSelector(coarse_grid_size=4, token_grid_size=4, dilation=0),
                ASCRRunConfig(run_name='test_initial_state', max_iterations=1, image_size=64, token_grid_size=4, output_dir=temp_dir),
            )
            initial_state = GenerationState(prompt='prompt', iteration=0, token_grid=[[7 for _ in range(4)] for _ in range(4)], metadata={'source': 'test'})
            summary = loop.run('prompt', project_root=temp_dir, initial_state=initial_state)
            self.assertTrue(summary['started_from_initial_state'])
            self.assertEqual(summary['stop_reason'], 'no_semantic_error')


if __name__ == '__main__':
    unittest.main()
