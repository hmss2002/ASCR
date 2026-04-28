import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from ascr.evaluators.local_vlm import LocalVLMEvaluator, score_prompt_alignment


class LocalVLMEvaluatorTests(unittest.TestCase):
    def test_scores_red_left_of_blue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'image.png'
            image = Image.new('RGB', (128, 128), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 40, 48, 88), fill=(230, 20, 20))
            draw.ellipse((80, 40, 120, 88), fill=(20, 20, 230))
            image.save(path)
            score = score_prompt_alignment('A red cube left of a blue sphere', path, image_size=128)
            self.assertTrue(score['supported'])
            self.assertGreater(score['score'], 0.8)

    def test_flags_wrong_spatial_relation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'image.png'
            image = Image.new('RGB', (128, 128), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((80, 40, 120, 88), fill=(230, 20, 20))
            draw.ellipse((8, 40, 48, 88), fill=(20, 20, 230))
            image.save(path)
            evaluator = LocalVLMEvaluator(image_size=128, pass_threshold=0.7)
            evaluation = evaluator.evaluate('A red cube left of a blue sphere', path, 0)
            self.assertTrue(evaluation.has_error)
            self.assertTrue(evaluation.regions)


if __name__ == '__main__':
    unittest.main()
