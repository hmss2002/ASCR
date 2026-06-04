import unittest

from ascr.cli.compare_stage1_variants import parse_arms


class ParseArmsTest(unittest.TestCase):
    def test_default_both_ordered(self):
        self.assertEqual(parse_arms("coarse,direct"), ["direct", "coarse"])

    def test_single_direct(self):
        self.assertEqual(parse_arms("direct"), ["direct"])

    def test_single_coarse(self):
        self.assertEqual(parse_arms("coarse"), ["coarse"])

    def test_whitespace_and_case(self):
        self.assertEqual(parse_arms(" Direct , COARSE "), ["direct", "coarse"])

    def test_invalid_arm_raises(self):
        with self.assertRaises(ValueError):
            parse_arms("coarse,bogus")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_arms(" , ")


if __name__ == "__main__":
    unittest.main()
