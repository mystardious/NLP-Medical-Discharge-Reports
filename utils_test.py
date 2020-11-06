import unittest

import read_data
from utils import *


class TestUtils(unittest.TestCase):
    def test_remove_empty_array_items(self):

        only_spaces = ["     ", "      "]
        empty_line = ["", "   Sneaky!   "]

        self.assertAlmostEqual(len(remove_empty_array_items(only_spaces)), 0)
        self.assertAlmostEqual(len(remove_empty_array_items(empty_line)), 1)

    def test_clean_header(self):

        def test(string):
            return clean_header(string)

        string1 = "    yes"
        string2 = "    Yes"
        string2_result = "yes"

        self.assertEqual(len(test(string1)), 3, "Test removal of whitespaces")
        self.assertEqual(test(string2), string2_result,
                         "Test if text is lowercase")

    def test_count_all_section_headers(self):

        count_all_section_headers(read_data.read_samples(100))
