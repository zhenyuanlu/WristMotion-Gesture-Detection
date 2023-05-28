"""
test_process_bioVid.py

Test the process_bioVid function in utils.py
"""

import unittest
import os
import numpy as np
from ..utils.process_pison import process_pison


class TestProcessBioVid(unittest.TestCase):
    def test_process_bioVid(self):
        # Directories for test data and output
        data_dir = r'Z:\Pison\pison_movement\data'
        output_dir = r'Z:\Pison\pison_movement\data\processed'

        # Call the process_bioVid function with test data and output directories
        process_pison(data_dir, output_dir)

        # Check if output files are created
        output_files = os.listdir(output_dir)
        self.assertGreater(len(output_files), 0, "No output files were created.")

        # Check if output files have the expected format
        for filename in output_files:
            file_path = os.path.join(output_dir, filename)
            with np.load(file_path) as data:
                self.assertIn("x", data, "The 'x' key is missing in the output file.")
                self.assertIn("y", data, "The 'y' key is missing in the output file.")
                self.assertEqual(data["x"].shape[0], data["y"].shape[0], "The number of samples in 'x' and 'y' "
                                                                         "should be the same.")


if __name__ == "__main__":
    unittest.main()
