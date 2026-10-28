#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of culebra.
#
# Culebra is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Culebra. If not, see <http://www.gnu.org/licenses/>.
#
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Unit test for :class:`culebra.tools.scripts.BatchResultsViewer`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from os import remove

from pandas import DataFrame

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.tools import Results
from culebra.tools.evaluation import Batch
from culebra.tools.scripts import BatchResultsViewer


# Serialized results file
results_obj = "my_results" + SERIALIZED_FILE_EXTENSION

# Key
key = "training_stats"


class BatchResultsViewerTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.BatchResultsViewer`."""

    @classmethod
    def setUpClass(cls):
        """Create the results."""
        results = Results()

        for key in Batch.ResultsKeys:
            results[key.value] = DataFrame(columns=[key.value])

        for file in (results_obj, BatchResultsViewer.default_results_obj):
            results.dump(file)

    @classmethod
    def tearDownClass(cls):
        """Remove the serialized results files."""
        for file in (results_obj, BatchResultsViewer.default_results_obj):
            if Path(file).exists():
                remove(file)

    def test_parse_args(self):
        """Test the parse_args method."""
        stderr = StringIO()
        with redirect_stderr(stderr):
            # Try a wrong results file
            wrong_file = "wrong.dat"
            with self.assertRaises(SystemExit):
                viewer = BatchResultsViewer(
                    ["--results-obj", wrong_file]
                )
                viewer.parse_args()

            # Try a wrong results file extension
            wrong_extension = "test.py"
            with self.assertRaises(SystemExit):
                viewer = BatchResultsViewer(
                    ["--results-obj", wrong_extension]
                )
                viewer.parse_args()

            # Try a wrong key
            wrong_key = "wrong"
            with self.assertRaises(SystemExit):
                viewer = BatchResultsViewer(
                    ["--key", wrong_key]
                )
                viewer.parse_args()

        # Try the default parameters
        viewer = BatchResultsViewer()
        viewer.parse_args()

        # Assess the parameters
        self.assertEqual(viewer._args.results_obj, viewer.default_results_obj)
        self.assertEqual(viewer._args.key, viewer.default_key)

        # Try custom parameters
        viewer = BatchResultsViewer(
            [
                "--results-obj", results_obj,
                "--key", key
            ]
        )
        viewer.parse_args()

        # Assess the parameters
        self.assertEqual(viewer._args.results_obj, results_obj)
        self.assertEqual(viewer._args.key, key)


    def test_process(self):
        """Test the process method."""
        # Try custom parameters
        viewer = BatchResultsViewer(
            [
                "--results-obj", results_obj,
                "--key", key
            ]
        )
        stdout = StringIO()
        with redirect_stdout(stdout):
            viewer.process()


if __name__ == '__main__':
    unittest.main()
