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

"""Unit test for :class:`culebra.tools.abc.scripts.BatchResultsAnalyzer`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr
from os import mkdir, rmdir

from culebra.tools.abc.scripts import BatchResultsAnalyzer


# Analysis method
method = "rank"

# List of batches
batches = ["batch1", "batch2"]


class MyBatchResultsAnalyzer(BatchResultsAnalyzer):
    """
    Dummy analyzer.
    """

    def process(self):
        pass


class BatchResultsAnalyzerTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.BatchResultsAnalyzer`."""

    @classmethod
    def setUpClass(cls):
        """Create the batch folders."""
        for batch in batches:
            mkdir(batch)

    @classmethod
    def tearDownClass(cls):
        """Remove the trainre folders."""
        for batch in batches:
            rmdir(batch)

    def test_parse_args(self):
        """Test the parse_args method."""
        stderr = StringIO()
        with redirect_stderr(stderr):
            # Try without parameters. Should fail ...
            with self.assertRaises(SystemExit):
                analyzer = MyBatchResultsAnalyzer()
                analyzer.parse_args()

            # Try without batches. Sould fail ...
            with self.assertRaises(SystemExit):
                analyzer = MyBatchResultsAnalyzer([method])
                analyzer.parse_args()

            # Try with only one batch. Sould fail ...
            with self.assertRaises(SystemExit):
                analyzer = MyBatchResultsAnalyzer([method] + [batches[0]])
                analyzer.parse_args()

            # Try with a wrong batch folder. Sould fail ...
            with self.assertRaises(SystemExit):
                analyzer = MyBatchResultsAnalyzer(
                    [method] + [batches[0], "wrong"]
                )
                analyzer.parse_args()

        # Try valid values
        analyzer = MyBatchResultsAnalyzer([method] + batches)
        analyzer.parse_args()

        self.assertEqual(analyzer._args.method, method)
        self.assertEqual(analyzer._args.batches, batches)


if __name__ == '__main__':
    unittest.main()
