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

"""Some common scripts.

This module provides the following classes:

* :class:`~culebra.tools.scripts.BatchResultsViewer`, to display the results
  of a trainer batch
"""

from pathlib import Path

import pandas as pd

from culebra import SERIALIZED_FILE_EXTENSION
from .abc import Script
from .evaluation import Batch
from .results import Results
from .constants import DEFAULT_RESULTS_BASE_FILENAME


class BatchResultsViewer(Script):
    """
    Batch results viewer script.

    The expected workflow is:

        viewer = BatchResultsViewer()
        viewer.process()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    """

    default_results_obj = (
        DEFAULT_RESULTS_BASE_FILENAME + SERIALIZED_FILE_EXTENSION
    )
    """Default serialized results."""

    default_key = Batch.ResultsKeys.TEST_BEST.value
    """Default results key."""

    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the viewer.

        Define the command-line arguments.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        super().__init__(args)

        self._parser.add_argument(
            "--results-obj",
            default=self.default_results_obj,
            metavar="RESULTS_OBJ",
            help="Serialized results. Default: %(default)s."
        )

        self._parser.add_argument(
            "--key",
            default=self.default_key,
            metavar="KEY",
            help="Results key. Default: %(default)s."
        )

    def parse_args(self) -> None:
        """
        Parse and validate command-line arguments.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        super().parse_args()

        # Check the serialized results file
        if not Path(self._args.results_obj).is_file():
            self._parser.error(
                "Can't open the serialized results file: "
                f"{self._args.results_obj}"
            )
        if not self._args.results_obj.endswith(
            SERIALIZED_FILE_EXTENSION
        ):
            self._parser.error(
                "Wrong serialized results file extension: "
                f"{self._args.results_obj}"
            )

        # Check the key
        valid_keys = [key.value for key in Batch.ResultsKeys]
        if self._args.key.lower() not in valid_keys:
            self._parser.error(
                f"Wrong key: {self._args.key}. "
                f"Valid keys: {', '.join(valid_keys)}"
            )

    def process(self) -> None:
        """
        Display the contents of the results key.
        """
        # Parse command-line arguments
        self.parse_args()

        results = Results.load(self._args.results_obj)

        # Print the analysis
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(results[self._args.key.lower()])


# Exported symbols for this module
__all__ = [
    'BatchResultsViewer'
]
