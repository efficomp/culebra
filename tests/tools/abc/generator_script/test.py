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

"""Unit test for :class:`culebra.tools.abc.GeneratorScript`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr

from culebra.tools.abc.scripts import GeneratorScript


class MyGenerator(GeneratorScript):
    """
    Dummy script.
    """
    def __init__(self, args=None):
        super().__init__(args)

        self._parser.add_argument(
            "param",
            metavar="PARAM",
            help="Parameter."
        )

    def process(self):
        pass

    def generate(self):
        pass


class GeneratorScriptTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.GeneratorScript`."""

    def test_init(self):
        """Test the constructor."""
        # Try without debug args
        generator = MyGenerator()
        self.assertIsNone(generator._debug_args)
        self.assertIsNone(generator._args)

        # Try with debug args
        args = ["4", "file.dat"]
        generator = MyGenerator(args)
        self.assertEqual(generator._debug_args, args)

    def test_parse_args(self):
        """Test the parse_args method."""

        stderr = StringIO()
        with redirect_stderr(stderr):
            # Try without parameters. Should fail ...
            with self.assertRaises(SystemExit):
                generator = MyGenerator()
                generator.parse_args()

        # Try a valid parameter
        params = ["1"]
        generator = MyGenerator(params)
        generator.parse_args()

        # Assess the parameter
        self.assertEqual(generator._args.param, params[0])


if __name__ == '__main__':
    unittest.main()
