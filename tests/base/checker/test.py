# !/usr/bin/env python3
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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for the :py:mod:`base.checker` module."""

import unittest
from numbers import Real
from culebra.base import (
    check_bool,
    check_str,
    check_limits,
    check_int,
    check_float,
    check_instance,
    check_subclass,
    check_func,
    check_func_params,
    check_sequence,
    check_filename
)


class TypeCheckerTester(unittest.TestCase):
    """Test the :py:mod:`base.checker` module."""

    def test_check_bool(self):
        """Test the :py:func:`base.check_bool` function."""
        # Check invalid types. Should fail
        invalid_values = ('a', 1.5, len)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_bool(value, "name")

        valid_values = (True, False)
        for value in valid_values:
            self.assertEqual(check_bool(value, "name"), value)

    def test_check_str(self):
        """Test the :py:func:`base.check_str` function."""
        # Check invalid types. Should fail
        invalid_values = (1.5, len)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_str(value, "name")

        # Check valid strings
        valid_values = ('a', "b", """c""")
        for value in valid_values:
            self.assertEqual(check_str(value, "name"), value)

        # Check valid chars
        check_str("hooo", "name", valid_chars="hello")
        with self.assertRaises(ValueError):
            check_str("hello", "name", valid_chars="j")

        # Check invalid chars
        check_str("hooo", "name", invalid_chars="jkf")
        with self.assertRaises(ValueError):
            check_str("hello", "name", valid_chars="h")

    def test_check_limits(self):
        """Test :py:func:`base.check_limits`."""
        # Check invalid values. Should fail
        with self.assertRaises(ValueError):
            check_limits(1, "name", lt=0)
        with self.assertRaises(ValueError):
            check_limits(1, "name", le=0)
        with self.assertRaises(ValueError):
            check_limits(0, "name", gt=1)
        with self.assertRaises(ValueError):
            check_limits(0, "name", gt=1)

        # Check valid values
        self.assertEqual(check_limits(0, "name", lt=1), 0)
        self.assertEqual(check_limits(0, "name", le=1), 0)
        self.assertEqual(check_limits(1, "name", le=1), 1)
        self.assertEqual(check_limits(1, "name", gt=0), 1)
        self.assertEqual(check_limits(1, "name", ge=0), 1)
        self.assertEqual(check_limits(0, "name", ge=0), 0)

    def test_check_int(self):
        """Test the :py:func:`base.check_int` function."""
        # Check invalid types. Should fail
        invalid_values = ('a', 1.5, len)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_int(value, "name")

        valid_values = (-5, 0, 10, -12.0, 0.0, 45.0)
        for value in valid_values:
            self.assertEqual(check_int(value, "name"), int(value))

        # Check valid values
        self.assertEqual(check_limits(0.0, "name", lt=1), 0)
        self.assertEqual(check_limits(0.0, "name", le=1), 0)
        self.assertEqual(check_limits(1.0, "name", le=1), 1)
        self.assertEqual(check_limits(1.0, "name", gt=0), 1)
        self.assertEqual(check_limits(1.0, "name", ge=0), 1)
        self.assertEqual(check_limits(0.0, "name", ge=0), 0)

    def test_check_float(self):
        """Test the :py:func:`base.check_float` function."""
        # Check invalid types. Should fail
        invalid_values = ('a', '1.5', len)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_float(value, "name")

        valid_values = (-5, 0, 10, -12.0, 0.0, 45.0)
        for value in valid_values:
            self.assertEqual(check_float(value, "name"), float(value))

        # Check valid values
        self.assertEqual(check_limits(0, "name", lt=1), 0.0)
        self.assertEqual(check_limits(0, "name", le=1), 0.0)
        self.assertEqual(check_limits(1, "name", le=1), 1.0)
        self.assertEqual(check_limits(1, "name", gt=0), 1.0)
        self.assertEqual(check_limits(1, "name", ge=0), 1.0)
        self.assertEqual(check_limits(0, "name", ge=0), 0.0)

    def test_check_instance(self):
        """Test :py:func:`base.check_instance`."""
        # Check invalid instances. Should fail
        invalid_values = ('a', 1.5)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_instance(value, "name", cls=int)

        # Check valid instances
        valid_values = (1, 2)
        for value in valid_values:
            self.assertEqual(
                check_instance(value, "name", cls=int), int(value)
            )

    def test_check_subclass(self):
        """Test :py:func:`base.check_subclass`."""
        # Check invalid classes. Should fail
        invalid_values = ('a', 1.5)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_subclass(value, "name", cls=Real)

        # Check invalid classes. Should fail
        invalid_values = (str, type)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_subclass(value, "name", cls=Real)

        # Check valid classes. Should fail
        invalid_values = (int, float)
        for value in invalid_values:
            self.assertEqual(
                check_subclass(value, "name", cls=Real), value
            )

    def test_check_func(self):
        """Test :py:func:`base.check_func`."""
        # Check invalid types. Should fail
        invalid_values = ('a', 1.5, 1)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_func(value, "name")

        valid_values = (len, max, int)
        for value in valid_values:
            self.assertEqual(check_func(value, "name"), value)

    def test_check_func_params(self):
        """Test :py:func:`base.check_func_params`."""
        # Check invalid types. Should fail
        invalid_values = ('a', 1.5, 1)
        for value in invalid_values:
            with self.assertRaises(TypeError):
                check_func_params(value, "name")

        # Check invalid keys. Should fail
        invalid_values = ({1: 1}, {'a': 1, 2: 2})
        for value in invalid_values:
            with self.assertRaises(ValueError):
                check_func_params(value, "name")

        valid_values = ({'a': 1}, {'a': 1, 'b': 2})
        for value in valid_values:
            self.assertEqual(check_func_params(value, "name"), value)

    def test_check_sequence(self):
        """Test :py:func:`base.check_sequence`."""
        # Check invalid sequences. Should fail
        invalid_sequences = (len, 1.5, 1)
        for seq in invalid_sequences:
            with self.assertRaises(TypeError):
                check_sequence(seq, "values")

        # Check valid sequences
        valid_sequences = ((1, 2), ['a', 'b', 'c'])
        for seq in valid_sequences:
            self.assertEqual(check_sequence(seq, "values"), list(seq))

        # Check invalid sequences sizes. Should fail
        invalid_sequences = ((1, 2), ['a', 'b', 'c'])
        for seq in invalid_sequences:
            with self.assertRaises(ValueError):
                check_sequence(seq, "values", size=5)

        # Check valid sequences sizes
        invalid_sequences = ((1, 2, 3), ['a', 'b', 'c'])
        for seq in invalid_sequences:
            self.assertEqual(
                check_sequence(seq, "values", size=3), list(seq)
            )

        # Check invalid item types. Should fail
        invalid_sequences = (('1', '2'), ['a', 'b', 'c'])
        for seq in invalid_sequences:
            with self.assertRaises(ValueError):
                check_sequence(seq, "values", item_checker=check_int)

        # Check valid item types
        invalid_sequences = ((1, 2.0), [-1.0, 2, 3.0])
        for seq in invalid_sequences:
            self.assertEqual(
                check_sequence(seq, "values", item_checker=check_int),
                [int(item) for item in seq]
            )

    def test_check_filename(self):
        """Test :py:func:`base.check_filename`."""
        # Check an invalid filename
        invalid_filename = 1
        with self.assertRaises(TypeError):
            check_filename(invalid_filename, "filename")

        # Try a valid filename
        valid_filename = "file.ext"
        check_filename(valid_filename, "filename")

        # Try an invalid extension type
        invalid_extension = 1
        with self.assertRaises(TypeError):
            check_filename(valid_filename, "filename", ext=invalid_extension)

        # Try an invalid extension value
        invalid_extension = "ext"
        with self.assertRaises(ValueError):
            check_filename(valid_filename, "filename", ext=invalid_extension)

        # Try a valid extension
        valid_extension = ".ext"
        check_filename(valid_filename, "filename", ext=valid_extension)


if __name__ == '__main__':
    unittest.main()
