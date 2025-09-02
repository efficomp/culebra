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

"""Unit test for :py:class:`culebra.abc.FitnessFunction`."""

import unittest

from os import remove

from culebra import DEFAULT_SIMILARITY_THRESHOLD, SERIALIZED_FILE_EXTENSION
from culebra.abc import FitnessFunction


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, 1)

    def evaluate(self, sol, index=None, representatives=None):
        """Evaluate one solution."""
        return (0, 0)


class FitnessFunctionTester(unittest.TestCase):
    """Test :py:class:`~culebra.abc.FitnessFunction`."""

    def test_num_obj(self):
        """Test the num_obj property."""
        # Fitness function to be tested
        func = MyFitnessFunction()
        self.assertEqual(func.num_obj, len(func.obj_weights))

    def test_obj_names(self):
        """Test the obj_names property."""
        # Fitness function to be tested
        func = MyFitnessFunction()
        self.assertEqual(func.obj_names, ("obj_0", "obj_1"))

    def test_obj_thresholds(self):
        """Test the obj_thresholds property."""
        # Try default objective similarity thresholds
        func = MyFitnessFunction()

        self.assertEqual(
            func.obj_thresholds,
            [DEFAULT_SIMILARITY_THRESHOLD] * func.num_obj
            )

        invalid_threshold_types = (type, {}, len)
        invalid_threshold_value = -1
        valid_thresholds = [0, 0.33, 0.5, 2]

        # Try invalid types for the thresholds. Should fail
        for threshold in invalid_threshold_types:
            with self.assertRaises(TypeError):
                func.obj_thresholds = threshold

        # Try invalid values for the threshold. Should fail
        with self.assertRaises(ValueError):
            func.obj_thresholds = invalid_threshold_value

        # Try a fixed value for all the thresholds
        for threshold in valid_thresholds:
            func.obj_thresholds = threshold
            # Check the length of the sequence
            self.assertEqual(len(func.obj_thresholds), func.num_obj)

            # Check that all the values match
            for th in func.obj_thresholds:
                self.assertEqual(threshold, th)

        # Try different values of threshold for each objective
        func.obj_thresholds = valid_thresholds[:func.num_obj]
        for th1, th2 in zip(
            valid_thresholds, func.obj_thresholds
        ):
            self.assertEqual(th1, th2)

        # Try a wrong number of thresholds
        with self.assertRaises(ValueError):
            func.obj_thresholds = valid_thresholds

    def test_is_noisy(self):
        """Test the is_noisy property."""
        # Fitness function to be tested
        func = MyFitnessFunction()
        self.assertEqual(func.is_noisy, False)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyFitnessFunction()
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def test_fitness_serialization(self):
        """Test the serialization of fitnesses."""
        func = MyFitnessFunction()
        fitness1 = func.fitness_cls(func.evaluate(None))

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        fitness1.dump(serialized_filename)
        fitness2 = fitness1.__class__.load(serialized_filename)

        # Check the serialization
        self.assertNotEqual(id(fitness1), id(fitness2))
        self.assertEqual(fitness1.wvalues, fitness2.wvalues)

        # Remove the serialized file
        remove(serialized_filename)


if __name__ == '__main__':
    unittest.main()
