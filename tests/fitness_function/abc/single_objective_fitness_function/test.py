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

"""Test the SingleObjectiveFitnessFunction class."""

import unittest

from culebra import DEFAULT_SIMILARITY_THRESHOLD
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction


class MySingleObjectiveFitnessFunction(SingleObjectiveFitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    def evaluate(self, sol, index, representatives):
        """Evaluate a solution."""
        sol.fitness.values = (0,)
        return sol.fitness


class SingleObjectiveFitnessFunctionTester(unittest.TestCase):
    """Test SingleObjectiveFitnessFunction."""

    def test_init(self):
        """Test the constructor."""
        # Check default parameter values
        func = MySingleObjectiveFitnessFunction()
        self.assertEqual(func.index, 0)

        # Check a valid index
        valid_index = 3
        func = MySingleObjectiveFitnessFunction(valid_index)
        self.assertEqual(func.index, valid_index)

        # Check an invalid index type
        invalid_index_type = 'a'
        with self.assertRaises(TypeError):
            MySingleObjectiveFitnessFunction(invalid_index_type)

        # Check an invalid index valur
        invalid_index_value = -1
        with self.assertRaises(ValueError):
            MySingleObjectiveFitnessFunction(invalid_index_value)

    def test_obj_names(self):
        """Test the obj_names property."""
        # Check default parameter values
        func = MySingleObjectiveFitnessFunction(index=8)
        self.assertEqual(func.index, 8)
        self.assertEqual(func.obj_names, ("obj_8",))

    def test_obj_thresholds(self):
        """Test the obj_thresholds property."""
        # Try default objective similarity thresholds
        func = MySingleObjectiveFitnessFunction()

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


if __name__ == '__main__':
    unittest.main()
