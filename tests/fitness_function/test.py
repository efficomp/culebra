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

"""Test the MultiObjectiveFitnessFunction class."""

import unittest

from culebra import DEFAULT_SIMILARITY_THRESHOLD
from culebra.abc import Solution, Species
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction
from culebra.fitness_function import MultiObjectiveFitnessFunction


class MySolution(Solution):
    """Dummy subclass to test the :class:`~culebra.abc.Solution` class."""


class MySpecies(Species):
    """Dummy subclass to test the :class:`~culebra.abc.Species` class."""

    def check(self, _):
        """Check a solution."""
        return True


class MySingleObjectiveFitnessFunction(SingleObjectiveFitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    def evaluate(self, sol, index, representatives):
        """Evaluate a solution."""
        sol.fitness.update_value(0, self.index)
        return sol.fitness


class AnotherSingleObjectiveFitnessFunction(SingleObjectiveFitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    def evaluate(self, sol, index, representatives):
        """Evaluate a solution."""
        sol.fitness.update_value(1, self.index)
        return sol.fitness


class MultiObjectiveFitnessFunctionTester(unittest.TestCase):
    """Test MultiObjectiveFitnessFunction."""

    def test_init(self):
        """Test the constructor."""
        # Objectives
        obj0 = MySingleObjectiveFitnessFunction()
        obj1 = AnotherSingleObjectiveFitnessFunction()

        # Try without objectives
        func = MultiObjectiveFitnessFunction()
        self.assertEqual(func.num_obj, 0)
        self.assertEqual(func.objectives, ())
        self.assertEqual(func.obj_weights, ())
        self.assertEqual(func.obj_names, ())

        # Try with only an objective
        func = MultiObjectiveFitnessFunction(obj0)
        self.assertEqual(func.num_obj, 1)
        self.assertEqual(func.objectives, (obj0,))
        self.assertEqual(func.obj_weights, obj0.obj_weights)
        self.assertEqual(func.obj_names, obj0.obj_names)
        self.assertEqual(obj0.index, 0)

        # Try a bi-objective fitness function
        func = MultiObjectiveFitnessFunction(obj0, obj1)

        self.assertEqual(func.num_obj, 2)
        self.assertEqual(func.objectives, (obj0, obj1))
        self.assertEqual(
            func.obj_weights, obj0.obj_weights + obj1.obj_weights
        )
        self.assertEqual(
            func.obj_names, obj0.obj_names + obj1.obj_names
        )
        self.assertEqual(obj0.index, 0)
        self.assertEqual(obj1.index, 1)

    def test_obj_thresholds(self):
        """Test the obj_thresholds property."""
        # Objectives
        obj0 = MySingleObjectiveFitnessFunction()
        obj1 = MySingleObjectiveFitnessFunction()
        func = MultiObjectiveFitnessFunction(obj0, obj1)

        # Test default threshold
        for th in func.obj_thresholds:
            self.assertEqual(th, DEFAULT_SIMILARITY_THRESHOLD)


        # Try differnt objective similarity thresholds
        obj_thresholds = [0.1, 0.2]
        func.obj_thresholds = obj_thresholds
        self.assertEqual(obj0.obj_thresholds, [obj_thresholds[0]])
        self.assertEqual(obj1.obj_thresholds, [obj_thresholds[1]])
        self.assertEqual(func.obj_thresholds, obj_thresholds)

        single_threshold = 0.4
        func.obj_thresholds = single_threshold
        self.assertEqual(obj0.obj_thresholds, [single_threshold])
        self.assertEqual(obj1.obj_thresholds, [single_threshold])
        self.assertEqual(func.obj_thresholds, [single_threshold]*func.num_obj)

        invalid_threshold_types = (type, {}, len)
        invalid_threshold_value = -1

        # Try invalid types for the thresholds. Should fail
        for threshold in invalid_threshold_types:
            with self.assertRaises(TypeError):
                func.obj_thresholds = threshold

        # Try invalid values for the threshold. Should fail
        with self.assertRaises(ValueError):
            func.obj_thresholds = invalid_threshold_value

        # Try a wrong number of thresholds
        with self.assertRaises(ValueError):
            func.obj_thresholds = [1, 2, 3]
        with self.assertRaises(ValueError):
            func.obj_thresholds = [1]

    def test_evaluate(self):
        """Test the evaluation method."""
        obj0 = MySingleObjectiveFitnessFunction()
        obj1 = AnotherSingleObjectiveFitnessFunction()

        func = MultiObjectiveFitnessFunction(obj0, obj1)

        sol = MySolution(MySpecies(), func.fitness_cls)

        fit_values = func.evaluate(sol).values

        self.assertEqual(sol.fitness.values, (0, 1))
        self.assertEqual(fit_values, sol.fitness.values)


if __name__ == '__main__':
    unittest.main()
