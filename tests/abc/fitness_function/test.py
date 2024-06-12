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

from culebra.abc import Species, Fitness, FitnessFunction


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = [0.001, 0.001]

    def evaluate(self, sol, index, representatives):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.
        """


class FitnessFunctionTester(unittest.TestCase):
    """Test :py:class:`~culebra.abc.FitnessFunction`."""

    def test_set_fitness_thresholds(self):
        """Test the set_fitness_thresholds class method."""
        invalid_threshold_types = (type, {}, len)
        invalid_threshold_value = -1
        valid_thresholds = [0.33, 0.5, 2]

        # Try invalid types for the thresholds. Should fail
        for threshold in invalid_threshold_types:
            with self.assertRaises(TypeError):
                MyFitnessFunction.set_fitness_thresholds(threshold)

        # Try invalid values for the threshold. Should fail
        with self.assertRaises(ValueError):
            MyFitnessFunction.set_fitness_thresholds(invalid_threshold_value)

        # Try a fixed value for all the thresholds
        for threshold in valid_thresholds:
            MyFitnessFunction.set_fitness_thresholds(threshold)
            # Check the length of the sequence
            self.assertEqual(
                len(MyFitnessFunction.Fitness.thresholds),
                len(MyFitnessFunction.Fitness.weights)
            )

            # Check that all the values match
            for th in MyFitnessFunction.Fitness.thresholds:
                self.assertEqual(threshold, th)

        # Try different values of threshold for each objective
        MyFitnessFunction.set_fitness_thresholds(
            valid_thresholds[:len(MyFitnessFunction.Fitness.weights)]
        )
        for th1, th2 in zip(
            valid_thresholds, MyFitnessFunction.Fitness.thresholds
        ):
            self.assertEqual(th1, th2)

        # Try a wrong number of thresholds
        with self.assertRaises(ValueError):
            MyFitnessFunction.set_fitness_thresholds(valid_thresholds)

    def test_get_objective_threshold(self):
        """Test :py:meth:~culebra.abc.Fitness.get_objective_threshold`."""
        # Try an invalid type for the objective name. Should fail ...
        with self.assertRaises(TypeError):
            MyFitnessFunction.get_fitness_objective_threshold(1)

        # Try an invalid objective name. Should fail ...
        with self.assertRaises(ValueError):
            MyFitnessFunction.get_fitness_objective_threshold(
                "invalid_obj_name"
            )

        MyFitnessFunction.set_fitness_thresholds(1)
        obj_index = 0
        obj_name = MyFitnessFunction.Fitness.names[obj_index]
        obj_threshold = MyFitnessFunction.Fitness.thresholds[obj_index]
        self.assertEqual(
            MyFitnessFunction.get_fitness_objective_threshold(obj_name),
            obj_threshold
        )

    def test_set_fitness_objective_threshold(self):
        """Test the set_fitness_objective_threshold class method."""
        # Try an invalid type for the objective name. Should fail ...
        with self.assertRaises(TypeError):
            MyFitnessFunction.set_fitness_objective_threshold(1, 0.5)

        # Try an invalid objective name. Should fail ...
        with self.assertRaises(ValueError):
            MyFitnessFunction.set_fitness_objective_threshold(
                "invalid_obj_name",
                0.5
            )

        MyFitnessFunction.set_fitness_thresholds(1)
        obj_index = 0
        obj_name = MyFitnessFunction.Fitness.names[obj_index]
        obj_threshold = MyFitnessFunction.Fitness.thresholds[obj_index]

        # Try an invalid type for the threshold. Should fail ...
        with self.assertRaises(TypeError):
            MyFitnessFunction.set_fitness_objective_threshold(obj_name, "a")

        # Try an invalid value for the threshold. Should fail ...
        with self.assertRaises(ValueError):
            MyFitnessFunction.set_fitness_objective_threshold(obj_name, -1)

        # Set valid thresholds
        for obj_index in range(len(MyFitnessFunction.Fitness.names)):
            obj_name = MyFitnessFunction.Fitness.names[obj_index]
            obj_threshold = MyFitnessFunction.Fitness.thresholds[obj_index]
            new_threshold = obj_threshold * 2
            MyFitnessFunction.set_fitness_objective_threshold(
                obj_name,
                new_threshold,
            )

            # Check the new threshold
            self.assertEqual(
                MyFitnessFunction.Fitness.thresholds[obj_index], new_threshold
            )

    def test_num_obj(self):
        """Test the num_obj property."""
        # Fitness function to be tested
        func = MyFitnessFunction()
        self.assertEqual(func.num_obj, len(MyFitnessFunction.Fitness.weights))

    def test_num_nodes(self):
        """Test the num_nodes property."""
        # Fitness function to be tested
        func = MyFitnessFunction()
        self.assertEqual(func.num_nodes, None)

    def test_heuristic(self):
        """Test the heuristic method."""
        # Fitness function to be tested

        func = MyFitnessFunction()
        self.assertEqual(func.heuristic(Species()), None)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyFitnessFunction()
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


if __name__ == '__main__':
    unittest.main()
