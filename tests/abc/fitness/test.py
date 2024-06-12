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

"""Unit test for :py:class:`culebra.abc.Fitness`."""

import unittest
import pickle
from copy import copy, deepcopy

from culebra.abc import Fitness


class MyFitness(Fitness):
    """Dummy fitness class."""

    weights = (1, -1)
    names = ("obj1", "obj2")
    thresholds = [0.1, 0.2]


class FitnessTester(unittest.TestCase):
    """Test :py:class:`culebra.abc.Fitness`."""

    def test_init(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__init__` constructor."""
        # Check default values
        fitness = MyFitness()
        self.assertEqual(fitness.weights, MyFitness.weights)
        self.assertEqual(fitness.names, MyFitness.names)
        self.assertEqual(fitness.thresholds, MyFitness.thresholds)

        # Delete the names and thresholds
        MyFitness.names = None
        MyFitness.thresholds = None

        # Check default objective names
        fitness = MyFitness()
        self.assertEqual(fitness.names, ("obj_0", "obj_1"))
        self.assertEqual(fitness.thresholds, [0, 0])

        # Try a wrong type for names. It should fail
        MyFitness.names = 3
        with self.assertRaises(TypeError):
            MyFitness()

        # Try a wrong number of names. It should fail
        MyFitness.names = ("a", "b", "c")
        with self.assertRaises(TypeError):
            MyFitness()

        # Try a wrong type for thresholds. It should fail
        MyFitness.names = None
        MyFitness.thresholds = 1
        with self.assertRaises(TypeError):
            MyFitness()

        # Try a wrong number of thresholds. It should fail
        MyFitness.thresholds = [0, 1, 2]
        with self.assertRaises(TypeError):
            MyFitness()

        # Try initial values
        MyFitness.thresholds = None
        values = (2, 3)

        fitness = MyFitness(values=values)
        self.assertEqual(fitness.values, values)

    def test_get_objective_threshold(self):
        """Test :py:meth:~culebra.abc.Fitness.get_objective_threshold`."""
        # Try an invalid type for the objective name. Should fail ...
        with self.assertRaises(TypeError):
            MyFitness.get_objective_threshold(1)

        # Try an invalid objective name. Should fail ...
        with self.assertRaises(ValueError):
            MyFitness.get_objective_threshold("invalid_obj_name")

        MyFitness.thresholds = [1, 1]
        obj_index = 0
        obj_name = MyFitness.names[obj_index]
        obj_threshold = MyFitness.thresholds[obj_index]
        self.assertEqual(
            MyFitness.get_objective_threshold(obj_name), obj_threshold
        )

    def test_set_objective_threshold(self):
        """Test :py:meth:~culebra.abc.Fitness.set_objective_threshold`."""
        # Try an invalid type for the objective name. Should fail ...
        with self.assertRaises(TypeError):
            MyFitness.set_objective_threshold(1, 0.5)

        # Try an invalid objective name. Should fail ...
        with self.assertRaises(ValueError):
            MyFitness.set_objective_threshold("invalid_obj_name", 0.5)

        MyFitness.thresholds = [1, 1]
        obj_index = 0
        obj_name = MyFitness.names[obj_index]
        obj_threshold = MyFitness.thresholds[obj_index]

        # Try an invalid type for the threshold. Should fail ...
        with self.assertRaises(TypeError):
            MyFitness.set_objective_threshold(obj_name, "a")

        # Try an invalid value for the threshold. Should fail ...
        with self.assertRaises(ValueError):
            MyFitness.set_objective_threshold(obj_name, -1)

        # Set valid thresholds
        for obj_index in range(len(MyFitness.names)):
            obj_name = MyFitness.names[obj_index]
            obj_threshold = MyFitness.thresholds[obj_index]
            new_threshold = obj_threshold * 2
            MyFitness.set_objective_threshold(obj_name, new_threshold)

            # Check the new threshold
            self.assertEqual(MyFitness.thresholds[obj_index], new_threshold)

    def test_del_values(self):
        """Test :py:meth:~culebra.abc.Fitness.delValues`."""
        # Construct a fitness
        fitness = MyFitness(values=(1, 2))

        # The fitness values and context should have been deleted
        self.assertEqual(fitness.values, (1, 2))

        # Try delValues
        fitness.delValues()

        # The fitness values and context should have been deleted
        self.assertEqual(fitness.values, ())

    def test_num_obj(self):
        """Test the :py:attr:`~culebra.abc.Fitness.num_obj` property."""
        fitness = MyFitness()
        self.assertEqual(fitness.num_obj, 2)

    def test_pheromone_amount(self):
        """Test the pheromone_amount property."""
        # Construct a fitness
        fitness = MyFitness(values=(2, 3))

        # The fitness values and context should have been deleted
        self.assertEqual(fitness.pheromone_amount, (2, 1/3))

    def test_dominates(self):
        """Test the :py:meth:`~culebra.abc.Fitness.dominates` method."""
        thresholds = [0, 2]
        min_max = (-1, 1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()

        # check minimization and maximization problems
        for type_opt in min_max:
            fitness_1.weights = (type_opt,) * fitness_1.num_obj
            fitness_2.weights = fitness_1.weights

            # Check different thresholds
            for threshold in thresholds:
                fitness_1.thresholds = (threshold,) * fitness_1.num_obj
                fitness_2.thresholds = fitness_1.thresholds

                for obj1 in range(7):
                    for obj2 in range(7):
                        fitness_1.setValues((3, 3))
                        fitness_2.setValues((obj1, obj2))
                        fitness_1_dominates_fitness_2 = False

                        for obj in range(2):
                            if (
                                fitness_1.wvalues[obj] - fitness_2.wvalues[obj]
                                > fitness_1.thresholds[obj]
                            ):
                                fitness_1_dominates_fitness_2 = True
                            elif (
                                fitness_2.wvalues[obj] - fitness_1.wvalues[obj]
                                > fitness_1.thresholds[obj]
                            ):
                                fitness_1_dominates_fitness_2 = False
                                break

                        self.assertEqual(
                            fitness_1.dominates(fitness_2),
                            fitness_1_dominates_fitness_2
                        )

    def test_le(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__le__` method."""
        thresholds = [0, 2]
        min_max = (-1, 1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()

        # check minimization and maximization problems
        for type_opt in min_max:
            fitness_1.weights = (type_opt,) * fitness_1.num_obj
            fitness_2.weights = fitness_1.weights

            # Check different thresholds
            for threshold in thresholds:
                fitness_1.thresholds = (threshold,) * fitness_1.num_obj
                fitness_2.thresholds = fitness_1.thresholds

                for obj1 in range(7):
                    for obj2 in range(7):
                        fitness_1.setValues((3, 3))
                        fitness_2.setValues((obj1, obj2))

                        fitness_1_le_fitness_2 = True
                        for obj in range(2):
                            if (
                                fitness_2.wvalues[obj] - fitness_1.wvalues[obj]
                                > fitness_1.thresholds[obj]
                            ):
                                break
                            if (
                                fitness_1.wvalues[obj] - fitness_2.wvalues[obj]
                                > fitness_1.thresholds[obj]
                            ):
                                fitness_1_le_fitness_2 = False
                                break

                        self.assertEqual(
                            fitness_1 <= fitness_2,
                            fitness_1_le_fitness_2
                        )

    def test_lt(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__lt__` method."""
        thresholds = [0, 2]
        min_max = (-1, 1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()

        # check minimization and maximization problems
        for type_opt in min_max:
            fitness_1.weights = (type_opt,) * fitness_1.num_obj
            fitness_2.weights = fitness_1.weights

            # Check different thresholds
            for threshold in thresholds:
                fitness_1.thresholds = (threshold,) * fitness_1.num_obj
                fitness_2.thresholds = fitness_1.thresholds

                for obj1 in range(7):
                    for obj2 in range(7):
                        fitness_1.setValues((3, 3))
                        fitness_2.setValues((obj1, obj2))

                        fitness_1_lt_fitness_2 = False
                        for obj in range(2):
                            if (
                                fitness_2.wvalues[obj] - fitness_1.wvalues[obj]
                                > fitness_1.thresholds[obj]
                            ):
                                fitness_1_lt_fitness_2 = True
                                break
                            if (
                                fitness_1.wvalues[obj] - fitness_2.wvalues[obj]
                                > fitness_1.thresholds[obj]
                            ):
                                break

                        self.assertEqual(
                            fitness_1 < fitness_2,
                            fitness_1_lt_fitness_2
                        )

    def test_eq(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__eq__` method."""
        thresholds = [0, 2]
        min_max = (-1, 1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()

        # check minimization and maximization problems
        for type_opt in min_max:
            fitness_1.weights = (type_opt,) * fitness_1.num_obj
            fitness_2.weights = fitness_1.weights

            # Check different thresholds
            for threshold in thresholds:
                fitness_1.thresholds = (threshold,) * fitness_1.num_obj
                fitness_2.thresholds = fitness_1.thresholds

                for obj1 in range(7):
                    for obj2 in range(7):
                        fitness_1.setValues((3, 3))
                        fitness_2.setValues((obj1, obj2))

                        fitness_1_eq_fitness_2 = True
                        for obj in range(2):
                            if (
                                abs(
                                    fitness_2.wvalues[obj] -
                                    fitness_1.wvalues[obj]
                                ) > fitness_1.thresholds[obj]
                            ):
                                fitness_1_eq_fitness_2 = False
                                break

                        self.assertEqual(
                            fitness_1 == fitness_2,
                            fitness_1_eq_fitness_2
                        )

    def test_copy(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__copy__` method."""
        fitness1 = MyFitness((1, 2))
        fitness2 = copy(fitness1)

        # Copy only copies the first level (fitness1 != fitness2)
        self.assertNotEqual(id(fitness1), id(fitness2))

        # The objects attributes are shared
        self.assertEqual(id(fitness1.wvalues), id(fitness2.wvalues))

    def test_deepcopy(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__deepcopy__` method."""
        fitness1 = MyFitness((1, 2))
        fitness2 = deepcopy(fitness1)

        # Check the copy
        self._check_deepcopy(fitness1, fitness2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.abc.Fitness.__setstate__` and
        :py:meth:`~culebra.abc.Fitness.__reduce__` methods.
        """
        fitness1 = MyFitness((1, 2))

        data = pickle.dumps(fitness1)
        fitness2 = pickle.loads(data)

        # Check the copy
        self._check_deepcopy(fitness1, fitness2)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        values = (1.0, 2.0)
        fitness1 = MyFitness(values)
        self.assertIsInstance(repr(fitness1), str)
        self.assertIsInstance(str(fitness1), str)

    def _check_deepcopy(self, fitness1, fitness2):
        """Check if *fitness1* is a deepcopy of *fitness2*.

        :param fitness1: The first fitness
        :type fitness1: :py:class:`~culebra.abc.Fitness`
        :param fitness2: The second fitness
        :type fitness2: :py:class:`~culebra.abc.Fitness`
        """
        # Copies all the levels
        self.assertNotEqual(id(fitness1), id(fitness2))
        self.assertEqual(fitness1.wvalues, fitness2.wvalues)


if __name__ == '__main__':
    unittest.main()
