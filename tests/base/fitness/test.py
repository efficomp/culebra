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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for :py:class:`base.Fitness`."""

import unittest
from copy import copy, deepcopy
import pickle
from culebra.base import Fitness, Dataset

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyFitness(Fitness):
    """Dummy fitness class."""

    weights = (1, -1)
    names = ("obj1", "obj2")
    thresholds = (0.1, 0.2)


class FitnessTester(unittest.TestCase):
    """Test :py:class:`base.Fitness`."""

    def test_init(self):
        """Test the :py:meth:`~base.Fitness.__init__` constructor."""
        # Check default values
        fitness = MyFitness()
        self.assertEqual(fitness.weights, MyFitness.weights)
        self.assertEqual(fitness.names, MyFitness.names)
        self.assertEqual(fitness.thresholds, MyFitness.thresholds)

        # Delete the names and thresholds
        MyFitness.names = None
        MyFitness.thresholds = None

        fitness = MyFitness()
        self.assertEqual(fitness.names, ("obj_0", "obj_1"))
        self.assertEqual(fitness.thresholds, (0, 0))

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
        MyFitness.thresholds = (0, 1, 2)
        with self.assertRaises(TypeError):
            MyFitness()

        # Try initial values
        MyFitness.thresholds = None
        values = (2, 3)

        fitness = MyFitness(values=values)
        self.assertEqual(fitness.values, values)

    def test_del_values(self):
        """Test :py:meth:~base.Fitness.delValues`."""
        # Construct a fitness
        fitness = MyFitness(values=(1, 2))

        # The fitness values and context should have been deleted
        self.assertEqual(fitness.values, (1, 2))

        # Try delValues
        fitness.delValues()

        # The fitness values and context should have been deleted
        self.assertEqual(fitness.values, ())

    def test_num_obj(self):
        """Test the :py:attr:`~base.Fitness.num_obj` property."""
        fitness = MyFitness()
        self.assertEqual(fitness.num_obj, 2)

    def test_dominates(self):
        """Test the :py:meth:`~base.Fitness.dominates` method."""
        thresholds = (0, 2)
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
        """Test the :py:meth:`~base.Fitness.__le__` method."""
        thresholds = (0, 2)
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
                            fitness_1.__le__(fitness_2),
                            fitness_1_le_fitness_2
                        )

    def test_lt(self):
        """Test the :py:meth:`~base.Fitness.__lt__` method."""
        thresholds = (0, 2)
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
                            fitness_1.__lt__(fitness_2),
                            fitness_1_lt_fitness_2
                        )

    def test_eq(self):
        """Test the :py:meth:`~base.Fitness.__eq__` method."""
        thresholds = (0, 2)
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
                            fitness_1.__eq__(fitness_2),
                            fitness_1_eq_fitness_2
                        )

    def test_copy(self):
        """Test the :py:meth:`~base.Fitness.__copy__` method."""
        fitness1 = MyFitness((1, 2))
        fitness2 = copy(fitness1)

        # Copy only copies the first level (fitness1 != fitness2)
        self.assertNotEqual(id(fitness1), id(fitness2))

        # The objects attributes are shared
        self.assertEqual(id(fitness1.wvalues), id(fitness2.wvalues))

    def test_deepcopy(self):
        """Test the :py:meth:`~base.Fitness.__deepcopy__` method."""
        fitness1 = MyFitness((1, 2))
        fitness2 = deepcopy(fitness1)

        # Check the copy
        self._check_deepcopy(fitness1, fitness2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~base.Fitness.__setstate__` and
        :py:meth:`~base.Fitness.__reduce__` methods.
        """
        fitness1 = MyFitness((1, 2))

        data = pickle.dumps(fitness1)
        fitness2 = pickle.loads(data)

        # Check the copy
        self._check_deepcopy(fitness1, fitness2)

    def _check_deepcopy(self, fitness1, fitness2):
        """Check if *fitness1* is a deepcopy of *fitness2*.

        :param fitness1: The first dataset
        :type fitness1: :py:class:`~base.Fitness`
        :param fitness2: The second dataset
        :type fitness2: :py:class:`~base.Fitness`
        """
        # Copies all the levels
        self.assertNotEqual(id(fitness1), id(fitness2))
        self.assertEqual(fitness1.wvalues, fitness2.wvalues)


if __name__ == '__main__':
    unittest.main()