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
from os import remove
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

    def test_get_objective_index(self):
        """Test :py:meth:~culebra.abc.Fitness.get_objective_index`."""
        # Try an invalid type for the objective name. Should fail ...
        with self.assertRaises(TypeError):
            MyFitness.get_objective_index(1)

        # Try an invalid objective name. Should fail ...
        with self.assertRaises(ValueError):
            MyFitness.get_objective_index("invalid_obj_name")

        # Try correct names
        for (index, name) in enumerate(MyFitness.names):
            self.assertEqual(index, MyFitness.get_objective_index(name))

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

    def test_get_objective_value(self):
        """Test :py:meth:~culebra.abc.Fitness.get_objective_value`."""
        # Construct a fitness
        fitness = MyFitness(values=(1, 2))

        # Check the fitness values
        for (name, value) in zip(fitness.names, fitness.values):
            self.assertEqual(fitness.get_objective_value(name), value)

    def test_get_objective_wvalue(self):
        """Test :py:meth:~culebra.abc.Fitness.get_objective_wvalue`."""
        # Construct a fitness
        fitness = MyFitness(values=(1, 2))

        # Check the fitness values
        for (name, wvalue) in zip(fitness.names, fitness.wvalues):
            self.assertEqual(fitness.get_objective_wvalue(name), wvalue)

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
        weights = (1, -1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()
        fitness_1.weights = fitness_2.weights = weights

        fitness_1.setValues((0.5, 0.5))

        # Try with thresholds = 0
        fitness_1.thresholds = (0,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # fitness_1 == fitness_2 -> fitness_1 should not dominate fitness_2
        fitness_2.setValues(fitness_1.values)
        self.assertFalse(fitness_1.dominates(fitness_2))

        # One objective is better and the other is worst
        # fitness_1 should not dominate fitness_2
        off = 0.1
        fitness_2.setValues(
            (fitness_1.values[0]+off, fitness_1.values[1]+off)
        )
        self.assertFalse(fitness_1.dominates(fitness_2))

        fitness_2.setValues(
            (fitness_1.values[0]-off, fitness_1.values[1]-off)
        )
        self.assertFalse(fitness_1.dominates(fitness_2))

        # One objective is equal and the other is worst
        # fitness_1 should not dominate fitness_2
        off = 0.1
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1]-off)
        )
        self.assertFalse(fitness_1.dominates(fitness_2))

        fitness_2.setValues(
            (fitness_1.values[0]+off, fitness_1.values[1])
        )
        self.assertFalse(fitness_1.dominates(fitness_2))

        # One objective is equal and the other is better
        # fitness_1 should dominate fitness_2
        off = 0.1
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1]+off)
        )
        self.assertTrue(fitness_1.dominates(fitness_2))

        fitness_2.setValues(
            (fitness_1.values[0]-off, fitness_1.values[1])
        )
        self.assertTrue(fitness_1.dominates(fitness_2))

        # The two objectives are better
        # fitness_1 should dominate fitness_2
        off = 0.1
        fitness_2.setValues(
            (fitness_1.values[0]-off, fitness_1.values[1]+off)
        )
        self.assertTrue(fitness_1.dominates(fitness_2))

# ----------------------------------

        # Try with thresholds = 0.1
        threshold = 0.1
        off = 2 * threshold
        fitness_1.thresholds = (threshold,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # Both objectives are within the threshold
        # fitness_1 should not dominate fitness_2
        offset_eq = (0, threshold/2, -threshold/2)
        for off_eq1 in offset_eq:
            for off_eq2 in offset_eq:
                fitness_2.setValues(
                    (fitness_1.values[0]+off_eq1, fitness_1.values[1]+off_eq2)
                )
                self.assertFalse(fitness_1.dominates(fitness_2))

        # One objective is better and the other is worst
        # fitness_1 should not dominate fitness_2
        fitness_2.setValues(
            (fitness_1.values[0]+off, fitness_1.values[1]+off)
        )
        self.assertFalse(fitness_1.dominates(fitness_2))

        fitness_2.setValues(
            (fitness_1.values[0]-off, fitness_1.values[1]-off)
        )
        self.assertFalse(fitness_1.dominates(fitness_2))

        for off_eq in offset_eq:
            # One objective is within the threshold and the other is worst
            # fitness_1 should not dominate fitness_2
            fitness_2.setValues(
                (fitness_1.values[0]+off_eq, fitness_1.values[1]-off)
            )
            self.assertFalse(fitness_1.dominates(fitness_2))

            fitness_2.setValues(
                (fitness_1.values[0]+off, fitness_1.values[1]+off_eq)
            )
            self.assertFalse(fitness_1.dominates(fitness_2))

            # One objective is within threshold  and the other is better
            # fitness_1 should dominate fitness_2
            fitness_2.setValues(
                (fitness_1.values[0]+off_eq, fitness_1.values[1]+off)
            )
            self.assertTrue(fitness_1.dominates(fitness_2))

            fitness_2.setValues(
                (fitness_1.values[0]-off, fitness_1.values[1]+off_eq)
            )
            self.assertTrue(fitness_1.dominates(fitness_2))

        # The two objectives are better
        # fitness_1 should dominate fitness_2
        fitness_2.setValues(
            (fitness_1.values[0]-off,
             fitness_1.values[1]+off)
        )
        self.assertTrue(fitness_1.dominates(fitness_2))

    def test_le(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__le__` method."""
        weights = (1, -1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()
        fitness_1.weights = fitness_2.weights = weights

        fitness_1.setValues((0.5, 0.5))

        # Try with thresholds = 0
        fitness_1.thresholds = (0,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # If the first component is lower, the second should not affect
        # fitness_1 should be always lower than or equal to fitness_2
        value_1 = fitness_1.values[0] + 0.1
        offset_2 = (0, 0.1, -0.1)
        for off in offset_2:
            fitness_2.setValues((value_1, fitness_1.values[1]+off))
            self.assertTrue(fitness_1 <= fitness_2)

        # If the first compoment is equal, the second decides
        off = 0.1

        # fitness_1 > fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1] + off)
        )
        self.assertFalse(fitness_1 <= fitness_2)

        # fitness_1 == fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1])
        )
        self.assertTrue(fitness_1 <= fitness_2)

        # fitness_1 < fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1] - off)
        )
        self.assertTrue(fitness_1 <= fitness_2)

        # Try with thresholds = 0.1
        threshold = 0.1
        fitness_1.thresholds = (threshold,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # If the first component is lower, the second should no affect
        # fitness_1 should be always lower than or equal to fitness_2
        value_1 = fitness_1.values[0] + threshold * 2
        offset_2 = (
            0,
            threshold/2,
            threshold*2,
            -threshold/2,
            -threshold*2
        )
        for off in offset_2:
            fitness_2.setValues((value_1, fitness_1.values[1]+off))
            self.assertTrue(fitness_1 <= fitness_2)

        # If the first compoment is within the threshold, the second decides
        off_eq = (0, threshold/2, -threshold/2)
        off_ne = threshold * 2
        for off1 in off_eq:

            # fitness_1 > fitness_2
            fitness_2.setValues(
                (fitness_1.values[0] + off1, fitness_1.values[1] + off_ne)
            )
            self.assertFalse(fitness_1 <= fitness_2)

            # fitness_1 == fitness_2
            for off2 in off_eq:
                fitness_2.setValues(
                    (fitness_1.values[0] + off1, fitness_1.values[1] + off2)
                )
                self.assertTrue(fitness_1 <= fitness_2)

            # fitness_1 < fitness_2
            fitness_2.setValues(
                (fitness_1.values[0] + off1, fitness_1.values[1] - off_ne)
            )
            self.assertTrue(fitness_1 <= fitness_2)

    def test_lt(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__lt__` method."""
        weights = (1, -1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()
        fitness_1.weights = fitness_2.weights = weights

        fitness_1.setValues((0.5, 0.5))

        # Try with thresholds = 0
        fitness_1.thresholds = (0,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # If the first component is lower, the second should not affect
        # fitness_1 should be always lower than fitness_2
        value_1 = fitness_1.values[0] + 0.1
        offset_2 = (0, 0.1, -0.1)
        for off in offset_2:
            fitness_2.setValues((value_1, fitness_1.values[1]+off))
            self.assertTrue(fitness_1 < fitness_2)

        # If the first compoment is equal, the second decides
        off = 0.1

        # fitness_1 > fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1] + off)
        )
        self.assertFalse(fitness_1 < fitness_2)

        # fitness_1 == fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1])
        )
        self.assertFalse(fitness_1 < fitness_2)

        # fitness_1 < fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1] - off)
        )
        self.assertTrue(fitness_1 < fitness_2)

        # Try with thresholds = 0.1
        threshold = 0.1
        fitness_1.thresholds = (threshold,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # If the first component is lower, the second should no affect
        # fitness_1 should be always lower than fitness_2
        value_1 = fitness_1.values[0] + threshold * 2
        offset_2 = (
            0,
            threshold / 2,
            threshold * 2,
            -threshold / 2,
            -threshold * 2
        )
        for off in offset_2:
            fitness_2.setValues((value_1, fitness_1.values[1]+off))
            self.assertTrue(fitness_1 < fitness_2)

        # If the first compoment is within the threshold, the second decides
        off_eq = (0, threshold / 2, -threshold / 2)
        off_ne = threshold * 2
        for off1 in off_eq:

            # fitness_1 > fitness_2
            fitness_2.setValues(
                (fitness_1.values[0] + off1, fitness_1.values[1] + off_ne)
            )
            self.assertFalse(fitness_1 < fitness_2)

            # fitness_1 == fitness_2
            for off2 in off_eq:
                fitness_2.setValues(
                    (fitness_1.values[0] + off1, fitness_1.values[1] + off2)
                )
                self.assertFalse(fitness_1 < fitness_2)

            # fitness_1 < fitness_2
            fitness_2.setValues(
                (fitness_1.values[0] + off1, fitness_1.values[1] - off_ne)
            )
            self.assertTrue(fitness_1 < fitness_2)

    def test_eq(self):
        """Test the :py:meth:`~culebra.abc.Fitness.__eq__` method."""
        weights = (1, -1)
        fitness_1 = MyFitness()
        fitness_2 = MyFitness()
        fitness_1.weights = fitness_2.weights = weights

        fitness_1.setValues((0.5, 0.5))

        # Try with thresholds = 0
        fitness_1.thresholds = (0,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # If the first component is not equal, the second should not affect
        # fitness_1 should be always not equal to fitness_2
        offset_1 = (0.1, -0.1)
        offset_2 = (0, 0.1, -0.1)
        for off1 in offset_1:
            for off2 in offset_2:
                fitness_2.setValues(
                    (fitness_1.values[0]+off1, fitness_1.values[1]+off2)
                )
                self.assertFalse(fitness_1 == fitness_2)

        # If the first compoment is equal, the second decides
        off = 0.1

        # fitness_1 > fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1] + off)
        )
        self.assertFalse(fitness_1 == fitness_2)

        # fitness_1 == fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1])
        )
        self.assertTrue(fitness_1 == fitness_2)

        # fitness_1 < fitness_2
        fitness_2.setValues(
            (fitness_1.values[0], fitness_1.values[1] - off)
        )
        self.assertFalse(fitness_1 == fitness_2)

        # Try with thresholds = 0.1
        threshold = 0.1
        fitness_1.thresholds = (threshold,) * fitness_1.num_obj
        fitness_2.thresholds = fitness_1.thresholds

        # If the first component is not equal, the second should no affect
        # fitness_1 should be always not equal to fitness_2
        offset_1 = (
            threshold * 2,
            -threshold * 2
        )
        offset_2 = (
            0,
            threshold / 2,
            threshold * 2,
            -threshold / 2,
            -threshold * 2
        )
        for off1 in offset_1:
            for off2 in offset_2:
                fitness_2.setValues(
                    (fitness_1.values[0]+off1, fitness_1.values[1]+off2)
                )
                self.assertFalse(fitness_1 == fitness_2)

        # If the first compoment is within the threshold, the second decides
        off_eq = (0, threshold / 2, -threshold / 2)
        off_ne = threshold * 2
        for off1 in off_eq:

            # fitness_1 > fitness_2
            fitness_2.setValues(
                (fitness_1.values[0] + off1, fitness_1.values[1] + off_ne)
            )
            self.assertFalse(fitness_1 == fitness_2)

            # fitness_1 == fitness_2
            for off2 in off_eq:
                fitness_2.setValues(
                    (fitness_1.values[0] + off1, fitness_1.values[1] + off2)
                )
                self.assertTrue(fitness_1 == fitness_2)

            # fitness_1 < fitness_2
            fitness_2.setValues(
                (fitness_1.values[0] + off1, fitness_1.values[1] - off_ne)
            )
            self.assertFalse(fitness_1 == fitness_2)

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
        :py:meth:`~culebra.abc.Fitness.__reduce__` methods,
        :py:meth:`~culebra.abc.Fitness.save_pickle` and
        :py:meth:`~culebra.abc.Fitness.load_pickle` methods.
        """
        fitness1 = MyFitness((1, 2))

        pickle_filename = "my_pickle.gz"
        fitness1.save_pickle(pickle_filename)
        fitness2 = MyFitness.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(fitness1, fitness2)

        # Remove the pickle file
        remove(pickle_filename)

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
