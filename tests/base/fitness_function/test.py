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

"""Unit test for :py:class:`base.FitnessFunction`."""

import unittest
from copy import copy, deepcopy
import pickle
from sklearn.neighbors import KNeighborsClassifier
from culebra.base import Fitness, FitnessFunction, Dataset, DEFAULT_CLASSIFIER

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = (0.001, 0.001)

    def evaluate(self, ind, index, representatives):
        """Dummy implementation of the evaluation function."""


class FitnessFunctionTester(unittest.TestCase):
    """Test :py:class:`~base.FitnessFunction`."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = MyFitnessFunction(dataset)
        self.assertTrue((dataset.inputs == func.training_data.inputs).all())
        self.assertTrue((dataset.outputs == func.training_data.outputs).all())
        self.assertEqual(func.test_data, None)
        self.assertEqual(func.test_prop, None)
        self.assertTrue(isinstance(func.classifier, DEFAULT_CLASSIFIER))

    def test_set_fitness_thresholds(self):
        """Test the set_fitness_thresholds class method."""
        invalid_threshold_types = (type, {}, len)
        invalid_threshold_value = -1
        valid_thresholds = (0.33, 0.5, 2)

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

    def test_num_obj(self):
        """Test the num_obj property."""
        # Fitness function to be tested
        func = MyFitnessFunction(Dataset())
        self.assertEqual(func.num_obj, len(MyFitnessFunction.Fitness.weights))

    def test_training_data(self):
        """Test the training_data property."""
        # Fitness function to be tested
        func = MyFitnessFunction(Dataset())

        # Check a valid value
        func.training_data = dataset
        self.assertEqual(func.training_data, dataset)

        # Check type
        with self.assertRaises(TypeError):
            func.training_data = 'a'

    def test_test_data(self):
        """Test the test_data property."""
        # Fitness function to be tested
        func = MyFitnessFunction(Dataset())

        # Check a valid value
        func.test_data = dataset
        self.assertEqual(func.test_data, dataset)

        # Check None
        func.test_data = None
        self.assertEqual(func.test_data, None)

        # Check type
        with self.assertRaises(TypeError):
            func.test_data = 'a'

    def test_test_prop(self):
        """Test the test_prop property."""
        # Fitness function to be tested
        func = MyFitnessFunction(Dataset())

        # Check a valid value
        func.test_prop = None
        self.assertEqual(func.test_prop, None)
        func.test_prop = 0.5
        self.assertEqual(func.test_prop, 0.5)

        # Check type
        with self.assertRaises(TypeError):
            func.test_prop = 'a'

        # Check invalid values
        with self.assertRaises(ValueError):
            func.test_prop = -0.1
        with self.assertRaises(ValueError):
            func.test_prop = 1.1

    def test_classifier(self):
        """Test the valid_prop property."""
        # Fitness function to be tested
        func = MyFitnessFunction(Dataset())

        # Check a valid value
        func.classifier = KNeighborsClassifier(n_neighbors=5)
        self.assertIsInstance(func.classifier, KNeighborsClassifier)

        # Check invalid values
        with self.assertRaises(TypeError):
            func.classifier = 'a'

    def test_final_training_test_data(self):
        """Test the generation of final training and test data."""
        training_data, test_data = dataset.split(0.3)
        func = MyFitnessFunction(training_data)

        # Try providing test data
        func.test_data = test_data
        training, test = func._final_training_test_data()
        self.assertTrue(training is training_data)
        self.assertTrue(test is test_data)

        # Try with test_data == None and test_prop == None.
        # Training data should not be split
        func.test_data = None
        training, test = func._final_training_test_data()
        self.assertTrue(training_data is training is test)

        # Set a test proportion
        func.test_prop = 0.3
        training, test = func._final_training_test_data()
        self.assertTrue(
            training.size, round(training_data.size * (1 - func.test_prop))
        )
        self.assertTrue(
            test.size, round(training_data.size * func.test_prop)
        )

    def test_copy(self):
        """Test the :py:meth:`~base.FitnessFunction.__copy__` method."""
        func1 = MyFitnessFunction(Dataset())
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))
        self.assertEqual(id(func1.classifier), id(func2.classifier))

    def test_deepcopy(self):
        """Test the :py:meth:`~base.FitnessFunction.__deepcopy__` method."""
        func1 = MyFitnessFunction(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~base.FitnessFunction.__setstate__` and
        :py:meth:`~base.FitnessFunction.__reduce__` methods.
        """
        func1 = MyFitnessFunction(Dataset())

        data = pickle.dumps(func1)
        func2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(func1, func2)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1: :py:class:`~base.FitnessFunction`
        :param func2: The second fitness function
        :type func2: :py:class:`~base.FitnessFunction`
        """
        # Copies all the levels
        self.assertNotEqual(id(func1), id(func2))
        self.assertNotEqual(id(func1.training_data), id(func2.training_data))

        self.assertTrue(
            (func1.training_data.inputs == func2.training_data.inputs).all()
        )
        self.assertTrue(
            (func1.training_data.outputs == func2.training_data.outputs).all()
        )


if __name__ == '__main__':
    unittest.main()
