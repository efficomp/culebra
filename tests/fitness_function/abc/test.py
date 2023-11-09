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

"""Test the abstract base feature selection fitness functions."""

import unittest
import pickle
from copy import copy, deepcopy

from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from culebra.abc import Fitness, Species
from culebra.fitness_function import DEFAULT_CLASSIFIER
from culebra.fitness_function.abc import (
    DatasetFitnessFunction,
    ClassificationFitnessFunction,
    FeatureSelectionFitnessFunction,
    RBFSVCFitnessFunction
)
from culebra.solution.feature_selection import Species as FSSpecies
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyDatasetFitnessFunction(DatasetFitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = (0.001, 0.001)

    def evaluate(self, sol, index, representatives):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.
        """


class MyClassificationFitnessFunction(ClassificationFitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = (0.001, 0.001)

    def evaluate(self, sol, index, representatives):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.
        """


class MyFeatureSelectionFitnessFunction(FeatureSelectionFitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = (0.001, 0.001)

    def evaluate(self, sol, index, representatives):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.
        """


class MyRBFSVCFitnessFunction(RBFSVCFitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = (0.001, 0.001)

    def evaluate(self, sol, index, representatives):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.
        """


class DatasetFitnessFunctionTester(unittest.TestCase):
    """Test DatasetFitnessFunction."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = MyDatasetFitnessFunction(dataset)
        self.assertTrue((dataset.inputs == func.training_data.inputs).all())
        self.assertTrue((dataset.outputs == func.training_data.outputs).all())
        self.assertEqual(func.test_data, None)
        self.assertEqual(func.test_prop, None)

    def test_training_data(self):
        """Test the training_data property."""
        # Fitness function to be tested
        func = MyDatasetFitnessFunction(Dataset())

        # Check a valid value
        func.training_data = dataset
        self.assertEqual(func.training_data, dataset)

        # Check type
        with self.assertRaises(TypeError):
            func.training_data = 'a'

    def test_test_data(self):
        """Test the test_data property."""
        # Fitness function to be tested
        func = MyDatasetFitnessFunction(Dataset())

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
        func = MyDatasetFitnessFunction(Dataset())

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

    def test_final_training_test_data(self):
        """Test the generation of final training and test data."""
        training_data, test_data = dataset.split(0.3)
        func = MyDatasetFitnessFunction(training_data)

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
        """Test the __copy__ method."""
        func1 = MyDatasetFitnessFunction(Dataset())
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = MyDatasetFitnessFunction(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test."""
        func1 = MyDatasetFitnessFunction(Dataset())

        data = pickle.dumps(func1)
        func2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(func1, func2)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyDatasetFitnessFunction(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.abc.DatasetFitnessFunction`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.abc.DatasetFitnessFunction`
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


class ClassificationFitnessFunctionTester(unittest.TestCase):
    """Test ClassificationFitnessFunction."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = ClassificationFitnessFunction(dataset)
        self.assertTrue((dataset.inputs == func.training_data.inputs).all())
        self.assertTrue((dataset.outputs == func.training_data.outputs).all())
        self.assertEqual(func.test_data, None)
        self.assertEqual(func.test_prop, None)
        self.assertTrue(isinstance(func.classifier, DEFAULT_CLASSIFIER))

    def test_classifier(self):
        """Test the valid_prop property."""
        # Fitness function to be tested
        func = ClassificationFitnessFunction(Dataset())

        # Check a valid value
        func.classifier = KNeighborsClassifier(n_neighbors=5)
        self.assertIsInstance(func.classifier, KNeighborsClassifier)

        # Check invalid values
        with self.assertRaises(TypeError):
            func.classifier = 'a'

    def test_copy(self):
        """Test the __copy__ method."""
        func1 = MyClassificationFitnessFunction(Dataset())
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))
        self.assertEqual(id(func1.classifier), id(func2.classifier))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = MyClassificationFitnessFunction(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test."""
        func1 = MyClassificationFitnessFunction(Dataset())

        data = pickle.dumps(func1)
        func2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(func1, func2)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyClassificationFitnessFunction(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.abc.ClassificationFitnessFunction`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.abc.ClassificationFitnessFunction`
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


class FeatureSelectionFitnessFunctionTester(unittest.TestCase):
    """Test FeatureSelectionFitnessFunction."""

    def test_num_nodes(self):
        """Test the num_nodes property."""
        func = MyFeatureSelectionFitnessFunction(dataset)
        self.assertEqual(func.num_nodes, dataset.num_feats)

    def test_heuristics(self):
        """Test the heuristics method."""
        func = MyFeatureSelectionFitnessFunction(dataset)

        # Try an invalid species. Should fail
        species = Species()
        with self.assertRaises(TypeError):
            func.heuristics(species)

        # Try a valid species
        num_feats = 10
        min_feat = 2
        max_feat = 8
        species = FSSpecies(
            num_feats=num_feats, min_feat=min_feat, max_feat=max_feat)
        (heuristics, ) = func.heuristics(species)
        self.assertIsInstance(heuristics, ndarray)
        self.assertEqual(heuristics.shape, (num_feats, num_feats))
        for row in range(num_feats):
            for column in range(num_feats):
                self.assertEqual(
                    heuristics[row][column],
                    0 if (
                        row == column or
                        row < min_feat or
                        row > max_feat or
                        column < min_feat or
                        column > max_feat
                    ) else 1
                )

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyFeatureSelectionFitnessFunction(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


class RBFSVCFitnessFunctionTester(unittest.TestCase):
    """Test RBFSVCFitnessFunction."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = MyRBFSVCFitnessFunction(dataset)
        self.assertTrue(isinstance(func.classifier, SVC))

    def test_classifier(self):
        """Test the valid_prop property."""
        # Fitness function to be tested
        func = MyRBFSVCFitnessFunction(Dataset())

        # Check a valid value
        func.classifier = SVC(kernel='rbf')
        self.assertIsInstance(func.classifier, SVC)
        self.assertEqual(func.classifier.kernel, 'rbf')

        # Check invalid values
        with self.assertRaises(TypeError):
            func.classifier = 'a'

        with self.assertRaises(ValueError):
            func.classifier = SVC(kernel='linear')

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyRBFSVCFitnessFunction(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


if __name__ == '__main__':
    unittest.main()
