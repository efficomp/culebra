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

"""Test the feature selection fitness functions."""

import unittest
from os import remove
from copy import copy, deepcopy

from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
)
from culebra.fitness_function.feature_selection import (
    NumFeats,
    KappaIndex,
    Accuracy,
    KappaNumFeats,
    AccuracyNumFeats
)
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class NumFeatsTester(unittest.TestCase):
    """Test NumFeats."""

    FitnessFunc = NumFeats

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Species for the solution
        species = Species(num_feats=dataset.num_feats)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        sol.fitness.values = func.evaluate(sol)

        # Check the fitness function
        self.assertEqual(sol.fitness.values, (sol.num_feats,))

    def test_copy(self):
        """Test the __copy__ method."""
        func1 = self.FitnessFunc(dataset)
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))
        self.assertEqual(id(func1.classifier), id(func2.classifier))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = self.FitnessFunc(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        func1 = self.FitnessFunc(Dataset())

        pickle_filename = "my_pickle.gz"
        func1.save_pickle(pickle_filename)
        func2 = self.FitnessFunc.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the pickle file
        remove(pickle_filename)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.abc.FeatureSelectionFitnessFunction`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.abc.FeatureSelectionFitnessFunction`
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

    def test_repr(self):
        """Test the repr and str dunder methods."""
        fitness_func = self.FitnessFunc(dataset)
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class KappaIndexTester(unittest.TestCase):
    """Test KappaIndex."""

    FitnessFunc = KappaIndex

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Species for the solution
        species = Species(num_feats=dataset.num_feats)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Check that Kappa is in [-1, 1]
        sol.fitness.values = func.evaluate(sol)
        self.assertGreaterEqual(sol.fitness.values[0], -1)
        self.assertLessEqual(sol.fitness.values[0], 1)

    test_copy = NumFeatsTester.test_copy
    test_deepcopy = NumFeatsTester.test_deepcopy
    test_serialization = NumFeatsTester.test_serialization
    _check_deepcopy = NumFeatsTester._check_deepcopy

    test_copy = NumFeatsTester.test_copy
    test_deepcopy = NumFeatsTester.test_deepcopy
    test_serialization = NumFeatsTester.test_serialization
    test_repr = NumFeatsTester.test_repr
    _check_deepcopy = NumFeatsTester._check_deepcopy


class AccuracyTester(unittest.TestCase):
    """Test Accuracy."""

    FitnessFunc = Accuracy

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Species for the solution
        species = Species(num_feats=dataset.num_feats)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Check that accuracy is in [0, 1]
        sol.fitness.values = func.evaluate(sol)
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)

    test_copy = NumFeatsTester.test_copy
    test_deepcopy = NumFeatsTester.test_deepcopy
    test_serialization = NumFeatsTester.test_serialization
    _check_deepcopy = NumFeatsTester._check_deepcopy

    test_copy = NumFeatsTester.test_copy
    test_deepcopy = NumFeatsTester.test_deepcopy
    test_serialization = NumFeatsTester.test_serialization
    test_repr = NumFeatsTester.test_repr
    _check_deepcopy = NumFeatsTester._check_deepcopy


class KappaNumFeatsTester(unittest.TestCase):
    """Test KappaNumFeats."""

    FitnessFunc = KappaNumFeats

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Species for the solution
        species = Species(num_feats=dataset.num_feats)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Evaluate the solution
        sol.fitness.values = func.evaluate(sol)

        # Check that Kappa is in [-1, 1]
        self.assertGreaterEqual(sol.fitness.values[0], -1)
        self.assertLessEqual(sol.fitness.values[0], 1)

        # Check the number of features
        self.assertEqual(sol.fitness.values[1], sol.num_feats)

    test_copy = NumFeatsTester.test_copy
    test_deepcopy = NumFeatsTester.test_deepcopy
    test_serialization = NumFeatsTester.test_serialization
    test_repr = NumFeatsTester.test_repr
    _check_deepcopy = NumFeatsTester._check_deepcopy


class AccuracyNumFeatsTester(unittest.TestCase):
    """Test AccuracyNumFeats."""

    FitnessFunc = AccuracyNumFeats

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Species for the solution
        species = Species(num_feats=dataset.num_feats)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Evaluate the solution
        sol.fitness.values = func.evaluate(sol)

        # Check that accuracy is in [0, 1]
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)

        # Check the number of features
        self.assertEqual(sol.fitness.values[1], sol.num_feats)

    test_copy = NumFeatsTester.test_copy
    test_deepcopy = NumFeatsTester.test_deepcopy
    test_serialization = NumFeatsTester.test_serialization
    test_repr = NumFeatsTester.test_repr
    _check_deepcopy = NumFeatsTester._check_deepcopy


if __name__ == '__main__':
    unittest.main()
