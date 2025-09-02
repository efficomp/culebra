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
from os import remove
from copy import copy, deepcopy
from random import random

from culebra import DEFAULT_SIMILARITY_THRESHOLD, SERIALIZED_FILE_EXTENSION
from culebra.fitness_function.dataset_score import DEFAULT_CV_FOLDS
from culebra.fitness_function.dataset_score.abc import DatasetScorer

from culebra.solution.feature_selection import (
    IntSolution as FSSolution,
    Species as FSSpecies
)
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class MyDatasetScorer(DatasetScorer):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    def _evaluate_train_test(self, training_data, test_data):
        """Evaluate with a training and test datasets."""
        return (1,)

    def _evaluate_mccv(self, sol, training_data):
        """Perform a Monte Carlo cross-validation."""
        # Random value in [2, 3)
        new_fitness_value = 2 + 0.5 * random()

        # Number of evaluations performed to this solution
        num_evals = sol.fitness.num_evaluations[self.index]

        # If previously evaluated
        if num_evals > 0:
            average_fitness_value = (
                new_fitness_value + sol.fitness.values[self.index] * num_evals
            ) / (num_evals + 1)
        else:
            average_fitness_value = new_fitness_value

        sol.fitness.num_evaluations[self.index] += 1

        return (average_fitness_value, )

    def _evaluate_kfcv(self, training_data):
        """Perform a k-fold cross-validation."""
        return (3,)

    def is_evaluable(self, sol):
        """Return True if the solution is evaluable."""
        return True


class DatasetScorerTester(unittest.TestCase):
    """Test DatasetScorer."""

    def test_init(self):
        """Test the constructor."""
        # Check default parameter values
        training_data, test_data = dataset.split(0.3)
        func = MyDatasetScorer(training_data=training_data)
        self.assertTrue(
            (func.training_data.inputs == training_data.inputs).all()
        )
        self.assertTrue(
            (func.training_data.outputs == training_data.outputs).all()
        )
        self.assertEqual(func.test_data, None)
        self.assertEqual(func.test_prop, None)
        self.assertEqual(func.cv_folds, DEFAULT_CV_FOLDS)
        self.assertEqual(func.index, 0)
        self.assertEqual(func.obj_thresholds, [DEFAULT_SIMILARITY_THRESHOLD])

        # Try an invalid training dataset, should fail
        with self.assertRaises(TypeError):
            func = MyDatasetScorer(training_data='a')

        # Try a valid test dataset
        func = MyDatasetScorer(
            training_data=training_data,
            test_data=test_data
        )
        self.assertTrue((func.test_data.inputs == test_data.inputs).all())
        self.assertTrue((func.test_data.outputs == test_data.outputs).all())

        # Try an invalid test dataset, should fail
        with self.assertRaises(TypeError):
            func = MyDatasetScorer(
                training_data=training_data,
                test_data='a'
                )

        # Try a valid value for test_prop
        valid_test_prop = 0.5
        func = MyDatasetScorer(
            training_data=training_data,
            test_prop=valid_test_prop
        )
        self.assertEqual(func.test_prop, valid_test_prop)

        # Try an invalid type for test_prop. Should fail
        with self.assertRaises(TypeError):
            MyDatasetScorer(
                training_data=training_data,
                test_prop='a'
            )

        # Try invalid values for test_prop. Should fail
        invalid_test_prop_values = (-0.1, 0, 1, 1.3)
        for invalid_value in invalid_test_prop_values:
            with self.assertRaises(ValueError):
                MyDatasetScorer(
                    training_data=training_data,
                    test_prop=invalid_value
                )

        # Try a valid value for cv_folds
        valid_cv_folds = 10
        func = MyDatasetScorer(
            training_data=training_data,
            cv_folds=valid_cv_folds
        )
        self.assertEqual(func.cv_folds, valid_cv_folds)

        # Try a invalid types for cv_folds. Should fail
        invalid_cv_folds_types = ('a', 1.1)
        for invalid_type in invalid_cv_folds_types:
            with self.assertRaises(TypeError):
                MyDatasetScorer(
                    training_data=training_data,
                    cv_folds=invalid_type
                )

        # Try invalid values for cv_folds. Should fail
        invalid_cv_folds_values = (-3, 0)
        for invalid_value in invalid_cv_folds_values:
            with self.assertRaises(ValueError):
                MyDatasetScorer(
                    training_data=training_data,
                    cv_folds=invalid_value
                )

        # Check a valid index
        valid_index = 3
        func = MyDatasetScorer(
            training_data=training_data,
            index=valid_index
        )
        self.assertEqual(func.index, valid_index)

    def test_is_noisy(self):
        """Test the is_noisy property."""
        training_data, test_data = dataset.split(0.3)

        # Fitness function to be tested
        func = MyDatasetScorer(training_data)

        func.test_prop = None
        self.assertEqual(func.is_noisy, False)
        func.test_prop = 0.5
        self.assertEqual(func.is_noisy, True)

        func = MyDatasetScorer(training_data, test_data)
        self.assertEqual(func.is_noisy, False)

    def test_final_training_test_data(self):
        """Test the generation of final training and test data."""
        training_data, test_data = dataset.split(0.3)

        # Try if no test data was provided
        func = MyDatasetScorer(training_data)
        final_training, final_test = func._final_training_test_data(None)
        self.assertTrue(
            (training_data.inputs == final_training.inputs).all()
        )
        self.assertTrue(
            (training_data.outputs == final_training.outputs).all()
        )
        self.assertEqual(final_test, None)

        # Try now with some test data
        func = MyDatasetScorer(training_data, test_data)
        final_training, final_test = func._final_training_test_data(None)
        self.assertTrue(
            (training_data.inputs == final_training.inputs).all()
        )
        self.assertTrue(
            (training_data.outputs == final_training.outputs).all()
        )
        self.assertTrue(
            (test_data.inputs == final_test.inputs).all()
        )
        self.assertTrue(
            (test_data.outputs == final_test.outputs).all()
        )

    def test_evaluate(self):
        """Test the evaluation method."""
        training_data, test_data = dataset.split(0.3)
        species = FSSpecies(training_data.num_feats)
        selected_feats = [0, 1, 2]

        func = MyDatasetScorer(training_data, test_data)
        sol = FSSolution(species, func.fitness_cls, features=selected_feats)
        sol.fitness.values = func.evaluate(sol)
        self.assertEqual(sol.fitness.values, (1, ))
        del sol.fitness.values

        func = MyDatasetScorer(training_data, test_prop=0.5)
        sol.fitness.values = func.evaluate(sol)
        self.assertTrue(2 < sol.fitness.values[0] < 3)
        sol.fitness.values = func.evaluate(sol)
        self.assertTrue(2 < sol.fitness.values[0] < 3)
        del sol.fitness.values

        func = MyDatasetScorer(training_data)
        sol.fitness.values = func.evaluate(sol)
        self.assertEqual(sol.fitness.values, (3, ))

    def test_copy(self):
        """Test the __copy__ method."""
        func1 = MyDatasetScorer(Dataset(), index=2)
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))

        # Check the index
        self.assertEqual(func1.index, func2.index)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = MyDatasetScorer(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test."""
        func1 = MyDatasetScorer(Dataset())

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        func1.dump(serialized_filename)
        func2 = MyDatasetScorer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyDatasetScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.dataset_score.abc.DatasetScorer`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.dataset_score.abc.DatasetScorer`
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
