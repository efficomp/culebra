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

from culebra import DEFAULT_SIMILARITY_THRESHOLD, SERIALIZED_FILE_EXTENSION
from culebra.fitness_func.dataset_score import (
    DEFAULT_CV_NUM_FOLDS,
    DEFAULT_CV_FIXED_FOLDS
    )
from culebra.fitness_func.dataset_score.abc import DatasetScorer

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

    @property
    def _worst_score(self):
        """Worst achievable score."""
        return -1

    def _evaluate_train_test(self, sol, training_data, test_data):
        """Evaluate with a training and test datasets."""
        return (1,)

    def _evaluate_kfcv(self, sol, training_data):
        """Perform a k-fold cross-validation."""
        return (3,)

    def is_evaluable(self, sol):
        """Return True if the solution is evaluable."""
        return True

    _score = None


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
        self.assertEqual(func.cv_num_folds, DEFAULT_CV_NUM_FOLDS)
        self.assertEqual(func.cv_fixed_folds, DEFAULT_CV_FIXED_FOLDS)
        self.assertEqual(func.index, 0)
        self.assertEqual(
            func.obj_thresholds, (DEFAULT_SIMILARITY_THRESHOLD,)
        )
        self.assertIsNotNone(func.cv_splitter)

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

        # Try a valid value for cv_num_folds
        valid_cv_num_folds = 10
        func = MyDatasetScorer(
            training_data=training_data,
            cv_num_folds=valid_cv_num_folds
        )
        self.assertEqual(func.cv_num_folds, valid_cv_num_folds)

        # Try a invalid types for cv_num_folds. Should fail
        invalid_cv_num_folds_types = ('a', 1.1)
        for invalid_type in invalid_cv_num_folds_types:
            with self.assertRaises(TypeError):
                MyDatasetScorer(
                    training_data=training_data,
                    cv_num_folds=invalid_type
                )

        # Try invalid values for cv_num_folds. Should fail
        invalid_cv_num_folds_values = (-3, 0)
        for invalid_value in invalid_cv_num_folds_values:
            with self.assertRaises(ValueError):
                MyDatasetScorer(
                    training_data=training_data,
                    cv_num_folds=invalid_value
                )

        # Try a valid value for cv_fixed_folds
        valid_cv_fixed_folds = True
        func = MyDatasetScorer(
            training_data=training_data,
            cv_fixed_folds=valid_cv_fixed_folds
        )
        self.assertEqual(func.cv_fixed_folds, valid_cv_fixed_folds)

        # Try invalid types for cv_fixed_folds. Should fail
        invalid_cv_fixed_folds_types = ('a', 1.1)
        for invalid_type in invalid_cv_fixed_folds_types:
            with self.assertRaises(TypeError):
                MyDatasetScorer(
                    training_data=training_data,
                    cv_fixed_folds=invalid_type
                )

        # Check a valid index
        valid_index = 3
        func = MyDatasetScorer(
            training_data=training_data,
            index=valid_index
        )
        self.assertEqual(func.index, valid_index)

    def test_cv_splitter(self):
        """Test the cv_splitter property."""
        training_data, _ = dataset.split(0.3)
        func = MyDatasetScorer(training_data)

        # Get the splitter
        cv_splitter = func.cv_splitter

        # Change the number of folds. The splitter should be reset
        func.cv_num_folds = 12
        self.assertNotEqual(func.cv_splitter, cv_splitter)
        cv_splitter = func.cv_splitter

        # Let the folds become not fixed
        func.cv_fixed_folds = False
        self.assertNotEqual(func.cv_splitter, cv_splitter)

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
        self.assertEqual(func.evaluate(sol), (1, ))
        del sol.fitness.values

        func = MyDatasetScorer(training_data)
        self.assertEqual(func.evaluate(sol), (3, ))

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
            :class:`~culebra.fitness_func.dataset_score.abc.DatasetScorer`
        :param func2: The second fitness function
        :type func2:
            :class:`~culebra.fitness_func.dataset_score.abc.DatasetScorer`
        """
        # Copies all the levels
        self.assertTrue(func1 is not func2)
        self.assertTrue(func1.training_data is not func2.training_data)

        self.assertTrue(
            (func1.training_data.inputs == func2.training_data.inputs).all()
        )
        self.assertTrue(
            (func1.training_data.outputs == func2.training_data.outputs).all()
        )


if __name__ == '__main__':
    unittest.main()
