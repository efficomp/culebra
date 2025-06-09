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

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

from culebra.abc import Fitness, Species
from culebra.fitness_function import (
    DEFAULT_CLASSIFIER,
    DEFAULT_CV_FOLDS,
    DEFAULT_THRESHOLD
)

from culebra.fitness_function.abc import (
    DatasetScorer,
    ClassificationScorer,
    ClassificationFSScorer,
    RBFSVCScorer
)
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

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("obj1",)
        thresholds = [DEFAULT_THRESHOLD]

    def _evaluate_train_test(self, training_data, test_data):
        return (1, )

    def _evaluate_mccv(self, training_data):
        return (2, )

    def _evaluate_kfcv(self, training_data):
        return (3, )


class MyClassificationScorer(
    ClassificationScorer
):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("Kappa",)
        thresholds = [DEFAULT_THRESHOLD]

    _score = cohen_kappa_score


class MyClassificationFSScorer(ClassificationFSScorer):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("Kappa",)
        thresholds = [DEFAULT_THRESHOLD]

    _score = cohen_kappa_score


class MyRBFSVCScorer(RBFSVCScorer):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("Kappa",)
        thresholds = [DEFAULT_THRESHOLD]

    _score = cohen_kappa_score


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
        self.assertEqual(
            MyDatasetScorer(training_data, test_data).evaluate(None),
            (1, )
        )
        self.assertEqual(
            MyDatasetScorer(
                training_data, test_prop=0.5
            ).evaluate(None),
            (2, )
        )
        self.assertEqual(
            MyDatasetScorer(training_data).evaluate(None),
            (3, )
        )

    def test_copy(self):
        """Test the __copy__ method."""
        func1 = MyDatasetScorer(Dataset())
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = MyDatasetScorer(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test."""
        func1 = MyDatasetScorer(Dataset())

        pickle_filename = "my_pickle.gz"
        func1.save_pickle(pickle_filename)
        func2 = MyDatasetScorer.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyDatasetScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.abc.DatasetScorer`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.abc.DatasetScorer`
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


class ClassificationScorerTester(unittest.TestCase):
    """Test ClassificationScorer."""

    def test_init(self):
        """Test the constructor."""
        # Check default parameter values
        training_data, test_data = dataset.split(0.3)
        func = MyClassificationScorer(
            training_data=training_data
        )
        self.assertTrue(
            (func.training_data.inputs == training_data.inputs).all()
        )
        self.assertTrue(
            (func.training_data.outputs == training_data.outputs).all()
        )
        self.assertEqual(func.test_data, None)
        self.assertEqual(func.test_prop, None)
        self.assertTrue(isinstance(func.classifier, DEFAULT_CLASSIFIER))

        # Try a valid classifier
        func = MyClassificationScorer(
            training_data,
            classifier=KNeighborsClassifier(n_neighbors=5)
        )
        self.assertIsInstance(func.classifier, KNeighborsClassifier)

        # Try an invalid classifier. Should fail
        with self.assertRaises(TypeError):
            MyClassificationScorer(
                training_data,
                classifier='a'
            )

    def test_is_noisy(self):
        """Test the is_noisy property."""
        training_data, test_data = dataset.split(0.3)

        # Fitness function to be tested
        func = MyClassificationScorer(training_data)

        func.test_prop = None
        self.assertEqual(func.is_noisy, False)
        func.test_prop = 0.5
        self.assertEqual(func.is_noisy, True)

        func = MyClassificationScorer(training_data, test_data)
        self.assertEqual(func.is_noisy, False)

    def test_evaluate_train_test(self):
        """Test the _evaluate_train_test method."""
        training_data, test_data = dataset.split(0.3)
        func = MyClassificationScorer(
            training_data=training_data,
            test_data=test_data
        )

        scores = func._evaluate_train_test(training_data, test_data)

        outputs_pred = func.classifier.fit(
            training_data.inputs,
            training_data.outputs
        ).predict(test_data.inputs)

        self.assertEqual(
            cohen_kappa_score(test_data.outputs, outputs_pred),
            scores[0]
        )

    def test_evaluate_mccv(self):
        """Test the _evaluate_mccv method."""
        training_data = dataset
        func = MyClassificationScorer(
            training_data=training_data,
            test_prop=0.9
        )

        scores1 = func._evaluate_mccv(training_data)
        scores2 = func._evaluate_mccv(training_data)

        self.assertNotEqual(scores1, scores2)

    def test_evaluate_kfcv(self):
        """Test the _evaluate_kfcv method."""
        training_data = dataset
        func = MyClassificationScorer(
            training_data=training_data,
        )

        cv_scores = func._evaluate_kfcv(training_data)

        fold_scores = cross_val_score(
            func.classifier,
            training_data.inputs,
            training_data.outputs,
            cv=StratifiedKFold(n_splits=func.cv_folds),
            scoring=make_scorer(func.__class__._score)
        )

        self.assertEqual(
            fold_scores.mean(),
            cv_scores[0]
        )

    def test_copy(self):
        """Test the __copy__ method."""
        func1 = MyClassificationScorer(Dataset())
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.training_data), id(func2.training_data))
        self.assertEqual(id(func1.classifier), id(func2.classifier))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = MyClassificationScorer(Dataset())
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test."""
        func1 = MyClassificationScorer(Dataset())

        pickle_filename = "my_pickle.gz"
        func1.save_pickle(pickle_filename)
        func2 = MyClassificationScorer.load_pickle(
            pickle_filename
        )

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyClassificationScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.abc.ClassificationScorer`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.abc.ClassificationScorer`
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


class ClassificationFSScorerTester(unittest.TestCase):
    """Test ClassificationFSScorer."""

    def test_final_training_test_data(self):
        """Test the _final_training_test_data method."""
        training_data, test_data = dataset.split(0.3)
        species = FSSpecies(training_data.num_feats)
        selected_feats = [0, 1, 2]

        # Try with a fitness function with training and test data
        func = MyClassificationFSScorer(
            training_data=training_data,
            test_data=test_data
        )
        sol = FSSolution(species, func.Fitness, features=selected_feats)
        final_training, final_test = func._final_training_test_data(sol)

        self.assertEqual(final_training.num_feats, len(selected_feats))
        self.assertEqual(final_test.num_feats, len(selected_feats))

        # Try with a fitness function with only training data
        func = MyClassificationFSScorer(training_data=training_data)
        sol = FSSolution(species, func.Fitness, features=selected_feats)
        final_training, final_test = func._final_training_test_data(sol)

        self.assertEqual(final_training.num_feats, len(selected_feats))
        self.assertEqual(final_test, None)

    def test_num_nodes(self):
        """Test the num_nodes property."""
        func = MyClassificationFSScorer(dataset)
        self.assertEqual(func.num_nodes, dataset.num_feats)

    def test_heuristic(self):
        """Test the heuristic method."""
        func = MyClassificationFSScorer(dataset)

        # Try an invalid species. Should fail
        species = Species()
        with self.assertRaises(TypeError):
            func.heuristic(species)

        # Try a valid species
        num_feats = 10
        min_feat = 2
        max_feat = 8
        species = FSSpecies(
            num_feats=num_feats, min_feat=min_feat, max_feat=max_feat)
        (heuristic, ) = func.heuristic(species)
        self.assertIsInstance(heuristic, np.ndarray)
        self.assertEqual(heuristic.shape, (num_feats, num_feats))
        for row in range(num_feats):
            for column in range(num_feats):
                self.assertEqual(
                    heuristic[row][column],
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
        func = MyClassificationFSScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


class RBFSVCScorerTester(unittest.TestCase):
    """Test RBFSVCScorer."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = MyRBFSVCScorer(dataset)
        self.assertTrue(isinstance(func.classifier, SVC))

    def test_classifier(self):
        """Test the classifier property."""
        # Fitness function to be tested
        func = MyRBFSVCScorer(Dataset())

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
        func = MyRBFSVCScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


if __name__ == '__main__':
    unittest.main()
