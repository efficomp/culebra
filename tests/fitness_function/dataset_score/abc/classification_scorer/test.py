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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer

from culebra import DEFAULT_SIMILARITY_THRESHOLD, SERIALIZED_FILE_EXTENSION
from culebra.fitness_function.dataset_score import (
    DEFAULT_CLASSIFIER,
    DEFAULT_CV_FOLDS
)
from culebra.fitness_function.dataset_score.abc import ClassificationScorer

from culebra.solution.feature_selection import (
    IntSolution as FSSolution,
    Species as FSSpecies
)
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class MyClassificationScorer(
    ClassificationScorer
):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("Kappa",)

    _score = cohen_kappa_score


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
        self.assertTrue(isinstance(func.classifier, DEFAULT_CLASSIFIER))
        self.assertEqual(func.cv_folds, DEFAULT_CV_FOLDS)
        self.assertEqual(func.index, 0)
        self.assertEqual(func.obj_thresholds, [DEFAULT_SIMILARITY_THRESHOLD])

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
        # Check a valid index
        valid_index = 3
        func = MyClassificationScorer(
            training_data=training_data,
            index=valid_index
        )
        self.assertEqual(func.index, valid_index)

    def test_evaluate_train_test(self):
        """Test the _evaluate_train_test method."""
        training_data, test_data = dataset.split(0.3)
        func = MyClassificationScorer(
            training_data=training_data,
            test_data=test_data
        )

        species = FSSpecies(training_data.num_feats)
        selected_feats = [0, 1, 2]
        sol = FSSolution(species, func.fitness_cls, features=selected_feats)

        fit_values = func._evaluate_train_test(
            sol,
            training_data,
            test_data
        ).values

        outputs_pred = func.classifier.fit(
            training_data.inputs,
            training_data.outputs
        ).predict(test_data.inputs)

        self.assertEqual(
            cohen_kappa_score(test_data.outputs, outputs_pred),
            sol.fitness.values[0]
        )
        self.assertEqual(fit_values, sol.fitness.values)

    def test_evaluate_kfcv(self):
        """Test the _evaluate_kfcv method."""
        training_data = dataset
        func = MyClassificationScorer(
            training_data=training_data,
        )

        species = FSSpecies(training_data.num_feats)
        selected_feats = [0, 1, 2]
        sol = FSSolution(species, func.fitness_cls, features=selected_feats)

        fit_values = func._evaluate_kfcv(sol, training_data).values

        fold_scores = cross_val_score(
            func.classifier,
            training_data.inputs,
            training_data.outputs,
            cv=StratifiedKFold(n_splits=func.cv_folds),
            scoring=make_scorer(func.__class__._score)
        )

        self.assertEqual(
            fold_scores.mean(),
            sol.fitness.values[0]
        )
        self.assertEqual(fit_values, sol.fitness.values)

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

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        func1.dump(serialized_filename)
        func2 = MyClassificationScorer.load(
            serialized_filename
        )

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyClassificationScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.dataset_score.abc.ClassificationScorer`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.dataset_score.abc.ClassificationScorer`
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
