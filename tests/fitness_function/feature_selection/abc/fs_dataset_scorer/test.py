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

from sklearn.metrics import cohen_kappa_score

from culebra.fitness_function.feature_selection.abc import FSDatasetScorer
from culebra.solution.parameter_optimization import (
    Species as ParamOptSpecies,
    Individual as ParamOptIndividual
)
from culebra.solution.feature_selection import (
    IntVector as FSIndividual,
    Species as FSSpecies
)
from culebra.tools import Dataset

# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class MyFSDatasetScorer(FSDatasetScorer):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("Kappa",)

    def _evaluate_train_test(self, sol, training_data, test_data):
        """Evaluate with a training and test datasets."""
        sol.fitness.update_value(1, self.index)
        return sol.fitness

    def _evaluate_kfcv(self, sol, training_data):
        """Perform a k-fold cross-validation."""
        sol.fitness.update_value(3, self.index)
        return sol.fitness

    _score = cohen_kappa_score


class FSDatasetScorerTester(unittest.TestCase):
    """Test FSDatasetScorer."""

    def test_evaluate(self):
        """Test the evaluate method."""
        func = MyFSDatasetScorer(dataset)

        # Try an invalid solution class. Should fail...
        invalid_species = ParamOptSpecies(
            lower_bounds=[0, 0],
            upper_bounds=[100000, 100000],
            names=["C", "gamma"]
        )

        invalid_ind = ParamOptIndividual(invalid_species, func.fitness_cls)
        with self.assertRaises(ValueError):
            func.evaluate(invalid_ind)

        # Try valid solutions
        training_data, test_data = dataset.split(0.3)
        species = FSSpecies(training_data.num_feats)
        selected_feats = [0, 1, 2]

        func = MyFSDatasetScorer(training_data, test_data)
        ind = FSIndividual(species, func.fitness_cls, features=selected_feats)
        fit_values = func.evaluate(ind).values
        self.assertEqual(ind.fitness.values, (1, ))
        self.assertEqual(fit_values, ind.fitness.values)
        del ind.fitness.values

        func = MyFSDatasetScorer(training_data)
        fit_values = func.evaluate(ind).values
        self.assertEqual(ind.fitness.values, (3, ))
        self.assertEqual(fit_values, ind.fitness.values)

    def test_final_training_test_data(self):
        """Test the _final_training_test_data method."""
        training_data, test_data = dataset.split(0.3)
        species = FSSpecies(training_data.num_feats)
        selected_feats = [0, 1, 2]

        # Try with a fitness function with training and test data
        func = MyFSDatasetScorer(
            training_data=training_data,
            test_data=test_data
        )
        ind = FSIndividual(species, func.fitness_cls, features=selected_feats)
        final_training, final_test = func._final_training_test_data(ind)

        self.assertEqual(final_training.num_feats, len(selected_feats))
        self.assertEqual(final_test.num_feats, len(selected_feats))

        # Try with a fitness function with only training data
        func = MyFSDatasetScorer(training_data=training_data)
        ind = FSIndividual(species, func.fitness_cls, features=selected_feats)
        final_training, final_test = func._final_training_test_data(ind)

        self.assertEqual(final_training.num_feats, len(selected_feats))
        self.assertEqual(final_test, None)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyFSDatasetScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


if __name__ == '__main__':
    unittest.main()
