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
from random import random

import numpy as np

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

    def _evaluate_train_test(self, training_data, test_data):
        return (1,)

    def _evaluate_mccv(self, sol, training_data):
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
        return (3,)

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
        ind.fitness.values = func.evaluate(ind)
        self.assertEqual(ind.fitness.values, (1, ))
        del ind.fitness.values

        func = MyFSDatasetScorer(training_data, test_prop=0.5)
        ind.fitness.values = func.evaluate(ind)
        self.assertTrue(2 < ind.fitness.values[0] < 3)
        ind.fitness.values = func.evaluate(ind)
        self.assertTrue(2 < ind.fitness.values[0] < 3)
        del ind.fitness.values

        func = MyFSDatasetScorer(training_data)
        ind.fitness.values = func.evaluate(ind)
        self.assertEqual(ind.fitness.values, (3, ))

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

    def test_num_nodes(self):
        """Test the num_nodes property."""
        func = MyFSDatasetScorer(dataset)
        self.assertEqual(func.num_nodes, dataset.num_feats)

    def test_heuristic(self):
        """Test the heuristic method."""
        func = MyFSDatasetScorer(dataset)

        # Try an invalid species. Should fail
        species = ParamOptSpecies(
            lower_bounds=[0, 0],
            upper_bounds=[100000, 100000],
            names=["C", "gamma"]
        )
        with self.assertRaises(TypeError):
            func.heuristic(species)

        # Try a valid species
        num_feats = 10
        min_feat = 2
        max_feat = 8
        species = FSSpecies(
            num_feats=num_feats, min_feat=min_feat, max_feat=max_feat
        )
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
        func = MyFSDatasetScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


if __name__ == '__main__':
    unittest.main()
