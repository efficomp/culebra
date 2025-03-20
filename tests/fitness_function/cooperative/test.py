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

"""Test the cooperative fitness functions."""

import unittest

from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Solution as ClassifierOptimizationSolution
)
from culebra.fitness_function.cooperative import (
    KappaNumFeatsC,
    KappaFeatsPropC,
    AccuracyNumFeatsC,
    AccuracyFeatsPropC
)
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Remove outliers
dataset.remove_outliers()

# Normalize inputs
dataset.robust_scale()

# Species to optimize a SVM-based classifier
hyperparams_species = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[100000, 100000],
    names=["C", "gamma"]
)

# Species for the feature selection problem
min_feat1 = 0
max_feat1 = dataset.num_feats // 2
features_species1 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=min_feat1,
    max_feat=max_feat1
)

min_feat2 = max_feat1 + 1
max_feat2 = dataset.num_feats - 1
features_species2 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=min_feat2,
    max_feat=max_feat2
)


class KappaNumFeatsCTester(unittest.TestCase):
    """Test KappaNumFeatsC."""

    FitnessFunc = KappaNumFeatsC

    def test_evaluate(self):
        """Test the evaluation method."""
        hyperparams_sol = ClassifierOptimizationSolution(
            hyperparams_species, self.FitnessFunc.Fitness
        )

        features_sol1 = FeatureSelectionSolution(
            features_species1, self.FitnessFunc.Fitness
        )

        features_sol2 = FeatureSelectionSolution(
            features_species2, self.FitnessFunc.Fitness
        )

        representatives = [hyperparams_sol, features_sol1, features_sol2]

        # Fitness function to be tested
        fitness_func = self.FitnessFunc(dataset)

        # Evaluate the solutions
        for index, sol in enumerate(representatives):
            sol.fitness.values = fitness_func.evaluate(
                sol, index, representatives
            )

        # Check that fitnesses match
        self.assertEqual(
            hyperparams_sol.fitness.values, features_sol1.fitness.values
        )
        self.assertEqual(
            features_sol1.fitness.values, features_sol2.fitness.values
        )

        # Try wrong solution species. Should fail
        with self.assertRaises(AttributeError):
            fitness_func.evaluate(features_sol1, 0, representatives)
        with self.assertRaises(AttributeError):
            fitness_func.evaluate(hyperparams_sol, 1, representatives)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        fitness_func = self.FitnessFunc(dataset)
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class KappaFeatsPropCTester(unittest.TestCase):
    """Test KappaFeatsPropC."""

    FitnessFunc = KappaFeatsPropC
    test_evaluate = KappaNumFeatsCTester.test_evaluate
    test_repr = KappaNumFeatsCTester.test_repr


class AccuracyNumFeatsCTester(unittest.TestCase):
    """Test AccuracyNumFeatsC."""

    FitnessFunc = AccuracyNumFeatsC
    test_evaluate = KappaNumFeatsCTester.test_evaluate
    test_repr = KappaNumFeatsCTester.test_repr


class AccuracyFeatsPropCTester(unittest.TestCase):
    """Test AccuracyFeatsPropC."""

    FitnessFunc = AccuracyFeatsPropC
    test_evaluate = KappaNumFeatsCTester.test_evaluate
    test_repr = KappaNumFeatsCTester.test_repr


if __name__ == '__main__':
    unittest.main()
