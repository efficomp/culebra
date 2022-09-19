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

"""Test the cooperative fitness functions."""

import unittest
from culebra.base import Dataset
from culebra.fitness_function.cooperative import KappaNumFeatsC
from culebra.genotype.feature_selection import (
    Species as FeatureSelectionSpecies
)
from culebra.genotype.feature_selection.individual import (
    BitVector as FeatureSelectionIndividual
)
from culebra.genotype.classifier_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class KappaNumFeatsCTester(unittest.TestCase):
    """Test :py:class:`~fitness_function.cooperative.KappaNumFeatsC`."""

    def test_evaluate(self):
        """Test the evaluation method."""
        # Species to optimize a SVM-based classifier
        hyperparams_species = ClassifierOptimizationSpecies(
            lower_bounds=[0, 0],
            upper_bounds=[100000, 100000],
            names=["C", "gamma"]
        )

        # Species for the feature selection problem
        min_feat1 = 0
        max_feat1 = dataset.num_feats / 2
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

        hyperparams_ind = ClassifierOptimizationIndividual(
            hyperparams_species, KappaNumFeatsC.Fitness
        )

        features_ind1 = FeatureSelectionIndividual(
            features_species1, KappaNumFeatsC.Fitness
        )

        features_ind2 = FeatureSelectionIndividual(
            features_species2, KappaNumFeatsC.Fitness
        )

        representatives = [hyperparams_ind, features_ind1, features_ind2]

        # Fitness function to be tested
        fitness_func = KappaNumFeatsC(dataset)

        # Evaluate the individuals
        for index, ind in enumerate(representatives):
            ind.fitness.values = fitness_func.evaluate(
                ind, index, representatives
            )

        # Check that fitnesses match
        self.assertEqual(
            hyperparams_ind.fitness.values, features_ind1.fitness.values
        )
        self.assertEqual(
            features_ind1.fitness.values, features_ind2.fitness.values
        )

        # Try wrong individual species. Should fail
        with self.assertRaises(AttributeError):
            fitness_func.evaluate(features_ind1, 0, representatives)
        with self.assertRaises(AttributeError):
            fitness_func.evaluate(hyperparams_ind, 1, representatives)


if __name__ == '__main__':
    unittest.main()
