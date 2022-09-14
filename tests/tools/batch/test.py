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

"""Unit test for :py:class:`tools.Batch`."""

import unittest
from shutil import rmtree
from pandas import DataFrame
from culebra.base import Dataset, FitnessFunction, Wrapper
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
from culebra.wrapper.single_pop import Elitist
from culebra.wrapper.multi_pop import ParallelCooperative
from culebra.tools import Results, Batch, DEFAULT_NUM_EXPERIMENTS

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Training fitness function, 50% of samples used for validation
training_fitness_function = KappaNumFeatsC(
    training_data=training_data, test_prop=0.5
)

# Test fitness function
test_fitness_function = KappaNumFeatsC(
    training_data=training_data, test_data=test_data
)

# Species to optimize a SVM-based classifier
classifierOptimizationSpecies = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[1000, 1000],
    names=["C", "gamma"]
)

# Species for the feature selection problem
featureSelectionSpecies1 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    max_feat=dataset.num_feats/2,
)
featureSelectionSpecies2 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=dataset.num_feats/2 + 1,
)

# Parameters for the wrapper
params = {
    "individual_classes": [
        ClassifierOptimizationIndividual,
        FeatureSelectionIndividual,
        FeatureSelectionIndividual
    ],
    "species": [
        classifierOptimizationSpecies,
        featureSelectionSpecies1,
        featureSelectionSpecies2
    ],
    "fitness_function": training_fitness_function,
    "subpop_wrapper_cls": Elitist,
    "representation_size": 2,
    "num_gens": 3,
    "pop_sizes": 2,
    # At least one hyperparameter will be mutated
    "gene_ind_mutation_probs": (
        1.0/classifierOptimizationSpecies.num_hyperparams
    ),
    "checkpoint_enable": False,
    "verbose": False
}

# Create the wrapper
wrapper = ParallelCooperative(**params)

num_experiments = 3


class BatchTester(unittest.TestCase):
    """Test :py:class:`~tools.Batch`."""

    def test_init(self):
        """Test the :py:meth:`~tools.Batch.__init__` constructor."""
        # Try default params
        batch = Batch(wrapper)

        self.assertEqual(batch.wrapper, wrapper)
        self.assertEqual(batch.test_fitness_function, None)
        self.assertEqual(batch.results, None)
        self.assertEqual(batch.num_experiments, DEFAULT_NUM_EXPERIMENTS)

        # Try a batch with custom parameters
        batch = Batch(
            wrapper,
            test_fitness_function,
            num_experiments
        )
        self.assertEqual(
            batch.test_fitness_function, test_fitness_function)
        self.assertEqual(batch.num_experiments, num_experiments)

    def test_from_config(self):
        """Test the from_config factory method."""
        # Generate the batch
        batch = Batch.from_config("config.py")

        # Check the wrapper
        self.assertIsInstance(batch.wrapper, Wrapper)

        # Check the test fitness function
        self.assertIsInstance(
            batch.test_fitness_function, FitnessFunction)

        # Check the number of experiments
        self.assertGreater(batch.num_experiments, DEFAULT_NUM_EXPERIMENTS)

    def test_exp_labels(self):
        """Test the exp_labels method."""
        # Try default params
        batch = Batch(wrapper)

        for num_exp in range(1, 15):
            batch.num_experiments = num_exp
            labels = batch.exp_labels
            self.assertEqual(len(labels), num_exp)
            self.assertEqual(labels[-1], "exp" + str(num_exp - 1))

    def test_reset(self):
        """Test the reset method."""
        batch = Batch(wrapper)
        batch._results = 1
        batch._results_indices = {1: 2}

        batch.reset()
        self.assertEqual(batch.results, None)
        self.assertEqual(batch._results_indices, {})

    def test_execute(self):
        """Test the _execute method."""
        # Create the batch
        batch = Batch(wrapper, test_fitness_function, num_experiments)

        # Execute the batch
        batch._execute()

        # Check the results
        self.assertNotEqual(batch.results, None)
        self.assertIsInstance(batch.results, Results)

        for key in Batch._ResultKeys.keys():
            self.assertTrue(key in batch.results.keys())

        for key in batch.results:
            self.assertIsInstance(batch.results[key], DataFrame)

        # Remove the experiments
        for exp in batch.exp_labels:
            rmtree(exp)


if __name__ == '__main__':
    unittest.main()
