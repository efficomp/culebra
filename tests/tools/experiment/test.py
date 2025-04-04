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

"""Unit test for :py:class:`~culebra.tools.Experiment`."""

import unittest
from os import remove
from os.path import isfile

from pandas import DataFrame

from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.cooperative import KappaNumFeatsC
from culebra.trainer.ea import ElitistEA
from culebra.trainer.ea import ParallelCooperativeEA
from culebra.tools import Dataset, Experiment, Results


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Scale inputs
dataset.scale()

# Remove outliers
dataset.remove_outliers()

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Training fitness function
training_fitness_function = KappaNumFeatsC(
    training_data=training_data, cv_folds=5
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
    max_feat=dataset.num_feats//2,
)
featureSelectionSpecies2 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=dataset.num_feats//2 + 1,
)

# Parameters for the wrapper
params = {
    "solution_classes": [
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
    "subtrainer_cls": ElitistEA,
    "representation_size": 2,
    "max_num_iters": 3,
    "pop_sizes": 2,
    "gene_ind_mutation_probs": (
        # At least one hyperparameter/feature will be mutated
        1.0/classifierOptimizationSpecies.num_params,
        2.0/dataset.num_feats,
        2.0/dataset.num_feats
    ),
    "checkpoint_enable": False,
    "verbose": False
}

# Create the trainer
trainer = ParallelCooperativeEA(**params)


class ExperimentTester(unittest.TestCase):
    """Test :py:class:`~culebra.tools.Experiment`."""

    def test_reset(self):
        """Test the reset method."""
        experiment = Experiment(trainer)
        experiment._results = 1
        experiment._best_solutions = 2
        experiment._best_representatives = 3
        experiment.reset()
        self.assertEqual(experiment.results, None)
        self.assertEqual(experiment.best_solutions, None)
        self.assertEqual(experiment.best_representatives, None)

    def test_execute(self):
        """Test the _execute method."""
        # Create the experiment
        experiment = Experiment(
            trainer, test_fitness_function,
            hyperparameters={"a": 0, "b": 1}
        )

        # Execute the trainer
        experiment.run()

        # Check the results
        self.assertNotEqual(experiment.results, None)
        self.assertNotEqual(experiment.best_solutions, None)
        self.assertNotEqual(experiment.best_representatives, None)
        self.assertIsInstance(experiment.results, Results)

        for key in Experiment._ResultKeys.keys():
            self.assertTrue(key in experiment.results.keys())

        for key in experiment.results:
            self.assertIsInstance(experiment.results[key], DataFrame)

        # Check the result files
        isfile(experiment.results_pickle_filename)
        isfile(experiment.results_excel_filename)

        # Remove the files
        remove(experiment.results_pickle_filename)
        remove(experiment.results_excel_filename)

        # Try a different results base filename
        filename = "the_results"
        experiment = Experiment(
            trainer, test_fitness_function, filename
        )

        # Execute the trainer
        experiment.run()

        # Check the result files
        isfile(experiment.results_pickle_filename)
        isfile(experiment.results_excel_filename)

        # Remove the files
        remove(experiment.results_pickle_filename)
        remove(experiment.results_excel_filename)


if __name__ == '__main__':
    unittest.main()
