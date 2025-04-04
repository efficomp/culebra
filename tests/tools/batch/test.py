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

"""Unit test for :py:class:`culebra.tools.Batch`."""

import unittest
from os import remove
from os.path import isfile, exists, join
from shutil import rmtree
from copy import copy, deepcopy

from pandas import DataFrame

from culebra import PICKLE_FILE_EXTENSION
from culebra.abc import FitnessFunction, Trainer
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
from culebra.tools import (
    Dataset,
    Results,
    Batch,
    DEFAULT_NUM_EXPERIMENTS,
    DEFAULT_RESULTS_BASENAME,
    DEFAULT_RUN_SCRIPT_FILENAME
)

# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Scale inputs
dataset.scale()

# Remove outliers
dataset.remove_outliers()

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Training fitness function
my_training_fitness_function = KappaNumFeatsC(
    training_data=training_data, cv_folds=5
)

# Test fitness function
my_test_fitness_function = KappaNumFeatsC(
    training_data=training_data, test_data=test_data
)

# Species to optimize a SVM-based classifier
classifierOptimizationSpecies = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[1000, 1000],
    names=["C", "gamma"]
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
    "fitness_function": my_training_fitness_function,
    "subtrainer_cls": ElitistEA,
    "representation_size": 2,
    "crossover_probs": 0.8,
    "mutation_probs": 0.2,
    "gene_ind_mutation_probs": (
        # At least one hyperparameter/feature will be mutated
        1.0/classifierOptimizationSpecies.num_params,
        2.0/dataset.num_feats,
        2.0/dataset.num_feats
    ),
    "max_num_iters": 3,
    "pop_sizes": 2,
    "checkpoint_enable": False,
    "verbose": False
}

# Create the trainer
my_trainer = ParallelCooperativeEA(**params)

my_num_experiments = 3


class BatchTester(unittest.TestCase):
    """Test :py:class:`~culebra.tools.Batch`."""

    def test_init(self):
        """Test the :py:meth:`~culebra.tools.Batch.__init__` constructor."""
        # Try default params
        batch = Batch(my_trainer)

        self.assertEqual(batch.trainer, my_trainer)
        self.assertEqual(batch.test_fitness_function, None)
        self.assertEqual(batch.results, None)
        self.assertEqual(
            batch.results_base_filename,
            DEFAULT_RESULTS_BASENAME
        )
        self.assertEqual(batch.hyperparameters, None)
        self.assertEqual(batch.num_experiments, DEFAULT_NUM_EXPERIMENTS)

        # Try a batch with custom parameters
        my_filename = "the_results"
        my_hyperparameters = {"a": 1}
        batch = Batch(
            my_trainer,
            my_test_fitness_function,
            my_filename,
            my_hyperparameters,
            my_num_experiments
        )
        self.assertEqual(
            batch.test_fitness_function, my_test_fitness_function)
        self.assertEqual(
            batch.results_base_filename, my_filename)
        self.assertEqual(
            batch.hyperparameters, my_hyperparameters)
        self.assertEqual(batch.num_experiments, my_num_experiments)

    def test_from_config(self):
        """Test the from_config factory method."""
        # Generate the batch
        batch = Batch.from_config("config.py")

        # Check the trainer
        self.assertIsInstance(batch.trainer, Trainer)

        # Check the test fitness function
        self.assertIsInstance(
            batch.test_fitness_function, FitnessFunction)

        # Check the hyperparameters
        self.assertEqual(
            batch.hyperparameters,
            {"representation_size": 2, "max_num_iters": 100}
        )

        # Check the number of experiments
        self.assertGreater(batch.num_experiments, DEFAULT_NUM_EXPERIMENTS)

    def test_experiment_labels(self):
        """Test the experiment_labels method."""
        # Try default params
        batch = Batch(my_trainer)

        for num_exp in range(1, 15):
            batch.num_experiments = num_exp
            labels = batch.experiment_labels
            self.assertEqual(len(labels), num_exp)
            self.assertEqual(labels[-1], "exp" + str(num_exp - 1))

    def test_reset(self):
        """Test the reset method."""
        batch = Batch(my_trainer)
        batch._results = 1
        batch._results_indices = {1: 2}

        batch.reset()
        self.assertEqual(batch.results, None)
        self.assertEqual(batch._results_indices, {})

    def test_setup(self):
        """Test the setup method."""
        # Create the batch
        batch = Batch(
            trainer=my_trainer,
            test_fitness_function=my_test_fitness_function,
            results_base_filename="res2",
            hyperparameters={"a": 1, "b": 2},
            num_experiments=my_num_experiments
        )
        batch.setup()

        experiment_filename = batch.experiment_basename + PICKLE_FILE_EXTENSION

        # Check that the experimetn has been pickled
        self.assertTrue(isfile(experiment_filename))

        # Check the experiment folders
        for exp_folder in batch.experiment_labels:
            self.assertTrue(exists(exp_folder))
            self.assertTrue(
                isfile(
                    join(exp_folder, DEFAULT_RUN_SCRIPT_FILENAME)
                )
            )

        # Remove the files
        remove(experiment_filename)
        for exp_folder in batch.experiment_labels:
            rmtree(exp_folder)

    def test_run(self):
        """Test the run method."""
        # Create the batch
        batch = Batch(
            trainer=my_trainer,
            test_fitness_function=my_test_fitness_function,
            results_base_filename="res2",
            hyperparameters={"a": 1, "b": 2},
            num_experiments=my_num_experiments
        )

        # Execute the batch
        batch.run()

        # Check the results
        self.assertNotEqual(batch.results, None)
        self.assertIsInstance(batch.results, Results)

        for key in Batch._ResultKeys.keys():
            self.assertTrue(key in batch.results.keys())

        for key in batch.results:
            self.assertIsInstance(batch.results[key], DataFrame)

        # Remove the experiments
        for exp in batch.experiment_labels:
            rmtree(exp)

        # Check the result files
        isfile(batch.results_pickle_filename)
        isfile(batch.results_excel_filename)

        # Remove the result files
        remove(batch.results_pickle_filename)
        remove(batch.results_excel_filename)

    def test_copy(self):
        """Test the :py:meth:`~culebra.tools.Batch.__copy__` method."""
        batch1 = Batch(
            trainer=my_trainer,
            test_fitness_function=my_test_fitness_function,
            results_base_filename="res2",
            hyperparameters={"a": 1, "b": 2},
            num_experiments=my_num_experiments
        )

        batch1.run()
        batch2 = copy(batch1)

        # Copy only copies the first level (batch1 != batch2)
        self.assertNotEqual(id(batch1), id(batch2))

        # The results are shared
        self.assertEqual(id(batch1._results), id(batch2._results))

        # The number of experiments match
        self.assertEqual(batch1.num_experiments, batch2.num_experiments)

        # Remove the experiments
        for exp in batch1.experiment_labels:
            rmtree(exp)

        # Remove the result files
        remove(batch1.results_pickle_filename)
        remove(batch1.results_excel_filename)

    def test_deepcopy(self):
        """Test :py:meth:`~culebra.tools.Batch.__deepcopy__`."""
        batch1 = Batch(
            trainer=my_trainer,
            test_fitness_function=my_test_fitness_function,
            results_base_filename="res2",
            hyperparameters={"a": 1, "b": 2},
            num_experiments=my_num_experiments
        )

        batch1.run()
        batch2 = deepcopy(batch1)

        # Check the copy
        self._check_deepcopy(batch1, batch2)

        # Remove the experiments
        for exp in batch1.experiment_labels:
            rmtree(exp)

        # Remove the result files
        remove(batch1.results_pickle_filename)
        remove(batch1.results_excel_filename)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.tools.Batch.__setstate__` and
        :py:meth:`~culebra.tools.Batch.__reduce__` methods.
        """
        batch1 = Batch(
            trainer=my_trainer,
            test_fitness_function=my_test_fitness_function,
            results_base_filename="res2",
            hyperparameters={"a": 1, "b": 2},
            num_experiments=my_num_experiments
        )
        batch1.run()

        pickle_filename = "my_pickle.gz"
        batch1.save_pickle(pickle_filename)
        batch2 = Batch.load_pickle(pickle_filename)

        # Check the copy
        self._check_deepcopy(batch1, batch2)

        # Remove the experiments
        for exp in batch1.experiment_labels:
            rmtree(exp)

        # Remove the result files
        remove(batch1.results_pickle_filename)
        remove(batch1.results_excel_filename)

        # Remove the pickle file
        remove(pickle_filename)

    def _check_deepcopy(self, batch1, batch2):
        """Check if *batch1* is a deepcopy of *batch2*.

        :param batch1: The first batch
        :type batch1: :py:class:`~culebra.tools.Batch`
        :param batch2: The second batch
        :type batch2: :py:class:`~culebra.tools.Batch`
        """
        # Copies all the levels
        self.assertNotEqual(id(batch1), id(batch2))
        if batch1.results is None:
            self.assertEqual(batch2.results, None)
        else:
            for key in batch1.results:
                self.assertTrue(
                    batch1.results[key].equals(batch2.results[key])
                )

        # The number of experiments match
        self.assertEqual(batch1.num_experiments, batch2.num_experiments)


if __name__ == '__main__':
    unittest.main()
