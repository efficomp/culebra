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

"""Unit test for :py:class:`culebra.tools.Evaluation`."""

import unittest
import pickle
from os import remove
from os.path import exists
from copy import copy, deepcopy

from pandas import DataFrame

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
    Evaluation,
    Results,
    DEFAULT_SCRIPT_FILENAME
)

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

# Parameters for the trainer
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
    # At least one hyperparameter will be mutated
    "gene_ind_mutation_probs": (
        1.0/classifierOptimizationSpecies.num_params
    ),
    "checkpoint_enable": False,
    "verbose": False
}

# Create the trainer
trainer = ParallelCooperativeEA(**params)


class MyEvaluation(Evaluation):
    """Dummy evaluation subclass."""

    def _execute(self):
        """Implement a dummy execution."""
        # Insert a dummy dataframe
        self._results["dummy"] = DataFrame()


class EvaluationTester(unittest.TestCase):
    """Test :py:class:`~culebra.tools.Evaluation`."""

    def test_init(self):
        """Test the constructor."""
        # Try default params
        evaluation = MyEvaluation(trainer)

        self.assertEqual(evaluation.trainer, trainer)
        self.assertEqual(evaluation.test_fitness_function, None)
        self.assertEqual(evaluation.results_base_filename, None)
        self.assertEqual(evaluation.results, None)

        # Try an evaluation with a test fitness function
        evaluation = MyEvaluation(trainer, test_fitness_function)
        self.assertEqual(
            evaluation.test_fitness_function, test_fitness_function)

    def test_from_config(self):
        """Test the from_config factory method."""
        # Generate the evaluation
        evaluation = MyEvaluation.from_config("config.py")

        # Check the trainer
        self.assertIsInstance(evaluation.trainer, Trainer)

        # Check the test fitness function
        self.assertIsInstance(
            evaluation.test_fitness_function, FitnessFunction)

    def test_reset(self):
        """Test the reset method."""
        evaluation = MyEvaluation(trainer)
        evaluation._results = 1
        evaluation.reset()
        self.assertEqual(evaluation.results, None)

    def test_execute(self):
        """Test the _execute method."""
        # Create the evaluation
        evaluation = MyEvaluation(trainer, test_fitness_function)

        # Init the results manager
        evaluation._results = Results(evaluation.results_base_filename)

        # Execute the trainer
        evaluation._execute()

        # Check the results
        self.assertIsInstance(evaluation.results, Results)
        for key in evaluation.results:
            self.assertIsInstance(evaluation.results[key], DataFrame)

    def test_generate_script(self):
        """Test the generate_script method."""
        # Generate the script with a default name
        MyEvaluation.generate_script()

        # Check that file exists
        self.assertTrue(exists(DEFAULT_SCRIPT_FILENAME))

        # Remove the file
        remove(DEFAULT_SCRIPT_FILENAME)

        # Try a custom name
        script_filename = "my_script.py"
        MyEvaluation.generate_script(script_filename=script_filename)

        # Check that file exists
        self.assertTrue(exists(script_filename))

        # Remove the file
        remove(script_filename)

    def test_run(self):
        """Test the run method."""
        # Create the evaluation
        evaluation = MyEvaluation(
            trainer, test_fitness_function, "the_results"
        )

        # Run the trainer
        evaluation.run()

        # Check the results
        self.assertIsInstance(evaluation.results, Results)

        # Check that backup and results files exist
        self.assertTrue(exists(evaluation.results.backup_filename))
        self.assertTrue(exists(evaluation.results.excel_filename))

        # Remove the files
        remove(evaluation.results.backup_filename)
        remove(evaluation.results.excel_filename)

    def test_copy(self):
        """Test the :py:meth:`~culebra.tools.Evaluation.__copy__` method."""
        evaluation1 = MyEvaluation(trainer, test_fitness_function)

        evaluation1.run()
        evaluation2 = copy(evaluation1)

        # Copy only copies the first level (evaluation1 != evaluation2)
        self.assertNotEqual(id(evaluation1), id(evaluation2))

        # The species attributes are shared
        self.assertEqual(id(evaluation1._results), id(evaluation2._results))

        # Remove the files
        remove(evaluation1.results.backup_filename)
        remove(evaluation1.results.excel_filename)

    def test_deepcopy(self):
        """Test :py:meth:`~culebra.tools.Evaluation.__deepcopy__`."""
        evaluation1 = MyEvaluation(trainer, test_fitness_function)
        evaluation1.run()
        evaluation2 = deepcopy(evaluation1)

        # Check the copy
        self._check_deepcopy(evaluation1, evaluation2)

        # Remove the files
        remove(evaluation1.results.backup_filename)
        remove(evaluation1.results.excel_filename)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.tools.Evaluation.__setstate__` and
        :py:meth:`~culebra.tools.Evaluation.__reduce__` methods.
        """
        evaluation1 = MyEvaluation(trainer, test_fitness_function)
        evaluation1.run()

        data = pickle.dumps(evaluation1)
        evaluation2 = pickle.loads(data)

        # Check the copy
        self._check_deepcopy(evaluation1, evaluation2)

        # Remove the files
        remove(evaluation1.results.backup_filename)
        remove(evaluation1.results.excel_filename)

    def _check_deepcopy(self, evaluation1, evaluation2):
        """Check if *evaluation1* is a deepcopy of *evaluation2*.

        :param evaluation1: The first evaluation
        :type evaluation1: :py:class:`~culebra.tools.Evaluation`
        :param evaluation2: The second evaluation
        :type evaluation2: :py:class:`~culebra.tools.Evaluation`
        """
        # Copies all the levels
        self.assertNotEqual(id(evaluation1), id(evaluation2))
        if evaluation1.results is None:
            self.assertEqual(evaluation2.results, None)
        else:
            for key in evaluation1.results:
                self.assertTrue(
                    evaluation1.results[key].equals(evaluation2.results[key])
                )


if __name__ == '__main__':
    unittest.main()
