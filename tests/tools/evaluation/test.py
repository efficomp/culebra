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

"""Unit test for :py:class:`culebra.tools.Evaluation`."""

import unittest
from os import remove
from os.path import exists
from copy import copy, deepcopy
from collections import Counter

from pandas import DataFrame
from sklearn.svm import SVC

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import FitnessFunction, Trainer
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats
)
from culebra.fitness_function.svc_optimization import C
from culebra.fitness_function.cooperative import FSSVCScorer
from culebra.trainer.ea import ElitistEA
from culebra.trainer.ea import ParallelCooperativeEA
from culebra.tools import (
    Dataset,
    Evaluation,
    Results,
    DEFAULT_RUN_SCRIPT_FILENAME,
    DEFAULT_RESULTS_BASENAME
)

SCRIPT_FILE_EXTENSION = ".py"
"""File extension for python scripts."""


# Fitness function
def KappaNumFeatsC(
    training_data, test_data=None, test_prop=None, cv_folds=None
):
    """Fitness Function."""
    return FSSVCScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            test_prop=test_prop,
            classifier=SVC(kernel='rbf'),
            cv_folds=cv_folds
        ),
        NumFeats(),
        C()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Oversample the training data to make all the clases have the same number
# of samples
training_data = training_data.oversample(random_seed=0)


# Training fitness function
training_fitness_function = KappaNumFeatsC(training_data, cv_folds=5)

# Set the training fitness similarity threshold
training_fitness_function.obj_thresholds = 0.001

# Untie fitness function to select the best solution
samples_per_class = Counter(training_data.outputs)
max_folds = samples_per_class[
    min(samples_per_class, key=samples_per_class.get)
]
untie_best_fitness_function = KappaNumFeatsC(training_data, cv_folds=max_folds)

# Test fitness function
test_fitness_function = KappaNumFeatsC(training_data, test_data)

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
        self.assertEqual(evaluation.untie_best_fitness_function, None)

        self.assertEqual(evaluation.test_fitness_function, None)
        self.assertEqual(
            evaluation.results_base_filename, DEFAULT_RESULTS_BASENAME
        )
        self.assertEqual(evaluation.hyperparameters, None)
        self.assertEqual(evaluation.results, None)

        # Try an invalid untie fitness function
        with self.assertRaises(TypeError):
            MyEvaluation(trainer, untie_best_fitness_function="a")

        # Try an invalid test fitness function
        with self.assertRaises(TypeError):
            MyEvaluation(trainer, test_fitness_function="a")

        # Try an invalid results base filename
        with self.assertRaises(TypeError):
            MyEvaluation(trainer, results_base_filename=1)

        # Try an invalid hyperparameter specification
        with self.assertRaises(TypeError):
            MyEvaluation(trainer, hyperparameters=1)

        # Try an invalid hyperparameter name
        with self.assertRaises(ValueError):
            MyEvaluation(trainer, hyperparameters={1: 1})

        # Try a reserved hyperparameter name
        with self.assertRaises(ValueError):
            MyEvaluation(trainer, hyperparameters={'Solution': 1})

        # Try an evaluation with a custom untie fitness function
        evaluation = MyEvaluation(
            trainer, untie_best_fitness_function=untie_best_fitness_function
        )
        self.assertEqual(
            evaluation.untie_best_fitness_function, untie_best_fitness_function
        )

        # Try an evaluation with a custom test fitness function
        evaluation = MyEvaluation(
            trainer, test_fitness_function=test_fitness_function
        )
        self.assertEqual(
            evaluation.test_fitness_function, test_fitness_function
        )

        # Try an evaluation with a custom results base name
        my_basename = "my_base"
        evaluation = MyEvaluation(trainer, results_base_filename=my_basename)
        self.assertEqual(
            evaluation.results_base_filename, my_basename
        )

        # Try an evaluation with custom hyperparameters
        my_hyperparameters = {"a": 1, "b": 2}
        evaluation = MyEvaluation(trainer, hyperparameters=my_hyperparameters)
        self.assertEqual(
            evaluation.hyperparameters, my_hyperparameters
        )

    def test_from_config(self):
        """Test the from_config factory method."""
        # Try invalid configuration filenames. Should fail...
        with self.assertRaises(TypeError):
            MyEvaluation.from_config(2)
        with self.assertRaises(ValueError):
            MyEvaluation.from_config("bad_config.txt")
        with self.assertRaises(RuntimeError):
            MyEvaluation.from_config("bad_config" + SCRIPT_FILE_EXTENSION)

        # Generate the evaluation
        evaluation = MyEvaluation.from_config()

        # Check the trainer
        self.assertIsInstance(evaluation.trainer, Trainer)

        # Check the untie fitness function
        self.assertIsInstance(
            evaluation.untie_best_fitness_function, FitnessFunction)

        # Check the test fitness function
        self.assertIsInstance(
            evaluation.test_fitness_function, FitnessFunction)

        # Check the results base filename
        self.assertEqual(evaluation.results_base_filename, "my_results")

        # Check the hyperparameters
        self.assertEqual(
            evaluation.hyperparameters,
            {"representation_size": 2, "max_num_iters": 100}
        )

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
        evaluation._results = Results()

        # Execute the trainer
        evaluation._execute()

        # Check the results
        self.assertIsInstance(evaluation.results, Results)
        for key in evaluation.results:
            self.assertIsInstance(evaluation.results[key], DataFrame)

    def test_generate_run_script(self):
        """Test the generate_script method."""
        # Try an invalid filename. Should fail...
        with self.assertRaises(TypeError):
            MyEvaluation.generate_run_script(5)

        # Try an invalid config_filename. Should fail...
        with self.assertRaises(ValueError):
            MyEvaluation.generate_run_script(
                config_filename="no_extension_filename"
            )

        # Try an invalid extension for config_filename. Should fail...
        with self.assertRaises(ValueError):
            MyEvaluation.generate_run_script(
                config_filename="filename.bad_ext"
            )

        # Try an invalid extension for config_filename. Should fail...
        with self.assertRaises(ValueError):
            MyEvaluation.generate_run_script(
                run_script_filename="filename.bad_ext"
            )

        # Generate the script with a default name
        MyEvaluation.generate_run_script()

        # Check that file exists
        self.assertTrue(exists(DEFAULT_RUN_SCRIPT_FILENAME))

        # Remove the file
        remove(DEFAULT_RUN_SCRIPT_FILENAME)

        # Try with a custom configuration script filename
        MyEvaluation.generate_run_script(
            config_filename="custom-config" + SCRIPT_FILE_EXTENSION
        )

        # Check that file exists
        self.assertTrue(exists(DEFAULT_RUN_SCRIPT_FILENAME))

        # Remove the file
        remove(DEFAULT_RUN_SCRIPT_FILENAME)

        # Try with a custom configuration serialized configuration
        MyEvaluation.generate_run_script(
            config_filename="custom-config" + SERIALIZED_FILE_EXTENSION
        )

        # Check that file exists
        self.assertTrue(exists(DEFAULT_RUN_SCRIPT_FILENAME))

        # Remove the file
        remove(DEFAULT_RUN_SCRIPT_FILENAME)

        # Try a custom script name
        my_run_script_filename = "my_script" + SCRIPT_FILE_EXTENSION
        MyEvaluation.generate_run_script(
            run_script_filename=my_run_script_filename
        )

        # Check that file exists
        self.assertTrue(exists(my_run_script_filename))

        # Remove the file
        remove(my_run_script_filename)

    def test_run(self):
        """Test the run method."""
        # Create the evaluation
        evaluation = MyEvaluation(
            trainer,
            untie_best_fitness_function,
            test_fitness_function,
            "the_results"
        )

        # Run the trainer
        evaluation.run()

        # Check the results
        self.assertIsInstance(evaluation.results, Results)

        # Check that backup and results files exist
        self.assertTrue(evaluation.serialized_results_filename)
        self.assertTrue(evaluation.excel_results_filename)

        # Remove the files
        remove(evaluation.serialized_results_filename)
        remove(evaluation.excel_results_filename)

    def test_copy(self):
        """Test the :py:meth:`~culebra.tools.Evaluation.__copy__` method."""
        evaluation1 = MyEvaluation(
            trainer,
            untie_best_fitness_function,
            test_fitness_function
        )

        evaluation1.run()
        evaluation2 = copy(evaluation1)

        # Copy only copies the first level (evaluation1 != evaluation2)
        self.assertNotEqual(id(evaluation1), id(evaluation2))

        # The results are shared
        self.assertEqual(id(evaluation1._results), id(evaluation2._results))

        # Remove the files
        remove(evaluation1.serialized_results_filename)
        remove(evaluation1.excel_results_filename)

    def test_deepcopy(self):
        """Test :py:meth:`~culebra.tools.Evaluation.__deepcopy__`."""
        evaluation1 = MyEvaluation(
            trainer,
            untie_best_fitness_function,
            test_fitness_function
        )
        evaluation1.run()
        evaluation2 = deepcopy(evaluation1)

        # Check the copy
        self._check_deepcopy(evaluation1, evaluation2)

        # Remove the files
        remove(evaluation1.serialized_results_filename)
        remove(evaluation1.excel_results_filename)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.tools.Evaluation.__setstate__` and
        :py:meth:`~culebra.tools.Evaluation.__reduce__` methods.
        """
        evaluation1 = MyEvaluation(
            trainer,
            untie_best_fitness_function,
            test_fitness_function
        )
        evaluation1.run()

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        evaluation1.dump(serialized_filename)
        evaluation2 = MyEvaluation.load(serialized_filename)

        # Check the copy
        self._check_deepcopy(evaluation1, evaluation2)

        # Remove the result files
        remove(evaluation1.serialized_results_filename)
        remove(evaluation1.excel_results_filename)

        # Remove the serialized file
        remove(serialized_filename)

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
