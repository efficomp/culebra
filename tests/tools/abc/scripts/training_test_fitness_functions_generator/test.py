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

"""Unit test for :class:`culebra.tools.abc.scripts.TrainingTestFitnessFunctionsGenerator`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path
from os import remove

from sklearn.neighbors import KNeighborsClassifier

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.fitness_func import MultiObjectiveFitnessFunction
from culebra.fitness_func.feature_selection import (
    KappaIndex,
    Accuracy,
    NumFeats
)
from culebra.tools import Dataset
from culebra.tools.abc.scripts import TrainingTestFitnessFunctionsGenerator


# Dataset ID
DATASET_ID = 52

# Number of neighbors for KNN
KNN_N_NEIGHBORS = 5

# Script parameters
training_data = "tr.dat"
test_data = "tst.dat"
training_cv_num_folds = "10"
training_sim_threshold = "0.01"
training_func_obj = "tr_func" + SERIALIZED_FILE_EXTENSION
test_func_obj = "tst_func" + SERIALIZED_FILE_EXTENSION


# Training fitness function class
def KappaNumFeats(
    training_data,
    test_data=None,
    cv_num_folds=None,
    cv_fixed_folds=None,
    classifier=None
):
    """Fitness Function."""
    return MultiObjectiveFitnessFunction(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            cv_num_folds=cv_num_folds,
            cv_fixed_folds=cv_fixed_folds,
            classifier=classifier
        ),
        NumFeats()
    )

# Test fitness function class
def AccuracyNumFeats(
    training_data,
    test_data=None,
    cv_num_folds=None,
    classifier=None
):
    """Fitness Function."""
    return MultiObjectiveFitnessFunction(
        Accuracy(
            training_data=training_data,
            test_data=test_data,
            cv_num_folds=cv_num_folds,
            classifier=classifier
        ),
        NumFeats()
    )


class MyTrainingTestFitnessFunctionsGenerator(
    TrainingTestFitnessFunctionsGenerator
   ):
    """
    Generate the training and test fitness functions.

    The training fitness function is multi-objective and evaluates the Kappa
    index and the number of features, while the test fitness function evaluates
    the accuracy and the number of features.
    """

    def process(self):
        """Define the training and test fitness functions."""
        # Parse command-line arguments
        self.parse_args()

        # Load the training and test data
        training_dataset = Dataset.from_text(
            self._args.training_data, output_index=-1
        )
        test_dataset = Dataset.from_text(self._args.test_data, output_index=-1)

        # Construct the classifier
        knn_classifier = KNeighborsClassifier(KNN_N_NEIGHBORS)

        # Construct the training fitness function
        self._training_fitness_func = KappaNumFeats(
            training_data=training_dataset,
            classifier=knn_classifier,
            cv_num_folds=self._args.training_cv_num_folds
        )

        # Set the training fitness similarity threshold
        self._training_fitness_func.obj_thresholds = (
            self._args.training_sim_threshold
        )

        # Construct the test fitness function
        self._test_fitness_func = AccuracyNumFeats(
            training_data=training_dataset,
            test_data=test_dataset,
            classifier=knn_classifier
        )


class TrainingTestFitnessFunctionsGeneratorTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.TrainingTestFitnessFunctionsGenerator`."""

    @classmethod
    def setUpClass(cls):
        """Generate the training and test data files."""

        # Load the dataset from UCI
        data = Dataset.from_uci(id_number=DATASET_ID).drop_missing().scale()

        # Split the data into training and test (stratified)
        (training_dataset, test_dataset) = data.split(
            test_prop=0.3
        )

        # Save the training and test datasets
        training_dataset.to_text(
            TrainingTestFitnessFunctionsGenerator.default_training_data
        )

        test_dataset.to_text(
            TrainingTestFitnessFunctionsGenerator.default_test_data
        )
        training_dataset.to_text(training_data)
        test_dataset.to_text(test_data)

    @classmethod
    def tearDownClass(cls):
        """Remove the datasets and fitness functions files."""
        for file in (
            TrainingTestFitnessFunctionsGenerator.default_training_data,
            TrainingTestFitnessFunctionsGenerator.default_test_data,
            TrainingTestFitnessFunctionsGenerator.default_training_func_obj,
            TrainingTestFitnessFunctionsGenerator.default_test_func_obj,
            training_data,
            test_data,
            training_func_obj,
            test_func_obj
        ):
            if Path(file).exists():
                remove(file)

    def test_parse_args(self):
        """Test the parse_args method."""
        stderr = StringIO()
        with redirect_stderr(stderr):
            # Try a wrong training dataset
            wrong_file = "wrong.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    ["--training-data", wrong_file]
                )
                generator.parse_args()

            # Try a wrong training dataset
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    ["--test-data", wrong_file]
                )
                generator.parse_args()

            # Try a wrong number of cv folds. Should fail ...
            wrong_cv_num_folds_values = ["a", "1.5", "-1", "1"]
            for wrong_cv_num_folds in wrong_cv_num_folds_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainingTestFitnessFunctionsGenerator(
                        ["--training-cv-num-folds", wrong_cv_num_folds]
                    )
                    generator.parse_args()

            # Try a wrong training obj similarity threshold. Should fail ...
            wrong_sim_threshold_values = ["a", "-1.5"]
            for wrong_sim_threshold in wrong_sim_threshold_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainingTestFitnessFunctionsGenerator(
                        ["--training-sim-threshold", wrong_sim_threshold]
                    )
                    generator.parse_args()

            # Try wrong training and test fitness function filepaths
            # Should fail ...
            wrong_filepath = "kk/file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    ["--training-func-obj", wrong_filepath]
                )
                generator.parse_args()

            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    ["--test-func-obj", wrong_filepath]
                )
                generator.parse_args()

            # Try wrong training and test fitness function file extension
            # Should fail ...
            wrong_file_ext = "file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    ["--training-func-obj", wrong_file_ext]
                )
                generator.parse_args()

            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    ["--test-func-obj", wrong_file_ext]
                )
                generator.parse_args()

        # Try the default parameters
        generator = MyTrainingTestFitnessFunctionsGenerator()
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(
            generator._args.training_data,
            generator.default_training_data
        )
        self.assertEqual(
            generator._args.test_data,
            generator.default_test_data
        )
        self.assertEqual(
            generator._args.training_cv_num_folds,
            generator.default_training_cv_num_folds
        )
        self.assertEqual(
            generator._args.training_sim_threshold,
            generator.default_training_sim_threshold
        )
        self.assertEqual(
            generator._args.training_func_obj,
            generator.default_training_func_obj
        )
        self.assertEqual(
            generator._args.test_func_obj,
            generator.default_test_func_obj
        )

        # Try valid values
        generator = MyTrainingTestFitnessFunctionsGenerator(
            [
                "--training-data", training_data,
                "--test-data", test_data,
                "--training-cv-num-folds", training_cv_num_folds,
                "--training-sim-threshold", training_sim_threshold,
                "--training-func-obj", training_func_obj,
                "--test-func-obj", test_func_obj
            ]
        )
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(generator._args.training_data, training_data)
        self.assertEqual(generator._args.test_data, test_data)
        self.assertEqual(
            generator._args.training_cv_num_folds,
            int(training_cv_num_folds)
        )
        self.assertEqual(
            generator._args.training_sim_threshold,
            float(training_sim_threshold)
        )
        self.assertEqual(
            generator._args.training_func_obj,
            training_func_obj
        )
        self.assertEqual(generator._args.test_func_obj, test_func_obj)

    def test_generate(self):
        """Test the generate method."""
        generator = MyTrainingTestFitnessFunctionsGenerator(
            [
                "--training-data", training_data,
                "--test-data", test_data,
                "--training-func-obj", training_func_obj,
                "--test-func-obj", test_func_obj
            ]
        )

        # Check that the training and test data files exist
        self.assertTrue(Path(training_data).exists())
        self.assertTrue(Path(test_data).exists())

        # Check that the fitness function files don't exist yet
        self.assertFalse(Path(training_func_obj).exists())
        self.assertFalse(Path(test_func_obj).exists())

        # Generate the fitness functions
        generator.generate()

        # Check that the fitness function files exist now
        self.assertTrue(Path(training_func_obj).exists())
        self.assertTrue(Path(test_func_obj).exists())


if __name__ == '__main__':
    unittest.main()
