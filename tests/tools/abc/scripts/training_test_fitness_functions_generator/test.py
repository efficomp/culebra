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


# Number of neighbors for KNN
KNN_N_NEIGHBORS = 5

# Script parameters
training_file = "training.data"
test_file = "test.data"
training_cv_num_folds = "5"
training_obj_threshold = "0.01"
training_func_file = "training_func" + SERIALIZED_FILE_EXTENSION
test_func_file = "test_func" + SERIALIZED_FILE_EXTENSION


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
        training_data = Dataset.from_text(
            self._args.training_file, output_index=-1
        )
        test_data = Dataset.from_text(self._args.test_file, output_index=-1)

        # Construct the classifier
        knn_classifier = KNeighborsClassifier(KNN_N_NEIGHBORS)

        # Construct the training fitness function
        self._training_fitness_func = KappaNumFeats(
            training_data=training_data,
            classifier=knn_classifier,
            cv_num_folds=self._args.training_cv_num_folds
        )

        # Set the training fitness similarity threshold
        self._training_fitness_func.obj_thresholds = (
            self._args.training_obj_threshold
        )

        # Construct the test fitness function
        self._test_fitness_func = AccuracyNumFeats(
            training_data=training_data,
            test_data=test_data,
            classifier=knn_classifier
        )


class TrainingTestFitnessFunctionsGeneratorTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.TrainingTestFitnessFunctionsGenerator`."""

    @classmethod
    def tearDownClass(cls):
        """Remove the training and test fitness functions files."""
        if Path(training_func_file).exists():
            remove(training_func_file)
        if Path(test_func_file).exists():
            remove(test_func_file)

    def test_parse_args(self):
        """Test the parse_args method."""
        stderr = StringIO()
        with redirect_stderr(stderr):
        # Try without parameters. Should fail ...
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator()
                generator.parse_args()

            # Try a wrong training dataset
            wrong_file = "wrong.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    [
                        wrong_file,
                        test_file,
                        training_cv_num_folds,
                        training_obj_threshold,
                        training_func_file,
                        test_func_file
                    ]
                )
                generator.parse_args()

            # Try a wrong training dataset
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    [
                        training_file,
                        wrong_file,
                        training_cv_num_folds,
                        training_obj_threshold,
                        training_func_file,
                        test_func_file
                    ]
                )
                generator.parse_args()

            # Try a wrong number of cv folds. Should fail ...
            wrong_cv_num_folds_values = ["a", "1.5", "-1", "1"]
            for wrong_cv_num_folds in wrong_cv_num_folds_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainingTestFitnessFunctionsGenerator(
                        [
                            training_file,
                            test_file,
                            wrong_cv_num_folds,
                            training_obj_threshold,
                            training_func_file,
                            test_func_file
                        ]
                    )
                    generator.parse_args()

            # Try a wrong training obj similarity threshold. Should fail ...
            wrong_obj_threshold_values = ["a", "-1.5"]
            for wrong_obj_threshold in wrong_obj_threshold_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainingTestFitnessFunctionsGenerator(
                        [
                            training_file,
                            test_file,
                            training_cv_num_folds,
                            wrong_obj_threshold,
                            training_func_file,
                            test_func_file
                        ]
                    )
                    generator.parse_args()

            # Try wrong training and test fitness function filepaths
            # Should fail ...
            wrong_filepath = "kk/file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    [
                        training_file,
                        test_file,
                        training_cv_num_folds,
                        training_obj_threshold,
                        wrong_filepath,
                        test_func_file
                    ]
                )
                generator.parse_args()

            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    [
                        training_file,
                        test_file,
                        training_cv_num_folds,
                        training_obj_threshold,
                        training_func_file,
                        wrong_filepath
                    ]
                )
                generator.parse_args()

            # Try wrong training and test fitness function file extension
            # Should fail ...
            wrong_file_ext = "file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    [
                        training_file,
                        test_file,
                        training_cv_num_folds,
                        training_obj_threshold,
                        wrong_file_ext,
                        test_func_file
                    ]
                )
                generator.parse_args()

            with self.assertRaises(SystemExit):
                generator = MyTrainingTestFitnessFunctionsGenerator(
                    [
                        training_file,
                        test_file,
                        training_cv_num_folds,
                        training_obj_threshold,
                        training_func_file,
                        wrong_file_ext
                    ]
                )
                generator.parse_args()

        # Try valid values
        generator = MyTrainingTestFitnessFunctionsGenerator(
            [
                training_file,
                test_file,
                training_cv_num_folds,
                training_obj_threshold,
                training_func_file,
                test_func_file
            ]
        )
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(generator._args.training_file, training_file)
        self.assertEqual(generator._args.test_file, test_file)
        self.assertEqual(
            generator._args.training_cv_num_folds,
            int(training_cv_num_folds)
        )
        self.assertEqual(
            generator._args.training_obj_threshold,
            float(training_obj_threshold)
        )
        self.assertEqual(
            generator._args.training_func_file,
            training_func_file
        )
        self.assertEqual(generator._args.test_func_file, test_func_file)

    def test_generate(self):
        """Test the generate method."""
        generator = MyTrainingTestFitnessFunctionsGenerator(
            [
                training_file,
                test_file,
                training_cv_num_folds,
                training_obj_threshold,
                training_func_file,
                test_func_file
            ]
        )

        # Check that the training and test data files exist
        self.assertTrue(Path(training_file).exists())
        self.assertTrue(Path(test_file).exists())

        # Check that the fitness function files don't exist yet
        self.assertFalse(Path(training_func_file).exists())
        self.assertFalse(Path(test_func_file).exists())

        # Generate the fitness functions
        generator.generate()

        # Check that the fitness function files exist now
        self.assertTrue(Path(training_func_file).exists())
        self.assertTrue(Path(test_func_file).exists())


if __name__ == '__main__':
    unittest.main()
