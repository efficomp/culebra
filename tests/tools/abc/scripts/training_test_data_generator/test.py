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

"""Unit test for :class:`culebra.tools.abc.scripts.TrainingTestDataGenerator`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path
from os import remove

import numpy as np

from culebra.tools import Dataset
from culebra.tools.abc.scripts import TrainingTestDataGenerator


# Dataset ID
DATASET_ID = 52

# Dataset dimensions
NUM_FEATS = 34
NUM_SAMPLES = 351
NUM_CLASSES = 2

# Script parameters
test_prop = "0.3"
training_file = "training.data"
test_file = "test.data"
seed = "0"


class MyTrainingTestDataGenerator(TrainingTestDataGenerator):
    """
    Load, validate and preprocess the Ionosphere dataset.
    """

    def get_dataset(self) -> Dataset:
        """
        Load and validate the Ionosphere dataset.

        :return: Loaded and validated dataset.
        :rtype: ~culebra.tools.Dataset

        :raises ValueError: If the loaded dataset does not match the expected
            number of features, samples, or classes.
        """
        # Load the dataset from UCI
        data = Dataset.from_uci(id_number=DATASET_ID)

        # Check the data
        if data.num_feats != NUM_FEATS:
            raise ValueError(
                f"Wrong number of features: expected {NUM_FEATS}, "
                f"got {data.num_feats}"
            )

        if data.size != NUM_SAMPLES:
            raise ValueError(
                f"Wrong number of samples: expected {NUM_SAMPLES}, "
                f"got {data.size}"
            )

        num_classes_found = len(np.unique(data.outputs))
        if num_classes_found != NUM_CLASSES:
            raise ValueError(
                f"Wrong number of classes: expected {NUM_CLASSES}, "
                f"got {num_classes_found}"
            )

        return data


class TrainingTestDataGeneratorTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.TrainingTestDataGenerator`."""

    @classmethod
    def tearDownClass(cls):
        """Remove the training and test data files."""
        if Path(training_file).exists():
            remove(training_file)
        if Path(test_file).exists():
            remove(test_file)

    def test_parse_args(self):
        """Test the parse_args method."""

        stderr = StringIO()
        with redirect_stderr(stderr):
            # Try without parameters. Should fail ...
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestDataGenerator()
                generator.parse_args()

            # Try a wrong test_prop. Should fail ...
            wrong_test_prop_values = ["a", "-1", "0", "1", "2"]
            for wrong_test_prop in wrong_test_prop_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainingTestDataGenerator(
                        [
                            wrong_test_prop,
                            training_file,
                            test_file,
                            seed
                        ]
                    )
                    generator.parse_args()

            # Try wrong training and test file paths. Should fail ...
            wrong_filepath = "kk/file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainingTestDataGenerator(
                    [
                        test_prop,
                        wrong_filepath,
                        test_file,
                        seed
                    ]
                )
                generator.parse_args()

            with self.assertRaises(SystemExit):
                generator = MyTrainingTestDataGenerator(
                    [
                        test_prop,
                        training_file,
                        wrong_filepath,
                        seed
                    ]
                )
                generator.parse_args()

            # Try a wrong seed. Should fail ...
            wrong_seed_values = ["a", "-1.5", "1.4"]
            for wrong_seed in wrong_seed_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainingTestDataGenerator(
                        [
                            test_prop,
                            training_file,
                            test_file,
                            wrong_seed
                        ]
                    )
                    generator.parse_args()

        # Try valid values
        generator = MyTrainingTestDataGenerator(
            [
                test_prop,
                training_file,
                test_file,
                seed
            ]
        )
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(generator._args.test_prop, float(test_prop))
        self.assertEqual(generator._args.training_file, training_file)
        self.assertEqual(generator._args.test_file, test_file)
        self.assertEqual(generator._args.seed, int(seed))

    def test_process(self):
        """Test the process method."""
        generator = MyTrainingTestDataGenerator(
            [
                test_prop,
                training_file,
                test_file,
                seed
            ]
        )

        # Check that the training and test datasets are None
        self.assertIsNone(generator._training_data)
        self.assertIsNone(generator._test_data)

        # Process the dataset
        generator.process()

        # Check that the training and test datasete arfe not None
        self.assertIsNotNone(generator._training_data)
        self.assertIsNotNone(generator._test_data)

    def test_generate(self):
        """Test the process method."""
        generator = MyTrainingTestDataGenerator(
            [
                test_prop,
                training_file,
                test_file,
                seed
            ]
        )

        # Check that the training and test files don't exist yet
        self.assertFalse(Path(training_file).exists())
        self.assertFalse(Path(test_file).exists())

        # Process the dataset
        generator.generate()

        # Check that the training and test files exist now
        self.assertTrue(Path(training_file).exists())
        self.assertTrue(Path(test_file).exists())


if __name__ == '__main__':
    unittest.main()
