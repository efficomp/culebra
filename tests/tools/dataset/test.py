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

"""Unit test for :py:class:`culebra.tools.Dataset`."""

import unittest
from os import remove
from copy import copy, deepcopy
from collections import Counter

import numpy as np

from culebra.tools import Dataset

AUSTRALIAN_PATH = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/'
    'australian/australian.dat')
"""Path to the Australian dataset."""

AUSTRALIAN_NUM_FEATS = 14
"""Number of features of the Australian dataset."""

AUSTRALIAN_SIZE = 690
"""Number of samples of the Australian dataset."""

WINE_NAME = "wine"
"""Name of the Wine dataset in the UCI ML repo."""

WINE_NUM_FEATS = 13
"""Number of features of the Wine dataset."""

WINE_SIZE = 178
"""Number of samples of the Wine dataset."""


class DatasetTester(unittest.TestCase):
    """Test :py:class:`culebra.tools.Dataset`."""

    def test_init(self):
        """Test the :py:meth:`~culebra.tools.Dataset.__init__` constructor."""
        # Empty dataset
        dataset = Dataset()

        # Check that input features is an empty ndarray
        self.assertIsInstance(dataset.inputs, np.ndarray)
        self.assertEqual(dataset.inputs.shape[0], 0)

        # Check that outputs is an empty ndarray
        self.assertIsInstance(dataset.outputs, np.ndarray)
        self.assertEqual(dataset.outputs.shape[0], 0)

        # Check that num_feats and size are coherent
        self.assertEqual(dataset.num_feats, 0)
        self.assertEqual(dataset.size, 0)

        # Try to load a mixed dataset with an invalid separator.
        # It should fail
        with self.assertRaises(TypeError):
            Dataset(files=("numeric_1.dat",), output_index=0, sep=1)

        # Try to load a mixed empty dataset. It should fail
        with self.assertRaises(RuntimeError):
            Dataset("empty.dat", output_index=0)

        # Try to load a mixed dataset with missing data. It should fail
        with self.assertRaises(RuntimeError):
            Dataset("missing.dat", output_index=0)

        # Try to load a mixed dataset with non-numeric input data.
        # It should fail
        with self.assertRaises(RuntimeError):
            Dataset("non_numeric.dat", output_index=1)

        # Try to load a mixed dataset with numeric labels.
        dataset = Dataset("numeric_1.dat", output_index=0)
        self.assertEqual(dataset.num_feats, 3)
        self.assertEqual(dataset.size, 10)

        # Try to load a mixed dataset with non-numeric labels.
        dataset = Dataset("non_numeric.dat", output_index=0)
        self.assertEqual(dataset.num_feats, 4)
        self.assertEqual(dataset.size, 10)

        # Try to load a mixed dataset from the Internet
        dataset = Dataset(AUSTRALIAN_PATH, output_index=-1)
        self.assertEqual(dataset.num_feats, AUSTRALIAN_NUM_FEATS)
        self.assertEqual(dataset.size, AUSTRALIAN_SIZE)

        # Try to load a dataset stored in one file, but without output_index.
        # It should fail
        with self.assertRaises(TypeError):
            Dataset("numeric_1.dat")

        # Try to load a split dataset with an invalid separator.
        # It should fail
        with self.assertRaises(TypeError):
            Dataset("numeric_1.dat", "non_numeric.dat", sep=1)

        # Try to load a split dataset with an empty labels file. It should fail
        with self.assertRaises(RuntimeError):
            Dataset("numeric_1.dat", "empty.dat")

        # Try to load a split dataset with an empty inputs file. It should fail
        with self.assertRaises(RuntimeError):
            Dataset("empty.dat", "numeric_1.dat")

        # Try to load a split dataset with missing input data. It should fail
        with self.assertRaises(RuntimeError):
            Dataset("missing.dat", "numeric_1.dat")

        # Try to load a split dataset with non-numeric input data.
        # It should fail
        with self.assertRaises(RuntimeError):
            Dataset("non_numeric.dat", "numeric_1.dat")

        # Try to load a split dataset with different number of inputs and
        # outputs. It should fail
        with self.assertRaises(RuntimeError):
            Dataset("numeric_1.dat", "numeric_2.dat")

        # Try to a split dataset with numeric labels.
        dataset = Dataset("numeric_1.dat", "numeric_1.dat")
        self.assertEqual(dataset.num_feats, 4)
        self.assertEqual(dataset.size, 10)

        # Try to load a mixed dataset with non-numeric labels.
        dataset = Dataset("numeric_1.dat", "non_numeric.dat")
        self.assertEqual(dataset.num_feats, 4)
        self.assertEqual(dataset.size, 10)

    def test_load_train_test(self):
        """Test :py:meth:`~culebra.tools.Dataset.load_train_test`."""
        # Try to load a mixed dataset
        datasets = Dataset.load_train_test("numeric_1.dat", output_index=0)

        # Check the 2 datasets are returned
        self.assertEqual(len(datasets), 2)

        # Check that training dataset is numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 10)

        # Check that test dataset is a copy of training data
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 10)

        # Try to load a mixed dataset and split it
        datasets = Dataset.load_train_test(
            "numeric_1.dat", output_index=0, test_prop=0.2)

        # Check that training dataset is 75% of numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 10*0.8)

        # Check that test dataset is 25% of numeric_1.dat
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 10*0.2)

        # For mixed datasets, if test_prop is not None, the second dataset
        # should be ignored
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "numeric_2.dat", test_prop=0.2, output_index=0)

        # Check that training dataset is 75% of numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 10*0.8)

        # Check that test dataset is 25% of numeric_1.dat
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 10*0.2)

        # Try to load a mixed dataset, split it and also normalize it
        datasets = Dataset.load_train_test(
            AUSTRALIAN_PATH, output_index=-1, test_prop=0.2, normalize=True)

        # Check that the minimum value for each feature is zero
        min_train_inputs = np.min(datasets[0].inputs, axis=0)
        min_test_inputs = np.min(datasets[1].inputs, axis=0)
        min_values = np.minimum(min_train_inputs, min_test_inputs)
        self.assertEqual(min(min_values), 0)
        self.assertEqual(max(min_values), 0)

        # Check that the maximum value for each feature is one
        max_train_inputs = np.max(datasets[0].inputs, axis=0)
        max_test_inputs = np.max(datasets[1].inputs, axis=0)
        max_values = np.maximum(max_train_inputs, max_test_inputs)
        self.assertEqual(max(max_values), 1)
        self.assertEqual(max(max_values), 1)

        # Try to load a mixed dataset and append it some random features
        datasets = Dataset.load_train_test(
            "numeric_1.dat", output_index=0, test_prop=0.2, random_feats=5)

        # Check that training dataset is 75% of numeric_1.dat, but having
        # 5 more features
        self.assertEqual(datasets[0].num_feats, 3 + 5)
        self.assertEqual(datasets[0].size, 10*0.8)

        # Check that test dataset is 25% of numeric_1.dat, but having
        # 5 more features
        self.assertEqual(datasets[1].num_feats, 3 + 5)
        self.assertEqual(datasets[1].size, 10*0.2)

        # Try to load two mixed datasets, the first for training and the second
        # for testing
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "numeric_2.dat", output_index=0)

        # Check the 2 datasets are returned
        self.assertEqual(len(datasets), 2)

        # Check that training dataset is numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 10)

        # Check that test dataset is a numeric_2
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 4)

        # Try to load two mixed datasets, the first for training and the second
        # for testing. It should fail because datasets have different number
        # of features
        self.assertRaises(RuntimeError, Dataset.load_train_test,
                          "numeric_1.dat", "non_numeric.dat", output_index=0)

        # Try to load a split dataset
        #  - numeric_1.dat conains the input features
        #  - non_numeric.dat contains the outputs. Onle the first column is
        #    considered
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "non_numeric.dat", test_prop=0.5, random_feats=3)
        # Check the 2 datasets are returned
        self.assertEqual(len(datasets), 2)

        # Check the training dataset
        self.assertEqual(datasets[0].num_feats, 4+3)
        self.assertEqual(datasets[0].size, 10*0.5)

        # Check the test dataset
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 4+3)
        self.assertEqual(datasets[1].size, 10*0.5)

        # Try to load two split datasets. Since test_prop is not None,
        # The second dataset should be ignored
        #  - numeric_1.dat conains the input features
        #  - non_numeric.dat contains the outputs. Onle the first column is
        #    considered
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "non_numeric.dat", "numeric_1.dat",
            "non_numeric.dat", test_prop=0.5)

        # Check the training dataset
        self.assertEqual(datasets[0].num_feats, 4)
        self.assertEqual(datasets[0].size, 10*0.5)

        # Check the test dataset
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 4)
        self.assertEqual(datasets[1].size, 10*0.5)

        # Try to load two split datasets
        #  - numeric_1.dat conains the input features
        #  - non_numeric.dat contains the outputs. Onle the first column is
        #    considered
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "non_numeric.dat", "numeric_1.dat",
            "non_numeric.dat")

        # Check the training dataset
        self.assertEqual(datasets[0].num_feats, 4)
        self.assertEqual(datasets[0].size, 10)

        # Check the test dataset
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 4)
        self.assertEqual(datasets[1].size, 10)

    def test_load_from_uci(self):
        """Test the load_from_uci class method."""
        # Dataset
        dataset = Dataset.load_from_uci(name=WINE_NAME)
        self.assertEqual(dataset.num_feats, WINE_NUM_FEATS)
        self.assertEqual(dataset.size, WINE_SIZE)

    def test_normalize(self):
        """Test the normalization method."""
        # Load the data
        dataset = Dataset(AUSTRALIAN_PATH, output_index=-1)
        (training_data, test_data) = dataset.split(
            test_prop=0.3, random_seed=0
        )
        training_data.normalize(test_data)

        # Check that the minimum value for each feature is zero
        min_train_inputs = np.min(training_data.inputs, axis=0)
        min_test_inputs = np.min(test_data.inputs, axis=0)
        min_values = np.minimum(min_train_inputs, min_test_inputs)
        self.assertEqual(min(min_values), 0)
        self.assertEqual(max(min_values), 0)

        # Check that the maximum value for each feature is one
        max_train_inputs = np.max(training_data.inputs, axis=0)
        max_test_inputs = np.max(test_data.inputs, axis=0)
        max_values = np.maximum(max_train_inputs, max_test_inputs)
        self.assertEqual(max(max_values), 1)
        self.assertEqual(max(max_values), 1)

    def test_robust_scale(self):
        """Test the robust_scale method."""
        num_feats = 1000
        size = 1000

        # Create a dataset with repeated samples, there isn't any outlier
        dataset = Dataset()
        dataset._inputs = np.concatenate(
            (
                np.ones((int(size/2), num_feats)),
                np.ones((int(size/2), num_feats)) * 10
            )
        )

        dataset._outputs = np.concatenate(
            (
                np.ones((int(size/2), 1)),
                np.ones((int(size/2), 1)) * 10
            )
        )

        # Try robust_scale
        dataset.robust_scale()

        # Check the scale
        self.assertEqual(np.min(dataset.inputs), -0.5)
        self.assertEqual(np.max(dataset.inputs), 0.5)

        # Insert outliers
        dataset._inputs = np.concatenate(
            (
                [[-1000] * num_feats],
                np.concatenate(
                    (
                        np.ones((int(size/2), num_feats)),
                        np.ones((int(size/2), num_feats)) * 10
                    )
                ),
                [[1000] * num_feats]
            )
        )
        dataset._outputs = np.concatenate(
            (
                [[-1000]],
                np.concatenate(
                    (
                        np.ones((int(size/2), 1)),
                        np.ones((int(size/2), 1)) * 10
                    )
                ),
                [[1000]]
            )
        )

        # Try robust_scale again
        dataset.robust_scale()

        # Remove the outliers
        dataset.remove_outliers()

        # The scale should be the same than without outliers
        self.assertEqual(np.min(dataset.inputs), -0.5)
        self.assertEqual(np.max(dataset.inputs), 0.5)

    def test_remove_outliers(self):
        """Test the outliers removal method."""
        num_feats = 10
        size = 100

        # Create a dataset with repeated samples, there isn't any outlier
        dataset = Dataset()
        dataset._inputs = np.ones((size, num_feats))
        dataset._outputs = np.ones((size, 1))

        # Try to remove the outliers
        dataset.remove_outliers()

        # Check that all the samples remain
        self.assertEqual(dataset.size, size)
        self.assertEqual(dataset._inputs.shape[0], size)
        self.assertEqual(dataset._outputs.shape[0], size)

        # Insert outliers
        dataset._inputs = np.concatenate(
            ([[-100] * num_feats], dataset._inputs, [[100] * num_feats])
        )
        dataset._outputs = np.concatenate(
            ([[-100]], dataset._outputs, [[100]])
        )

        # Check that the size has increased
        self.assertEqual(dataset.size, size + 2)
        self.assertEqual(dataset._inputs.shape[0], size + 2)
        self.assertEqual(dataset._outputs.shape[0], size + 2)

        # Try to remove the outliers again
        dataset.remove_outliers()

        # The outlier should have dissapeared
        self.assertEqual(dataset.size, size)
        self.assertEqual(dataset._inputs.shape[0], size)
        self.assertEqual(dataset._outputs.shape[0], size)
        self.assertTrue((dataset._inputs == 1).all())
        self.assertTrue((dataset._outputs == 1).all())

        # Try also with a test dataset having outliers
        test_dataset = copy(dataset)

        # Insert outliers
        test_dataset._inputs = np.concatenate(
            ([[-100] * num_feats], test_dataset._inputs, [[100] * num_feats])
        )
        test_dataset._outputs = np.concatenate(
            ([[-100]], test_dataset._outputs, [[100]])
        )

        # Check that the size has increased
        self.assertEqual(dataset.size, size)
        self.assertEqual(dataset._inputs.shape[0], size)
        self.assertEqual(dataset._outputs.shape[0], size)
        self.assertEqual(test_dataset.size, size + 2)
        self.assertEqual(test_dataset._inputs.shape[0], size + 2)
        self.assertEqual(test_dataset._outputs.shape[0], size + 2)

        # Try to remove the outliers again
        dataset.remove_outliers(test_dataset)

        # The outlier should have dissapeared
        self.assertEqual(dataset.size, size)
        self.assertEqual(dataset._inputs.shape[0], size)
        self.assertEqual(dataset._outputs.shape[0], size)
        self.assertTrue((dataset._inputs == 1).all())
        self.assertTrue((dataset._outputs == 1).all())

        self.assertEqual(test_dataset.size, size)
        self.assertEqual(test_dataset._inputs.shape[0], size)
        self.assertEqual(test_dataset._outputs.shape[0], size)
        self.assertTrue((test_dataset._inputs == 1).all())
        self.assertTrue((test_dataset._outputs == 1).all())

    def test_oversample(self):
        """Test the :py:meth:`~culebra.tools.Dataset.oversample` method."""
        dataset1 = Dataset("numeric_1.dat", output_index=0)
        samples_per_class_dataset1 = Counter(dataset1.outputs)
        samples_majority_class = max(samples_per_class_dataset1.values())
        dataset2 = dataset1.oversample()
        samples_per_class_dataset2 = Counter(dataset2.outputs)
        self.assertTrue(
            all(
                count == samples_majority_class
                for count in samples_per_class_dataset2.values()
            )
        )

    def test_copy(self):
        """Test the :py:meth:`~culebra.tools.Dataset.__copy__` method."""
        dataset1 = Dataset("numeric_1.dat", output_index=0)
        dataset2 = copy(dataset1)

        # Copy only copies the first level (dataset1 != dataset2)
        self.assertNotEqual(id(dataset1), id(dataset2))

        # The objects attributes are shared
        self.assertEqual(id(dataset1._inputs), id(dataset2._inputs))
        self.assertEqual(id(dataset1._outputs), id(dataset2._outputs))

    def test_deepcopy(self):
        """Test the :py:meth:`~culebra.abc.Base.__deepcopy__` method."""
        dataset1 = Dataset("numeric_1.dat", output_index=0)
        dataset2 = deepcopy(dataset1)

        # Check the copy
        self._check_deepcopy(dataset1, dataset2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.abc.Base.__setstate__` and
        :py:meth:`~culebra.abc.Base.__reduce__` methods.
        """
        dataset1 = Dataset("numeric_1.dat", output_index=0)

        pickle_filename = "my_pickle.gz"
        dataset1.save_pickle(pickle_filename)
        dataset2 = Dataset.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(dataset1, dataset2)

        # Remove the pickle file
        remove(pickle_filename)

    def _check_deepcopy(self, dataset1, dataset2):
        """Check if *dataset1* is a deepcopy of *dataset2*.

        :param dataset1: The first dataset
        :type dataset1: :py:class:`~culebra.tools.Dataset`
        :param dataset2: The second dataset
        :type dataset2: :py:class:`~culebra.tools.Dataset`
        """
        # Copies all the levels
        self.assertNotEqual(id(dataset1), id(dataset2))
        self.assertNotEqual(id(dataset1._inputs), id(dataset2._inputs))
        self.assertNotEqual(id(dataset1._outputs), id(dataset2._outputs))

        self.assertTrue((dataset1.inputs == dataset2.inputs).all())
        self.assertTrue((dataset1.outputs == dataset2.outputs).all())


if __name__ == '__main__':
    unittest.main()
