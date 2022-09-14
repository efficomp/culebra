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

"""Test the Dataset class."""

import unittest
from copy import copy, deepcopy
import pickle
import numpy as np
from culebra.base import Dataset

AUSTRALIAN_PATH = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/'
    'australian/australian.dat')
"""Path to the Australian dataset."""

AUSTRALIAN_NUM_FEATS = 14
"""Number of features of the Australian dataset."""

AUSTRALIAN_SIZE = 690
"""Number of samples of the Australian dataset."""


class DatasetTester(unittest.TestCase):
    """Test :py:class:`base.Dataset`."""

    def test_init(self):
        """Test the :py:meth:`~base.Dataset.__init__` constructor."""
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
        self.assertEqual(dataset.size, 8)

        # Try to load a mixed dataset with non-numeric labels.
        dataset = Dataset("non_numeric.dat", output_index=0)
        self.assertEqual(dataset.num_feats, 4)
        self.assertEqual(dataset.size, 8)

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
        self.assertEqual(dataset.size, 8)

        # Try to load a mixed dataset with non-numeric labels.
        dataset = Dataset("numeric_1.dat", "non_numeric.dat")
        self.assertEqual(dataset.num_feats, 4)
        self.assertEqual(dataset.size, 8)

    def test_load_train_test(self):
        """Test the :py:meth:`~base.Dataset.load_train_test` method."""
        # Try to load a mixed dataset
        datasets = Dataset.load_train_test("numeric_1.dat", output_index=-1)

        # Check the 2 datasets are returned
        self.assertEqual(len(datasets), 2)

        # Check that training dataset is numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 8)

        # Check that test dataset is a copy of training data
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 8)

        # Try to load a mixed dataset and split it
        datasets = Dataset.load_train_test(
            "numeric_1.dat", output_index=-1, test_prop=0.25)

        # Check that training dataset is 75% of numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 8*0.75)

        # Check that test dataset is 25% of numeric_1.dat
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 8*0.25)

        # For mixed datasets, if test_prop is not None, the second dataset
        # should be ignored
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "numeric_2.dat", test_prop=0.25, output_index=-1)

        # Check that training dataset is 75% of numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 8*0.75)

        # Check that test dataset is 25% of numeric_1.dat
        self.assertEqual(datasets[1].num_feats, 3)
        self.assertEqual(datasets[1].size, 8*0.25)

        # Try to load a mixed dataset, split it and also normalize it
        datasets = Dataset.load_train_test(
            AUSTRALIAN_PATH, output_index=-1, test_prop=0.25, normalize=True)

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
            "numeric_1.dat", output_index=-1, test_prop=0.25, random_feats=5)

        # Check that training dataset is 75% of numeric_1.dat, but having
        # 5 more features
        self.assertEqual(datasets[0].num_feats, 3 + 5)
        self.assertEqual(datasets[0].size, 8*0.75)

        # Check that test dataset is 25% of numeric_1.dat, but having
        # 5 more features
        self.assertEqual(datasets[1].num_feats, 3 + 5)
        self.assertEqual(datasets[1].size, 8*0.25)

        # Try to load two mixed datasets, the first for training and the second
        # for testing
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "numeric_2.dat", output_index=-1)

        # Check the 2 datasets are returned
        self.assertEqual(len(datasets), 2)

        # Check that training dataset is numeric_1.dat
        self.assertEqual(datasets[0].num_feats, 3)
        self.assertEqual(datasets[0].size, 8)

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
        self.assertEqual(datasets[0].size, 8*0.5)

        # Check the test dataset
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 4+3)
        self.assertEqual(datasets[1].size, 8*0.5)

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
        self.assertEqual(datasets[0].size, 8*0.5)

        # Check the test dataset
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 4)
        self.assertEqual(datasets[1].size, 8*0.5)

        # Try to load two split datasets
        #  - numeric_1.dat conains the input features
        #  - non_numeric.dat contains the outputs. Onle the first column is
        #    considered
        datasets = Dataset.load_train_test(
            "numeric_1.dat", "non_numeric.dat", "numeric_1.dat",
            "non_numeric.dat")

        # Check the training dataset
        self.assertEqual(datasets[0].num_feats, 4)
        self.assertEqual(datasets[0].size, 8)

        # Check the test dataset
        self.assertTrue(datasets[0] is not datasets[1])
        self.assertEqual(datasets[1].num_feats, 4)
        self.assertEqual(datasets[1].size, 8)

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

    def test_remove_outliers(self):
        """Test the outliers removal method."""
        # Try with the whole data
        dataset = Dataset(AUSTRALIAN_PATH, output_index=-1)
        size_before = dataset.size
        dataset.remove_outliers()
        size_after = dataset.size
        self.assertLessEqual(size_after, size_before)

        # Try a split dataset
        dataset = Dataset(AUSTRALIAN_PATH, output_index=-1)
        (training_data, test_data) = dataset.split(
            test_prop=0.3, random_seed=0
        )
        size_training_before = training_data.size
        size_test_before = test_data.size

        training_data.remove_outliers(test_data)
        size_training_after = training_data.size
        size_test_after = test_data.size
        self.assertLessEqual(size_training_after, size_training_before)
        self.assertLessEqual(size_test_after, size_test_before)

    def test_copy(self):
        """Test the :py:meth:`~base.Dataset.__copy__` method."""
        dataset1 = Dataset("numeric_1.dat", output_index=0)
        dataset2 = copy(dataset1)

        # Copy only copies the first level (dataset1 != dataset2)
        self.assertNotEqual(id(dataset1), id(dataset2))

        # The objects attributes are shared
        self.assertEqual(id(dataset1._inputs), id(dataset2._inputs))
        self.assertEqual(id(dataset1._outputs), id(dataset2._outputs))

    def test_deepcopy(self):
        """Test the :py:meth:`~base.Base.__deepcopy__` method."""
        dataset1 = Dataset("numeric_1.dat", output_index=0)
        dataset2 = deepcopy(dataset1)

        # Check the copy
        self._check_deepcopy(dataset1, dataset2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~base.Base.__setstate__` and
        :py:meth:`~base.Base.__reduce__` methods.
        """
        dataset1 = Dataset("numeric_1.dat", output_index=0)

        data = pickle.dumps(dataset1)
        dataset2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(dataset1, dataset2)

    def _check_deepcopy(self, dataset1, dataset2):
        """Check if *dataset1* is a deepcopy of *dataset2*.

        :param dataset1: The first dataset
        :type dataset1: :py:class:`~base.Dataset`
        :param dataset2: The second dataset
        :type dataset2: :py:class:`~base.Dataset`
        """
        # Copies all the levels
        self.assertNotEqual(id(dataset1), id(dataset2))
        self.assertNotEqual(id(dataset1._inputs), id(dataset2._inputs))
        self.assertNotEqual(id(dataset1._outputs), id(dataset2._outputs))

        self.assertTrue((dataset1.inputs == dataset2.inputs).all())
        self.assertTrue((dataset1.outputs == dataset2.outputs).all())


if __name__ == '__main__':
    unittest.main()
