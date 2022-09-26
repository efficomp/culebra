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

"""Unit test for :py:class:`tools.Results`."""

import unittest
import pickle
from collections import UserDict
from os import remove
from os.path import exists
from copy import copy, deepcopy
from pandas import DataFrame
from culebra.tools import (
    Results,
    DEFAULT_RESULTS_BACKUP_FILENAME,
    DEFAULT_RESULTS_EXCEL_FILENAME
)


class ResultsTester(unittest.TestCase):
    """Test :py:class:`~tools.Results`."""

    def test_init(self):
        """Test the constructor."""
        # Try an empty results manager
        results = Results()

        # Check that the results manager is empty
        self.assertEqual(len(results.keys()), 0)

        # Check that the results manager subclass of dict
        self.assertIsInstance(results, Results)
        self.assertIsInstance(results, UserDict)

    def test_setitem(self):
        """Test the :py:meth:`~tools.Results.__setitem__` method."""
        results = Results()

        data = DataFrame()
        data["inputs"] = [1, 2, 3]
        data["outputs"] = [4, 5, 6]
        key = "data"

        # Try to add an item with an invalid key
        with self.assertRaises(TypeError):
            results[1] = data

        # Try to add an item which is not a dataframe
        with self.assertRaises(TypeError):
            results[key] = 1

        # Try a valid key and data
        results[key] = data

        # Check the key
        self.assertTrue(key in results)

        # Check the data
        self.assertTrue(results[key].equals(data))

    def test_from_csv_files(self):
        """Test the :py:meth:`~tools.Results.from_csv_files` class method."""
        results = Results()

        bad_filename = "bad.csv"
        bad_key = 1
        bad_sep = 2
        good_filenames = ("test_fitness.csv", "execution_metrics.csv")
        good_keys = ("test_fitness", "execution_metrics")

        # Try to read a non-existing file
        with self.assertRaises(FileNotFoundError):
            Results.from_csv_files(
                (bad_filename, *good_filenames),
                ("a", *good_keys)
            )

        # Try to a bad key
        with self.assertRaises(ValueError):
            Results.from_csv_files(
                good_filenames,
                ("a", bad_key)
            )

        # Try a bad separator
        with self.assertRaises(TypeError):
            Results.from_csv_files(
                good_filenames, good_keys, sep=bad_sep
            )

        # Try sequences with different lengthd
        with self.assertRaises(ValueError):
            Results.from_csv_files(
                good_filenames, ("a", *good_keys)
            )

        # Try default keys
        results = Results.from_csv_files(good_filenames)
        for key1, key2 in zip(good_keys, results.keys()):
            self.assertEqual(key1, key2)
            self.assertIsInstance(results[key1], DataFrame)

    def test_load_save(self):
        """Test the save and load methods."""
        data_filenames = ("test_fitness.csv", "execution_metrics.csv")
        data_keys = ("test_fitness", "execution_metrics")
        bad_backup_filename_type = 1
        bad_backup_filename_values = ["file", "file.tar"]
        good_backup_filename = "myresults.gz"

        results = Results.from_csv_files(data_filenames, data_keys)

        # Try saving with a wrong filename type
        with self.assertRaises(TypeError):
            results.save(bad_backup_filename_type)

        # Try saving with wrong filename values
        for filename in bad_backup_filename_values:
            with self.assertRaises(ValueError):
                results.save(filename)

        # Try saving with a custom filename
        results.save(good_backup_filename)

        # Try loading with a wrong filename type
        with self.assertRaises(TypeError):
            Results.load(bad_backup_filename_type)

        # Try loading with wrong filename values
        for filename in bad_backup_filename_values:
            with self.assertRaises(ValueError):
                results.save(filename)

        # Try loading with a custom filename
        results2 = Results.load(good_backup_filename)

        # Check keys and data
        for key in data_keys:
            self.assertTrue(key in results2)
            self.assertTrue(results2[key].equals(results[key]))

        # Remove the file
        remove(good_backup_filename)

        # Try saving with the default filename
        results.save()

        # Try loading with the default filename
        results3 = Results.load()

        # Check keys and data
        for key in data_keys:
            self.assertTrue(key in results3)
            self.assertTrue(results3[key].equals(results[key]))

        # Remove the file
        remove(DEFAULT_RESULTS_BACKUP_FILENAME)

    def test_to_excel(self):
        """Test the to_excel method."""
        data_filenames = ("test_fitness.csv", "execution_metrics.csv")
        data_keys = ("test_fitness", "execution_metrics")
        bad_excel_filename_type = 1
        bad_excel_filename_values = ["file", "file.tar"]
        good_excel_filename = "myresults.xlsx"

        results = Results.from_csv_files(data_filenames, data_keys)

        # Try saving with a wrong filename type
        with self.assertRaises(TypeError):
            results.to_excel(bad_excel_filename_type)

        # Try saving with wrong filename values
        for filename in bad_excel_filename_values:
            with self.assertRaises(ValueError):
                results.to_excel(filename)

        # Try saving with a custom filename
        results.to_excel(good_excel_filename)
        self.assertTrue(exists(good_excel_filename))

        # Remove the file
        remove(good_excel_filename)

        # Try saving with to the default file
        results.to_excel()
        self.assertTrue(exists(DEFAULT_RESULTS_EXCEL_FILENAME))

        # Remove the file
        remove(DEFAULT_RESULTS_EXCEL_FILENAME)

    def test_copy(self):
        """Test the :py:meth:`~tools.Results.__copy__` method."""
        data_filenames = ("test_fitness.csv", "execution_metrics.csv")
        data_keys = ("test_fitness", "execution_metrics")

        results1 = Results.from_csv_files(data_filenames, data_keys)
        results2 = copy(results1)

        # Copy only copies the first level (results1 != results2)
        self.assertNotEqual(id(results1), id(results2))

        # The results data are shared
        for key in results1:
            self.assertEqual(id(results1[key]), id(results2[key]))

    def test_deepcopy(self):
        """Test :py:meth:`~tools.Results.__deepcopy__`."""
        data_filenames = ("test_fitness.csv", "execution_metrics.csv")
        data_keys = ("test_fitness", "execution_metrics")

        results1 = Results.from_csv_files(data_filenames, data_keys)
        results2 = deepcopy(results1)

        # Check the copy
        self._check_deepcopy(results1, results2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~tools.Results.__setstate__` and
        :py:meth:`~tools.Results.__reduce__` methods.
        """
        data_filenames = ("test_fitness.csv", "execution_metrics.csv")
        data_keys = ("test_fitness", "execution_metrics")

        results1 = Results.from_csv_files(data_filenames, data_keys)

        data = pickle.dumps(results1)
        results2 = pickle.loads(data)

        # Check the copy
        self._check_deepcopy(results1, results2)

    def _check_deepcopy(self, results1, results2):
        """Check if *results1* is a deepcopy of *results2*.

        :param results1: The first results
        :type results1: :py:class:`~tools.Results`
        :param results2: The second results
        :type results2: :py:class:`~tools.Results`
        """
        # Copies all the levels
        self.assertNotEqual(id(results1), id(results2))

        for key in results1:
            self.assertTrue(
                results1[key].equals(results2[key])
            )


if __name__ == '__main__':
    unittest.main()
