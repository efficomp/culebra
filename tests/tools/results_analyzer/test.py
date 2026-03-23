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

"""Unit test for :class:`culebra.tools.ResultsAnalyzer`."""

import unittest
import random
from collections import UserDict

import numpy as np
from pandas import DataFrame
from scipy.stats import (
    shapiro,
    normaltest,
    bartlett,
    levene,
    fligner
)

from culebra.tools import Results, ResultsAnalyzer

CSV_FILE_EXTENSION = ".csv"
"""Extension for csv files."""

N_SAMPLES = 200
"""Number of samples per batch."""


class ResultsAnalyzerTester(unittest.TestCase):
    """Test the :class:`~culebra.tools.ResultsAnalyzer` class."""

    def setUp(self):
        """Set up each test."""
        self.valid_batches = ["Valid0", "Valid1", "Valid2"]
        self.discarded_batches = ["Discarded0", "Discarded1"]
        self.dataframe_key = "DataframeKey"
        self.invalid_dataframe_key = "InvalidDataframe"
        self.column_key = "ColumnKey"
        self.invalid_column_key = "InvalidColumn"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        self.analyzer = ResultsAnalyzer()

        # Add the valid batches
        for batch in self.valid_batches:
            self.analyzer[batch] = Results()
            self.analyzer[batch][self.dataframe_key] = DataFrame()
            self.analyzer[batch][self.dataframe_key][self.column_key] = (
                np.random.normal(size=N_SAMPLES)
            )

        # Add the batches that should not be considered
        batches = self.discarded_batches
        dataframe_keys = [self.dataframe_key, "OtherDataframeKey"]
        column_keys = ["OtherColumnKey", self.column_key]
        for batch, dataframe_key, column_key in zip(
            batches, dataframe_keys, column_keys
        ):
            self.analyzer[batch] = Results()
            self.analyzer[batch][dataframe_key] = DataFrame()
            self.analyzer[batch][dataframe_key][column_key] = (
                np.random.normal(size=N_SAMPLES)
            )

    def check_invalid_df_col(self, method):
        df_keys = [self.dataframe_key, "Invalid"]
        col_keys = ["invalid", self.column_key]
        
        for df_key, col_key in zip(df_keys, col_keys):
            with self.assertRaises(ValueError):
                method(df_key, col_key)

    def check_invalid_alpha(self, method):
        """Check alpha for the given method"""
        df_key = self.dataframe_key
        col_key = self.column_key

        # Try invalid significance level types
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                method(df_key, col_key, alpha=alpha)

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                method(df_key, col_key, alpha=alpha)

    def check_invalid_test_func(self, method):
        """Check the test_func for the given method"""
        # Try an unsupported test function
        df_key = self.dataframe_key
        col_key = self.column_key
        not_valid_tests = (1, len, int)
        for test in not_valid_tests:
            with self.assertRaises(ValueError):
                method(df_key, col_key, test=test)

    def test_init(self):
        """Test the constructor."""
        # Create an empty analyzer
        self.analyzer = ResultsAnalyzer()

        # Check that the results analyzer is empty
        self.assertEqual(len(self.analyzer.keys()), 0)

        # Check that the results manager subclass of UserDict
        self.assertIsInstance(self.analyzer, UserDict)

    def test_setitem(self):
        """Test the __setitem__ dunder method."""
        # Create an empty analyzer
        self.analyzer = ResultsAnalyzer()
        results = Results.from_csv_files(
            (
                "execution_metrics" + CSV_FILE_EXTENSION,
                "test_fitness" + CSV_FILE_EXTENSION
            )
        )
        key = "my_wrapper"

        # Try to add an item with an invalid key
        with self.assertRaises(TypeError):
            self.analyzer[1] = results

        # Try to add an item which is not a results manager
        with self.assertRaises(TypeError):
            self.analyzer[key] = 1

        # Try a valid key and results
        self.analyzer[key] = results

        # Check the key
        self.assertTrue(key in self.analyzer)

        # Check the results
        for results_key in results:
            self.assertTrue(
                self.analyzer[key][results_key].equals(results[results_key])
            )

    def test_gather_data(self):
        """Test the _gather_data private method."""
        df_key = self.dataframe_key
        col_key = self.column_key

        batches = self.valid_batches
        data = [
            np.random.normal(size=N_SAMPLES),
            np.random.exponential(scale=3.0, size=N_SAMPLES),
            np.ones((N_SAMPLES,))
        ]

        for batch, d in zip(batches, data):
            self.analyzer[batch][df_key][col_key] = d

        gathered_data = self.analyzer._gather_data(df_key, col_key)

        # Check the results
        for batch, d in zip(batches, data):
            self.assertTrue((gathered_data[batch] == d).all())

    def test_normality_test(self):
        """Test the normality_test method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        data = [
            np.random.normal(size=N_SAMPLES),
            np.random.exponential(scale=3.0, size=N_SAMPLES),
            np.ones((N_SAMPLES,))
        ]
        expected_success = [True, False, False]
        method = self.analyzer.normality_test
        valid_tests = (shapiro, normaltest)

        # Prepare the data
        for batch, d in zip(self.valid_batches, data):
            self.analyzer[batch][df_key][col_key] = d

        # Check the test parameters
        self.check_invalid_df_col(method)
        self.check_invalid_alpha(method)
        self.check_invalid_test_func(method)

        # Try all the supported tests
        for test in valid_tests:
            test_result = method(df_key, col_key, test=test)

            # Check that all the valid batches have been considered
            for batch in self.valid_batches:
                self.assertTrue(batch in test_result.batches)

            # Check that invalid batches have not been considered
            for batch in self.discarded_batches:
                self.assertFalse(batch in test_result.batches)

            # Assert the success
            self.assertTrue(
                (test_result.success == expected_success).all()
            )

    def test_homoscedasticity_test(self):
        """Test the homoscedasticity_test method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.homoscedasticity_test
        valid_tests = (bartlett, levene, fligner)

        # Check the test parameters
        self.check_invalid_df_col(method)
        self.check_invalid_alpha(method)
        self.check_invalid_test_func(method)

        def update_0():
            """No changes. All the series are homoscedastic.
            
            :return: The expected success
            """
            return True
    
        def update_1():
            """Try with non constant heteroscedastic samples.
            
            :return: The expected success
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, scale=5)
            )
            return False

        def update_2():
            """Try with constant and non constant heteroscedastic samples.
                        
            :return: The expected success
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            return False

        def update_3():
            """Try with constant samples.
                        
            :return: The expected success
            """
            for batch in self.valid_batches:
                self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            return True
        
        update_funcs = [update_0, update_1, update_2, update_3]
        
        for func in update_funcs:
            expected_success = func()
    
            for test in valid_tests:
                test_result = method(df_key, col_key, test=test)
    
                # Check that all the valid batches have been considered
                for batch in self.valid_batches:
                    self.assertTrue(batch in test_result.batches)
    
                # Check that invalid batches have not been considered
                for batch in self.discarded_batches:
                    self.assertFalse(batch in test_result.batches)

                # Assert the success
                self.assertEqual(expected_success, test_result.success)
    
    def test_parametric_test(self):
        """Test the parametric_test method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.parametric_test

        # Check the test parameters
        self.check_invalid_df_col(method)
        self.check_invalid_alpha(method)

        def update_0():
            """No changes. The three series follow the same distribution.
            
            :return: The expected success and applied test
            """
            return True, "One-way ANOVA test"
    
        def update_1():
            """Change the first series.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, loc=5)
            )
            return False, "One-way ANOVA test"

        def update_2():
            """Delete the first series.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[0]
            del self.analyzer[batch]
            return True, "T-test"

        def update_3():
            """Change the first series.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, loc=5)
            )
            return False, "T-test"

        update_funcs = [update_0, update_1, update_2, update_3]
        
        for func in update_funcs:
            expected_success, expected_test = func()
    
            test_result = method(df_key, col_key)
            
            # Assert the test
            self.assertEqual(expected_test, test_result.test)

            # Assert the success
            self.assertEqual(expected_success, test_result.success)

            # Check that all the valid batches have been considered
            first_valid = 1 if expected_test == "T-test" else 0
            for batch in self.valid_batches[first_valid:]:
                self.assertTrue(batch in test_result.batches)

            # Check that invalid batches have not been considered
            for batch in self.discarded_batches:
                self.assertFalse(batch in test_result.batches)

    def test_non_parametric_test(self):
        """Test the non_parametric_test method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.non_parametric_test

        # Check the test parameters
        self.check_invalid_df_col(method)
        self.check_invalid_alpha(method)

        def update_0():
            """No changes. The three series follow the same distribution.
            
            :return: The expected success and applied test
            """
            return True, "Kruskal-Wallis H-test"

        def update_1():
            """Change the first series.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = (
                np.random.exponential(scale=3.0, size=N_SAMPLES)
            )
            return False, "Kruskal-Wallis H-test"

        def update_2():
            """Change the first series for a constant.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = np.zeros((N_SAMPLES,))
            return False, "Kruskal-Wallis H-test"

        def update_3():
            """All the series are constant but different.
            
            :return: The expected success and applied test
            """
            for batch in self.valid_batches[1:]:
                self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            return False, "Kruskal-Wallis H-test"

        def update_4():
            """All the series are constant and equal.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            return True, "Kruskal-Wallis H-test"

        def update_5():
            """Delete the first series.
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[0]
            del self.analyzer[batch]
            return True, "Mann-Whitney U test"

        def update_6():
            """Both series are constant but different
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = np.zeros((N_SAMPLES,))
            return False, "Mann-Whitney U test"

        def update_7():
            """One normal and another constant
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES)
            )
            return False, "Mann-Whitney U test"

        def update_8():
            """Two different distributions
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[2]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, loc=200)
            )
            return False, "Mann-Whitney U test"

        def update_9():
            """The same distribution
            
            :return: The expected success and applied test
            """
            batch = self.valid_batches[2]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES)
            )
            return True, "Mann-Whitney U test"

        update_funcs = [
            update_0, update_1, update_2, update_3, update_4,
            update_5, update_6, update_7, update_8, update_9
        ]

        for func in update_funcs:
            expected_success, expected_test = func()
    
            test_result = method(df_key, col_key)

            # Assert the test
            self.assertEqual(expected_test, test_result.test)

            # Assert the success
            self.assertEqual(expected_success, test_result.success)

            # Check that all the valid batches have been considered
            first_valid = 1 if expected_test == "Mann-Whitney U test" else 0
            for batch in self.valid_batches[first_valid:]:
                self.assertTrue(batch in test_result.batches)

            # Check that invalid batches have not been considered
            for batch in self.discarded_batches:
                self.assertFalse(batch in test_result.batches)

    def test_parametric_pairwise_test(self):
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.parametric_pairwise_test
        
        # Prepare the data
        batch = self.valid_batches[0]
        self.analyzer[batch][df_key][col_key] = (
            np.random.normal(size=N_SAMPLES, loc=5)
        )
        expected_success = [
            [True, False, False],
            [False, True, True],
            [False, True, True],
        ]

        # Check the test parameters
        self.check_invalid_df_col(method)
        self.check_invalid_alpha(method)

        # Try the comparison
        test_result = method(df_key, col_key)

        # Check that all the valid batches have been considered
        for batch in self.valid_batches:
            self.assertTrue(batch in test_result.batches)

        # Check that invalid batches have not been considered
        for batch in self.discarded_batches:
            self.assertFalse(batch in test_result.batches)

            # Assert the success
            self.assertTrue(
                (test_result.success == expected_success).all()
            )

    def test_non_parametric_pairwise_test(self):
        """Test the non_parametric_pairwise_test method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.non_parametric_pairwise_test

        # Check the test parameters
        self.check_invalid_df_col(method)
        self.check_invalid_alpha(method)
        
        # Try without p-adjustment
        method(df_key, col_key, p_adjust=None)

        # Try a wrong p-adjustment method
        with self.assertRaises(ValueError):
            method(df_key, col_key, p_adjust="Wrong")

        def update_0():
            """No changes. The three series follow the same distribution.
            
            :return: The expected success
            """
            num_batches = len(self.valid_batches)
            return [
                [True for _ in range(num_batches)] for _ in range(num_batches)
            ]

        def update_1():
            """Change the first series.
            
            :return: The expected success
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, loc=5)
            )
            
            return [
                [True, False, False], [False, True, True], [False, True, True]
            ]

        def update_2():
            """Change the second series for a constant.
            
            :return: The expected success
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            return [
                [True, False, False], [False, True, False], [False, False, True]
            ]

        def update_3():
            """Change the first series for a constant.
            
            :return: The expected success
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = np.full((N_SAMPLES,), 100)
            
            return [
                [True, False, False], [False, True, False], [False, False, True]
            ]

        def update_4():
            """Change the third series for a constant.
            
            :return: The expected success
            """
            batch = self.valid_batches[2]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            return [
                [True, False, False], [False, True, True], [False, True, True]
            ]

        def update_5():
            """All the series ar constant and equal.
            
            :return: The expected success
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            num_batches = len(self.valid_batches)
            return [
                [True for _ in range(num_batches)] for _ in range(num_batches)
            ]


        update_funcs = [
            update_0, update_1, update_2, update_3, update_4, update_5
        ]

        for func in update_funcs:
            expected_success = func()
    
            test_result = method(df_key, col_key)

            # Check that all the valid batches have been considered
            for batch in self.valid_batches:
                self.assertTrue(batch in test_result.batches)

            # Check that invalid batches have not been considered
            for batch in self.discarded_batches:
                self.assertFalse(batch in test_result.batches)

            # Assert the success
            self.assertTrue((expected_success==test_result.success).all())

    def test_effect_size(self):
        """Test the rank method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.effect_size

        def gen_cut_off(
            val:float | None = None,
            where: list[tuple[int, int]] | None = None
        ):
            """Generate a cut off matrix.
            
            :param val: Cut off value
            :param where: Coordinates for *val*
            """
            num_batches = len(self.valid_batches)
            
            cut_off = [
                [0 for _ in range(num_batches)] for _ in range(num_batches)
            ]
            
            if val is not None and where is not None:
                for coor in where:
                    cut_off[coor[0]][coor[1]] = val

            return cut_off
            
        
        def update_0():
            """No changes. The three series follow the same distribution.
            
            :return: The cut-off threshold
            """
            first_batch = self.valid_batches[0]
            first_batch_data = self.analyzer[first_batch][df_key][col_key]

            for batch in self.valid_batches[1:]:
                self.analyzer[batch][df_key][col_key] = first_batch_data
            
            return gen_cut_off()
            
        def update_1():
            """The first series is shifted.
            
            :return: The cut-off threshold
            """
            batch = self.valid_batches[0]
            diff = 0.2

            self.analyzer[batch][df_key][col_key] += diff
            
            return gen_cut_off(diff, [(0, 1), (0, 2), (1, 0), (2, 0)])
                    
        def update_2():
            """The first series is constant.
            
            :return: The cut-off threshold
            """
            batch = self.valid_batches[0]
            diff = 1.5

            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            return gen_cut_off(diff, [(0, 1), (0, 2), (1, 0), (2, 0)])

        def update_3():
            """The second series is constant and equal to the first one.
            
            :return: The cut-off threshold
            """
            batch = self.valid_batches[1]
            diff = 1.5

            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            return gen_cut_off(diff, [(0, 2), (1, 2), (2, 0), (2, 1)])

        def update_4():
            """The third series is constant but different.
            
            :return: The cut-off threshold
            """
            batch = self.valid_batches[2]
            diff = np.inf

            self.analyzer[batch][df_key][col_key] = np.zeros((N_SAMPLES,))
            
            return gen_cut_off(diff, [(0, 2), (1, 2), (2, 0), (2, 1)])
        
        update_funcs = [
            update_0, update_1, update_2, update_3, update_4
        ]

        for func in update_funcs:
            cut_off = func()
    
            test_result = method(df_key, col_key)

            # Check that all the valid batches have been considered
            for batch in self.valid_batches:
                self.assertTrue(batch in test_result.batches)

            # Check that invalid batches have not been considered
            for batch in self.discarded_batches:
                self.assertFalse(batch in test_result.batches)

            # Assert the success
            self.assertTrue((test_result.value <= cut_off).all())
    
    def test_compare(self):
        """Test the compare method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        method = self.analyzer.compare

        def update_0():
            """The three series follow the same normal distribution.
            
            :return: The pairwise comparison success and applied test
            """
            first_batch = self.valid_batches[0]
            first_batch_data = self.analyzer[first_batch][df_key][col_key]
            num_batches = len(self.valid_batches)

            for batch in self.valid_batches[1:]:
                self.analyzer[batch][df_key][col_key] = (
                    first_batch_data +
                    np.random.uniform(low=-0.001, high=0.001, size=N_SAMPLES)
                )
            
                success = [
                    [True for _ in range(num_batches)]
                    for _ in range(num_batches)
                ]
            return success, "One-way ANOVA test"

        def update_1():
            """Change the first series.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, loc=5)
            )
            
            success = [
                [True, False, False], [False, True, True], [False, True, True]
            ]
            return success, "Tukey's HSD test"
        
        def update_2():
            """Change the first series for an exponential distribution.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = (
                np.random.exponential(scale=3.0, size=N_SAMPLES)
            )
            
            success = [
                [True, False, False], [False, True, True], [False, True, True]
            ]
            return success, "Dunn's test"

        def update_3():
            """The three series follow the same exponential distribution.
            
            :return: The pairwise comparison success and applied test
            """
            first_batch = self.valid_batches[0]
            first_batch_data = self.analyzer[first_batch][df_key][col_key]
            num_batches = len(self.valid_batches)

            for batch in self.valid_batches[1:]:
                self.analyzer[batch][df_key][col_key] = (
                    first_batch_data +
                    np.random.uniform(low=-0.001, high=0.001, size=N_SAMPLES)
                )
            
                success = [
                    [True for _ in range(num_batches)]
                    for _ in range(num_batches)
                ]
            return success, "Kruskal-Wallis H-test"

        def update_4():
            """Change the second series for a constant.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            success = [
                [True, False, True], [False, True, False], [True, False, True]
            ]
            return success, "Dunn's test"

        def update_5():
            """Change the first series for another constant.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[0]
            self.analyzer[batch][df_key][col_key] = np.full((N_SAMPLES,), 100)
            
            success = [
                [True, False, False], [False, True, False], [False, False, True]
            ]
            return success, "Dunn's test"

        def update_6():
            """The second and third series are constant and equal.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[2]
            self.analyzer[batch][df_key][col_key] = np.ones((N_SAMPLES,))
            
            success = [
                [True, False, False], [False, True, True], [False, True, True]
            ]
            return success, "Dunn's test"

        def update_7():
            """Delete the first series. Two constant equal distributions remain.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[0]
            del self.analyzer[batch]
            
            success = [
                [True, True], [True, True]
            ]
            return success, "Mann-Whitney U test"

        def update_8():
            """Two constant but different distributions.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = np.zeros((N_SAMPLES,))
            
            success = [
                [True, False], [False, True]
            ]
            return success, "Mann-Whitney U test"

        def update_9():
            """One constant and another exponential.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = (
                np.random.exponential(scale=3.0, size=N_SAMPLES)
            )
            
            success = [
                [True, False], [False, True]
            ]
            return success, "Mann-Whitney U test"

        def update_10():
            """One normal and another exponential.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[2]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES)
            )
            
            success = [
                [True, False], [False, True]
            ]
            return success, "Mann-Whitney U test"

        def update_11():
            """Two normal and equal distributions.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES)
            )
            
            success = [
                [True, True], [True, True]
            ]
            return success, "T-test"

        def update_12():
            """Two normal but different distributions.
            
            :return: The pairwise comparison success and applied test
            """
            batch = self.valid_batches[1]
            self.analyzer[batch][df_key][col_key] = (
                np.random.normal(size=N_SAMPLES, loc=5)
            )
            
            success = [
                [True, False], [False, True]
            ]
            return success, "T-test"

        update_funcs = [
            update_0, update_1, update_2, update_3, update_4, update_5, 
            update_6, update_7, update_8, update_9, update_10, update_11,
            update_12
        ]

        for func in update_funcs:
            expected_success, expected_test = func()
    
            test_result = method(df_key, col_key).pairwise_comparison

            # Assert the test
            self.assertTrue(test_result.test.startswith(expected_test))

            # Assert the success
            self.assertTrue((expected_success == test_result.success).all())

            # Check that all the valid batches have been considered
            first_valid = (
                1 if any(
                    test.startswith(expected_test)
                    for test in ("T-test", "Mann-Whitney U test")
                )
                else 0
            )
            for batch in self.valid_batches[first_valid:]:
                self.assertTrue(batch in test_result.batches)

            # Check that invalid batches have not been considered
            for batch in self.discarded_batches:
                self.assertFalse(batch in test_result.batches)


    def test_rank(self):
        """Test the rank method."""
        df_key = self.dataframe_key
        col_key = self.column_key
        self.analyzer = ResultsAnalyzer()

        # Prepare the batches
        batch_names = [f"Results{i}" for i in range(5)]
        data = [
            np.random.normal(size=N_SAMPLES),
            np.random.normal(size=N_SAMPLES),
            np.random.normal(size=N_SAMPLES, loc=5),
            np.random.normal(size=N_SAMPLES, loc=20),
            np.random.normal(size=N_SAMPLES, loc=20)
        ]
        for batch, d in zip(batch_names, data):
            self.analyzer[batch] = Results()
            self.analyzer[batch][df_key] = DataFrame()
            self.analyzer[batch][df_key][col_key] = d
        
        # Expected ranks
        expected_ranks = [0.5, 0.5, 2, 3.5, 3.5]

        # Rank
        weight = -1
        ranked_results = self.analyzer.rank(df_key, col_key, weight)

        # Check the ranks
        for batch, rank in zip(batch_names, expected_ranks):
            self.assertEqual(ranked_results[batch], rank)

    def test_multiple_rank(self):
        """Test the multiple_rank method."""
        # Test fitness related results
        test_fitness_key = "test_fitness"
        kappa_key = "Kappa"
        nf_key = "NF"

        # Execution metrics related results
        execution_metrics_key = "execution_metrics"
        runtime_key = "Runtime"

        # Prepare the batches
        self.analyzer = ResultsAnalyzer()
        batch_names = [f"Results{i}" for i in range(4)]

        kappa_data = [
            np.random.normal(size=N_SAMPLES, loc=0.9),
            np.random.normal(size=N_SAMPLES, loc=0.8),
            np.random.normal(size=N_SAMPLES, loc=0.4),
            np.random.normal(size=N_SAMPLES, loc=0.9),
        ]
        nf_data = [
            np.random.normal(size=N_SAMPLES, loc=6),
            np.random.normal(size=N_SAMPLES, loc=6),
            np.random.normal(size=N_SAMPLES, loc=8),
            np.random.normal(size=N_SAMPLES, loc=4)
        ]
        runtime_data = [
            np.random.normal(size=N_SAMPLES, loc=125),
            np.random.normal(size=N_SAMPLES, loc=80),
            np.random.normal(size=N_SAMPLES, loc=50),
            np.random.normal(size=N_SAMPLES, loc=80)
        ]
        for batch, kappa, nf, runtime in zip(
            batch_names, kappa_data, nf_data, runtime_data
        ):
            self.analyzer[batch] = Results()
            self.analyzer[batch][test_fitness_key] = DataFrame()
            self.analyzer[batch][test_fitness_key][kappa_key] = kappa
            self.analyzer[batch][test_fitness_key][nf_key] = nf
            self.analyzer[batch][execution_metrics_key] = DataFrame()
            self.analyzer[batch][execution_metrics_key][runtime_key] = runtime
        
        # Expected ranks
        expected_ranks = {
            nf_key: [1.5, 1.5, 3.0, 0.0],
            kappa_key: [1.0, 1.0, 3.0, 1.0],
            runtime_key: [3.0, 1.5, 0.0, 1.5]
        }

        # Rank
        dataframe_keys = (
            test_fitness_key, test_fitness_key, execution_metrics_key
        )
        column_keys = (nf_key, kappa_key, runtime_key)
        weights = (-1, 1, -1)
        multiple_ranked_results = self.analyzer.multiple_rank(
            dataframe_keys, column_keys, weights
        )

        # Check the ranks
        for df_key, col_key in zip(dataframe_keys, column_keys):
            for batch_idx, batch in enumerate(batch_names):
                self.assertEqual(
                    multiple_ranked_results[df_key][col_key][batch],
                    expected_ranks[col_key][batch_idx]
                )


if __name__ == '__main__':
    unittest.main()
