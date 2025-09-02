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

"""Unit test for :py:class:`culebra.tools.ResultsAnalyzer`."""

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


class ResultsAnalyzerTester(unittest.TestCase):
    """Test the :py:class:`~culebra.tools.ResultsAnalyzer` class."""

    def test_init(self):
        """Test the constructor."""
        # Create the analyzer
        analyzer = ResultsAnalyzer()

        # Check that the results analyzer is empty
        self.assertEqual(len(analyzer.keys()), 0)

        # Check that the results manager subclass of UserDict
        self.assertIsInstance(analyzer, UserDict)

    def test_setitem(self):
        """Test the __setitem__ dunder method."""
        # Create the analyzer
        analyzer = ResultsAnalyzer()
        results = Results.from_csv_files(
            (
                "execution_metrics" + CSV_FILE_EXTENSION,
                "test_fitness" + CSV_FILE_EXTENSION
            )
        )
        key = "my_wrapper"

        # Try to add an item with an invalid key
        with self.assertRaises(TypeError):
            analyzer[1] = results

        # Try to add an item which is not a results manager
        with self.assertRaises(TypeError):
            analyzer[key] = 1

        # Try a valid key and results
        analyzer[key] = results

        # Check the key
        self.assertTrue(key in analyzer)

        # Check the results
        for results_key in results:
            self.assertTrue(
                analyzer[key][results_key].equals(results[results_key])
            )

    def test_gather_data(self):
        """Test the _gather_data private method."""
        normal_column_key = "Normal"
        not_normal_column_key = "NotNormal"
        another_column_key = "AnotherColumnKey"
        dataframe_key = "NormalityTest"
        another_dataframe_key = "AnotherDataframeKey"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare a first DataFrame with some data
        data1 = DataFrame()
        data1[normal_column_key] = np.random.normal(size=100)
        data1[not_normal_column_key] = np.random.beta(a=1, b=5, size=100)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare a second DataFrame with some data
        data2 = DataFrame()
        data2[normal_column_key] = np.random.normal(size=200)
        data2[not_normal_column_key] = np.random.normal(size=200)
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Insert a third dataframe with another dataframe key
        # Should not be considerted in normality tests
        data3 = DataFrame()
        data3[normal_column_key] = np.random.normal(size=50)
        results3 = Results()
        results3[another_dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another column key
        # Should not be considerted in normality tests
        data4 = DataFrame()
        data4[another_column_key] = np.random.normal(size=300)
        results4 = Results()
        results4[dataframe_key] = data4
        analyzer["Results4"] = results4

        data = analyzer._gather_data(dataframe_key, normal_column_key)

        # Check the results
        self.assertTrue(
            (
                analyzer['Results1'][dataframe_key][normal_column_key] ==
                data['Results1']
            ).all()
        )
        self.assertTrue(
            (
                analyzer['Results2'][dataframe_key][normal_column_key] ==
                data['Results2']
            ).all()
        )

    def test_normality_test(self):
        """Test the normality_test method."""
        normal_column_key = "Normal"
        not_normal_column_key = "NotNormal"
        another_column_key = "AnotherColumnKey"
        invalid_column_key = "InvalidColumn"
        dataframe_key = "NormalityTest"
        another_dataframe_key = "AnotherDataframeKey"
        invalid_dataframe_key = "InvalidDataframe"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare a first DataFrame with some data
        data1 = DataFrame()
        data1[normal_column_key] = np.random.normal(size=200)
        data1[not_normal_column_key] = np.random.beta(a=1, b=5, size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare a second DataFrame with some data
        data2 = DataFrame()
        data2[normal_column_key] = np.random.normal(size=200)
        data2[not_normal_column_key] = np.random.normal(size=200)
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Insert a third dataframe with another dataframe key
        # Should not be considered in normality tests
        data3 = DataFrame()
        data3[normal_column_key] = np.random.normal(size=200)
        results3 = Results()
        results3[another_dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another column key
        # Should not be considered in normality tests
        data4 = DataFrame()
        data4[another_column_key] = np.random.normal(size=200)
        results4 = Results()
        results4[dataframe_key] = data4
        analyzer["Results4"] = results4

        # Try invalid significance level types
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                analyzer.normality_test(
                    dataframe_key, normal_column_key, alpha=alpha
                )

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                analyzer.normality_test(
                    dataframe_key, normal_column_key, alpha=alpha
                )

        # Valid significance levels
        valid_alpha_values = (0, 0.05, 0.5, 0.95, 1)
        for alpha in valid_alpha_values:
            analyzer.normality_test(
                dataframe_key, normal_column_key, alpha=alpha
            )

        # Try an unsupported test function
        not_valid_tests = (1, len, int)
        for test in not_valid_tests:
            with self.assertRaises(ValueError):
                analyzer.normality_test(
                    dataframe_key, normal_column_key, test=test
                )

        # Try an invalid dataframe key
        with self.assertRaises(ValueError):
            analyzer.normality_test(
                invalid_dataframe_key, normal_column_key
            )

        # Try an invalid column key
        with self.assertRaises(ValueError):
            analyzer.normality_test(
                dataframe_key, invalid_column_key
            )

        # Try all the supported tests
        valid_tests = (shapiro, normaltest)
        for test in valid_tests:
            # Try normal results
            test_result = analyzer.normality_test(
                dataframe_key, normal_column_key, test=test
            )

            # Normality should be true
            self.assertTrue(test_result.success.all())

            # Check that results1 and results2 have been considered
            self.assertTrue("Results1" in test_result.batches)
            self.assertTrue("Results2" in test_result.batches)

            # Check that results3 and results4 have not been considered
            self.assertFalse("Results3" in test_result.batches)
            self.assertFalse("Results4" in test_result.batches)

            # Try not normal results
            test_result = analyzer.normality_test(
                dataframe_key, not_normal_column_key, test=test
            )

            # Normality should be false
            self.assertFalse(test_result.success.all())

            # Check that results1 and results2 have been considered
            self.assertTrue("Results1" in test_result.batches)
            self.assertTrue("Results2" in test_result.batches)

            # Check that results3 and results4 have not been considered
            self.assertFalse("Results3" in test_result.batches)
            self.assertFalse("Results4" in test_result.batches)

    def test_homoscedasticity_test(self):
        """Test the homoscedasticity_test method."""
        homoscedastic_column_key = "Homoscedastic"
        heteroscedastic_column_key = "Heteroscedastic"
        another_column_key = "AnotherColumnKey"
        invalid_column_key = "InvalidColumn"
        dataframe_key = "HomoscedasticityTest"
        another_dataframe_key = "AnotherDataframeKey"
        invalid_dataframe_key = "InvalidDataframe"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[homoscedastic_column_key] = np.random.normal(size=200)
        data1[heteroscedastic_column_key] = np.random.normal(size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[homoscedastic_column_key] = np.random.normal(size=200)
        data2[heteroscedastic_column_key] = np.random.normal(
            size=200, scale=5
        )
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Insert a third dataframe with another dataframe key
        # Should not be considered in homoscedasticity tests
        data3 = DataFrame()
        data3[homoscedastic_column_key] = np.random.normal(size=200)
        results3 = Results()
        results3[another_dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another column key
        # Should not be considered in homoscedasticity tests
        data4 = DataFrame()
        data4[another_column_key] = np.random.normal(size=200)
        results4 = Results()
        results4[dataframe_key] = data4
        analyzer["Results4"] = results4

        # Try an unsupported test function
        not_valid_tests = (1, len, int)
        for test in not_valid_tests:
            with self.assertRaises(ValueError):
                analyzer.homoscedasticity_test(
                    dataframe_key, homoscedastic_column_key, test=test
                )

        # Try invalid significance level typess
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                analyzer.homoscedasticity_test(
                    dataframe_key, homoscedastic_column_key, alpha=alpha
                )

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                analyzer.homoscedasticity_test(
                    dataframe_key, homoscedastic_column_key, alpha=alpha
                )
        # Try an invalid dataframe key
        with self.assertRaises(ValueError):
            analyzer.homoscedasticity_test(
                invalid_dataframe_key, homoscedastic_column_key
            )

        # Try an invalid column key
        with self.assertRaises(ValueError):
            analyzer.homoscedasticity_test(
                dataframe_key, invalid_column_key
            )

        # Valid significance levels
        valid_alpha_values = (0, 0.05, 0.5, 0.95, 1)
        for alpha in valid_alpha_values:
            analyzer.homoscedasticity_test(
                dataframe_key, homoscedastic_column_key, alpha=alpha
            )

        # Try all the supported tests
        valid_tests = (bartlett, levene, fligner)
        for test in valid_tests:
            # Try homoscedastic results
            test_result = analyzer.homoscedasticity_test(
                dataframe_key, homoscedastic_column_key, test=test
            )

            # Homoscedasticity should be true
            self.assertTrue(test_result.success)

            # Check that results1 and results2 have been considered
            self.assertTrue("Results1" in test_result.batches)
            self.assertTrue("Results2" in test_result.batches)

            # Check that results3 and results4 have not been considered
            self.assertFalse("Results3" in test_result.batches)
            self.assertFalse("Results4" in test_result.batches)

            # Try heteroscedastic results
            test_result = analyzer.homoscedasticity_test(
                dataframe_key, heteroscedastic_column_key, test=test
            )

            # Homoscedasticity should be false
            self.assertFalse(test_result.success)

            # Check that results1 and results2 have been considered
            self.assertTrue("Results1" in test_result.batches)
            self.assertTrue("Results2" in test_result.batches)

            # Check that results3 and results4 have not been considered
            self.assertFalse("Results3" in test_result.batches)
            self.assertFalse("Results4" in test_result.batches)

    def test_parametric_test(self):
        """Test the parametric_test method."""
        equal_column_key = "Equal"
        different_column_key = "Different"
        another_column_key = "AnotherColumnKey"
        invalid_column_key = "InvalidColumn"
        dataframe_key = "Parametric"
        another_dataframe_key = "AnotherDataframeKey"
        invalid_dataframe_key = "InvalidDataframe"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[equal_column_key] = np.random.normal(size=200)
        data1[different_column_key] = np.random.normal(size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Try to analyze not enough results
        with self.assertRaises(ValueError):
            analyzer.parametric_test(
                dataframe_key, equal_column_key
            )

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[equal_column_key] = np.random.normal(size=200)
        data2[different_column_key] = np.random.normal(
            size=200, loc=5
        )
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the thrid results manager
        data3 = DataFrame()
        data3[equal_column_key] = np.random.normal(size=200)
        data3[different_column_key] = np.random.normal(
            size=200, loc=10
        )
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another dataframe key
        # Should not be considered in the tests
        data4 = DataFrame()
        data4[equal_column_key] = np.random.normal(size=200)
        results4 = Results()
        results4[another_dataframe_key] = data4
        analyzer["Results4"] = results4

        # Insert a fifth dataframe with another column key
        # Should not be considered in the tests
        data5 = DataFrame()
        data5[another_column_key] = np.random.normal(size=200)
        results5 = Results()
        results5[dataframe_key] = data5
        analyzer["Results5"] = results5

        # Try invalid significance level typess
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                analyzer.parametric_test(
                    dataframe_key, equal_column_key, alpha=alpha
                )

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                analyzer.parametric_test(
                    dataframe_key, equal_column_key, alpha=alpha
                )
        # Try an invalid dataframe key
        with self.assertRaises(ValueError):
            analyzer.parametric_test(
                invalid_dataframe_key, equal_column_key
            )

        # Try an invalid column key
        with self.assertRaises(ValueError):
            analyzer.parametric_test(
                dataframe_key, invalid_column_key
            )

        # Valid significance levels
        valid_alpha_values = (0, 0.05, 0.5, 0.95, 1)
        for alpha in valid_alpha_values:
            analyzer.parametric_test(
                dataframe_key, equal_column_key, alpha=alpha
            )

        # Try ANOVA with equal distributions
        test_result = analyzer.parametric_test(
            dataframe_key, equal_column_key
        )

        # The test should be successful
        self.assertTrue(test_result.success)

        # The test should be ANOVA
        self.assertEqual(test_result.test, "One-way ANOVA test")

        # Check that results1-results3 have been considered
        self.assertTrue("Results1" in test_result.batches)
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        # Try ANOVA with different distributions
        test_result = analyzer.parametric_test(
            dataframe_key, different_column_key
        )

        # The test should not be successful
        self.assertFalse(test_result.success)

        # The test should be ANOVA
        self.assertEqual(test_result.test, "One-way ANOVA test")

        # Check that results1-results3 have been considered
        self.assertTrue("Results1" in test_result.batches)
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        # Try a t-test with equal distributions
        del analyzer['Results1']
        test_result = analyzer.parametric_test(
            dataframe_key, equal_column_key
        )

        # The test should be successful
        self.assertTrue(test_result.success)

        # The test should be t-test
        self.assertEqual(test_result.test, "T-test")

        # Check that results2 and results3 have been considered
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        # Try t-test with different distributions
        test_result = analyzer.parametric_test(
            dataframe_key, different_column_key
        )

        # The test should not be successful
        self.assertFalse(test_result.success)

        # The test should be t-test
        self.assertEqual(test_result.test, "T-test")

        # Check that results2 and results3 have been considered
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

    def test_non_parametric_test(self):
        """Test the non_parametric_test method."""
        equal_column_key = "Equal"
        different_column_key = "Different"
        another_column_key = "AnotherColumnKey"
        invalid_column_key = "InvalidColumn"
        dataframe_key = "NonParametric"
        another_dataframe_key = "AnotherDataframeKey"
        invalid_dataframe_key = "InvalidDataframe"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[equal_column_key] = np.random.normal(size=200)
        data1[different_column_key] = np.random.normal(size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Try to analyze not enough results
        with self.assertRaises(ValueError):
            analyzer.non_parametric_test(
                dataframe_key, equal_column_key
            )

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[equal_column_key] = np.random.normal(size=200)
        data2[different_column_key] = np.random.normal(
            size=200, loc=5
        )
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the thrid results manager
        data3 = DataFrame()
        data3[equal_column_key] = np.random.normal(size=200)
        data3[different_column_key] = np.random.normal(
            size=200, loc=10
        )
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another dataframe key
        # Should not be considered in the tests
        data4 = DataFrame()
        data4[equal_column_key] = np.random.normal(size=200)
        results4 = Results()
        results4[another_dataframe_key] = data4
        analyzer["Results4"] = results4

        # Insert a fifth dataframe with another column key
        # Should not be considered in the tests
        data5 = DataFrame()
        data5[another_column_key] = np.random.normal(size=200)
        results5 = Results()
        results5[dataframe_key] = data5
        analyzer["Results5"] = results5

        # Try invalid significance level typess
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                analyzer.non_parametric_test(
                    dataframe_key, equal_column_key, alpha=alpha
                )

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                analyzer.non_parametric_test(
                    dataframe_key, equal_column_key, alpha=alpha
                )
        # Try an invalid dataframe key
        with self.assertRaises(ValueError):
            analyzer.non_parametric_test(
                invalid_dataframe_key, equal_column_key
            )

        # Try an invalid column key
        with self.assertRaises(ValueError):
            analyzer.non_parametric_test(
                dataframe_key, invalid_column_key
            )

        # Valid significance levels
        valid_alpha_values = (0, 0.05, 0.5, 0.95, 1)
        for alpha in valid_alpha_values:
            analyzer.non_parametric_test(
                dataframe_key, equal_column_key, alpha=alpha
            )

        # Try Kruskal-Wallis with equal distributions
        test_result = analyzer.non_parametric_test(
            dataframe_key, equal_column_key
        )

        # The test should be successful
        self.assertTrue(test_result.success)

        # The test should be Kruskal-Wallis
        self.assertEqual(test_result.test, "Kruskal-Wallis H-test")

        # Check that results1-results3 have been considered
        self.assertTrue("Results1" in test_result.batches)
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        # Try Kruskal-Wallis with different distributions
        test_result = analyzer.non_parametric_test(
            dataframe_key, different_column_key
        )

        # The test should not be successful
        self.assertFalse(test_result.success)

        # The test should be Kruskal-Wallis
        self.assertEqual(test_result.test, "Kruskal-Wallis H-test")

        # Check that results1-results3 have been considered
        self.assertTrue("Results1" in test_result.batches)
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        # Try a Mann-Whitney with equal distributions
        del analyzer['Results1']
        test_result = analyzer.non_parametric_test(
            dataframe_key, equal_column_key
        )

        # The test should be successful
        self.assertTrue(test_result.success)

        # The test should be Mann-Whitney
        self.assertEqual(test_result.test, "Mann-Whitney U test")

        # Check that results2 and results3 have been considered
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        # Try Mann-Whitney with different distributions
        test_result = analyzer.non_parametric_test(
            dataframe_key, different_column_key
        )

        # The test should not be successful
        self.assertFalse(test_result.success)

        # The test should be Mann-Whitney
        self.assertEqual(test_result.test, "Mann-Whitney U test")

        # Check that results2 and results3 have been considered
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

    def test_parametric_pairwise_test(self):
        """Test the parametric_pairwise_test method."""
        column_key = "Data"
        another_column_key = "AnotherColumnKey"
        invalid_column_key = "InvalidColumn"
        dataframe_key = "ParametricPairwise"
        another_dataframe_key = "AnotherDataframeKey"
        invalid_dataframe_key = "InvalidDataframe"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[column_key] = np.random.normal(size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[column_key] = np.random.normal(size=200, loc=5)
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the thrid results manager
        data3 = DataFrame()
        data3[column_key] = np.random.normal(size=200)
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another dataframe key
        # Should not be considered in the tests
        data4 = DataFrame()
        data4[column_key] = np.random.normal(size=200)
        results4 = Results()
        results4[another_dataframe_key] = data4
        analyzer["Results4"] = results4

        # Insert a fifth dataframe with another column key
        # Should not be considered in the tests
        data5 = DataFrame()
        data5[another_column_key] = np.random.normal(size=200, loc=5)
        results5 = Results()
        results5[dataframe_key] = data5
        analyzer["Results5"] = results5

        # Try invalid significance level typess
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                analyzer.parametric_pairwise_test(
                    dataframe_key, column_key, alpha=alpha
                )

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                analyzer.parametric_pairwise_test(
                    dataframe_key, column_key, alpha=alpha
                )
        # Try an invalid dataframe key
        with self.assertRaises(ValueError):
            analyzer.parametric_pairwise_test(
                invalid_dataframe_key, column_key
            )

        # Try an invalid column key
        with self.assertRaises(ValueError):
            analyzer.parametric_pairwise_test(
                dataframe_key, invalid_column_key
            )

        # Valid significance levels
        valid_alpha_values = (0, 0.05, 0.5, 0.95, 1)
        for alpha in valid_alpha_values:
            analyzer.parametric_pairwise_test(
                dataframe_key, column_key, alpha=alpha
            )

        # Try the comparison
        test_result = analyzer.parametric_pairwise_test(
            dataframe_key, column_key
        )

        # The test should be successful when comparing each result
        # with itself
        self.assertTrue(test_result.success[0][0])
        self.assertTrue(test_result.success[1][1])
        self.assertTrue(test_result.success[2][2])

        # Check that results1-results3 have been considered
        self.assertTrue("Results1" in test_result.batches)
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        test_result = analyzer.parametric_pairwise_test(
            dataframe_key, column_key
        )

    def test_non_parametric_pairwise_test(self):
        """Test the non_parametric_pairwise_test method."""
        column_key = "Data"
        another_column_key = "AnotherColumnKey"
        invalid_column_key = "InvalidColumn"
        dataframe_key = "NonParametricPairwise"
        another_dataframe_key = "AnotherDataframeKey"
        invalid_dataframe_key = "InvalidDataframe"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[column_key] = np.random.normal(size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[column_key] = np.random.normal(size=200, loc=5)
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the thrid results manager
        data3 = DataFrame()
        data3[column_key] = np.random.normal(size=200)
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Insert a fourth dataframe with another dataframe key
        # Should not be considered in the tests
        data4 = DataFrame()
        data4[column_key] = np.random.normal(size=200)
        results4 = Results()
        results4[another_dataframe_key] = data4
        analyzer["Results4"] = results4

        # Insert a fifth dataframe with another column key
        # Should not be considered in the tests
        data5 = DataFrame()
        data5[another_column_key] = np.random.normal(size=200, loc=5)
        results5 = Results()
        results5[dataframe_key] = data5
        analyzer["Results5"] = results5

        # Try invalid significance level typess
        not_valid_alpha_types = (int, len)
        for alpha in not_valid_alpha_types:
            with self.assertRaises(TypeError):
                analyzer.non_parametric_pairwise_test(
                    dataframe_key, column_key, alpha=alpha
                )

        # Try invalid significance level values
        not_valid_alpha_values = (-1, 3, -1.3, 2.8)
        for alpha in not_valid_alpha_values:
            with self.assertRaises(ValueError):
                analyzer.non_parametric_pairwise_test(
                    dataframe_key, column_key, alpha=alpha
                )
        # Valid significance levels
        valid_alpha_values = (0, 0.05, 0.5, 0.95, 1)
        for alpha in valid_alpha_values:
            analyzer.non_parametric_pairwise_test(
                dataframe_key, column_key, alpha=alpha
            )

        # Try without p-adjustment
        analyzer.non_parametric_pairwise_test(
            dataframe_key, column_key, p_adjust=None
        )

        # Try a wrong p-adjustment method
        with self.assertRaises(ValueError):
            analyzer.non_parametric_pairwise_test(
                dataframe_key, column_key, p_adjust="Wrong"
            )

        # Try an invalid dataframe key
        with self.assertRaises(ValueError):
            analyzer.non_parametric_pairwise_test(
                invalid_dataframe_key, column_key
            )

        # Try an invalid column key
        with self.assertRaises(ValueError):
            analyzer.non_parametric_pairwise_test(
                dataframe_key, invalid_column_key
            )

        # Try the comparison
        test_result = analyzer.non_parametric_pairwise_test(
            dataframe_key, column_key
        )

        # The test should be successful when comparing each result
        # with itself
        self.assertTrue(test_result.success[0][0])
        self.assertTrue(test_result.success[1][1])
        self.assertTrue(test_result.success[2][2])

        # Check that results1-results3 have been considered
        self.assertTrue("Results1" in test_result.batches)
        self.assertTrue("Results2" in test_result.batches)
        self.assertTrue("Results3" in test_result.batches)

        # Check that results4 and results5 have not been considered
        self.assertFalse("Results4" in test_result.batches)
        self.assertFalse("Results5" in test_result.batches)

        test_result = analyzer.non_parametric_pairwise_test(
            dataframe_key, column_key
        )

    def test_compare(self):
        """Test the compare method."""
        parametric_column_key = "Parametric"
        non_parametric_column_key = "NonParametric"
        dataframe_key = "Comparison"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[parametric_column_key] = np.random.normal(size=200)
        data1[non_parametric_column_key] = np.random.beta(a=1, b=5, size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[parametric_column_key] = (
            data1[parametric_column_key] +
            np.random.uniform(low=-0.001, high=0.001, size=200)
        )
        data2[non_parametric_column_key] = (
            data1[non_parametric_column_key] +
            np.random.uniform(low=-0.001, high=0.001, size=200)
        )
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Try a comparison of two equal normal and homoscedastic results
        comparison_result = analyzer.compare(
            dataframe_key, parametric_column_key
        )
        self.assertNotEqual(comparison_result.normality, None)
        self.assertTrue(comparison_result.normality.success.all())
        self.assertNotEqual(comparison_result.homoscedasticity, None)
        self.assertTrue(comparison_result.homoscedasticity.success.all())
        self.assertTrue(comparison_result.global_comparison.success.all())
        self.assertNotEqual(comparison_result.pairwise_comparison, None)

        # Try a comparison of two equal but not normal results
        comparison_result = analyzer.compare(
            dataframe_key, non_parametric_column_key
        )
        self.assertNotEqual(comparison_result.normality, None)
        self.assertFalse(comparison_result.normality.success.all())
        self.assertEqual(comparison_result.homoscedasticity, None)
        self.assertTrue(comparison_result.global_comparison.success.all())
        self.assertNotEqual(comparison_result.pairwise_comparison, None)

        # Prepare the DataFrame with some data for the third results manager
        data3 = DataFrame()
        data3[parametric_column_key] = (
            data1[parametric_column_key] +
            np.random.uniform(low=-0.001, high=0.001, size=200)
        )
        data3[non_parametric_column_key] = (
            data1[non_parametric_column_key] +
            np.random.uniform(low=-0.001, high=0.001, size=200)
        )
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Try a comparison of three equal normal and homoscedastic results
        comparison_result = analyzer.compare(
            dataframe_key, parametric_column_key
        )
        self.assertNotEqual(comparison_result.normality, None)
        self.assertTrue(comparison_result.normality.success.all())
        self.assertNotEqual(comparison_result.homoscedasticity, None)
        self.assertTrue(comparison_result.homoscedasticity.success.all())
        self.assertTrue(comparison_result.global_comparison.success.all())
        self.assertNotEqual(comparison_result.pairwise_comparison, None)

        # Try a comparison of three equal heteroscedastic results
        comparison_result = analyzer.compare(
            dataframe_key, non_parametric_column_key
        )
        self.assertNotEqual(comparison_result.normality, None)
        self.assertFalse(comparison_result.normality.success.all())
        self.assertEqual(comparison_result.homoscedasticity, None)
        self.assertTrue(comparison_result.global_comparison.success.all())
        self.assertNotEqual(comparison_result.pairwise_comparison, None)

        # Change data3 to store different distributions
        data3[parametric_column_key] = np.random.normal(
            size=200, loc=5
        )
        data3[non_parametric_column_key] = (
            data3[parametric_column_key] +
            np.random.uniform(low=-0.001, high=0.001, size=200)
        )

        # Try a comparison of three different normal and homoscedastic results
        comparison_result = analyzer.compare(
            dataframe_key, parametric_column_key
        )
        self.assertNotEqual(comparison_result.normality, None)
        self.assertTrue(comparison_result.normality.success.all())
        self.assertNotEqual(comparison_result.homoscedasticity, None)
        self.assertTrue(comparison_result.homoscedasticity.success.all())
        self.assertFalse(comparison_result.global_comparison.success.all())
        self.assertNotEqual(comparison_result.pairwise_comparison, None)

        # Try a comparison of three different heteroscedastic results
        comparison_result = analyzer.compare(
            dataframe_key, non_parametric_column_key
        )
        self.assertNotEqual(comparison_result.normality, None)
        self.assertFalse(comparison_result.normality.success.all())
        self.assertEqual(comparison_result.homoscedasticity, None)
        self.assertFalse(comparison_result.global_comparison.success.all())
        self.assertNotEqual(comparison_result.pairwise_comparison, None)

    def test_rank(self):
        """Test the rank method."""
        column_key = "Data"
        dataframe_key = "Rank"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[column_key] = np.random.normal(size=200)
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[column_key] = np.random.normal(size=200, loc=5)
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the third results manager
        data3 = DataFrame()
        data3[column_key] = np.random.normal(size=200)
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Prepare the DataFrame with some data for the fourth results manager
        data4 = DataFrame()
        data4[column_key] = np.random.normal(size=200, loc=20)
        results4 = Results()
        results4[dataframe_key] = data4
        analyzer["Results4"] = results4

        weight = -1
        ranked_results = analyzer.rank(dataframe_key, column_key, weight)

        # Check ranks
        self.assertEqual(ranked_results["Results1"], 0.5)
        self.assertEqual(ranked_results["Results2"], 2.0)
        self.assertEqual(ranked_results["Results3"], 0.5)
        self.assertEqual(ranked_results["Results4"], 3.0)

    def test_multiple_rank(self):
        """Test the multiple_rank method."""
        # Test fitness related results
        test_fitness_key = "test_fitness"
        kappa_key = "Kappa"
        nf_key = "NF"

        # Execution metrics related results
        execution_metrics_key = "execution_metrics"
        runtime_key = "Runtime"

        # Fix the random seed
        random.seed(0)
        np.random.seed(0)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        test_fitness_1 = DataFrame()
        test_fitness_1[kappa_key] = np.random.normal(size=200, loc=0.9)
        test_fitness_1[nf_key] = np.random.normal(size=200, loc=6)
        execution_metrics_1 = DataFrame()
        execution_metrics_1[runtime_key] = np.random.normal(size=200, loc=125)
        results1 = Results()
        results1[test_fitness_key] = test_fitness_1
        results1[execution_metrics_key] = execution_metrics_1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        test_fitness_2 = DataFrame()
        test_fitness_2[kappa_key] = np.random.normal(size=200, loc=0.8)
        test_fitness_2[nf_key] = np.random.normal(size=200, loc=6)
        execution_metrics_2 = DataFrame()
        execution_metrics_2[runtime_key] = np.random.normal(size=200, loc=80)
        results2 = Results()
        results2[test_fitness_key] = test_fitness_2
        results2[execution_metrics_key] = execution_metrics_2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the third results manager
        test_fitness_3 = DataFrame()
        test_fitness_3[kappa_key] = np.random.normal(size=200, loc=0.4)
        test_fitness_3[nf_key] = np.random.normal(size=200, loc=8)
        execution_metrics_3 = DataFrame()
        execution_metrics_3[runtime_key] = np.random.normal(size=200, loc=50)
        results3 = Results()
        results3[test_fitness_key] = test_fitness_3
        results3[execution_metrics_key] = execution_metrics_3
        analyzer["Results3"] = results3

        # Prepare the DataFrame with some data for the fourth results manager
        test_fitness_4 = DataFrame()
        test_fitness_4[kappa_key] = np.random.normal(size=200, loc=0.9)
        test_fitness_4[nf_key] = np.random.normal(size=200, loc=4)
        execution_metrics_4 = DataFrame()
        execution_metrics_4[runtime_key] = np.random.normal(size=200, loc=80)
        results4 = Results()
        results4[test_fitness_key] = test_fitness_4
        results4[execution_metrics_key] = execution_metrics_4
        analyzer["Results4"] = results4

        dataframe_keys = (
            test_fitness_key, test_fitness_key, execution_metrics_key
        )

        columns = (nf_key, kappa_key, runtime_key)
        weights = (-1, 1, -1)

        multiple_ranked_results = analyzer.multiple_rank(
            dataframe_keys, columns, weights
        )

        self.assertEqual(
            multiple_ranked_results[test_fitness_key][kappa_key]["Results3"],
            3
        )
        self.assertEqual(
            multiple_ranked_results[test_fitness_key][nf_key]["Results4"],
            0
        )
        self.assertEqual(
            multiple_ranked_results[
                execution_metrics_key][runtime_key]["Results1"],
            3
        )

    def test_effect_size(self):
        """Test the rank method."""
        column_key = "Data"
        dataframe_key = "EffectSize"

        # Fix the random seed
        random.seed(1)
        np.random.seed(1)

        # Create the results analyzer
        analyzer = ResultsAnalyzer()

        # Prepare the DataFrame with some data for the first results manager
        data1 = DataFrame()
        data1[column_key] = 10 * np.random.randn(10000) + 60
        results1 = Results()
        results1[dataframe_key] = data1
        analyzer["Results1"] = results1

        # Prepare the DataFrame with some data for the second results manager
        data2 = DataFrame()
        data2[column_key] = 10 * np.random.randn(10000) + 55
        results2 = Results()
        results2[dataframe_key] = data2
        analyzer["Results2"] = results2

        # Prepare the DataFrame with some data for the second results manager
        data3 = DataFrame()
        data3[column_key] = np.array([5, 7, 9, 11, 13])
        results3 = Results()
        results3[dataframe_key] = data3
        analyzer["Results3"] = results3

        # Prepare the DataFrame with some data for the second results manager
        data4 = DataFrame()
        data4[column_key] = np.array([6, 8, 10, 12, 14])
        results4 = Results()
        results4[dataframe_key] = data4
        analyzer["Results4"] = results4

        effect_size = analyzer.effect_size(dataframe_key, column_key)
        self.assertAlmostEqual(effect_size.value[0][1], 0.5, places=3)
        self.assertAlmostEqual(effect_size.value[2][3], 0.31622776601683794)


if __name__ == '__main__':
    unittest.main()
