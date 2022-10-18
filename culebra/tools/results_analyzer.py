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
"""Statistical analysis of batches results."""

from __future__ import annotations
from typing import Callable, NamedTuple, Dict
from collections import UserDict, namedtuple
from collections.abc import Sequence
from functools import partial
from warnings import catch_warnings, simplefilter
import numpy as np
from pandas import Series
from scipy.stats import (
    shapiro,
    normaltest,
    bartlett,
    levene,
    fligner,
    ttest_ind,
    f_oneway,
    mannwhitneyu,
    kruskal,
    tukey_hsd
)
from scikit_posthocs import posthoc_dunn
from tabulate import tabulate
from culebra.base import Base, check_instance, check_float, check_str
from culebra.tools import Results

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_ALPHA = 0.05
"""Default significance level for statistical tests."""

DEFAULT_NORMALITY_TEST = shapiro
"""Default normality test."""

DEFAULT_HOMOSCEDASTICITY_TEST = bartlett
"""Default homoscedasticity test."""

DEFAULT_P_ADJUST = 'fdr_tsbky'
"""Default method for adjusting the p-values with the Dunn's test."""

_DEFAULT_PRECISION = 5
"""Default precision to print the results."""

_TEST_NAMES = {
    shapiro:      "Shapiro-Wilk test",
    normaltest:   "D'Agostino-Pearson test",
    bartlett:     "Bartlett's test",
    levene:       "Levene's test",
    fligner:      "Fligner-Killeen test",
    ttest_ind:    "T-test",
    f_oneway:     "One-way ANOVA test",
    mannwhitneyu: "Mann-Whitney U test",
    kruskal:      "Kruskal-Wallis H-test",
    tukey_hsd:    "Tukey's HSD test",
    posthoc_dunn: "Dunn's test"
}
"""Names of the statistical tests."""

_P_ADJUST_NAMES = {
    'bonferroni': "Bonferroni correction",
    'sidak': "Šidák correction",
    'holm-sidak': "Holm-Šídák correction",
    'holm': "Holm correction",
    'simes-hochberg': "Simes-Hochberg correction",
    'hommel': "Homel correction",
    'fdr_bh': "Benjamini-Hochberg FDR correction",
    'fdr_by': "Benjamini-Yekutieli FDR correction",
    'fdr_tsbh': "two stage Benjamini-Hochberg FDR correction",
    'fdr_tsbky': "two-stage Benjamini-Krieger-Yekutieli FDR correction"
}
"""Names of p-values adjustment methods."""


class TestOutcome(NamedTuple):
    """Outcome from a statistical test."""

    test: str
    """Name of the test."""

    data: str
    """Key for the dataframe containing the data"""

    column: str
    """Column key in the dataframe"""

    alpha: float
    """Significance level."""

    batches: list
    """Labels of all the analyzed batches."""

    pvalue: np.ndarray
    """p-value(s) returned by the test."""

    def __str__(self) -> str:
        """Pretty print of the success and p-values returned by a test."""
        sorted_batches = list(
            sorted(
                dict(
                    enumerate(self.batches)
                ).items(),
                key=lambda item: item[1]
            )
        )

        data = []
        if len(self.pvalue.shape) == 1:
            if self.pvalue.size == 1:
                headers = ('Success', 'p-value')
                index = None
                data = [
                    [
                        self.success,
                        np.round(self.pvalue, _DEFAULT_PRECISION)
                    ]
                ]
            else:
                headers = ('Batch', 'Success', 'p-value')
                index = [
                    batch for (_, batch) in sorted_batches
                ]
                data = [
                    [
                        self.success[i],
                        np.round(self.pvalue[i], _DEFAULT_PRECISION)
                    ]
                    for (i, _) in sorted_batches
                ]
        else:
            headers = ('Batch1', 'Batch2', 'Success', 'p-value')
            index = None
            num_batches = len(self.batches)
            for i in range(num_batches):
                index_i, batch_i = sorted_batches[i]
                for j in range(i+1, num_batches):
                    index_j, batch_j = sorted_batches[j]
                    data += [
                        [
                            batch_i,
                            batch_j,
                            self.success[index_i][index_j],
                            round(
                                self.pvalue[index_i][index_j],
                                _DEFAULT_PRECISION
                            )
                        ]
                    ]

        return tabulate(data, headers=headers, showindex=index)

    def __repr__(self) -> str:
        """Print all the input parameters and outputs returned by a test."""
        inputs = [
            [field.capitalize(), self.__getattribute__(field)]
            for field in ('test', 'data', 'column', 'alpha')
        ]

        return tabulate(inputs, tablefmt='plain') + "\n\n" + self.__str__()

    @property
    def success(self) -> np.ndarray:
        """Return a boolean array showing where the null hypothesis is met."""
        return self.pvalue > self.alpha


class ResultsComparison(NamedTuple):
    """Outcome from the comparison of several results."""

    normality: TestOutcome
    """Outcome of the normality test."""

    homoscedasticity: TestOutcome | None
    """Outcome of the homoscedasticity test. :py:data:`None` if the compared
    results did not follow a normal distribution."""

    global_comparison: TestOutcome
    """Outcome of the global comparison of all the results."""

    pairwise_comparison: TestOutcome | None
    """Outcome of the pairwise comparison of the results. Only performed if
    all the analyzed results did not follow the same distribution."""

    def __str__(self) -> str:
        """Pretty print of the comparison result."""

        def title(text: str, underline: str = '-') -> str:
            """Make a title for the output report.

            :param text: Text to be displayed
            :type text: :py:class:`str`
            :param underline: Symbol used to underline the title
            :type underline: :py:class:`str`
            :return: The formatted title
            :rtype: :py:class:`str`
            """
            # Check the underline char
            underline_name = "underline character"
            underline = check_str(underline, underline_name)
            if len(underline) != 1:
                raise ValueError(
                    f"The {underline_name} must be a single character: "
                    f"{underline}"
                )

            # Return the formatted title
            return text + "\n" + (underline * len(text)) + "\n\n"

        # Title for the results analysis
        output = title("Results comparison", '=')

        # Get batches as a string
        batches = ""
        for batch_name in sorted(self.normality.batches):
            if len(batches) == 0:
                batches += batch_name
            else:
                batches += ", " + batch_name

        # Add the input parameters to the output
        inputs = [
            ["Batches:", batches]
        ]

        inputs += [
            [
                field.capitalize() + ":",
                self.normality.__getattribute__(field)
            ]
            for field in ('data', 'column', 'alpha')
        ]

        output += tabulate(inputs, tablefmt='plain') + "\n\n\n"

        # Output the result of the normality test
        output += title("Normality: " + self.normality.test, '-')
        normality_success = self.normality.success.all()

        output += "Results "
        if not normality_success:
            output += "do not "
        output += "follow a normal distribution "
        output += f"(alpha = {self.normality.alpha})\n\n"
        output += self.normality.__str__() + "\n\n\n"

        # If the results are normal, check homoscedasticity
        homoscedasticity_success = False
        if normality_success:
            output += title(
                "Homoscedasticity: " + self.homoscedasticity.test, '-'
            )
            homoscedasticity_success = self.homoscedasticity.success.all()

            output += "Results are "
            if not homoscedasticity_success:
                output += "not "
            output += "homoscedastic "
            output += f"(alpha = {self.homoscedasticity.alpha})\n\n"
            output += self.homoscedasticity.__str__() + "\n\n\n"

        # Output the global comparison results
        output += title(
            "Global comparison: " + self.global_comparison.test, '-'
        )
        global_comparison_success = self.global_comparison.success.all()

        output += "Results "
        if not global_comparison_success:
            output += "do not "
        output += "follow the same distribution "
        output += f"(alpha = {self.global_comparison.alpha})\n\n"
        output += self.global_comparison.__str__() + "\n\n\n"

        # Output from a pairwise comparison
        if self.pairwise_comparison is not None:
            output += title(
                "Pairwise comparison: " + self.pairwise_comparison.test, '-'
            )

            output += "Results following the same distribution "
            output += f"(alpha = {self.pairwise_comparison.alpha})\n\n"
            output += self.pairwise_comparison.__str__()

        return output


class ResultsAnalyzer(UserDict, Base):
    """Perform statistical analyses over the results of several batches."""

    def __init__(self) -> None:
        """Create an empty results analyzer."""
        return super().__init__()

    def __setitem__(self, batch_key: str, batch_results: Results) -> Results:
        """Overriden to verify the *batch_key* and *batch_results*.

        Assure that *batch_key* is a :py:class:`str` and *batch_results* is a
        :py:class:`~tools.Results`.

        :param batch_key: Key to identify the *batch_results* within the
            analyzer
        :type batch_key: :py:class:`str`
        :param batch_results: The results of a batch of experiments
        :type batch_results: :py:class:`~tools.Results`
        :return: The inserted results
        :rtype: :py:class:`~tools.Results`
        """
        return super().__setitem__(
            check_instance(batch_key, "key for the batch results", str),
            check_instance(batch_results, "batch results", Results)
        )

    def _gather_data(
        self,
        dataframe_key: str,
        column: str,
    ) -> Dict['str', Series]:
        """Gather data from the results in the analyzer.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :return: The gathered results. One entry for each
            :py:class:`~tools.Results` in this analyzer containing the
            provided *dataframe_key* and *column*.
        :rtype: :py:class:`dict`
        """
        data = {}

        for batch_key in self.keys():
            if (
                    dataframe_key in self[batch_key].keys() and
                    column in self[batch_key][dataframe_key].keys()
            ):
                data[batch_key] = self[batch_key][dataframe_key][column]

        return data

    def normality_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        test: Callable[
            [Sequence],
            namedtuple
        ] = DEFAULT_NORMALITY_TEST
    ) -> TestOutcome:
        """Assess the normality of the same results for several batches.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :param test: Normality test to be applied, defaults to
            :py:data:`DEFAULT_NORMALITY_TEST`
        :type test: :py:class:`~collections.abc.Callable`, optional
        :return: The results of the normality test
        :rtype: :py:class:`TestOutcome`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *test* is not a valid normality test
        :raises ValueError: If there aren't any data in the analyzed results
            with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Check the test
        valid_tests = (shapiro, normaltest)
        if test not in valid_tests:
            raise ValueError(f"Not a valid normality test: {test}")

        # Gather the data
        data = self._gather_data(dataframe_key, column)

        # Check that the number of distributions is ok
        if len(data.keys()) == 0:
            raise ValueError(
                "No data in results with such dataframe key and column label"
            )

        # Apply the test to all the distributions
        pvalues = []
        for batch_key in data.keys():
            with catch_warnings():
                simplefilter("ignore")
                results = test(data[batch_key])
            pvalues += (results.pvalue,)

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[test],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=np.asarray(pvalues)
        )

    def homoscedasticity_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        test: Callable[
            [Sequence],
            namedtuple
        ] = DEFAULT_HOMOSCEDASTICITY_TEST
    ) -> TestOutcome:
        """Assess the homoscedasticity of the same results for several batches.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :param test: Homoscedasticity test to be applied, defaults to
            :py:data:`DEFAULT_HOMOSCEDASTICITY_TEST`
        :type test: :py:class:`~collections.abc.Callable`, optional
        :return: The results of the homoscedasticity test
        :rtype: :py:class:`TestOutcome`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *test* is not a valid homoscedasticity test
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Check the test
        valid_tests = (bartlett, levene, fligner)
        if test not in valid_tests:
            raise ValueError(f"Not a valid homoscedasticity test: {test}")

        # Gather the data
        data = self._gather_data(dataframe_key, column)

        # Check that the number of distributions is ok
        if len(data.keys()) < 2:
            raise ValueError(
                "Less than two results with such dataframe key and column "
                "label"
            )

        # Apply the test
        with catch_warnings():
            simplefilter("ignore")
            results = test(*data.values())

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[test],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=np.asarray([results.pvalue])
        )

    def parametric_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA
    ) -> TestOutcome:
        """Compare the results of several batches.

        Data should be independent, follow a normal distribution and also be
        homoscedastic. If only two results are analyzed, the T-test
        (:py:func:`~scipy.stats.ttest_ind`) is applied. For more
        results, the one-way ANOVA test (:py:func:`~scipy.stats.f_oneway`) is
        used instead.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :return: The results of the comparison
        :rtype: :py:class:`TestOutcome`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Gather the data
        data = self._gather_data(dataframe_key, column)

        # Check that the number of distributions is ok
        test = None
        if len(data.keys()) < 2:
            raise ValueError(
                "Less than two results with such dataframe key and column "
                "label"
            )
        elif len(data.keys()) == 2:
            # Apply the T-test
            test = ttest_ind
        else:
            # Apply the one-way ANOVA test
            test = f_oneway

        # Apply the test
        with catch_warnings():
            simplefilter("ignore")
            results = test(*data.values())

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[test],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=np.asarray([results.pvalue])
        )

    def non_parametric_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA
    ) -> TestOutcome:
        """Compare the results of several batches.

        Data should be independent. If only two results are analyzed,
        the Mann-Whitney U-test (:py:func:`~scipy.stats.mannwhitneyu`) is
        applied. For more results, the Kruskal-Wallis H-test
        (:py:func:`~scipy.stats.kruskal`) is used instead.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :return: The results of the comparison
        :rtype: :py:class:`TestOutcome`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Gather the data
        data = self._gather_data(dataframe_key, column)

        # Check that the number of distributions is ok
        test = None
        if len(data.keys()) < 2:
            raise ValueError(
                "Less than two results with such dataframe key and column "
                "label"
            )
        elif len(data.keys()) == 2:
            # Apply the Mann-Whitney U-test
            test = mannwhitneyu
        else:
            # Apply the Kruskal-Wallis H-test
            test = kruskal

        # Apply the test
        with catch_warnings():
            simplefilter("ignore")
            results = test(*data.values())

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[test],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=np.asarray([results.pvalue])
        )

    def parametric_pairwise_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA
    ) -> TestOutcome:
        """Pairwise comparison the results of several batches.

        Data should be independent, follow a normal distribution and also be
        homoscedastic.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :return: The results of the comparison
        :rtype: :py:class:`TestOutcome`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Gather the data
        data = self._gather_data(dataframe_key, column)

        # Check that the number of distributions is ok
        if len(data.keys()) < 2:
            raise ValueError(
                "Less than two results with such dataframe key and column "
                "label"
            )

        # Apply the test
        with catch_warnings():
            simplefilter("ignore")
            results = tukey_hsd(*data.values())

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[tukey_hsd],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=results.pvalue
        )

    def non_parametric_pairwise_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        p_adjust: str = DEFAULT_P_ADJUST,
    ) -> TestOutcome:
        """Pairwise comparison the results of several batches.

        Data should be independent.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :param p_adjust: Method for adjusting the p-values, defaults to
            :py:data:`DEFAULT_P_ADJUST`
        :type p_adjust: :py:class:`str` or :py:data:`None`, optional
        :return: The results of the comparison
        :rtype: :py:class:`TestOutcome`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *p_adjust* is not :py:data:`None` or any valid
            p-value adjustment method.
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Check p-adjust
        if p_adjust is not None and p_adjust not in _P_ADJUST_NAMES.keys():
            raise ValueError(
                f"Not valid p-value adjustment method: {p_adjust}"
            )

        # Gather the data
        data = self._gather_data(dataframe_key, column)

        # Check that the number of distributions is ok
        if len(data.keys()) < 2:
            raise ValueError(
                "Less than two results with such dataframe key and column "
                "label"
            )

        # Apply the test
        with catch_warnings():
            simplefilter("ignore")
            results = posthoc_dunn([*data.values()], p_adjust=p_adjust)

        # Return the results
        test_name = (
            _TEST_NAMES[posthoc_dunn] +
            " (" +
            (
                "No p-values correction"
                if p_adjust is None
                else _P_ADJUST_NAMES[p_adjust]
            ) +
            ")"
        )

        return TestOutcome(
            test=test_name,
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=results.to_numpy()
        )

    def compare(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        normality_test: Callable[
            [Sequence],
            namedtuple
        ] = DEFAULT_NORMALITY_TEST,
        homoscedasticity_test: Callable[
                    [Sequence],
                    namedtuple
                ] = DEFAULT_HOMOSCEDASTICITY_TEST,
        p_adjust: str = DEFAULT_P_ADJUST,
    ) -> TestOutcome:
        """Pairwise comparison the results of several batches.

        Data should be independent.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: :py:class:`str`
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: :py:class:`str`
        :param alpha: Significance level, defaults to :py:data:`DEFAULT_ALPHA`
        :type alpha: :py:class:`float`, optional
        :param normality_test: Normality test to be applied, defaults to
            :py:data:`DEFAULT_NORMALITY_TEST`
        :type normality_test: :py:class:`~collections.abc.Callable`, optional
        :param homoscedasticity_test: Homoscedasticity test to be applied,
            defaults to :py:data:`DEFAULT_HOMOSCEDASTICITY_TEST`
        :type homoscedasticity_test: :py:class:`~collections.abc.Callable`,
            optional
        :param p_adjust: Method for adjusting the p-values, defaults to
            :py:data:`DEFAULT_P_ADJUST`
        :type p_adjust: :py:class:`str` or :py:data:`None`, optional
        :return: The results of the comparison
        :rtype: :py:class:`ResultsComparison`
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *normality_test* is not a valid normality test
        :raises ValueError: If *homoscedasticity_test* is not a valid
            homoscedasticity test
        :raises ValueError: If *p_adjust* is not :py:data:`None` or any valid
            p-value adjustment method.
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check normality
        normality_result = self.normality_test(
            dataframe_key, column, alpha=alpha, test=normality_test
        )
        normality_success = normality_result.success.all()

        # Check homoscedasticity
        homoscedasticity_result = None
        homoscedasticity_success = False
        if normality_success:
            homoscedasticity_result = self.homoscedasticity_test(
                dataframe_key, column, alpha=alpha, test=homoscedasticity_test
            )
            homoscedasticity_success = homoscedasticity_result.success.all()

        pairwise_comparison_result = None

        # If all the results are normal and homoscedastic,
        # try a parametric test
        if normality_success and homoscedasticity_success:
            global_compare_meth = self.parametric_test
            pairwise_compare_meth = self.parametric_pairwise_test
        # If not, try a non parametric test
        else:
            global_compare_meth = self.non_parametric_test
            pairwise_compare_meth = partial(
                self.non_parametric_pairwise_test, p_adjust=p_adjust
            )

        # Perform a global comparison
        global_comparison_result = global_compare_meth(
            dataframe_key, column, alpha=alpha
        )

        # If there are more than two results and they are not equal,
        # a pairwise comparison is necessary
        if (
                len(self.keys()) > 2 and not
                global_comparison_result.success.all()
        ):
            pairwise_comparison_result = pairwise_compare_meth(
                dataframe_key, column, alpha=alpha
            )

        # Return the results
        return ResultsComparison(
            normality=normality_result,
            homoscedasticity=homoscedasticity_result,
            global_comparison=global_comparison_result,
            pairwise_comparison=pairwise_comparison_result
        )


# Exported symbols for this module
__all__ = [
    'TestOutcome',
    'ResultsComparison',
    'ResultsAnalyzer',
    'DEFAULT_ALPHA',
    'DEFAULT_NORMALITY_TEST',
    'DEFAULT_HOMOSCEDASTICITY_TEST',
    'DEFAULT_P_ADJUST'
]
