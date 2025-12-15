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

"""Statistical analysis of batches results."""

from __future__ import annotations

from typing import NamedTuple
from collections import UserDict, namedtuple
from collections.abc import Sequence, Callable
from functools import partial
from warnings import catch_warnings, simplefilter
from math import isclose

import numpy
from pandas import Series, DataFrame, MultiIndex
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
    tukey_hsd,
    ConstantInputWarning
)
from scikit_posthocs import posthoc_dunn
from tabulate import tabulate

from culebra.abc import Base
from culebra.checker import (
    check_int,
    check_instance,
    check_float,
    check_str
)
from culebra.tools import Results


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
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


def __title(text: str, underline: str = '-') -> str:
    """Make a title for the output report.

    :param text: Text to be displayed
    :type text: str
    :param underline: Symbol used to underline the title
    :type underline: str
    :return: The formatted title
    :rtype: str
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


class TestOutcome(NamedTuple):

    """Outcome from a statistical test"""

    test: str
    """Name of the test"""

    data: str
    """Key for the dataframe containing the data"""

    column: str
    """Column key in the dataframe"""

    alpha: float
    """Significance level"""

    batches: list
    """Labels of all the analyzed batches"""

    pvalue: numpy.ndarray
    """p-value(s) returned by the test"""

    @property
    def success(self) -> numpy.ndarray:
        """Check the success of the test.

        :return: A boolean array showing where the null hypothesis is met
        :rtype: ~numpy.ndarray[bool]
        """
        return self.pvalue > self.alpha

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
                headers = ('Batch', 'Success', 'p-value')
                index = None
                data = [
                    [
                        '\n'.join(self.batches),
                        self.success,
                        numpy.round(self.pvalue, _DEFAULT_PRECISION)
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
                        numpy.round(self.pvalue[i], _DEFAULT_PRECISION)
                    ]
                    for (i, _) in sorted_batches
                ]
        else:
            headers = ('Batch0', 'Batch1', 'Success', 'p-value')
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
        output = __title(self.test)
        # Add the input parameters to the output
        inputs = [
            ["Batches:", ", ".join(sorted(self.batches))]
        ]

        inputs += [
            [field.capitalize(), self.__getattribute__(field)]
            for field in ('data', 'column', 'alpha')
        ]

        return (
            output +
            tabulate(inputs, tablefmt='plain') + "\n\n" +
            self.__str__()
        )


class ResultsComparison(NamedTuple):
    """Outcome from the comparison of several results."""

    normality: TestOutcome
    """Outcome of the normality test"""

    homoscedasticity: TestOutcome | None
    """Outcome of the homoscedasticity test. :data:`None` if the compared
    results did not follow a normal distribution"""

    global_comparison: TestOutcome
    """Outcome of the global comparison of all the results"""

    pairwise_comparison: TestOutcome | None
    """Outcome of the pairwise comparison of the results. Only performed if
    more than two results are analyzed and not all the analyzed results follow
    the same distribution"""

    def __repr__(self) -> str:
        """Print all the input parameters and outputs returned by a test."""
        # Title for the results analysis
        output = __title("Results comparison", '=')

        # Get batches as a string
        batches = ", ".join(sorted(self.normality.batches))

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
        output += __title("Normality: " + self.normality.test, '-')
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
            output += __title(
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
        output += __title(
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
            output += __title(
                "Pairwise comparison: " +
                self.pairwise_comparison.test +
                f", alpha = {self.pairwise_comparison.alpha}",
                '-'
            )

            output += self.pairwise_comparison.__str__()

        return output

    def __str__(self) -> str:
        """Pretty print of the success and p-values returned by a test."""
        return self.pairwise_comparison.__str__()


class EffectSize(NamedTuple):
    """Outcome from a effect size estimation."""

    test: str
    """Name of the test."""

    data: str
    """Key for the dataframe containing the data"""

    column: str
    """Column key in the dataframe"""

    batches: list
    """Labels of all the analyzed batches."""

    value: numpy.ndarray
    """Effect size values."""

    def __str__(self) -> str:
        """Pretty print of the effect sizes."""
        sorted_batches = list(
            sorted(
                dict(
                    enumerate(self.batches)
                ).items(),
                key=lambda item: item[1]
            )
        )

        data = []
        headers = ('Batch0', 'Batch1', 'Effect size')
        num_batches = len(self.batches)
        for i in range(num_batches):
            index_i, batch_i = sorted_batches[i]
            for j in range(i+1, num_batches):
                index_j, batch_j = sorted_batches[j]
                data += [
                    [
                        batch_i,
                        batch_j,
                        round(
                            self.value[index_i][index_j],
                            _DEFAULT_PRECISION
                        )
                    ]
                ]

        return tabulate(data, headers=headers, showindex=None)

    def __repr__(self) -> str:
        """Print all the input parameters and outputs."""
        output = __title(self.test)
        # Add the input parameters to the output
        inputs = [
            ["Batches:", ", ".join(sorted(self.batches))]
        ]

        inputs += [
            [field.capitalize(), self.__getattribute__(field)]
            for field in ('data', 'column')
        ]

        return (
            output +
            tabulate(inputs, tablefmt='plain') + "\n\n" +
            self.__str__()
        )


class ResultsAnalyzer(UserDict, Base):
    """Perform statistical analyses over the results of several batches."""

    def normality_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        test: Callable[
            [Sequence],
            NamedTuple
        ] = DEFAULT_NORMALITY_TEST
    ) -> TestOutcome:
        """Assess the normality of the same results for several batches.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :param test: Normality test to be applied, defaults to
            :data:`~culebra.tools.DEFAULT_NORMALITY_TEST`
        :type test: ~collections.abc.Callable
        :return: The results of the normality test
        :rtype: ~culebra.tools.TestOutcome
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
        for batch_val in data.values():
            with catch_warnings():
                simplefilter("ignore")
                results = test(batch_val)
            pvalues += (results.pvalue,)

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[test],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=numpy.asarray(pvalues)
        )

    def homoscedasticity_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        test: Callable[
            [Sequence],
            NamedTuple
        ] = DEFAULT_HOMOSCEDASTICITY_TEST
    ) -> TestOutcome:
        """Assess the homoscedasticity of the same results for several batches.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :param test: Homoscedasticity test to be applied, defaults to
            :data:`~culebra.tools.DEFAULT_HOMOSCEDASTICITY_TEST`
        :type test: ~collections.abc.Callable
        :return: The results of the homoscedasticity test
        :rtype: ~culebra.tools.TestOutcome
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
            pvalue=numpy.asarray([results.pvalue])
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
        (:func:`~scipy.stats.ttest_ind`) is applied. For more
        results, the one-way ANOVA test (:func:`~scipy.stats.f_oneway`) is
        used instead.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :return: The results of the comparison
        :rtype: ~culebra.tools.TestOutcome
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
        if len(data.keys()) == 2:
            # Apply the T-test
            test = ttest_ind
        else:
            # Apply the one-way ANOVA test
            test = f_oneway

        # Apply the test
        with catch_warnings(record=True) as w:
            simplefilter("always")
            results = test(*data.values())
            if len(w) > 0 and isinstance(w[0].message, ConstantInputWarning):
                F_onewayResult = namedtuple(
                    'F_onewayResult', ('statistic', 'pvalue')
                )
                results = F_onewayResult(results.statistic, 1)

        # Return the results
        return TestOutcome(
            test=_TEST_NAMES[test],
            data=dataframe_key,
            column=column,
            alpha=alpha,
            batches=list(data.keys()),
            pvalue=numpy.asarray([results.pvalue])
        )

    def non_parametric_test(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA
    ) -> TestOutcome:
        """Compare the results of several batches.

        Data should be independent. If only two results are analyzed,
        the Mann-Whitney U-test (:func:`~scipy.stats.mannwhitneyu`) is
        applied. For more results, the Kruskal-Wallis H-test
        (:func:`~scipy.stats.kruskal`) is used instead.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :return: The results of the comparison
        :rtype: ~culebra.tools.TestOutcome
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
        if len(data.keys()) == 2:
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
            pvalue=numpy.asarray([results.pvalue])
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
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :return: The results of the comparison
        :rtype: ~culebra.tools.TestOutcome
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
        p_adjust: str = DEFAULT_P_ADJUST
    ) -> TestOutcome:
        """Pairwise comparison the results of several batches.

        Data should be independent.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :param p_adjust: Method for adjusting the p-values, defaults to
            :data:`~culebra.tools.DEFAULT_P_ADJUST`
        :type p_adjust: str
        :return: The results of the comparison
        :rtype: ~culebra.tools.TestOutcome
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *p_adjust* is not :data:`None` or any valid
            p-value adjustment method.
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        # Check alpha
        alpha = check_float(alpha, "significance level", ge=0, le=1)

        # Check p-adjust
        if p_adjust is not None and p_adjust not in _P_ADJUST_NAMES:
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

    def effect_size(
        self,
        dataframe_key: str,
        column: str
    ) -> TestOutcome:
        """Effect size calculation for several batches.

        The effect size is calculated with the Cohen's d index.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :return: The results of the effect sizes estimation
        :rtype: ~culebra.tools.EffectSize
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        def cohens_d(group1: Series, group2: Series) -> float:
            """Calculate the Cohen's d index fot two groups.

            :param group1: The first group
            :type group1: ~pandas.Series
            :param group2: The second group
            :type group2: ~pandas.Series
            :return: The effect size
            :rtype: float
            """
            # Calculating means of the two groups
            mean1, mean2 = group1.mean(), group2.mean()

            # Calculating pooled standard deviation
            std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
            n1, n2 = len(group1), len(group2)
            pooled_std = numpy.sqrt(
                ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
            )

            # Calculating Cohen's d
            if isclose(pooled_std, 0):
                d = float('inf')
            else:
                d = abs(mean1 - mean2) / pooled_std

            return d

        # Gather the data
        data = self._gather_data(dataframe_key, column)
        num_batches = len(data.keys())

        # Check that the number of distributions is ok
        if num_batches < 2:
            raise ValueError(
                "Less than two results with such dataframe key and column "
                "label"
            )

        effect_size = numpy.zeros((num_batches, num_batches))

        batch_names = tuple(data.keys())
        for i, batch_i in enumerate(batch_names):
            for j in range(i+1, num_batches):
                batch_j = batch_names[j]
                effect_size[i][j] = effect_size[j][i] = cohens_d(
                    data[batch_i], data[batch_j]
                )

        return EffectSize(
            test="Cohen's d index",
            data=dataframe_key,
            column=column,
            batches=list(data.keys()),
            value=effect_size
        )

    def compare(
        self,
        dataframe_key: str,
        column: str,
        alpha: float = DEFAULT_ALPHA,
        normality_test: Callable[
            [Sequence],
            NamedTuple
        ] = DEFAULT_NORMALITY_TEST,
        homoscedasticity_test: Callable[
                    [Sequence],
                    NamedTuple
                ] = DEFAULT_HOMOSCEDASTICITY_TEST,
        p_adjust: str = DEFAULT_P_ADJUST
    ) -> ResultsComparison:
        """Pairwise comparison the results of several batches.

        Data should be independent.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :param normality_test: Normality test to be applied, defaults to
            :data:`~culebra.tools.DEFAULT_NORMALITY_TEST`
        :type normality_test: ~collections.abc.Callable
        :param homoscedasticity_test: Homoscedasticity test to be applied,
            defaults to :data:`~culebra.tools.DEFAULT_HOMOSCEDASTICITY_TEST`
        :type homoscedasticity_test: ~collections.abc.Callable
        :param p_adjust: Method for adjusting the p-values, defaults to
            :data:`~culebra.tools.DEFAULT_P_ADJUST`
        :type p_adjust: str
        :return: The results of the comparison
        :rtype: ~culebra.tools.ResultsComparison
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *normality_test* is not a valid normality test
        :raises ValueError: If *homoscedasticity_test* is not a valid
            homoscedasticity test
        :raises ValueError: If *p_adjust* is not :data:`None` or any valid
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
        else:
            num_batches = len(self)
            pairwise_comparison_result = TestOutcome(
                test=global_comparison_result.test,
                data=global_comparison_result.data,
                column=global_comparison_result.column,
                alpha=global_comparison_result.alpha,
                batches=global_comparison_result.batches,
                pvalue=numpy.full(
                    (num_batches, num_batches),
                    global_comparison_result.pvalue[0]
                )
            )

        # Return the results
        return ResultsComparison(
            normality=normality_result,
            homoscedasticity=homoscedasticity_result,
            global_comparison=global_comparison_result,
            pairwise_comparison=pairwise_comparison_result
        )

    def rank(
        self,
        dataframe_key: str,
        column: str,
        weight: int,
        alpha: float = DEFAULT_ALPHA,
        normality_test: Callable[
            [Sequence],
            NamedTuple
        ] = DEFAULT_NORMALITY_TEST,
        homoscedasticity_test: Callable[
                    [Sequence],
                    NamedTuple
                ] = DEFAULT_HOMOSCEDASTICITY_TEST,
        p_adjust: str = DEFAULT_P_ADJUST
    ) -> Series:
        """Rank the batches according to a concrete result.

        Batches are ranked according to the procedure proposed in
        [Gonzalez2021]_. The rank of each batch is calculated as the number of
        batches whose result is better (with a statistical significative
        difference) than that of it. Batches with a statistically similar
        result share the same rank.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :param weight: Used for the comparison of the batches results
            selected by the *dataframe_key* and *column* provided. A negative
            value implies minimization (lower values are better), while a
            positive weight implies maximization.
        :type weight: int
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :param normality_test: Normality test to be applied, defaults to
            :data:`~culebra.tools.DEFAULT_NORMALITY_TEST`
        :type normality_test: ~collections.abc.Callable
        :param homoscedasticity_test: Homoscedasticity test to be applied,
            defaults to :data:`~culebra.tools.DEFAULT_HOMOSCEDASTICITY_TEST`
        :type homoscedasticity_test: ~collections.abc.Callable
        :param p_adjust: Method for adjusting the p-values, defaults to
            :data:`~culebra.tools.DEFAULT_P_ADJUST`
        :type p_adjust: str
        :return: The ranked batches
        :rtype: ~pandas.Series
        :raises TypeError: If *weight* is not an integer number
        :raises ValueError: If *weight* is 0
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *normality_test* is not a valid normality test
        :raises ValueError: If *homoscedasticity_test* is not a valid
            homoscedasticity test
        :raises ValueError: If *p_adjust* is not :data:`None` or any valid
            p-value adjustment method.
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with such dataframe key and column label
        """
        def average_ranks(first: int, last: int) -> list[float]:
            """Return a list of averaged ranks.

            :param first: The first considered rank
            :type first: int
            :param last: The last considered rank
            :type last: int
            :return: The list of averaged ranks
            :rtype: list[float]
            """
            if first == last:
                ranks = [float(first)]
            else:
                ranks = [
                    float((first + last) / 2)
                ] * (last - first + 1)

            return ranks

        # Check weight
        weight = check_int(weight, "weight", ne=0)

        # Get the mean value for each batch
        means = {
            (
                index,
                name,
                data.mean()
            )
            for (index, (name, data))
            in enumerate(self._gather_data(dataframe_key, column).items())
        }

        # Sort the means according to weight
        # Better means first
        sorted_means = list(
            sorted(
                means,
                key=lambda item: item[2]*(-weight)
            )
        )

        # Compare the results
        comparison = self.compare(
            dataframe_key,
            column,
            alpha,
            normality_test,
            homoscedasticity_test,
            p_adjust
        )

        batch_names = [name for (_, name, _) in sorted_means]

        num_batches = len(means)
        # If all the batches results are equal
        if comparison.global_comparison.success.all():
            ranks = average_ranks(0, num_batches-1)
        # Only two different batches
        elif num_batches == 2:
            ranks = [0, 1]
        # More than two different batches
        else:
            # Start with an empty list of ranks
            ranks = []

            # First rank and index of the next interval of equal batches
            first_rank = 0
            first_index = sorted_means[first_rank][0]

            # Shortcut for cleaner code
            pairwise_equal = comparison.pairwise_comparison.success

            # Iterate over the batches sorted by their mean result
            for rank in range(0, num_batches):
                # Index of the current batch
                index = sorted_means[rank][0]

                # If this batch is not equal than the previous ones ...
                if not pairwise_equal[first_index][index]:
                    # Append the previous interval of equal batches
                    ranks += average_ranks(first_rank, rank-1)

                    # A new interval starts
                    first_rank = rank
                    first_index = sorted_means[first_rank][0]

            # Append the last interval
            ranks += average_ranks(first_rank, rank)

        # Return the ranks
        rank_series = Series(ranks, index=batch_names, name=column)
        rank_series.sort_index(inplace=True)

        return rank_series

    def multiple_rank(
        self,
        dataframe_keys: Sequence[str],
        columns: Sequence[str],
        weights: Sequence[int],
        alpha: float = DEFAULT_ALPHA,
        normality_test: Callable[
            [Sequence],
            NamedTuple
        ] = DEFAULT_NORMALITY_TEST,
        homoscedasticity_test: Callable[
                    [Sequence],
                    NamedTuple
                ] = DEFAULT_HOMOSCEDASTICITY_TEST,
        p_adjust: str = DEFAULT_P_ADJUST
    ) -> DataFrame:
        """Rank the batches according to multiple results.

        Batches are ranked according to the procedure proposed in
        [Gonzalez2021]_. The rank of each batch is calculated as the number of
        batches whose result is better (with a statistical significative
        difference) than that of it. Batches with a statistically similar
        result share the same rank.

        The *dataframe_keys*, *columns* and *weights* must have the same
        length, and will be used to obtain different ranks for the results.

        :param dataframe_keys: Sequence of dataframe keys to select the
            different results of the batches
        :type dataframe_keys: ~collections.abc.Sequence[str]
        :param columns: Sequence of column labels to select the different
            results of the batches
        :type columns: ~collections.abc.Sequence[str]
        :param weights: Sequence of weights to be applied to the different
            results of the batches. Negative values imply minimization (lower
            values are better), while a positive weights imply maximization.
        :type weights: ~collections.abc.Sequence[int]
        :param alpha: Significance level, defaults to
            :data:`~culebra.tools.DEFAULT_ALPHA`
        :type alpha: float
        :param normality_test: Normality test to be applied, defaults to
            :data:`~culebra.tools.DEFAULT_NORMALITY_TEST`
        :type normality_test: ~collections.abc.Callable
        :param homoscedasticity_test: Homoscedasticity test to be applied,
            defaults to :data:`~culebra.tools.DEFAULT_HOMOSCEDASTICITY_TEST`
        :type homoscedasticity_test: ~collections.abc.Callable
        :param p_adjust: Method for adjusting the p-values, defaults to
            :data:`~culebra.tools.DEFAULT_P_ADJUST`
        :type p_adjust: str
        :return: The ranked batches
        :rtype: ~pandas.DataFrame
        :raises TypeError: If any weight is not an integer number
        :raises ValueError: If any weight is 0
        :raises TypeError: If *alpha* is not a real number
        :raises ValueError: If *alpha* is not in [0, 1]
        :raises ValueError: If *normality_test* is not a valid normality test
        :raises ValueError: If *homoscedasticity_test* is not a valid
            homoscedasticity test
        :raises ValueError: If *p_adjust* is not :data:`None` or any valid
            p-value adjustment method.
        :raises ValueError: If there aren't sufficient data in the analyzed
            results with any given dataframe key and column label
        """
        multiple_ranking = DataFrame()
        index_tuples = []
        for (
            dataframe_key,
            column,
            weight
        ) in zip(dataframe_keys, columns, weights):
            ranked_results = self.rank(
                dataframe_key,
                column,
                weight,
                alpha,
                normality_test,
                homoscedasticity_test,
                p_adjust
            )
            index_tuples += [(dataframe_key, column)]
            multiple_ranking[column] = ranked_results

        multi_index = MultiIndex.from_tuples(
            index_tuples, names=(
                "DataFrame", "Batch"
            )
        )
        multiple_ranking.columns = multi_index
        multiple_ranking.sort_index(axis=1, inplace=True)
        return multiple_ranking

    def __setitem__(self, batch_key: str, batch_results: Results) -> Results:
        """Overridden to verify the *batch_key* and *batch_results*.

        Assure that *batch_key* is a :class:`str` and *batch_results* is a
        :class:`~culebra.tools.Results`.

        :param batch_key: Key to identify the *batch_results* within the
            analyzer
        :type batch_key: str
        :param batch_results: The results of a batch of experiments
        :type batch_results: ~culebra.tools.Results
        :return: The inserted results
        :rtype: ~culebra.tools.Results
        """
        return super().__setitem__(
            check_instance(batch_key, "key for the batch results", str),
            check_instance(batch_results, "batch results", Results)
        )

    def _gather_data(
        self,
        dataframe_key: str,
        column: str,
    ) -> dict[str, Series | None]:
        """Gather data from the results in the analyzer.

        :param dataframe_key: Key to select a dataframe from the results
            of all the batches
        :type dataframe_key: str
        :param column: Column label to be analyzed in the selected dataframes
            from the results of all the batches
        :type column: str
        :return: The gathered results. One entry for each
            ~culebra.tools.Results in this analyzer containing the
            provided *dataframe_key* and *column*.
        :rtype: dict
        """
        data = {}

        for batch_key in self:
            if (
                    dataframe_key in self[batch_key].keys() and
                    column in self[batch_key][dataframe_key].keys()
            ):
                data[batch_key] = self[batch_key][dataframe_key][column]

        return data


# Exported symbols for this module
__all__ = [
    'TestOutcome',
    'ResultsComparison',
    'EffectSize',
    'ResultsAnalyzer',
    'DEFAULT_ALPHA',
    'DEFAULT_NORMALITY_TEST',
    'DEFAULT_HOMOSCEDASTICITY_TEST',
    'DEFAULT_P_ADJUST'
]
