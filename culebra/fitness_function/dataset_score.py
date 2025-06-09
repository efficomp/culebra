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

"""Feature selection related fitness functions.

This sub-module provides several abstract fitness functions to score
dataset-related problems:

  * :py:class:`~culebra.fitness_function.feature_selection.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for
    classification problems.

  * :py:class:`~culebra.fitness_function.feature_selection.Accuracy`:
    Single-objective function that maximizes the Accuracy for classification
    problems.
"""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score, accuracy_score

from culebra.abc import Fitness
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.abc import DatasetScorer


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class KappaIndex(DatasetScorer):
    """Single-objective fitness function for classification problems.

    Calculate the Kohen's Kappa index.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.KappaIndex.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = (1.0,)
        """Maximizes the validation Kappa index."""

        names = ("Kappa",)
        """Name of the objective."""

        thresholds = [DEFAULT_THRESHOLD]
        """Similarity threshold for fitness comparisons."""

    _score = cohen_kappa_score
    """Score function to be used in the evaluation."""

    @property
    def _worst_score(self) -> float:
        """Worst achievable score.

        :type: :py:class:`float`
        """
        return -1


KappaIndex._score.__doc__ = """
Use :py:func:`~sklearn.metrics.cohen_kappa_score` to score.
"""


class Accuracy(DatasetScorer):
    """Single-objective fitness function for classification problems.

    Calculate the accuracy.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.feature_selection.Accuracy.evaluate`
        method within a
        :py:class:`~culebra.solution.feature_selection.Solution`.
        """

        weights = (1.0,)
        """Maximizes the validation accuracy."""

        names = ("Accuracy",)
        """Name of the objective."""

        thresholds = [DEFAULT_THRESHOLD]
        """Similarity threshold for fitness comparisons."""

    _score = accuracy_score
    """Score function to be used in the evaluation."""

    @property
    def _worst_score(self) -> float:
        """Worst achievable score.

        :type: :py:class:`float`
        """
        return 0


Accuracy._score.__doc__ = """
Use :py:func:`~sklearn.metrics.accuracy_score` to score.
"""


# Exported symbols for this module
__all__ = [
    'KappaIndex',
    'Accuracy'
]
