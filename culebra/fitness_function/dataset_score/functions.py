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

"""Fitness functions related to dataset scoring."""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score, accuracy_score

from culebra.fitness_function.dataset_score.abc import DatasetScorer


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class KappaIndex(DatasetScorer):
    """Single-objective fitness function for classification problems.

    Calculate the Kohen's Kappa index.
    """

    _score = cohen_kappa_score
    """Score function to be used in the evaluation."""

    @property
    def obj_weights(self):
        """Maximize the validation Kappa index."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("Kappa",)

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

    _score = accuracy_score
    """Score function to be used in the evaluation."""

    @property
    def obj_weights(self):
        """Maximize the validation accuracy."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("Accuracy",)

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
