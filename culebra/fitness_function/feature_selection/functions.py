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

"""Feature selection related fitness functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple

from culebra.abc import Fitness
from culebra.fitness_function.feature_selection.abc import (
    FSScorer,
    FSClassificationScorer
)

from culebra.fitness_function.dataset_score import (
    KappaIndex as DatasetKappaIndex,
    Accuracy as DatasetAccuracy
)
from culebra.solution.feature_selection import Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class NumFeats(FSScorer):
    """Dummy single-objective fitness function for testing purposes.

    Return the number of selected features by a solution.
    """

    @property
    def obj_weights(self) -> Tuple[int, ...]:
        """Objective weights.

        Minimize the number of features that a solution has selected.

        :return: (-1, )
        :rtype: tuple[int]
        """
        return (-1, )

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :return: ("NF",)
        :rtype: tuple[str]
        """
        return ("NF",)

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.feature_selection.Solution
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: int
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        :raises ValueError: If *sol* is not evaluable
        """
        if not self.is_evaluable(sol):
            raise ValueError("The solution is not evaluable")

        sol.fitness.update_value(sol.num_feats, self.index)

        return sol.fitness


class FeatsProportion(FSScorer):
    """Dummy single-objective fitness function for testing purposes.

    Return the proportion of selected features chosen by a solution. That is,
    0 if no features are selected or 1 if all the features have been chosen.
    """

    @property
    def obj_weights(self) -> Tuple[int, ...]:
        """Objective weights.

        Minimize the proportion of features that a solution has selected.

        :return: (-1, )
        :rtype: tuple[int]
        """
        return (-1, )

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :return: ("FP",)
        :rtype: tuple[str]
        """
        return ("FP",)

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.feature_selection.Solution
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: int
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        :raises ValueError: If *sol* is not evaluable
        """
        if not self.is_evaluable(sol):
            raise ValueError("The solution is not evaluable")

        sol.fitness.update_value(
            float(sol.num_feats)/sol.species.num_feats,
            self.index
        )

        return sol.fitness


class KappaIndex(FSClassificationScorer, DatasetKappaIndex):
    """Single-objective fitness function for feature selection problems.

    Calculate the Kohen's Kappa index.
    """


class Accuracy(FSClassificationScorer, DatasetAccuracy):
    """Single-objective fitness function for feature selection problems.

    Calculate the accuracy.
    """


# Exported symbols for this module
__all__ = [
    'NumFeats',
    'FeatsProportion',
    'KappaIndex',
    'Accuracy'
]
