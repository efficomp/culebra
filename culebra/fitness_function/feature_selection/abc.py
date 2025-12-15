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

This sub-module provides several abstract fitness functions to solve feature
selecion problems:

* :class:`~culebra.fitness_function.feature_selection.abc.FSClassificationScorer`:
  Designed to support feature selection on classification problems
* :class:`~culebra.fitness_function.feature_selection.abc.FSDatasetScorer`:
  Allows the definition of FS dataset-related fitness functions.
* :class:`~culebra.fitness_function.feature_selection.abc.FSScorer`:
  Abstract single-objective fitness function for FS problems.
"""

from __future__ import annotations

from collections.abc import Sequence

from culebra.abc import Fitness
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction
from culebra.fitness_function.dataset_score.abc import (
    DatasetScorer,
    ClassificationScorer
)

from culebra.solution.feature_selection import Solution
from culebra.tools import Dataset


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class FSScorer(SingleObjectiveFitnessFunction):
    """Abstract base fitness function for FS problems."""

    def is_evaluable(self, sol: Solution) -> bool:
        """Assess the evaluability of a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.feature_selection.Solution
        :return: :data:`True` if the solution can be evaluated
        :rtype: bool
        :raises NotImplementedError: If has not been overridden
        """
        if isinstance(sol, Solution):
            return True

        return False


class FSDatasetScorer(DatasetScorer, FSScorer):
    """Abstract base fitness function for FS problems."""

    def evaluate(
        self,
        sol: Solution,
        index: int | None = None,
        representatives: Sequence[Solution] | None = None
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

        if sol.features.size > 0:
            super().evaluate(sol, index, representatives)
        else:
            sol.fitness.update_value(self._worst_score, self.index)

        return sol.fitness

    def _final_training_test_data(
        self,
        sol: Solution
    ) -> tuple[Dataset, Dataset]:
        """Get the final training and test data.

        :param sol: Solution to be evaluated. It is used to select the
          features from the datasets
        :type sol: ~culebra.abc.Solution

        :return: The final training and test datasets
        :rtype: tuple[~culebra.tools.Dataset]
        """
        return (
            self.training_data.select_features(sol.features),
            None if self.test_data is None else
            self.test_data.select_features(sol.features)
        )


class FSClassificationScorer(FSDatasetScorer, ClassificationScorer):
    """Abstract base fitness function for FS problems."""


# Exported symbols for this module
__all__ = [
    'FSScorer',
    'FSDatasetScorer',
    'FSClassificationScorer'
]
