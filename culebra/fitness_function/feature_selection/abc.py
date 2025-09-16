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

  * :py:class:`~culebra.fitness_function.feature_selection.abc.FSScorer`:
    Abstract single-objective fitness function for FS problems.

  * :py:class:`~culebra.fitness_function.feature_selection.abc.FSDatasetScorer`:
    Allows the definition of FS dataset-related fitness functions.

  * :py:class:`~culebra.fitness_function.feature_selection.abc.FSClassificationScorer`:
    Designed to support feature selection on classification problems
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, Optional

import numpy as np

from culebra.checker import check_instance
from culebra.abc import Fitness
from culebra.fitness_function.abc import (
    ACOFitnessFunction,
    SingleObjectiveFitnessFunction
)
from culebra.fitness_function.dataset_score.abc import (
    DatasetScorer,
    ClassificationScorer
)

from culebra.solution.feature_selection import Species, Solution
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
        """Return :py:data:`True` if the solution can be evaluated.

        :param sol: Solution to be evaluated.
        :type sol:
            :py:class:`~culebra.solution.parameter_optimization.Solution`
        """
        if isinstance(sol, Solution):
            return True

        return False


class FSDatasetScorer(DatasetScorer, FSScorer, ACOFitnessFunction):
    """Abstract base fitness function for FS problems."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.feature_selection.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness for *sol*
        :rtype: :py:class:`~culebra.abc.Fitness`
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
    ) -> Tuple[Dataset, Dataset]:
        """Get the final training and test data.

        :param sol: Solution to be evaluated. It is used to select the
          features from the datasets
        :type sol: :py:class:`~culebra.abc.Solution`

        :return: The final training and test datasets
        :rtype: :py:class:`tuple` of :py:class:`~culebra.tools.Dataset`
        """
        return (
            self.training_data.select_features(sol.features),
            None if self.test_data is None else
            self.test_data.select_features(sol.features)
        )

    @property
    def num_nodes(self) -> int:
        """Return the problem graph's number of nodes for ACO-based trainers.

        :return: The problem graph's number of nodes
        :rtype: :py:class:`int`
        """
        return self.training_data.num_feats

    def heuristic(self, species: Species) -> Sequence[np.ndarray, ...]:
        """Get the heuristic matrices for ACO-based trainers.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.solution.feature_selection.Species`
        :raises TypeError: If *species* is not an instance of
            :py:class:`~culebra.solution.feature_selection.Species`

        :return: A tuple with only one heuristic matrix. Arcs between
            selectable features have a heuristic value of 1, while arcs
            involving any non-selectable feature or arcs from a feature to
            itself have a heuristic value of 0.
        :rtype: :py:class:`~collections.abc.Sequence` of
            :py:class:`~numpy.ndarray`
        """
        check_instance(species, "species", cls=Species)

        num_feats = species.num_feats

        # All the features should be considered
        heuristic = np.ones((num_feats, num_feats))

        # Ignore features with an index lower than min_feat
        min_feat = species.min_feat
        if min_feat > 0:
            for feat in range(num_feats):
                for ignored in range(min_feat):
                    heuristic[feat][ignored] = 0
                    heuristic[ignored][feat] = 0

        # Ignore features with an index greater than max_feat
        max_feat = species.max_feat
        if max_feat < num_feats - 1:
            for feat in range(num_feats):
                for ignored in range(max_feat + 1, num_feats):
                    heuristic[feat][ignored] = 0
                    heuristic[ignored][feat] = 0

        # The distance from a feature to itself is also ignored
        for index in range(min_feat, max_feat+1):
            heuristic[index][index] = 0

        return (heuristic, )


class FSClassificationScorer(FSDatasetScorer, ClassificationScorer):
    """Abstract base fitness function for FS problems."""


# Exported symbols for this module
__all__ = [
    'FSScorer',
    'FSDatasetScorer',
    'FSClassificationScorer'
]
