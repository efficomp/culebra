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
from copy import deepcopy
from typing import Optional

from numpy import ndarray

from culebra.abc import Fitness, Species
from culebra.fitness_function.feature_selection.abc import (
    FSScorer,
    FSClassificationScorer
)
from culebra.fitness_function.abc import ACOFitnessFunction
from culebra.fitness_function import MultiObjectiveFitnessFunction

from culebra.fitness_function.feature_selection.abc import (
    FSDatasetScorer
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
    def obj_weights(self):
        """Minimize the number of features that a solution has selected."""
        return (-1, )

    @property
    def obj_names(self):
        """Name of the objective."""
        return ("NF",)

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

        sol.fitness.update_value(sol.num_feats, self.index)

        return sol.fitness


class FeatsProportion(FSScorer):
    """Dummy single-objective fitness function for testing purposes.

    Return the proportion of selected features chosen by a solution. That is,
    0 if no features are selected or 1 if all the features have been chosen.
    """

    @property
    def obj_weights(self):
        """Minimize the proportion of features that a solution has selected."""
        return (-1, )

    @property
    def obj_names(self):
        """Name of the objective."""
        return ("FP",)

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


class FSMultiObjectiveDatasetScorer(
    MultiObjectiveFitnessFunction,
    ACOFitnessFunction
):
    """Base class for multi-objective classification FS problems."""

    def __init__(
        self,
        fs_classification_scorer,
        *remaining_objectives
    ) -> None:
        """Construct a multi-objective fitness function.

        :param fs_classification_scorer: A classification FS objective
            responsible of the
            :py:attr:`~culebra.fitness_function.feature_selection.FSMultiObjectiveDatasetScorer.num_nodes`
            property and the
            :py:meth:`~culebra.fitness_function.feature_selection.FSMultiObjectiveDatasetScorer.heuristic`
            implementations
        :type fs_classification_scorer:
            :py:class:`~culebra.fitness_function.feature_selection.abc.FSDatasetScorer`

        :param remaining_objectives: Remaining objectives for this fitness
            function
        :type remaining_objectives:
            :py:class:`~culebra.fitness_function.abc.SingleObjectiveFitnessFunction`
        """
        # Check the first objective
        if not isinstance(fs_classification_scorer, FSDatasetScorer):
            raise ValueError(
                "The first objective is not an instance of "
                "FSDatasetScorer"
            )
        super().__init__(fs_classification_scorer, *remaining_objectives)

    @property
    def num_nodes(self) -> int:
        """Return the problem graph's number of nodes for ACO-based trainers.

        :return: The problem graph's number of nodes
        :rtype: :py:class:`int`
        """
        return self.objectives[0].num_nodes

    def heuristic(self, species: Species) -> Sequence[ndarray, ...]:
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
        return self.objectives[0].heuristic(species)

    def __copy__(self) -> FSMultiObjectiveDatasetScorer:
        """Shallow copy the fitness function."""
        cls = self.__class__
        result = cls(self.objectives[0])
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> FSMultiObjectiveDatasetScorer:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the fitness funtion
        :rtype:
            :py:class:`~culebra.fitness_function.abc.FSMultiObjectiveDatasetScorer`
        """
        cls = self.__class__
        result = cls(self.objectives[0])
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the fitness function.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.objectives[0],), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> FSMultiObjectiveDatasetScorer:
        """Return an FS multi-objective dataset scorer from a state.

        :param state: The state.
        :type state: :py:class:`~dict`
        """
        obj = cls(state['_objectives'][0])
        obj.__setstate__(state)
        return obj


# Exported symbols for this module
__all__ = [
    'NumFeats',
    'FeatsProportion',
    'KappaIndex',
    'Accuracy',
    'FSMultiObjectiveDatasetScorer'
]
