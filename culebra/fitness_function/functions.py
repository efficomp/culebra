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

"""Fitness functions.

This sub-module provides the class
:class:`~culebra.fitness_function.MultiObjectiveFitnessFunction`,
which allows the aggregation of several single-objective fitness functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, Optional, List
from functools import partial

from culebra import DEFAULT_SIMILARITY_THRESHOLD
from culebra.abc import Fitness, FitnessFunction, Solution
from culebra.checker import (
    check_float, check_instance, check_sequence
)
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class MultiObjectiveFitnessFunction(FitnessFunction):
    """Base class for multi-objective fitness functions."""

    def __init__(
        self,
        *objectives: Tuple[SingleObjectiveFitnessFunction, ...]
    ) -> None:
        """Construct a multi-objective fitness function.

        :param objectives: Objectives for this fitness function
        :type objectives:
            tuple[~culebra.fitness_function.abc.SingleObjectiveFitnessFunction]
        """
        # Init the objective list
        self._objectives = check_sequence(
            objectives,
            "objectives",
            item_checker=partial(
                check_instance,
                cls=SingleObjectiveFitnessFunction
            )
        )

        # Assign an index to each objective
        for idx, obj in enumerate(self.objectives):
            obj.index = idx

    @property
    def obj_weights(self) -> Tuple[int, ...]:
        """Objective weights.

        :rtype: tuple[int]
        """
        weights = ()

        for obj in self.objectives:
            weights += obj.obj_weights

        return weights

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :rtype: tuple[str]
        """
        names = ()

        for obj in self.objectives:
            names += obj.obj_names

        return names

    @property
    def obj_thresholds(self) -> List[float]:
        """Objective similarity thresholds.

        :rtype: list[float]
        :setter: Set new thresholds.
        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :type values: float | ~collections.abc.Sequence[float] | None
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        thresholds = []

        for obj in self.objectives:
            thresholds += obj.obj_thresholds

        return thresholds

    @obj_thresholds.setter
    def obj_thresholds(
        self, values: float | Sequence[float] | None
    ) -> None:
        """Set new objective similarity thresholds.

        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :type values: float | ~collections.abc.Sequence[float] | None
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if isinstance(values, Sequence):
            thresholds = check_sequence(
                values,
                "objective similarity thresholds",
                size=self.num_obj,
                item_checker=partial(check_float, ge=0)
            )
        elif values is not None:
            thresholds = [
                check_float(values, "objective similarity threshold", ge=0)
            ] * self.num_obj
        else:
            thresholds = (
                [DEFAULT_SIMILARITY_THRESHOLD] * self.num_obj
            )

        for obj, th in zip(self.objectives, thresholds):
            obj.obj_thresholds = th

    @property
    def objectives(self) -> List[SingleObjectiveFitnessFunction]:
        """List of objectives.

        :rtype:
            list[~culebra.fitness_function.abc.SingleObjectiveFitnessFunction]
        """
        return self._objectives

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        Parameters *representatives* and *index* are used only for cooperative
        evaluations

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem, optional
        :type index: int
        :param representatives: Representative solutions of each species
            being optimized, optional
        :type representatives:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        """
        for obj in self.objectives:
            obj.evaluate(sol, index, representatives)

        return sol.fitness


# Exported symbols for this module
__all__ = [
    'MultiObjectiveFitnessFunction'
]
