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

"""Decision managers.

This module provides several decision manager implementations:

* The :class:`~culebra.tools.decision_manager.LexicographicDM` class, which
  ranks the best solutions found by the trainer lexicographically and selects
  the first one.
* The :class:`~culebra.tools.decision_manager.LexicographicWithRepeatedCVDM`
  class, which applies repeated cross-validation and then selects the best
  solution lexicographically.

"""
from __future__ import annotations

from collections.abc import Sequence
from copy import copy, deepcopy
from functools import partial

import numpy as np

from culebra import DEFAULT_SIMILARITY_THRESHOLD
from culebra.abc import Fitness, Solution, Trainer
from culebra.fitness_func.dataset_score.abc import DatasetScorer
from culebra.checker import check_int, check_float, check_sequence
from .abc import DecisionManager


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_CV_REPEATS = 10
"""Default value for the number of cross-validation repeats."""


class LexicographicDM(DecisionManager):
    """Lexicographic decision manager."""

    def __init__(
        self,
        trainer: Trainer,
        obj_thresholds : float | None = None
    ) -> None:
        """
        Init the decesion manager.

        :param trainer: The trainer
        :type trainer: ~culebra.abc.Trainer
        :param obj_thresholds: Similarity thresholds for fitness values.
            If omitted,
            :attr:`~culebra.tools.decision_manager.LexicographicDM._default_similarity_threshold`
            is used. Defaults to :data:`None`
        :type obj_thresholds: float
        """
        # Init the superclass
        super().__init__(trainer)

        # Set the attributes to default values
        self.obj_thresholds = obj_thresholds

    @property
    def _default_similarity_threshold(self) -> float:
        """Default similarity threshold for fitnesses.

        :return: :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :rtype: float
        """
        return DEFAULT_SIMILARITY_THRESHOLD

    @property
    def obj_thresholds(self) -> tuple[float]:
        """Objective similarity thresholds.

        :rtype: tuple[float]
        :setter: Set new thresholds.
        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.tools.decision_manager.LexicographicDM._default_similarity_threshold`
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if self._obj_thresholds is None:
            return (
                self._default_similarity_threshold,
            ) * self.trainer.fitness_func.num_obj

        return self._obj_thresholds

    @obj_thresholds.setter
    def obj_thresholds(
        self, values: float | Sequence[float] | None
    ) -> None:
        """Set new objective similarity thresholds.

        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.tools.decision_manager.LexicographicDM._default_similarity_threshold`
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if values is None:
            self._obj_thresholds = None
        elif isinstance(values, Sequence):
            self._obj_thresholds = tuple(
                check_sequence(
                    values,
                    "objective similarity thresholds",
                    size=self.trainer.fitness_func.num_obj,
                    item_checker=partial(check_float, ge=0)
                )
            )
        else:
            self._obj_thresholds = (
                check_float(values, "objective similarity threshold", ge=0),
            ) * self.trainer.fitness_func.num_obj

    def _evaluate(
        self,
        all_combinations: list[list[Solution, ...]]
    ) -> list[Fitness, ...]:
        """Evaluate all the combinations of Pareto optimal solutions.

        The trainer's training fitness function is used to evaluate each
        combination.

        :param all_combinations: All the combinations of the Pareto optimal
            solutions from each species
        :type all_combinations: list[list[~culebra.abc.Solution, ...]]
        :return: The fitness for each combination
        :rtype: list[Fitness, ...]
        """
        fitness_func = deepcopy(self.trainer.fitness_func)
        fitness_func.obj_thresholds = 0
        fitness_cls = fitness_func.fitness_cls

        # Evaluate each combination
        all_combinations_fitness = []
        for comb in all_combinations:
            all_combinations_fitness.append(
                fitness_cls(fitness_func.evaluate(comb[0], 0, comb))
            )

        return all_combinations_fitness

    def _choose(
        self,
        all_combinations: list[list[Solution, ...]],
        all_combinations_fitness: list[Fitness, ...]
    ) -> tuple[Solution, ...] | None:
        """Choose one combination of solutions.

        The combinations are ranked lexicographically and the first one
        is selected.

        :param all_combinations: All the combinations of the Pareto optimal
            solutions from each species
        :type all_combinations: list[list[~culebra.abc.Solution, ...]]
        :all_combinations_fitness: The fitness for each combination
        :type all_combinations_fitness: list[Fitness, ...]
        :return: The chosen combination
        :rtype: tuple[~culebra.abc.Solution, ...]
        """
        class CombSolution(list):
            """Solution to keep a combination."""

            def __init__(self, comb, fitness):
                """Add a fitness to the combination.

                :param comb: The combination
                :type comb: list[culebra.abc.Solution, ...]
                :param fitness: The fitness
                :type fitness: culebra.abc.Fitness
                """
                super().__init__(comb)
                self.fitness = fitness
                self.working_fitness = copy(fitness)

        # If there aren't any combination...
        if len(all_combinations) == 0:
            return None

        # Selected combinations
        selected_combs = [
            CombSolution(comb, comb_fitness)
            for comb, comb_fitness in zip(
                all_combinations, all_combinations_fitness
            )
        ]

        # Keep in the Pareto Front only those solutions that differ from the
        # best fitness below the similarity threshold
        for obj_idx, obj_th in enumerate(self.obj_thresholds):
            # Sort lexicographically
            sorted_combs = sorted(
                selected_combs, key=lambda x: x.working_fitness, reverse=True
            )
            best_comb = sorted_combs[0]
            best_obj_fitness = best_comb.working_fitness.values[obj_idx]
            selected_combs = [best_comb]
            for comb in sorted_combs[1:]:
                fitness_values = comb.working_fitness.values
                if abs(best_obj_fitness - fitness_values[obj_idx]) <= obj_th:
                    comb.working_fitness.values = (
                        fitness_values[:obj_idx] +
                        (best_obj_fitness,) +
                        fitness_values[obj_idx+1:]
                    )
                    selected_combs.append(comb)
                else:
                    break

        if len(selected_combs) == 1:
            return tuple(selected_combs[0])

        # Do not use the similarity threshold to break ties
        sorted_combs = sorted(
            selected_combs, key=lambda x: x.fitness, reverse=True
        )

        return tuple(sorted_combs[0])


class LexicographicWithRepeatedCVDM(LexicographicDM):
    """Lexicographic decision manager with repeated cross-validation."""

    def __init__(
        self,
        trainer: Trainer,
        obj_thresholds : float | None = None,
        cv_repeats: int | None = None
    ) -> None:
        """
        Init the decesion manager.

        :param trainer: The trainer
        :type trainer: ~culebra.abc.Trainer
        :param obj_thresholds: Similarity thresholds for fitness values.
            If omitted,
            :attr:`~culebra.tools.decision_manager.LexicographicDM._default_similarity_threshold`
            is used. Defaults to :data:`None`
        :type obj_thresholds: float
        :param cv_repeats: Number of cross-validation repeats.
            If omitted,
            :attr:`~culebra.tools.decision_manager.LexicographicWithRepeatedCVDM._default_cv_repeats`
            is used. Defaults to :data:`None`
        :type cv_repeats: int
        """
        # Init the superclass
        super().__init__(trainer, obj_thresholds)

        # Set the attributes to default values
        self.cv_repeats = cv_repeats

    @property
    def _default_cv_repeats(self) -> int:
        """Default number of repeats for cross-validation.

        :return:
            :attr:`~~culebra.tools.decision_manager.DEFAULT_CV_REPEATS`
        :rtype: int
        """
        return DEFAULT_CV_REPEATS

    @property
    def cv_repeats(self) -> int:
        """Number of cross-validation repeats.

        :rtype: int

        :setter: Set a new value for the number of cross-validation repeats
        :param value: A positive integer value. If set to :data:`None`,
            :attr:`~culebra.tools.decision_manager.LexicographicWithRepeatedCVDM._default_cv_repeats`
            is assumed
        :type value: int
        :raises TypeError: If *value* is not an integer value
        :raises ValueError: If *value* is not positive
        """
        return (
            self._default_cv_repeats
            if self._cv_repeats is None
            else self._cv_repeats
        )

    @cv_repeats.setter
    def cv_repeats(self, value: int | None) -> None:
        """Set a value for the number of cross-validation repeats.

        :param value: A positive integer value. If set to :data:`None`,
            :attr:`~culebra.tools.decision_manager.LexicographicWithRepeatedCVDM._default_cv_repeats`
            is assumed
        :type value: int
        :raises TypeError: If *value* is not an integer value
        :raises ValueError: If *value* is not positive
        """
        self._cv_repeats = (
            None if value is None else check_int(
                value, "number of cross-validation repeats", gt=0
            )
        )

    def _evaluate(
        self,
        all_combinations: list[list[Solution, ...]]
    ) -> list[Fitness, ...]:
        """Evaluate all the combinations of Pareto optimal solutions.

        The trainer's training fitness function is used to perform a
        repeated cross-validaton process.

        :param all_combinations: All the combinations of the Pareto optimal
            solutions from each species
        :type all_combinations: list[list[~culebra.abc.Solution, ...]]
        :return: The fitness for each combination
        :rtype: list[Fitness, ...]
        """
        fitness_func = deepcopy(self.trainer.fitness_func)
        fitness_func.obj_thresholds = 0
        fitness_cls = fitness_func.fitness_cls
        for obj in fitness_func.objectives:
            if isinstance(obj, DatasetScorer):
                obj.cv_fixed_folds = False

        # Evaluate each combination
        all_combinations_fitness = []
        for comb in all_combinations:
            repated_evaluations_fitness_values = []
            for _ in range(self.cv_repeats):
                repated_evaluations_fitness_values.append(
                    fitness_func.evaluate(comb[0], 0, comb)
                )
            mean_fitness_values = np.mean(
                repated_evaluations_fitness_values, axis=0
            )
            all_combinations_fitness.append(
                fitness_cls(mean_fitness_values)
            )

        return all_combinations_fitness


# Exported symbols for this module
__all__ = [
    'LexicographicDM',
    'LexicographicWithRepeatedCVDM',
    'DEFAULT_CV_REPEATS'
]
