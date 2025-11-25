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

"""Abstract base fitness functions.

This sub-module provides several abstract classes that help defining other
fitness functions. The following classes are provided:

* :class:`~culebra.fitness_function.abc.SingleObjectiveFitnessFunction`:
  Abstract base class for all the single-objective fitness functions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Tuple, List, Optional
from functools import partial

from culebra import DEFAULT_SIMILARITY_THRESHOLD
from culebra.abc import FitnessFunction, Solution
from culebra.checker import check_int, check_float, check_sequence


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class SingleObjectiveFitnessFunction(FitnessFunction):
    """Base class for single-objective fitness functions."""

    def __init__(
        self,
        index: Optional[int] = None
    ) -> None:
        """Construct the fitness function.

        :param index: Index of this objective when it is used for
            multi-objective fitness functions, optional
        :type index: int
        :raises RuntimeError: If the number of objectives is not 1
        :raises TypeError: If *index* is not an integer number
        :raises ValueError: If *index* is not positive
        """
        # Init the superclasses
        if self.num_obj != 1:
            raise RuntimeError(
                f"Class {self.__class__} should have only one objective"
            )

        super().__init__()
        self.index = index
        self.obj_thresholds = None

    @property
    def index(self) -> int:
        """Objective index.

        :rtype: int

        :setter: Set a new index
        :param value: The new index
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is not positive
        """
        return self._index

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a new index for the objective.

        :param value: The new index
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is not positive
        """
        if value is None:
            self._index = 0
        else:
            self._index = check_int(value, "objective index", ge=0)

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :rtype: tuple[str]
        """
        return (f"obj_{self.index}",)

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
            :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :type values: float | ~collections.abc.Sequence[float] | None
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if isinstance(values, Sequence):
            self._obj_thresholds = check_sequence(
                values,
                "objective similarity thresholds",
                size=self.num_obj,
                item_checker=partial(check_float, ge=0)
            )
        elif values is not None:
            self._obj_thresholds = [
                check_float(values, "objective similarity threshold", ge=0)
            ] * self.num_obj
        else:
            self._obj_thresholds = (
                [DEFAULT_SIMILARITY_THRESHOLD] * self.num_obj
            )

    @abstractmethod
    def is_evaluable(self, sol: Solution) -> bool:
        """Assess the evaluability of a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :return: :data:`True` if the solution can be evaluated
        :rtype: bool
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The is_evaluable method has not been implemented in the "
            f"{self.__class__.__name__} class")


# Exported symbols for this module
__all__ = [
    'SingleObjectiveFitnessFunction'
]
