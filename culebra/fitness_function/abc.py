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

from culebra import DEFAULT_INDEX
from culebra.abc import FitnessFunction, Solution
from culebra.checker import check_int


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
        index: int | None = None
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
        super().__init__()

        # Check the number of objectives
        if self.num_obj != 1:
            raise RuntimeError(
                f"Class {self.__class__} should have only one objective"
            )

        # Set the index
        self.index = index

    @property
    def _default_index(self) -> int:
        """Default index.

        :return: :attr:`~culebra.DEFAULT_INDEX`
        :rtype: int
        """
        return DEFAULT_INDEX

    @property
    def index(self) -> int:
        """Objective index.

        :rtype: int

        :setter: Set a new index
        :param value: The new index. If set to :data:`None`,
            :attr:`~culebra.fitness_function.abc.SingleObjectiveFitnessFunction._default_index`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is a negative number
        """
        return self._index

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a new index for the objective.

        :param value: The new index. If set to :data:`None`,
            :attr:`~culebra.fitness_function.abc.SingleObjectiveFitnessFunction._default_index`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is a negative number
        """
        # Check the value
        self._index = (
            self._default_index if value is None else check_int(
                value, "objective index", ge=0
            )
        )

    @property
    def obj_names(self) -> tuple[str, ...]:
        """Objective names.

        :rtype: tuple[str]
        """
        return (f"obj_{self.index}",)

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
