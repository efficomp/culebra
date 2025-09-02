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

  * :py:class:`~culebra.fitness_function.abc.ACOFitnessFunction`:
    Abstract base class for the all the fitness functions for ACO-based
    trainers.
  * :py:class:`~culebra.fitness_function.abc.SingleObjectiveFitnessFunction`:
    Abstract base class for all the single-objective fitness functions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Tuple, Optional
from numpy import ndarray

from culebra.abc import FitnessFunction, Species, Solution
from culebra.checker import check_int


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class ACOFitnessFunction(FitnessFunction):
    """Base class for fitness functions for ACO-based trainers."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Return the problem graph's number of nodes for ACO-based trainers.

        This property must be overridden by subclasses to return the problem
        graph's number of nodes.

        :return: The problem graph's number of nodes
        :rtype: :py:class:`int`
        :raises NotImplementedError: if has not been overridden
        """
        raise NotImplementedError(
            "The num_nodes property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def heuristic(self, species: Species) -> Sequence[ndarray, ...]:
        """Get the heuristic matrices for ACO-based trainers.

        This property must be overridden by subclasses.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.abc.Species`
        :return: A sequence of heuristic matrices
        :rtype: :py:class:`~collections.abc.Sequence` of
            :py:class:`~numpy.ndarray`
        """
        raise NotImplementedError(
            "The heuristic method has not been implemented in the "
            f"{self.__class__.__name__} class")


class SingleObjectiveFitnessFunction(FitnessFunction):
    """Base class for single-objective fitness functions."""

    def __init__(
        self,
        index: Optional[int] = None
    ) -> None:
        """Construct the fitness function.

        :param index: Index of this objective when it is used for
            multi-objective fitness functions
        :type index: :py:class:`int`, optional
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

    @property
    def index(self) -> int:
        """Get and set the index of this objective.

        :getter: Return the current index
        :setter: Set a new index
        :type: :py:class:`int`
        :raises TypeError: If the index is not an integer number
        :raises ValueError: If the index is not positive
        """
        return self._index

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a new index for the objective.

        :param value: The new index
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is not positive
        """
        if value is None:
            self._index = 0
        else:
            self._index = check_int(value, "objective index", ge=0)

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Get the objective names.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        suffix_len = len(str(self.num_obj-1 + self.index))

        return tuple(
            f"obj_{i+self.index:0{suffix_len}d}" for i in range(self.num_obj)
        )

    @abstractmethod
    def is_evaluable(self, sol: Solution) -> bool:
        """Return :py:data:`True` if the solution can be evaluated.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.abc.Solution`
        :raises NotImplementedError: if has not been overridden
        """
        raise NotImplementedError(
            "The is_evaluable method has not been implemented in the "
            f"{self.__class__.__name__} class")


# Exported symbols for this module
__all__ = [
    'ACOFitnessFunction',
    'SingleObjectiveFitnessFunction'
]
