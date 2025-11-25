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

"""Abstract base classes for solutions for different metaheuristics.

The :mod:`~culebra.solution.abc` module provides solutions targeted for
different metaheuristics. Currently, the following are defined:

* An :class:`~culebra.solution.abc.Ant` class, which adds the path handling
  stuff to the :class:`~culebra.abc.Solution` class to support the
  implementation of ant colony trainers.
* An :class:`~culebra.solution.abc.Individual` class, which adds the crossover
  and mutation operators to the :class:`~culebra.abc.Solution` class to support
  the implementation of evolutionary trainers.
"""

from __future__ import annotations

from typing import Tuple, Sequence
from abc import abstractmethod

from culebra.abc import Solution


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class Individual(Solution):
    """Abstract base class for individuals.

    Add the crossover and mutation operators to the
    :class:`~culebra.abc.Solution` class.
    """

    @abstractmethod
    def crossover(self, other: Individual) -> Tuple[Individual, Individual]:
        """Cross this individual with another one.

        This method must be overridden by subclasses to return a correct
        value.

        :param other: The other individual
        :type other: ~culebra.solution.abc.Individual
        :return: The two offspring
        :rtype: tuple[~culebra.solution.abc.Individual]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The crossover operator has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def mutate(self, indpb: float) -> Tuple[Individual]:
        """Mutate the individual.

        This method must be overridden by subclasses to return a correct
        value.

        :param indpb: Independent probability for each gene to be mutated.
        :type indpb: float
        :return: The mutant
        :rtype: tuple[~culebra.solution.abc.Individual]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The mutation operator has not been implemented in the "
            f"{self.__class__.__name__} class"
        )


class Ant(Solution):
    """Abstract base class for ants.

    Add the path handling stuff to the :class:`~culebra.abc.Solution` class.
    """

    @property
    @abstractmethod
    def path(self) -> Sequence[int]:
        """Path traveled by the ant.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Sequence[int]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The path property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    def current(self) -> int:
        """Current node in the path.

        :rtype: int
        """
        return self.path[-1] if len(self.path) > 0 else None

    @property
    @abstractmethod
    def discarded(self) -> Sequence[int]:
        """Nodes discarded by the ant.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Sequence[int]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The discarded property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def append(self, node: int) -> None:
        """Append a new node to the ant's path.

        This method must be overridden by subclasses to return a correct
        value.

        :param node: The node
        :type node: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The append method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def discard(self, node: int) -> None:
        """Discard a node.

        The discarded node is not appended to the ant's path.

        This method must be overridden by subclasses to return a correct
        value.

        :param node: The node
        :type node: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The discard method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def __str__(self) -> str:
        """Ant as a string.

        :rtype: str
        """
        return str(self.path)

    def __repr__(self) -> str:
        """Ant representation.

        :rtype: str
        """
        cls_name = self.__class__.__name__
        species_info = str(self.species)
        fitness_info = self.fitness.values

        return (
            f"{cls_name}(species={species_info}, fitness={fitness_info}, "
            f"path={str(self.path)}, discarded={self.discarded})"
        )


# Exported symbols for this module
__all__ = [
    'Individual',
    'Ant'
]
