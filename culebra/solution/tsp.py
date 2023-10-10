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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Species and solutions for the traveling salesman problem.

This module provides all the classes necessary to solve the traveling salesman
problem with culebra. The possible solutions to the problem are handled by:

  * A :py:class:`~culebra.solution.tsp.Species` class to define
    the constraints that the desired paths should meet.

  * A :py:class:`~culebra.solution.tsp.Solution` abstract class
    defining the interface for solutions to the problem.

In order to make possible the application of ACO approaches to this
problem, the :py:class:`~culebra.solution.tsp.Ant` class is provided.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Type, Optional
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from culebra.abc import (
    Species as BaseSpecies,
    Solution as BaseSolution,
    Fitness
)
from culebra.checker import check_int
from culebra.solution.abc import Ant as BaseAnt


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Species(BaseSpecies):
    """Species for tsp solutions."""

    def __init__(
        self,
        num_nodes: int,
        banned_nodes: Optional[Sequence[int]] = None
    ) -> None:
        """Create a new species.

        :param num_nodes: Number of nodes considered
        :type num_nodes: :py:class:`int`
        :param banned_nodes: Sequence of banned nodes. If provided, each node
            index in the sequence must be in the interval [0, *num_nodes*).
            Defaults to :py:data:`None`
        :type banned_nodes: :py:class:`~collections.abc.Sequence` of
            :py:class:`int`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__()

        self._num_nodes = check_int(num_nodes, "number of nodes", gt=0)
        self.banned_nodes = banned_nodes

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes for this species.

        :type: :py:class:`int`
        """
        return self._num_nodes

    @property
    def banned_nodes(self) -> Sequence[int]:
        """Get and set the sequence of banned nodes.

        :type: :py:class:`~collections.abc.Sequence` of :py:class:`int`
        """
        return self._banned_nodes

    @banned_nodes.setter
    def banned_nodes(self, values: Sequence[int] | None) -> None:
        """Set a new number of nodes.

        :type values: :py:class:`~collections.abc.Sequence` of :py:class:`int`
            ot :pt:data:`None`
        :raises TypeError: If *values* is not a Sequence
        :raises ValueError: If any item in *values* is not a valid node index
        """
        if values is None:
            self._banned_nodes = np.empty(shape=(0,), dtype=int)
        else:
            self._banned_nodes = np.sort(
                np.unique(np.asarray(values, dtype=int))
            )
        if (
            np.any(self._banned_nodes < 0) or
            np.any(self._banned_nodes >= self.num_nodes)
        ):
            raise ValueError("Invalid index for banned_nodes")

    def is_banned(self, node: int) -> bool:
        """Return :py:data:`True` if the node index provided is banned.

        :param node: The node index
        :type node: :py:class:`int`
        """
        return node in self._banned_nodes

    def is_feasible(self, node: int) -> bool:
        """Return :py:data:`True` if the node index provided is feasible.

        :param node: The node index
        :type node: :py:class:`int`
        """
        if node < 0 or node >= self.num_nodes or self.is_banned(node):
            return False
        return True

    def is_member(self, sol: Solution) -> bool:
        """Check if a solution meets the constraints imposed by the species.

        :param sol: The solution
        :type sol: :py:class:`~culebra.solution.tsp.Solution`
        :return: :py:data:`True` if the solution belongs to the species.
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        for node in sol.path:
            if not self.is_feasible(node):
                return False

        return True

    def __copy__(self) -> Species:
        """Shallow copy the species."""
        cls = self.__class__
        result = cls(self.num_nodes)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Species:
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the species
        :rtype: :py:class:`~culebra.solution.tsp.Species`
        """
        cls = self.__class__
        result = cls(self.num_nodes)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the species.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.num_nodes,), self.__dict__)


class Solution(BaseSolution):
    """Abstract base class for all the tsp solutions."""

    species_cls = Species
    """Class for the species used by the
    :py:class:`~culebra.solution.tsp.Solution` class to constrain
    all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness],
        path: Optional[Sequence[int]] = None
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species:
            :py:class:`~culebra.solution.tsp.Solution.species_cls`
        :param fitness: The solution's fitness class
        :type fitness: :py:class:`~culebra.abc.Fitness`
        :param path: Initial path
        :type path: :py:class:`~collections.abc.Sequence` of
            :py:class:`int`
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)

        if path is not None:
            self.path = path
        else:
            self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Init the nodes of this solution.

        This method must be overridden by subclasses.

        :raises NotImplementedError: if has not been overridden
        """
        raise NotImplementedError(
            "The _setup method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    def path(self) -> Sequence[int]:
        """Get and set the path.

        :getter: Return the path.
        :setter: Set the new path. An array-like object of node indices is
            expected
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If the new path does not meet the species
            constraints.
        """
        return self._path

    @path.setter
    def path(self, value: Sequence[int]) -> None:
        """Set a new path.

        :param value: The new path
        :type value: :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If *value* does not meet the species constraints.
        """
        # Get the new path
        self._path = np.asarray(value, dtype=int)

        # Check if there are any duplicate node
        if len(np.unique(self._path)) < len(self._path):
            raise ValueError(
                "The path provided has repeated nodes"
            )

        # Check if it is valid
        if not self.species.is_member(self):
            raise ValueError(
                "The path provided do not meet the species constraints"
            )

    def __str__(self) -> str:
        """Return the solution as a string.

        The path is rolled to start with the node with smallest index.
        """
        offset = np.argwhere(self.path == np.min(self.path)).flatten()[0]
        return str(np.roll(self.path, -offset))

    def __repr__(self) -> str:
        """Return the solution representation."""
        cls_name = self.__class__.__name__
        species_info = str(self.species)
        fitness_info = self.fitness.values

        return (
            f"{cls_name}(species={species_info}, fitness={fitness_info}, "
            f"path={str(self)})"
        )


class Ant(Solution, BaseAnt):
    """Ant to apply ACO to Fs problems."""

    def _setup(self) -> None:
        """Set the default path for ants.

        An empty path is set.
        """
        self._path = np.empty(shape=(0,), dtype=int)

    def append(self, node: int) -> None:
        """Append a new node to the ant's path.

        :raises ValueError: If *node* does not meet the species
            constraints.
        :raises ValueError: If *node* is already in the path.
        """
        if node in self.path:
            raise ValueError(
                f"Node {node} is already in the path"
            )

        if not self.species.is_feasible(node):
            raise ValueError(
                f"Node {node} does not meet the species constraints"
            )

        self._path = np.append(self.path, (node))


# Exported symbols for this module
__all__ = [
    'Species',
    'Solution',
    'Ant'
]
