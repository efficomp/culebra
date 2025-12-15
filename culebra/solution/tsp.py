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

"""Species and solutions for the traveling salesman problem.

This module provides all the classes necessary to solve the traveling salesman
problem with culebra. The possible solutions to the problem are handled by:

* A :class:`~culebra.solution.tsp.Solution` abstract class defining the
  interface for solutions to the problem.
* A :class:`~culebra.solution.tsp.Species` class to define the constraints
  that the desired paths should meet.

In order to make possible the application of ACO approaches to this
problem, the :class:`~culebra.solution.tsp.Ant` class is provided.
"""

from __future__ import annotations

from abc import abstractmethod
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
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Species(BaseSpecies):
    """Species for tsp solutions."""

    def __init__(
        self,
        num_nodes: int,
        banned_nodes: Sequence[int] | None = None
    ) -> None:
        """Create a new species.

        :param num_nodes: Number of nodes considered
        :type num_nodes: int
        :param banned_nodes: Sequence of banned nodes. If provided, each node
            index in the sequence must be in the interval [0, *num_nodes*).
            Defaults to :data:`None`
        :type banned_nodes: ~collections.abc.Sequence[int]
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__()

        self._num_nodes = check_int(num_nodes, "number of nodes", gt=0)
        self.banned_nodes = banned_nodes

    @property
    def num_nodes(self) -> int:
        """Number of nodes.

        :rtype: int
        """
        return self._num_nodes

    @property
    def banned_nodes(self) -> np.ndarray[int]:
        """Banned nodes.

        :rtype: ~numpy.ndarray[int]
        :setter: Set new banned nodes
        :param values: The new banned nodes or :data:`None` if all the nodes
            are allowed
        :type values: ~collections.abc.Sequence[int]
        :raises TypeError: If *values* is not a Sequence
        :raises ValueError: If any item in *values* is not a valid node index
        """
        return self._banned_nodes

    @banned_nodes.setter
    def banned_nodes(self, values: Sequence[int] | None) -> None:
        """Set new banned nodes.

        :param values: The new banned nodes or :data:`None` if all the nodes
            are allowed
        :type values: ~collections.abc.Sequence[int]
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
        """Check if a node is banned.

        :param node: The node index
        :type node: int
        :return: :data:`True` if the node index provided is banned
        :rtype: bool
        """
        return node in self._banned_nodes

    def is_feasible(self, node: int) -> bool:
        """Check if a node is feasible.

        :param node: The node index
        :type node: int
        :return: :data:`True` if the node index provided is feasible.
        :rtype: bool
        """
        if node < 0 or node >= self.num_nodes or self.is_banned(node):
            return False
        return True

    def is_member(self, sol: Solution) -> bool:
        """Check if a solution meets the constraints imposed by the species.

        :param sol: The solution
        :type sol: ~culebra.solution.tsp.Solution
        :return: :data:`True` if the solution belongs to the species.
            :data:`False` otherwise
        :rtype: bool
        """
        for node in sol.path:
            if not self.is_feasible(node):
                return False

        return True

    def __copy__(self) -> Species:
        """Shallow copy the species.

        :return: The copied species
        :rtype: ~culebra.solution.tsp.Species
        """
        cls = self.__class__
        result = cls(self.num_nodes)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Species:
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: dict
        :return: The copied species
        :rtype: ~culebra.solution.tsp.Species
        """
        cls = self.__class__
        result = cls(self.num_nodes)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the species.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, (self.num_nodes,), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Species:
        """Return a species from a state.

        :param state: The state
        :type state: dict
        :return: The species
        :rtype: ~culebra.solution.tsp.Species
        """
        obj = cls(
            state['_num_nodes']
        )
        obj.__setstate__(state)
        return obj


class Solution(BaseSolution):
    """Abstract base class for all the tsp solutions."""

    species_cls = Species
    """Class for the species used by the
    :class:`~culebra.solution.tsp.Solution` class to constrain
    all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: type[Fitness],
        path: Sequence[int] | None = None
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: ~culebra.solution.tsp.Species
        :param fitness_cls: The solution's fitness class
        :type fitness_cls: type[~culebra.abc.Fitness]
        :param path: Initial path
        :type path: ~collections.abc.Sequence[int]
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

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _setup method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    def path(self) -> np.ndarray[int]:
        """Path.

        :rtype: ~numpy.ndarray[int]
        :setter: Set a new path
        :param value: The new path
        :type value: ~collections.abc.Sequence[int]
        :raises ValueError: If *value* does not meet the species constraints.
        """
        return self._path

    @path.setter
    def path(self, value: Sequence[int]) -> None:
        """Set a new path.

        :param value: The new path
        :type value: ~collections.abc.Sequence[int]
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
        """Solution as a string.

        A symmetric tsp problem is assumed. Thus, all rotations of paths
        [0, 1, ..., *n*] and [*n*, ..., 1, 0] are considered the same path.

        The path is rolled to start with the node with smallest index.

        :rtype: str
        """
        if len(self.path) > 0:
            offset = np.argwhere(self.path == np.min(self.path)).flatten()[0]
            the_path = np.roll(self.path, -offset)
            if len(the_path) > 1 and the_path[-1] < the_path[1]:
                the_path[1:] = np.flip(the_path[1:])
        else:
            the_path = self.path

        return str(the_path)

    def __repr__(self) -> str:
        """Solution representation.

        :rtype: str
        """
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

    @property
    def discarded(self) -> np.ndarray[int]:
        """Nodes discarded by the ant.

        Return an empty array since all nodes must be visited for the TSP
        problem.

        :rtype: ~numpy.ndarray[int]
        """
        return np.empty(shape=(0,), dtype=int)

    def append(self, node: int) -> None:
        """Append a new node to the ant's path.

        :param node: The node
        :type node: int
        :raises TypeError: If *node* is not an integer number
        :raises ValueError: If *node* does not meet the species constraints
        :raises ValueError: If *node* is already in the path.
        """
        # Check the node type
        node = check_int(node, "node index")

        if node in self.path:
            raise ValueError(
                f"Node {node} is already in the path"
            )

        if not self.species.is_feasible(node):
            raise ValueError(
                f"Node {node} does not meet the species constraints"
            )

        self._path = np.append(self.path, (node))
        del self.fitness.values

    def discard(self, node: int) -> None:
        """Discard a node.

        This method raises an exception since nodes can not be discarded for
        the TSP problem.

        :param node: The node
        :type node: int
        :raises RuntimeError: If called
        """
        raise RuntimeError(
            "Nodes can not be discarded for the TSP problem"
        )

    def __repr__(self) -> str:
        """Return the ant representation."""
        return BaseAnt.__repr__(self)


# Exported symbols for this module
__all__ = [
    'Species',
    'Solution',
    'Ant'
]
