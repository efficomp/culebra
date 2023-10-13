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

"""Traveling salesman problem related fitness functions.

This sub-module fitness functions to solve tsp problems. Currently:

  * :py:class:`~culebra.fitness_function.tsp.PathLength`: Single-objective
    function that minimizes the length of a path.
"""

from __future__ import annotations

from typing import Tuple, Optional
from collections.abc import Sequence
from copy import deepcopy
import random

import numpy as np

from culebra.abc import Fitness, FitnessFunction
from culebra.checker import check_instance, check_matrix
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.solution.tsp import Species, Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class PathLength(FitnessFunction):
    """Single-objective fitness function for tsp problems.

    Evaluate the length of the path encoded by a solution.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.tap.PathLength.evaluate`
        method within a :py:class:`~culebra.solution.tsp.Solution`.
        """

        weights = (-1.0,)
        """Minimize the path length of a solution."""

        names = ("Len",)
        """Name of the objective."""

        thresholds = (DEFAULT_THRESHOLD,)
        """Similarity threshold for fitness comparisons."""

    def __init__(self, distances: Sequence[Sequence[float], ...]) -> None:
        """Construct a fitness function.

        :param distances: Distances between each pair of nodes
        :type distances: Two-dimensional array-like object
        :raises TypeError: If *distances* is not an array-like object
        :raises ValueError: If *distances* has any not float element
        :raises ValueError: If *distances* has not an homogeneous shape
        :raises ValueError: If *distances* has not two dimensions
        :raises ValueError: If *distances* has negative values
        """
        # Init the superclass
        super().__init__()
        self._distances = check_matrix(
            distances,
            "distances matrix",
            square=True,
            ge=0
        )

    @classmethod
    def fromPath(cls, path: Sequence[int]) -> PathLength:
        """Create an instance from an optimum path.

        This class method has been designed for testing purposes.

        :param path: The nodes permutation with an optimum path.
        :type path: :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises TypeError: If *path* is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If *path* has any not float element
        :raises ValueError: If *path* has not an homogeneous shape
        :raises ValueError: If *path* has not two dimensions
        """
        # Check path's type and elements type
        try:
            the_path = np.asarray(path, dtype=int)
        except TypeError as err:
            raise TypeError(
                "Invalid path: It is not a sequence"
            ) from err
        except ValueError as err:
            raise ValueError(
                "Invalid path: The path must have only one dimension and "
                "can only contain int values"
            ) from err

        # Check the path's shape
        if len(the_path.shape) != 1:
            raise ValueError("Invalid path: Only vectors are allowed")

        # Check the path's values
        num_nodes = len(the_path)
        if num_nodes == 0:
            raise ValueError("Invalid path: Empty path")

        if np.min(the_path) != 0:
            raise ValueError("Invalid path: The minimum value is not 0")

        if len(np.unique(the_path)) != num_nodes:
            raise ValueError("Invalid path: Repeated nodes")

        if np.max(the_path) != num_nodes - 1:
            raise ValueError("Invalid path: Missing nodes")

        # Create the distances matrix
        distances = np.full((num_nodes, num_nodes), 10)

        # Distances from a node to itself are banned
        for node in range(num_nodes):
            distances[node][node] = 0

        # Fix the path in the distances matrix
        org = path[-1]
        for dest in path:
            distances[org][dest] = 1
            distances[dest][org] = 1
            org = dest

        # Return the fitness function
        return cls(distances)

    @property
    def distances(self) -> Sequence[Sequence[float, ...]]:
        """Return the distances matrix.

        :rtype: :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Sequence` of :py:class:`int`
        """
        return self._distances

    @property
    def num_nodes(self) -> int:
        """Return the problem graph's number of nodes.

        :return: The problem graph's number of nodes
        :rtype: :py:class:`int`
        """
        return self.distances.shape[0]

    def heuristics(self, species: Species) -> Tuple[np.ndarray]:
        """Get the heuristics matrix for ACO-based trainers.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.solution.tsp.Species`
        :raises TypeError: If *species* is not an instance of
            :py:class:`~culebra.solution.tsp.Species`

        :return: A tuple with only one heuristics matrix. Arcs involving any
            banned node or arcs from a node to itself have a heuristic value
            of 0. For the rest of arcs, the reciprocal of their nodes distance
            is used as heuristic
        :rtype: :py:class:`tuple` of :py:class:`~numpy.ndarray`
        """
        check_instance(species, "species", cls=Species)

        # All the distances should be considered
        with np.errstate(divide='ignore'):
            heuristics = np.where(
                self.distances != 0.,
                1 / self.distances,
                0.
            )

        # Ignore banned nodes and arcs from a node to itself
        for node in range(species.num_nodes):
            heuristics[node][node] = 0
            for ignored in species.banned_nodes:
                heuristics[node][ignored] = 0
                heuristics[ignored][node] = 0

        return (heuristics, )

    def greedy_solution(self, species: Species) -> Solution:
        """Return a greddy solution for the problem.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.solution.tsp.Species`
        :raises TypeError: If *species* is not an instance of
            :py:class:`~culebra.solution.tsp.Species`

        :return: The greddy solution
        :rtype: :py:class:`~culebra.solution.tsp.Solution`
        """
        # Maximum solution's length
        max_len = species.num_nodes - len(species.banned_nodes)

        # Current path
        current_path = []

        # If the path can be constructed ...
        if max_len > 0:
            heuristics = self.heuristics(species)[0]

            # Start with a feasible node
            current_node = random.randint(0, species.num_nodes-1)
            while (current_node in species.banned_nodes):
                current_node = random.randint(0, species.num_nodes-1)
            current_path.append(current_node)

            # Complete the greedy path
            while len(current_path) < max_len:
                current_heuristics = heuristics[current_node]
                current_heuristics[current_path] = 0
                current_node = np.argwhere(
                    current_heuristics == np.max(current_heuristics)
                ).flatten()[0]
                current_path.append(current_node)

        # Construct the solution form the greedy path
        sol = Solution(species, self.Fitness, current_path)
        sol.fitness.values = self.evaluate(sol)
        return sol

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.tsp.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Return the solution's path length
        path_len = 0
        if (len(sol.path) > 0):
            org = sol.path[-1]
            for dest in sol.path:
                path_len += self.distances[org][dest]
                org = dest

        return (path_len,)

    def __copy__(self) -> PathLength:
        """Shallow copy the fitness function."""
        cls = self.__class__
        result = cls(self.distances)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> PathLength:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the fitness function
        :rtype:
            :py:class::py:class:`~culebra.fitness_function.tsp.PathLength`
        """
        cls = self.__class__
        result = cls(self.distances)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the fitness function.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.distances,), self.__dict__)


# Exported symbols for this module
__all__ = [
    'PathLength'
]
