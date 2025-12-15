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

"""Abstract base fitness functions for the TSP problem.

This sub-module provides several abstract classes that help defining other
fitness functions. The following classes are provided:

* :class:`~culebra.fitness_function.tsp.abc.TSPFitnessFunction`: Abstract base
  class for the all the fitness functions for the TSP problem.
"""

from __future__ import annotations

from abc import abstractmethod

from numpy import ndarray

from culebra.abc import FitnessFunction


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class TSPFitnessFunction(FitnessFunction):
    """Base class for fitness functions for the TSP problem."""

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes of the problem graph.

        This property must be overridden by subclasses to return the problem
        graph's number of nodes.

        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The num_nodes property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def heuristic(self) -> tuple[ndarray[float], ...]:
        """Heuristic matrices.

        This property must be overridden by subclasses.

        :return: A sequence of heuristic matrices. One for each objective.
            Arcs from a node to itself have a heuristic value of 0. For the
            rest of arcs, the reciprocal of their nodes distance is used as
            heuristic
        :rtype: tuple[~numpy.ndarray[float]]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The heuristic method has not been implemented in the "
            f"{self.__class__.__name__} class")


# Exported symbols for this module
__all__ = [
    'TSPFitnessFunction',
]
