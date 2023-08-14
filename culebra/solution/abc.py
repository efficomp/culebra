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

"""Abstract base classes for solutions for different metaheuristics.

The :py:mod:`~culebra.solution.abc` module provides solutions targeted for
different metaheuristics. Currently, the following are defined:

      * An :py:class:`~culebra.solution.abc.Individual` class, which adds the
        crossover and mutation operators to the
        :py:class:`~culebra.abc.Solution` class to support the implementation
        of evolutionary trainers.
"""

from __future__ import annotations

from typing import Tuple
from abc import abstractmethod

from culebra.abc import Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Individual(Solution):
    """Abstract base class for individuals.

    Add the crossover and mutation operators to the
    :py:class:`~culebra.abc.Solution` class.
    """

    @abstractmethod
    def crossover(self, other: Individual) -> Tuple[Individual, Individual]:
        """Cross this individual with another one.

        This method must be overriden by subclasses to return a correct
        value.

        :param other: The other individual
        :type other: :py:class:`~culebra.solution.abc.Individual`
        :raises NotImplementedError: if has not been overriden
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError(
            "The crossover operator has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def mutate(self, indpb: float) -> Tuple[Individual]:
        """Mutate the individual.

        This method must be overriden by subclasses to return a correct
        value.

        :param indpb: Independent probability for each gene to be mutated.
        :type indpb: :py:class:`float`
        :raises NotImplementedError: if has not been overriden
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError(
            "The mutation operator has not been implemented in the "
            f"{self.__class__.__name__} class"
        )


# Exported symbols for this module
__all__ = [
    'Individual'
]
