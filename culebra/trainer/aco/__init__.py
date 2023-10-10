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

"""Implementation of some ant colony optimization trainers.

This module is composed by:

  * The :py:mod:`~culebra.trainer.aco.abc` sub-module, where some abstract base
    classes are defined to support the ACO trainers developed in this module

  * Some popular single-colony ACO algorithms:

      * The :py:class:`~culebra.trainer.aco.AntSystem` class, which implements
        the Ant System algorithm
"""

from . import abc

from .single_pop_aco import (
    AntSystem,
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_PHEROMONE_EVAPORATION_RATE
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


# Exported symbols for this module
__all__ = [
    'abc',
    'AntSystem',
    'DEFAULT_PHEROMONE_INFLUENCE',
    'DEFAULT_HEURISTIC_INFLUENCE',
    'DEFAULT_PHEROMONE_EVAPORATION_RATE'
]