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

r"""Implementation of some ant colony optimization trainers.

This module is composed by:

* The :mod:`~culebra.trainer.aco.abc` sub-module, where some abstract base
  classes are defined to support the ACO trainers developed in this module

* Some popular single-objective ACO algorithms:

  * Algorithms relying on pheromone evaporation:

    * The :class:`~culebra.trainer.aco.AntSystem` class, which implements
      the Ant System algorithm
    * The :class:`~culebra.trainer.aco.ElitistAntSystem` class, which
      implements the Elitist Ant System algorithm
    * The :class:`~culebra.trainer.aco.MMAS` class, which implements the
      :math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` Ant System algorithm

  * Population-based approaches:

    * The :class:`~culebra.trainer.aco.AgeBasedPACO` class, which
      implements a PACO approach with an age-based population update strategy
    * The :class:`~culebra.trainer.aco.QualityBasedPACO` class, which
      implements a PACO approach with a quality-based population update
      strategy

* Some multi-objective ACO algorithms:

  * The :class:`~culebra.trainer.aco.CPACO` class, which implements the
    Crowding PACO algorithm
  * The :class:`~culebra.trainer.aco.PACOMO` class, which implements the
    PACO-MO algorithm

* ACO-based wrappers for Feature Selection problems:

  * The :class:`~culebra.trainer.aco.ACOFSConvergenceDetector` class,
    designed to enable stopping :class:`~culebra.trainer.aco.abc.ACOFS`
    trainers upon convergence
  * The :class:`~culebra.trainer.aco.ElitistACOFS` class. A new
    elitist ACO-based wrapper for Feature Selection problems
  * The :class:`~culebra.trainer.aco.PACOFS` class. A new
    population-based wrapper ACO approach for Feature Selection problems
"""

from .constants import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_EXPLOITATION_PROB,
    DEFAULT_PHEROMONE_DEPOSIT_WEIGHT,
    DEFAULT_PHEROMONE_EVAPORATION_RATE,
    DEFAULT_CONVERGENCE_CHECK_FREQ,
    DEFAULT_ACOFS_INITIAL_PHEROMONE,
    DEFAULT_ACOFS_HEURISTIC_INFLUENCE,
    DEFAULT_ACOFS_EXPLOITATION_PROB,
    DEFAULT_ACOFS_DISCARD_PROB
)

from . import abc

from .single_obj_aco import (
    AntSystem,
    ElitistAntSystem,
    MMAS,
    AgeBasedPACO,
    QualityBasedPACO,
    DEFAULT_AS_EXPLOITATION_PROB,
    DEFAULT_ELITE_WEIGHT,
    DEFAULT_MMAS_ITER_BEST_USE_LIMIT
)

from .multi_obj_aco import (
    PACOMO,
    CPACO
)

from .aco_fs import (
    ACOFSConvergenceDetector,
    PACOFS,
    ElitistACOFS
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


# Exported symbols for this module
__all__ = [
    'abc',
    'AntSystem',
    'ElitistAntSystem',
    'MMAS',
    'AgeBasedPACO',
    'QualityBasedPACO',
    'PACOMO',
    'CPACO',
    'ACOFSConvergenceDetector',
    'PACOFS',
    'ElitistACOFS',
    'DEFAULT_PHEROMONE_INFLUENCE',
    'DEFAULT_HEURISTIC_INFLUENCE',
    'DEFAULT_EXPLOITATION_PROB',
    'DEFAULT_PHEROMONE_DEPOSIT_WEIGHT',
    'DEFAULT_PHEROMONE_EVAPORATION_RATE',
    'DEFAULT_CONVERGENCE_CHECK_FREQ',
    'DEFAULT_AS_EXPLOITATION_PROB',
    'DEFAULT_ELITE_WEIGHT',
    'DEFAULT_MMAS_ITER_BEST_USE_LIMIT',
    'DEFAULT_ACOFS_INITIAL_PHEROMONE',
    'DEFAULT_ACOFS_HEURISTIC_INFLUENCE',
    'DEFAULT_ACOFS_EXPLOITATION_PROB',
    'DEFAULT_ACOFS_DISCARD_PROB'
]
