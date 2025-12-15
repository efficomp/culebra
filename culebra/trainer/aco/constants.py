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

"""Constants of the module."""


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_PHEROMONE_INFLUENCE = 1.0
r"""Default pheromone influence (:math:`{\alpha}`)."""

DEFAULT_HEURISTIC_INFLUENCE = 2.0
r"""Default heuristic influence (:math:`{\beta}`)."""

DEFAULT_EXPLOITATION_PROB = 0.9
r"""Default exploitation probability (:math:`{q_0}`)."""

DEFAULT_PHEROMONE_DEPOSIT_WEIGHT = 1.0
"""Default pheromone deposit weight."""

DEFAULT_PHEROMONE_EVAPORATION_RATE = 0.1
r"""Default pheromone evaporation rate (:math:`{\rho}`)."""

DEFAULT_CONVERGENCE_CHECK_FREQ = 100
"""Default frequency to check if an elitist ACO has converged."""

DEFAULT_ACOFS_INITIAL_PHEROMONE = 1
"""Default initial pheromone for ACO-FS approaches."""

DEFAULT_ACOFS_HEURISTIC_INFLUENCE = 0
r"""Default heuristic influence (:math:`{\beta}`) for ACO-FS approaches."""

DEFAULT_ACOFS_EXPLOITATION_PROB = 0
r"""Default exploitation probability (:math:`{q_0}`) for ACO-FS approaches."""

DEFAULT_ACOFS_DISCARD_PROB = 0.5
"""Default probability of discarding a node (feature) for ACO-FS approaches."""


# Exported symbols for this module
__all__ = [
    'DEFAULT_PHEROMONE_INFLUENCE',
    'DEFAULT_HEURISTIC_INFLUENCE',
    'DEFAULT_EXPLOITATION_PROB',
    'DEFAULT_PHEROMONE_DEPOSIT_WEIGHT',
    'DEFAULT_PHEROMONE_EVAPORATION_RATE',
    'DEFAULT_CONVERGENCE_CHECK_FREQ',
    'DEFAULT_ACOFS_INITIAL_PHEROMONE',
    'DEFAULT_ACOFS_HEURISTIC_INFLUENCE',
    'DEFAULT_ACOFS_EXPLOITATION_PROB',
    'DEFAULT_ACOFS_DISCARD_PROB'
]
