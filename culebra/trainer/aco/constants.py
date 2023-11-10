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
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_PHEROMONE_INFLUENCE = 1.0
r"""Default pheromone influence (:math:`{\alpha}`)."""

DEFAULT_HEURISTIC_INFLUENCE = 2.0
r"""Default heuristic influence (:math:`{\beta}`)."""

DEFAULT_CONVERGENCE_CHECK_FREQ = 100
"""Default frequency to check if an elitist ACO has converged."""

DEFAULT_ELITE_WEIGHT = 0.3
"""Default weight for the elite ants (best-so-far ants) respect to the
iteration-best ant."""


# Exported symbols for this module
__all__ = [
    'DEFAULT_PHEROMONE_INFLUENCE',
    'DEFAULT_HEURISTIC_INFLUENCE',
    'DEFAULT_CONVERGENCE_CHECK_FREQ',
    'DEFAULT_ELITE_WEIGHT'
]
