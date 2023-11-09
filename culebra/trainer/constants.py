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

from deap.tools import selTournament

from .topology import ring_destinations


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_NUM_SUBTRAINERS = 1
"""Default number of subtrainers."""

DEFAULT_REPRESENTATION_SIZE = 5
"""Default value for the number of representatives sent to the other
subtrainers.
"""

DEFAULT_REPRESENTATION_FREQ = 10
"""Default value for the number of iterations between representatives
sending."""

DEFAULT_REPRESENTATION_SELECTION_FUNC = selTournament
"""Default selection policy function to choose the representatives."""

DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS = {'tournsize': 3}
"""Default parameters for the representatives selection policy function."""

DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC = ring_destinations
"""Default topology function for the islands model."""

DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS = {}
"""Parameters for the default topology function in the islands model."""


# Exported symbols for this module
__all__ = [
    'DEFAULT_NUM_SUBTRAINERS',
    'DEFAULT_REPRESENTATION_SIZE',
    'DEFAULT_REPRESENTATION_FREQ',
    'DEFAULT_REPRESENTATION_SELECTION_FUNC',
    'DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS',
    'DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC',
    'DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS'
]
