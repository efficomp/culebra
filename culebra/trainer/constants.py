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

from functools import partial

from deap.tools import selTournament

from .topology import ring_destinations, full_connected_destinations


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_MAX_NUM_ITERS = 100
"""Default maximum number of iterations."""

DEFAULT_CHECKPOINT_ACTIVATION = True
"""Default checkpointing activation for a :class:`~culebra.trainer.abc.CentralizedTrainer`."""

DEFAULT_CHECKPOINT_FREQ = 10
"""Default checkpointing frequency for a :class:`~culebra.trainer.abc.CentralizedTrainer`."""

DEFAULT_CHECKPOINT_BASENAME = "checkpoint"
"""Default basename for checkpointing files."""

DEFAULT_VERBOSITY = __debug__
"""Default verbosity for a :class:`~culebra.trainer.abc.CentralizedTrainer`."""

DEFAULT_COOPERATIVE_TOPOLOGY_FUNC = full_connected_destinations
"""Default topology function for the cooperative model."""

DEFAULT_ISLANDS_TOPOLOGY_FUNC = ring_destinations
"""Default topology function for the islands model."""

DEFAULT_NUM_REPRESENTATIVES = 5
"""Default value for the number of representatives sent to the other
subtrainers."""

DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ = 10
"""Default value for the number of iterations between representatives
sending."""

DEFAULT_REPRESENTATIVES_SELECTION_FUNC_PARAMS = {'tournsize': 3}
"""Default parameters for the representatives selection policy function."""

DEFAULT_REPRESENTATIVES_SELECTION_FUNC = partial(
    selTournament, **DEFAULT_REPRESENTATIVES_SELECTION_FUNC_PARAMS
)
"""Default selection policy function to choose the representatives."""


# Exported symbols for this module
__all__ = [
    'DEFAULT_MAX_NUM_ITERS',
    'DEFAULT_CHECKPOINT_ACTIVATION',
    'DEFAULT_CHECKPOINT_FREQ',
    'DEFAULT_CHECKPOINT_BASENAME',
    'DEFAULT_VERBOSITY',
    'DEFAULT_COOPERATIVE_TOPOLOGY_FUNC',
    'DEFAULT_ISLANDS_TOPOLOGY_FUNC',
    'DEFAULT_NUM_REPRESENTATIVES',
    'DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ',
    'DEFAULT_REPRESENTATIVES_SELECTION_FUNC'
]
