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

"""Constants of the module."""


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


import numpy as np


DEFAULT_STATS_NAMES = ('Iter', 'Pop', 'NEvals')
"""Default statistics calculated for each iteration of the
:py:class:`~culebra.abc.Trainer`.
"""

DEFAULT_OBJECTIVE_STATS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max,
}
"""Default statistics calculated for each objective within a
:py:class:`~culebra.abc.Trainer`.
"""

DEFAULT_POP_SIZE = 100
"""Default population size."""

DEFAULT_MAX_NUM_ITERS = 100
"""Default maximum number of iterations."""

DEFAULT_CHECKPOINT_ENABLE = True
"""Default checkpointing enablement for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_CHECKPOINT_FREQ = 10
"""Default checkpointing frequency for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_CHECKPOINT_FILENAME = "checkpoint.gz"
"""Default checkpointing file name for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_VERBOSITY = __debug__
"""Default verbosity for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_INDEX = 0
"""Default :py:class:`~culebra.abc.Trainer` index. Only used within a
distributed approaches.
"""


__all__ = [
    'DEFAULT_STATS_NAMES',
    'DEFAULT_OBJECTIVE_STATS',
    'DEFAULT_POP_SIZE',
    'DEFAULT_MAX_NUM_ITERS',
    'DEFAULT_CHECKPOINT_ENABLE',
    'DEFAULT_CHECKPOINT_FREQ',
    'DEFAULT_CHECKPOINT_FILENAME',
    'DEFAULT_VERBOSITY',
    'DEFAULT_INDEX'
]
