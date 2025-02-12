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


DEFAULT_MAX_NUM_ITERS = 100
"""Default maximum number of iterations."""

PICKLE_FILE_EXTENSION = "gz"
"""Extension for files containing pickled objects."""

DEFAULT_CHECKPOINT_ENABLE = True
"""Default checkpointing enablement for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_CHECKPOINT_FREQ = 10
"""Default checkpointing frequency for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_CHECKPOINT_BASENAME = "checkpoint"
"""Default basename for checkpointing files."""

DEFAULT_CHECKPOINT_FILENAME = (
    DEFAULT_CHECKPOINT_BASENAME + "." + PICKLE_FILE_EXTENSION
)
"""Default checkpointing file name for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_VERBOSITY = __debug__
"""Default verbosity for a :py:class:`~culebra.abc.Trainer`."""

DEFAULT_INDEX = 0
"""Default :py:class:`~culebra.abc.Trainer` index. Only used within a
distributed approaches.
"""


__all__ = [
    'DEFAULT_MAX_NUM_ITERS',
    'PICKLE_FILE_EXTENSION',
    'DEFAULT_CHECKPOINT_ENABLE',
    'DEFAULT_CHECKPOINT_FREQ',
    'DEFAULT_CHECKPOINT_BASENAME',
    'DEFAULT_CHECKPOINT_FILENAME',
    'DEFAULT_VERBOSITY',
    'DEFAULT_INDEX'
]
