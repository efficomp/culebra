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

"""Implementation of some trainers.

This module is composed by:

  * The :py:mod:`~culebra.trainer.abc` module, which defines some abstract base
    classes to implement several types of trainers.
  * The :py:class:`~culebra.trainer.ea` class, which implements several
    evolutionary algorithms-based trainers.
  * The :py:mod:`~culebra.trainer.topology` sub-module, which provides several
    tolpologies for distributed trainers.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


from .constants import (
    DEFAULT_NUM_SUBPOPS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)
from . import topology
from . import abc
from . import ea

__all__ = [
    'topology',
    'abc',
    'ea',
    'DEFAULT_NUM_SUBPOPS',
    'DEFAULT_REPRESENTATION_SIZE',
    'DEFAULT_REPRESENTATION_FREQ',
    'DEFAULT_REPRESENTATION_TOPOLOGY_FUNC',
    'DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS',
    'DEFAULT_REPRESENTATION_SELECTION_FUNC',
    'DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS',
    'DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC',
    'DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS',
    'DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC',
    'DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS'
]
