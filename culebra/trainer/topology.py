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

"""Topologies for distributed trainer algorithms."""

from __future__ import annotations

from typing import List

__author__ = "Jesús González"
__copyright__ = "Copyright 2023, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.2.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


DEFAULT_RING_OFFSET = 1
"""Default offset for the ring topology."""


def ring_destinations(
    origin: int,
    num_subpops: int,
    offset: int = DEFAULT_RING_OFFSET
) -> List[int]:
    """Return the destinations reachable from *origin*.

    :param origin: The index of the origin subpopulation trainer
    :type origin: :py:class:`int`
    :param num_subpops: The number of subpopulations
    :type num_subpops: :py:class:`int`
    :param offset: Offset applied to *origin*, defaults to
        :py:attr:`~culebra.trainer.topology.DEFAULT_RING_OFFSET`
    :type offset: :py:class:`int`, optional
    :return: The direct destinations from *origin*
    :rtype: :py:class:`list` of subpopulation trainer indexes
    """
    return [(origin + offset) % num_subpops]


def full_connected_destinations(
    origin: int,
    num_subpops: int
) -> List[int]:
    """Return the destinations reachable from *origin*.

    :param origin: The index of the origin subpopulation trainer
    :type origin: :py:class:`int`
    :param num_subpops: The number of subpopulations
    :type num_subpops: :py:class:`int`
    :return: The direct destinations from *origin*
    :rtype: :py:class:`list` of subpopulation trainer indexes
    """
    destinations = list(range(num_subpops))
    destinations.remove(origin)
    return destinations


__all__ = [
    'ring_destinations',
    'full_connected_destinations',
    'DEFAULT_RING_OFFSET'
]
