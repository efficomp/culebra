# !/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""Unit test for :py:mod:`~culebra.trainer.topology`."""

import unittest

from culebra.trainer.topology import (
    ring_destinations,
    full_connected_destinations,
    DEFAULT_RING_OFFSET
)


class TopologyTester(unittest.TestCase):
    """Test the :py:mod:`~culebra.trainer.topology` module."""

    def test_ring_destinations(self):
        """Test :py:func:`~culebra.trainer.topology.ring_destinations`."""
        # Maximum number of islands for the test
        max_islands = 10

        # Try different offsets
        for num_islands in range(max_islands):
            for origin in range(num_islands):
                for offset in range(num_islands):
                    destinations = ring_destinations(
                        origin,
                        num_islands,
                        offset
                    )
                    self.assertIsInstance(destinations, list)
                    self.assertEqual(len(destinations), 1)
                    self.assertEqual(
                        destinations[0],
                        (origin + offset) % num_islands
                    )

        # Try deffault offset
        for num_islands in range(max_islands):
            for origin in range(num_islands):
                destinations = ring_destinations(origin, num_islands)
                self.assertIsInstance(destinations, list)
                self.assertEqual(len(destinations), 1)
                self.assertEqual(
                    destinations[0],
                    (origin + DEFAULT_RING_OFFSET) % num_islands
                )

    def test_full_connected_destinations(self):
        """Test full_connected_destinations."""
        # Maximum number of islands for the test
        max_islands = 10

        for num_islands in range(max_islands):
            for origin in range(num_islands):
                destinations = full_connected_destinations(origin, num_islands)
                self.assertIsInstance(destinations, list)
                self.assertEqual(len(destinations), num_islands - 1)
                for dest in range(num_islands):
                    if dest != origin:
                        self.assertTrue(dest in destinations)


if __name__ == '__main__':
    unittest.main()
