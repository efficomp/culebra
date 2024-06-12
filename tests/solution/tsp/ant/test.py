#!/usr/bin/env python3
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
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Unit test for the tsp ants."""

import unittest

import numpy as np

from culebra.abc import Fitness
from culebra.solution.tsp import Species, Ant


class MyFitness(Fitness):
    """Dummy fitness."""

    weights = (1.0, 1.0)
    names = ("obj1", "obj2")
    thresholds = [0.001, 0.001]


class AntTester(unittest.TestCase):
    """Test :py:class:`culebra.solution.tsp.Ant`."""

    def test_path(self):
        """Test the path property."""
        num_nodes = 10
        species = Species(num_nodes)
        ant = Ant(species, MyFitness)

        # Invalid paths
        invalid_paths = [
            [1, 1, 2],
            [0, -1, 2],
            [0, 1, num_nodes]
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                ant.path = path

        valid_path = [node for node in range(num_nodes)]
        ant.path = valid_path
        self.assertTrue(np.all(valid_path == ant.path))

    def test_append_current(self):
        """Test the append method."""
        num_nodes = 10
        species = Species(num_nodes)
        ant = Ant(species, MyFitness)

        # All possible indices for the species
        indices = np.arange(0, num_nodes)

        # Test repeated nodes, should fail
        for index in indices:
            ant.append(index)
            with self.assertRaises(ValueError):
                ant.append(index)

        # Test invalid values, should fail
        with self.assertRaises(ValueError):
            ant.append(num_nodes)
        with self.assertRaises(ValueError):
            ant.append(-1)

        # Try correct indices
        ant = Ant(species, MyFitness)
        for index, node in enumerate(indices):
            ant.append(node)
            self.assertEqual(index + 1, len(ant.path))
            self.assertEqual(node, ant.current)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_nodes = 10
        species = Species(num_nodes)
        ant = Ant(species, MyFitness)
        self.assertIsInstance(repr(ant), str)
        self.assertIsInstance(str(ant), str)


if __name__ == '__main__':
    unittest.main()
