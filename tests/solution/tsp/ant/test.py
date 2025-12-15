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

from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength

# Default number of nodes
num_nodes = 25

# Default species
species = Species(num_nodes)

# Default fitness function
optimum_path = np.random.permutation(num_nodes)
fitness_function = PathLength.from_path(optimum_path)

# Default fitness class
fitness_cls = fitness_function.fitness_cls


class AntTester(unittest.TestCase):
    """Test :class:`culebra.solution.tsp.Ant`."""

    def test_path(self):
        """Test the path property."""
        ant = Ant(species, fitness_cls)

        # Invalid paths
        invalid_paths = [
            [1, 1, 2],
            [0, -1, 2],
            [0, 1, num_nodes]
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                ant.path = path

        valid_path = list(range(num_nodes))
        ant.path = valid_path
        self.assertTrue(np.all(valid_path == ant.path))

    def test_discarded(self):
        """Test the discarded property."""
        ant = Ant(species, fitness_cls)

        discarded_nodes = ant.discarded

        self.assertIsInstance(discarded_nodes, np.ndarray)
        self.assertEqual(len(discarded_nodes), 0)

    def test_append_current(self):
        """Test the append method."""
        ant = Ant(species, fitness_cls)

        # Test invalid feature types
        invalid_features = ('a', self, None)
        for feature in invalid_features:
            with self.assertRaises(TypeError):
                ant.append(feature)

        # All possible indices for the species
        indices = np.arange(0, num_nodes)

        # Test repeated nodes, should fail
        for index in indices:
            # Evaluate the ant
            fitness_function.evaluate(ant)

            # Check that the ant has been evaluated
            self.assertNotEqual(ant.fitness.values, (None, ))

            # Append a new node
            ant.append(index)

            # Check that the ant has not been evaluated yet
            self.assertEqual(ant.fitness.values, (None, ))

            # Try to append a repeated node, should fail
            with self.assertRaises(ValueError):
                ant.append(index)

        # Test invalid values, should fail
        with self.assertRaises(ValueError):
            ant.append(num_nodes)
        with self.assertRaises(ValueError):
            ant.append(-1)

        # Try correct indices
        ant = Ant(species, fitness_cls)
        for index, node in enumerate(indices):
            # Evaluate the ant
            fitness_function.evaluate(ant)

            # Append a new node
            ant.append(node)

            # Check that the ant has not been evaluated yet
            self.assertEqual(ant.fitness.values, (None, ))

            # Check the number of features and the current node
            self.assertEqual(index + 1, len(ant.path))
            self.assertEqual(node, ant.current)

    def test_discard(self):
        """Test the append method."""
        ant = Ant(species, fitness_cls)

        with self.assertRaises(RuntimeError):
            ant.discard(1)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        ant = Ant(species, fitness_cls)
        self.assertIsInstance(repr(ant), str)
        self.assertIsInstance(str(ant), str)


if __name__ == '__main__':
    unittest.main()
