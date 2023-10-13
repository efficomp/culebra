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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for the tsp solutions."""

import unittest
import pickle
import copy

import numpy as np

from culebra.abc import Species as BaseSpecies, Fitness
from culebra.solution.tsp import Species, Solution


class MyFitness(Fitness):
    """Dummy fitness."""

    weights = (1.0, 1.0)
    names = ("obj1", "obj2")
    thresholds = (0.001, 0.001)


class MySolution(Solution):
    """Dummy solution."""

    def _setup(self):
        """Set the default path for ants.

        An empty path is set.
        """
        self._path = self._path = np.empty(shape=(0,), dtype=int)


class SolutionTester(unittest.TestCase):
    """Tester for the tsp numpy-based solutions."""

    def test_constructor(self):
        """Test the behavior of a tsp solution constructor.

        The constructor is executed under different combinations of values for
        the number of nodes, minimum node and maximum node.
        """
        # Check the type of arguments
        with self.assertRaises(TypeError):
            MySolution(BaseSpecies(), MyFitness)
        with self.assertRaises(TypeError):
            MySolution(Species(), Species)

        # Check default solution
        num_nodes = 10
        species = Species(num_nodes)
        sol = MySolution(species, MyFitness)
        self.assertEqual(len(sol.path), 0)

    def test_path(self):
        """Test the path property."""
        num_nodes = 10
        species = Species(num_nodes)
        sol = MySolution(species, MyFitness)

        # Invalid paths
        invalid_paths = [
            [1, 1, 2],
            [0, -1, 2],
            [0, 1, num_nodes]
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                sol.path = path

        valid_path = [node for node in range(num_nodes)]
        sol.path = valid_path
        self.assertTrue(np.all(valid_path == sol.path))

    def test_serialization(self):
        """Serialization test."""
        num_nodes = 10
        species = Species(num_nodes)
        sol1 = MySolution(species, MyFitness, [1, 3, 5])

        data = pickle.dumps(sol1)
        sol2 = pickle.loads(data)

        self.assertTrue(np.all(sol1.path == sol2.path))

    def test_copy(self):
        """Copy test."""
        num_nodes = 10
        species = Species(num_nodes)
        sol1 = MySolution(species, MyFitness, [1, 5, 8])

        sol2 = copy.copy(sol1)
        self.assertTrue(np.all(sol1.path == sol2.path))

        sol3 = copy.deepcopy(sol1)
        self.assertTrue(np.all(sol1.path == sol3.path))

    def test_str(self):
        """Test the __str__ dunder method."""
        num_nodes = 10
        species = Species(num_nodes)
        sol = MySolution(species, MyFitness, [3, 4, 1, 2])
        self.assertEqual(str(sol), "[1 2 3 4]")

        sol = MySolution(species, MyFitness, [4, 3, 2, 1])
        self.assertEqual(str(sol), "[1 2 3 4]")

        sol = MySolution(species, MyFitness, [])
        self.assertEqual(str(sol), "[]")

        sol = MySolution(species, MyFitness, [3])
        self.assertEqual(str(sol), "[3]")


# Tests the classes in this file
if __name__ == '__main__':
    unittest.main()
