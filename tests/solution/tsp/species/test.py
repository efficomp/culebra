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

"""Unit test for the tsp species."""

import unittest
import pickle
from copy import copy, deepcopy

import numpy as np

from culebra.solution.tsp import Species


class SpeciesTester(unittest.TestCase):
    """Test :py:class:`~culebra.solution.tsp.Species`."""

    def test_init(self):
        """Test the constructor."""
        # Default parameters
        valid_num_nodes = (10, 50, 1000)
        for num_nodes in valid_num_nodes:
            species = Species(num_nodes)

            # Check the default values for the attributes
            self.assertEqual(species.num_nodes, num_nodes)
            self.assertEqual(len(species.banned_nodes), 0)

        # Invalid num_nodes types
        invalid_num_nodes = (len, 'a')
        for num_nodes in invalid_num_nodes:
            with self.assertRaises(TypeError):
                Species(num_nodes=num_nodes)

        # Invalid num_nodes values
        invalid_num_nodes = (-1, 0)
        for num_nodes in invalid_num_nodes:
            with self.assertRaises(ValueError):
                Species(num_nodes=num_nodes)

        # Invalid banned_nodes types
        num_nodes = 10
        invalid_banned_nodes = (len, Species)
        for banned_nodes in invalid_banned_nodes:
            with self.assertRaises(TypeError):
                Species(num_nodes=num_nodes, banned_nodes=banned_nodes)

        # Invalid banned_nodes values
        invalid_banned_nodes = ([-1, 3], [3, 10], "as")
        for banned_nodes in invalid_banned_nodes:
            with self.assertRaises(ValueError):
                Species(num_nodes=num_nodes, banned_nodes=banned_nodes)

        # Valid min_node values
        valid_banned_nodes = [0, 5, 9]
        species = Species(num_nodes=num_nodes, banned_nodes=valid_banned_nodes)
        self.assertTrue(np.all(species.banned_nodes == valid_banned_nodes))

    def test_is_banned_feasible(self):
        """Test the is_banned and is_feasible methods."""
        num_nodes = 10
        banned_nodes = [1, 3]
        species = Species(num_nodes, banned_nodes=banned_nodes)

        for node in range(num_nodes):
            if node in banned_nodes:
                self.assertTrue(species.is_banned(node))
            else:
                self.assertTrue(species.is_feasible(node))

        unfeasible_nodes = [-1, num_nodes]
        for node in unfeasible_nodes:
            self.assertFalse(species.is_feasible(node))

    def test_copy(self):
        """Test the __copy__ method."""
        species1 = Species(15, banned_nodes=[1, 3])
        species2 = copy(species1)

        # Copy only copies the first level (species1 != species2)
        self.assertNotEqual(id(species1), id(species2))

        # The species attributes are shared
        self.assertEqual(id(species1._num_nodes), id(species2._num_nodes))
        self.assertEqual(id(species1.banned_nodes), id(species2.banned_nodes))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        species1 = Species(18, banned_nodes=[1, 3])
        species2 = deepcopy(species1)

        # Check the copy
        self._check_deepcopy(species1, species2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        species1 = Species(18)

        data = pickle.dumps(species1)
        species2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(species1, species2)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        species = Species(18)
        self.assertIsInstance(repr(species), str)
        self.assertIsInstance(str(species), str)

    def _check_deepcopy(self, species1, species2):
        """Check if *species1* is a deepcopy of *species2*.

        :param species1: The first species
        :type species1: :py:class:`~culebra.solution.tsp.Species`
        :param species2: The second species
        :type species2: :py:class:`~culebra.solution.tsp.Species`
        """
        # Copies all the levels
        self.assertNotEqual(id(species1), id(species2))
        self.assertEqual(species1._num_nodes, species2._num_nodes)
        self.assertTrue(np.all(species1.banned_nodes == species2.banned_nodes))


if __name__ == '__main__':
    unittest.main()
