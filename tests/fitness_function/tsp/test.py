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

"""Test the tsp fitness functions."""

import unittest
import pickle
from copy import copy, deepcopy
from itertools import repeat

import numpy as np

from culebra.solution.tsp import Species, Solution
from culebra.fitness_function.tsp import PathLength


class PathLengthTester(unittest.TestCase):
    """Test the :py:class:`~culebra.fitness_function.tsp.PathLength` class."""

    def test_init(self):
        """Test the constructor."""
        # Try invalid distances matrix types. Should fail
        invalid_distances_types = [
            len,
            [Species],
            [
                [1, 2],
                [3, max]
            ]
        ]
        for distances in invalid_distances_types:
            with self.assertRaises(TypeError):
                PathLength(distances)

        # Try invalid distances matrix values. Should fail
        invalid_distances_values = [
            'as',
            [1, 'a'],
            [
                [1, 2],
                [3, 'a']
            ]
        ]
        for distances in invalid_distances_values:
            with self.assertRaises(ValueError):
                PathLength(distances)

        # Try non homogeneous shapes. Should fail
        invalid_distances_values = [
            [
                [1, 2],
                [3, 3, 4]
            ],
            [
                [1, 2, 5],
                [3]
            ],
            [
                [1, 2, 5],
                [2, 4, 3],
                [3, 1]
            ]
        ]
        for distances in invalid_distances_values:
            with self.assertRaises(ValueError):
                PathLength(distances)

        # Try invalid dimensions. Should fail
        invalid_distances_values = [
            [1, 2],
            [
                [
                    [1, 2, 5],
                    [3, 2, 4]
                ],
                [
                    [1, 2, 5],
                    [3, 2, 4]
                ]
            ]
        ]
        for distances in invalid_distances_values:
            with self.assertRaises(ValueError):
                PathLength(distances)

        # Try not square matrices. Should fail
        invalid_distances_values = [
            [
                [1, 2, 3],
                [3, 3, 4]
            ],
            [
                [1, 2],
                [3, 0],
                [4, 5],
            ]
        ]
        for distances in invalid_distances_values:
            with self.assertRaises(ValueError):
                PathLength(distances)

        # Try negative diatances. Should fail
        distances = [
            [0, -1, 3],
            [4, 0, 6],
            [7, 8, 0]
        ]
        with self.assertRaises(ValueError):
            PathLength(distances)

        # Try a valid distances matrix
        distances = [
            [0, 2, 3],
            [4, 0, 6],
            [7, 8, 0]
        ]
        fitness_func = PathLength(distances)
        self.assertTrue(
            np.all(
                np.asarray(distances, dtype=float) == fitness_func.distances
            )
        )

    def test_from_path(self):
        """Test the fromPath class method."""
        # Try invalid path types. Should fail
        invalid_path_types = [
            len,
            [Species],
            [3, max]
        ]
        for path in invalid_path_types:
            with self.assertRaises(TypeError):
                PathLength.fromPath(path)

        # Try invalid path values. Should fail
        invalid_path_values = [
            'as',
            [1, 'a'],
            [3, 'a']
        ]
        for path in invalid_path_values:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try non homogeneous shapes. Should fail
        invalid_path_values = [
            [
                [1, 2],
                [3, 3, 4]
            ],
            [
                [1, 2, 5],
                [3]
            ],
            [
                [1, 2, 5],
                [2, 4, 3],
                [3, 1]
            ]
        ]
        for path in invalid_path_values:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try invalid dimensions. Should fail
        invalid_path_values = [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [
                    [1, 2, 5],
                    [3, 2, 4]
                ],
                [
                    [1, 2, 5],
                    [3, 2, 4]
                ]
            ]
        ]
        for path in invalid_path_values:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try invalid permutations
        invalid_path_values = [
            [],
            [1, 2, 3],
            [0, 1, 1],
            [0, 2, 3],
        ]
        for path in invalid_path_values:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try valid permutations
        times = 100
        num_nodes = 5
        for _ in repeat(None, times):
            path = np.random.permutation(num_nodes)
            fitness_func = PathLength.fromPath(path)

            # Check the distances
            for org_idx, org in enumerate(path):
                dest_1 = path[org_idx - 1]
                dest_2 = path[(org_idx + 1) % num_nodes]

                for node in range(num_nodes):
                    if node == dest_1 or node == dest_2:
                        self.assertEqual(
                            fitness_func.distances[org][node], 1
                        )
                    elif node == org:
                        self.assertEqual(
                            fitness_func.distances[org][node], 0
                        )
                    else:
                        self.assertEqual(
                            fitness_func.distances[org][node], 10
                        )

    def test_num_nodes(self):
        """Test the num_nodes property."""
        max_num_nodes = 10
        for num_nodes in range(2, max_num_nodes):
            path = np.random.permutation(num_nodes)
            fitness_func = PathLength.fromPath(path)
            self.assertEqual(fitness_func.num_nodes, num_nodes)

    def test_heuristic(self):
        """Test the heuristic method."""
        num_nodes = 10
        path = np.random.permutation(num_nodes)
        fitness_func = PathLength.fromPath(path)
        banned_nodes = [0, num_nodes-1]
        species = Species(num_nodes, banned_nodes=banned_nodes)
        (heuristic, ) = fitness_func.heuristic(species)

        # Check the heuristic
        for org_idx, org in enumerate(path):
            dest_1 = path[org_idx - 1]
            dest_2 = path[(org_idx + 1) % num_nodes]

            for node in range(num_nodes):
                if (
                    org in banned_nodes or
                    node in banned_nodes or
                    node == org
                ):
                    self.assertEqual(
                        heuristic[org][node], 0
                    )
                elif node == dest_1 or node == dest_2:
                    self.assertEqual(
                        heuristic[org][node], 1
                    )
                else:
                    self.assertEqual(
                        heuristic[org][node], 0.1
                    )

    def test_greddy_solution(self):
        """Test the greedy solution method."""
        num_nodes = 10
        path = np.random.permutation(num_nodes)
        fitness_func = PathLength.fromPath(path)
        banned_nodes = [0, num_nodes-1]
        species = Species(num_nodes, banned_nodes=banned_nodes)

        # Try feasible solutions
        times = 1000
        for _ in repeat(None, times):
            sol = fitness_func.greedy_solution(species)
            self.assertTrue(species.is_member(sol))
            self.assertEqual(len(sol.path), num_nodes - len(banned_nodes))

        # Try an unfeasible solution
        banned_nodes = list(node for node in range(num_nodes))
        species = Species(num_nodes, banned_nodes=banned_nodes)
        sol = fitness_func.greedy_solution(species)
        self.assertEqual(len(sol.path), 0)

    def test_evaluate(self):
        """Test the evaluate method."""
        # Try valid permutations
        times = 100
        num_nodes = 5
        species = Species(num_nodes)
        for _ in repeat(None, times):
            optimum_path = np.random.permutation(num_nodes)
            fitness_func = PathLength.fromPath(optimum_path)
            # Evaluate the optimum path
            sol = Solution(species, PathLength.Fitness, optimum_path)
            self.assertEqual(fitness_func.evaluate(sol), (num_nodes,))

            # Evaluate another paths
            for _ in repeat(None, times):
                # Generate a not optimum path
                other_path = np.random.permutation(num_nodes)

                sol = Solution(species, PathLength.Fitness, other_path)
                self.assertGreaterEqual(
                    fitness_func.evaluate(sol), (num_nodes,)
                )

        # Try an unfeasible solution
        banned_nodes = list(node for node in range(num_nodes))
        species = Species(num_nodes, banned_nodes=banned_nodes)
        sol = fitness_func.greedy_solution(species)
        self.assertGreaterEqual(
            fitness_func.evaluate(sol), (0,)
        )

    def test_copy(self):
        """Test the __copy__ method."""
        num_nodes = 5
        path = np.random.permutation(num_nodes)
        func1 = PathLength.fromPath(path)
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.distances), id(func2.distances))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        num_nodes = 5
        path = np.random.permutation(num_nodes)
        func1 = PathLength.fromPath(path)
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        num_nodes = 5
        path = np.random.permutation(num_nodes)
        func1 = PathLength.fromPath(path)

        data = pickle.dumps(func1)
        func2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(func1, func2)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1: :py:class:`~culebra.fitness_function.tsp.PathLength`
        :param func2: The second fitness function
        :type func2: :py:class:`~culebra.fitness_function.tsp.PathLength`
        """
        # Copies all the levels
        self.assertNotEqual(id(func1), id(func2))
        self.assertNotEqual(id(func1.distances), id(func2.distances))

        self.assertTrue((func1.distances == func2.distances).all())

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_nodes = 5
        path = np.random.permutation(num_nodes)
        fitness_func = PathLength.fromPath(path)
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


if __name__ == '__main__':
    unittest.main()
