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
from os import remove
from collections.abc import Sequence
from copy import copy, deepcopy
from itertools import repeat
import io

import numpy as np

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import Species as BaseSpecies
from culebra.solution.tsp import Species, Solution
from culebra.fitness_function.tsp import PathLength, MultiObjectivePathLength
from culebra.fitness_function.feature_selection import NumFeats


class PathLengthTester(unittest.TestCase):
    """Test the :class:`~culebra.fitness_function.tsp.PathLength` class."""

    def test_init(self):
        """Test the constructor."""
        # Try an invalid values within a matrix. Should fail
        invalid_matrices = [
            [
                [max, 1],
                [1, 0]
            ],
            [
                [0, 'a'],
                [1, 0]
            ],
            [
                [0, 1],
                [1, int]
            ]
        ]
        for matrix in invalid_matrices:
            with self.assertRaises(ValueError):
                PathLength(matrix)

        # Try non homogeneous matrices. Should fail
        non_homogeneous_matrices = [
            [
                [0],
                [1, 0]
            ],
            [
                [0, 1],
                [1]
            ]
        ]
        for matrix in non_homogeneous_matrices:
            with self.assertRaises(ValueError):
                PathLength(matrix)

        # Try matrices ith an invalid number of dimensions. Should fail
        invalid_dimension_matrices = [
            [1, 0],
            [
                [
                    [0, 1],
                    [1, 0]
                ],
                [
                    [0, 1],
                    [1, 0]
                ]
            ]
        ]
        for matrix in invalid_dimension_matrices:
            with self.assertRaises(ValueError):
                PathLength(matrix)

        # Try non square matrices. Should fail
        non_square_matrices = [
            [
                [0, 1, 2],
                [1, 0, 4]
            ],
            [
                [0, 1],
                [1, 2],
                [3, 1],
                [2, 5]
            ]
        ]
        for matrix in non_square_matrices:
            with self.assertRaises(ValueError):
                PathLength(matrix)

        # Try matrices with negative values. Should fail
        matrices_with_negative_values = [
            [
                [-1, 1],
                [1, 0]
            ],
            [
                [0, -1],
                [1, 0]
            ],
            [
                [0, 1],
                [-1, 0]
            ],
            [
                [0, 1],
                [1, -1]
            ]
        ]
        for matrix in matrices_with_negative_values:
            with self.assertRaises(ValueError):
                PathLength(matrix)

        # Try empty matrices. Should fail
        empty_matrices = [
            [],
            [
                []
            ],
            [
                [
                    []
                ]
            ],
            [
                [
                    [],
                    []
                ]
            ]
        ]
        for matrices in empty_matrices:
            with self.assertRaises(ValueError):
                PathLength(matrix)

        # Try valid distance matrices
        valid_matrices = [
            [
                [0, 2],
                [2, 0]
            ],
            [
                [0, 4, 7],
                [4, 0, 8],
                [7, 8, 0]
            ]
        ]
        for matrix in valid_matrices:
            fitness_func = PathLength(matrix)
            for dist1, dist2 in zip(matrix, fitness_func.distance):
                self.assertTrue(
                    np.all(
                        np.asarray(dist1, dtype=float) == dist2
                    )
                )

        # Check a valid index
        valid_index = 3
        func = PathLength([[0, 1], [1, 0]], index=valid_index)
        self.assertEqual(func.index, valid_index)

        # Check an invalid index type
        invalid_index_type = 'a'
        with self.assertRaises(TypeError):
            PathLength([[0, 1], [1, 0]], index=invalid_index_type)

        # Check an invalid index valur
        invalid_index_value = -1
        with self.assertRaises(ValueError):
            PathLength([[0, 1], [1, 0]], index=invalid_index_value)

    def test_obj_names(self):
        """Test the obj_names property."""
        # Check default parameter values
        func = PathLength([[0, 1], [1, 0]], index=8)
        self.assertEqual(func.index, 8)
        self.assertEqual(func.obj_names, ("Len",))

    def test_num_nodes(self):
        """Test the num_nodes property."""
        max_num_nodes = 10
        for num_nodes in range(2, max_num_nodes):
            fitness_func = PathLength(np.ones((num_nodes, num_nodes)))
            self.assertEqual(fitness_func.num_nodes, num_nodes)

    def test_heuristic(self):
        """Test the heuristic method."""
        distance_matrix = [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 6, 7, 8, 9],
            [2, 6, 0, 1, 2, 3],
            [3, 7, 1, 0, 4, 5],
            [4, 8, 2, 4, 0, 6],
            [5, 9, 3, 5, 6, 0]
        ]

        fitness_func = PathLength(distance_matrix)

        heuristic = fitness_func.heuristic
        self.assertIsInstance(heuristic, Sequence)

        # Check the heuristic_matrix
        for i in range(fitness_func.num_nodes):
            for j in range(fitness_func.num_nodes):
                if i == j:
                    self.assertEqual(heuristic[0][i][j], 0)
                else:
                    self.assertAlmostEqual(
                        heuristic[0][i][j],
                        1/distance_matrix[i][j]
                    )

    def test_greddy_solution(self):
        """Test the greedy solution method."""
        distance_matrix = [
            [0, 1, 2],
            [1, 0, 6],
            [2, 6, 0]
        ]
        fitness_func = PathLength(distance_matrix)
        num_nodes = fitness_func.num_nodes
        banned_nodes = [0, fitness_func.num_nodes-1]
        species = Species(
            fitness_func.num_nodes,
            banned_nodes=banned_nodes
        )

        # Try a wrong species. Should fail...
        with self.assertRaises(TypeError):
            fitness_func.greedy_solution(BaseSpecies())

        # Try a feasible solution
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
        distance_matrix = [
            [0, 4, 1],
            [4, 0, 2],
            [1, 2, 0]
        ]

        fitness_func = PathLength(distance_matrix)
        species = Species(fitness_func.num_nodes)

        # Try valid permutations
        times = 100
        for _ in repeat(None, times):
            for _ in repeat(None, times):
                sol = Solution(
                    species,
                    fitness_func.fitness_cls,
                    np.random.permutation(fitness_func.num_nodes)
                )
                fit_values = fitness_func.evaluate(sol).values
                self.assertIsInstance(sol.fitness.values, tuple)
                self.assertEqual(len(sol.fitness.values), 1)
                for i in range(fitness_func.num_obj):
                    self.assertGreater(sol.fitness.values[i], 0)

                self.assertEqual(fit_values, sol.fitness.values)

        # Try an unfeasible solution
        banned_nodes = list(node for node in range(fitness_func.num_nodes))
        species = Species(fitness_func.num_nodes, banned_nodes=banned_nodes)
        sol = fitness_func.greedy_solution(species)
        fit_values = fitness_func.evaluate(sol).values
        self.assertGreaterEqual(
            sol.fitness.values, (0,)
        )
        self.assertEqual(fit_values, sol.fitness.values)

    def test_from_path(self):
        """Test the fromPath class method."""
        # Try invalid path sequence lengths. Should fail
        invalid_path_sequence_lengths = [
            [],
            [
                [0, 1, 2]
            ],
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 1, 0]
            ]
        ]
        for path in invalid_path_sequence_lengths:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try invalid path types. Should fail
        invalid_path_types = [
            [
                1
            ],
            [
                len
            ],
            [
                Species
            ]
        ]
        for path in invalid_path_types:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try invalid path values. Should fail
        invalid_path_values = [
            [
                "hi"
            ],
            [
                [0, max]
            ],
            [
                [1, -1]
            ],
            [
                [0, 1.5]
            ]
        ]
        for path in invalid_path_values:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try invalid paths. Should fail
        invalid_paths = [
            [
                []
            ],
            [
                [0, 1, 1, 2]
            ],
            [
                [1, 3, 2]
            ],
            [
                [0, 3]
            ],
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                PathLength.fromPath(path)

        # Try valid permutations
        times = 100
        num_nodes = 5
        for _ in repeat(None, times):
            path = np.random.permutation(num_nodes)

            fitness_func = PathLength.fromPath(path)

            # Check the distance
            dist = fitness_func.distance
            for org_idx, org in enumerate(path):
                dest_1 = path[org_idx - 1]
                dest_2 = path[(org_idx + 1) % num_nodes]

                for node in range(num_nodes):
                    if node == dest_1 or node == dest_2:
                        self.assertEqual(
                            dist[org][node], 1
                        )
                    elif node == org:
                        self.assertEqual(
                            dist[org][node], 0
                        )
                    else:
                        self.assertEqual(
                            dist[org][node], 10
                        )

    def test_from_tsplib_parameters(self):
        """Test the parameter parsing part of the fromTSPLib class method."""
        # Try invalid buffer types. Should fail
        invalid_buffer_types = [
            len,
            Species,
            4
        ]
        for buffer in invalid_buffer_types:
            with self.assertRaises(AttributeError):
                PathLength.fromTSPLib(buffer)

        # Try invalid file paths. Should fail
        invalid_paths = [
            "hi",
            "/dev/hi",
            "https://none.none/none"
        ]
        for path in invalid_paths:
            with self.assertRaises(RuntimeError):
                PathLength.fromTSPLib(path)

        # Try an invalid problem type. Should fail
        buffer = """TYPE: new"""
        with self.assertRaises(RuntimeError):
            PathLength.fromTSPLib(io.StringIO(buffer))

        # Try invalid dimensions. Should fail
        invalid_buffers = [
            "DIMENSION: new",
            "DIMENSION: 1.4",
            "DIMENSION: -2",
            "DIMENSION: 0"
            ]
        for buffer in invalid_buffers:
            with self.assertRaises(RuntimeError):
                PathLength.fromTSPLib(io.StringIO(buffer))

        # Try an invalid edge weight type. Should fail
        buffer = "EDGE_WEIGHT_TYPE: new"
        with self.assertRaises(RuntimeError):
            PathLength.fromTSPLib(io.StringIO(buffer))

        # Try an invalid edge weight format. Should fail
        buffer = "EDGE_WEIGHT_FORMAT: new"
        with self.assertRaises(RuntimeError):
            PathLength.fromTSPLib(io.StringIO(buffer))

    def test_from_tsplib_node_coord_section(self):
        """Test the node coord section parsing of fromTSPLib."""
        # Try a buffer with missing DIMENSION. Should fail
        buffer = "NODE_COORD_SECTION"
        with self.assertRaises(RuntimeError):
            PathLength.fromTSPLib(io.StringIO(buffer))

        # Try buffers with missing nodes. Should fail
        invalid_buffers = [
            "DIMENSION: 2\n"
            "EDGE_WEIGHT_TYPE: MAN_2D\n"
            "NODE_COORD_SECTION\n"
            "2 1 1",
            "DIMENSION: 2\n"
            "EDGE_WEIGHT_TYPE: MAN_2D\n"
            "NODE_COORD_SECTION\n"
            "1 1 1",
            "DIMENSION: 3\n"
            "EDGE_WEIGHT_TYPE: MAN_2D\n"
            "NODE_COORD_SECTION\n"
            "1 1 1\n"
            "3 3 3"
            ]
        for buffer in invalid_buffers:
            with self.assertRaises(RuntimeError):
                PathLength.fromTSPLib(io.StringIO(buffer))

        # Try a valid buffer
        buffer = (
            "TYPE: TSP\n"
            "DIMENSION: 3\n"
            "EDGE_WEIGHT_TYPE: MAN_2D\n"
            "NODE_COORD_SECTION\n"
            "1 0 0\n"
            "2 1 1\n"
            "3 2 2\n"
        )
        fitness_func = PathLength.fromTSPLib(io.StringIO(buffer))
        self.assertEqual(fitness_func.num_nodes, 3)
        self.assertTrue(
            np.all(
                fitness_func.distance ==
                np.asarray(
                    [
                        [0, 2, 4],
                        [2, 0, 2],
                        [4, 2, 0]
                    ]
                )
            )
        )

        # Try a valid local file
        fitness_func = PathLength.fromTSPLib("test.tsp")
        self.assertEqual(fitness_func.num_nodes, 5)
        self.assertTrue(
            np.all(
                fitness_func.distance ==
                np.asarray(
                    [
                        [0, 1, 2, 3, 4],
                        [1, 0, 1, 2, 3],
                        [2, 1, 0, 1, 2],
                        [3, 2, 1, 0, 1],
                        [4, 3, 2, 1, 0]
                    ]
                )
            )
        )

        # Try a valid url
        fitness_func = PathLength.fromTSPLib(
            "https://raw.githubusercontent.com/mastqe/tsplib/master/"
            "kroA100.tsp"
        )
        expected_num_nodes = 100
        self.assertEqual(fitness_func.num_nodes, expected_num_nodes)
        self.assertEqual(
            fitness_func.distance.shape,
            (expected_num_nodes, expected_num_nodes)
        )

    def test_from_tsplib_edge_weight_section(self):
        """Test the edge weight section parsing of fromTSPLib."""
        # Try a full matrix with missing values. Should fail
        buffer = (
            "TYPE: TSP\n"
            "DIMENSION: 3\n"
            "EDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
            "EDGE_WEIGHT_SECTION\n"
            "1 0 0\n"
            "2 1 1\n"
        )
        with self.assertRaises(RuntimeError):
            PathLength.fromTSPLib(io.StringIO(buffer))

        # Try a full matrix with an excessive number of values. Should fail
        buffer += "1 2 3 4"
        with self.assertRaises(RuntimeError):
            PathLength.fromTSPLib(io.StringIO(buffer))

        # Try a correct full matrix
        fitness_func = PathLength.fromTSPLib(
            "https://raw.githubusercontent.com/mastqe/tsplib/master/bays29.tsp"
        )
        expected_num_nodes = 29
        self.assertEqual(fitness_func.num_nodes, expected_num_nodes)
        self.assertEqual(
            fitness_func.distance.shape,
            (expected_num_nodes, expected_num_nodes)
        )

        # Try a correct upper row matrix
        upper_row_buffer = (
            "TYPE: TSP\n"
            "DIMENSION: 5\n"
            "EDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: UPPER_ROW\n"
            "EDGE_WEIGHT_SECTION\n"
            "1 2 3 4 \n"
            "  5 6 7 \n"
            "    8 9 \n"
            "      1 \n"
        )
        upper_row_fitness_func = PathLength.fromTSPLib(
            io.StringIO(upper_row_buffer)
        )
        expected_num_nodes = 5
        self.assertEqual(
            upper_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            upper_row_fitness_func.distance.shape,
            (expected_num_nodes, expected_num_nodes)
        )

        # Try the corresponding lower row matrix
        lower_row_buffer = (
            "TYPE: TSP\n"
            "DIMENSION: 5\n"
            "EDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: LOWER_ROW\n"
            "EDGE_WEIGHT_SECTION\n"
            "1       \n"
            "2 5     \n"
            "3 6 8   \n"
            "4 7 9 1 \n"
        )
        lower_row_fitness_func = PathLength.fromTSPLib(
            io.StringIO(lower_row_buffer)
        )
        self.assertEqual(
            lower_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            lower_row_fitness_func.distance.shape,
            (expected_num_nodes, expected_num_nodes)
        )
        self.assertTrue(
            np.all(
                upper_row_fitness_func.distance ==
                lower_row_fitness_func.distance
            )
        )

        # Try a correct upper row matrix with diagonal
        upper_diag_row_buffer = (
            "TYPE: TSP\n"
            "DIMENSION: 5\n"
            "EDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW\n"
            "EDGE_WEIGHT_SECTION\n"
            "9 1 2 3 4 \n"
            "  9 5 6 7 \n"
            "    9 8 9 \n"
            "      9 1 \n"
            "        9 \n"
        )
        upper_diag_row_fitness_func = PathLength.fromTSPLib(
            io.StringIO(upper_diag_row_buffer)
        )
        expected_num_nodes = 5
        self.assertEqual(
            upper_diag_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            upper_diag_row_fitness_func.distance.shape,
            (expected_num_nodes, expected_num_nodes)
        )

        # Try the corresponding lower row matrix
        lower_diag_row_buffer = (
            "TYPE: TSP\n"
            "DIMENSION: 5\n"
            "EDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW\n"
            "EDGE_WEIGHT_SECTION\n"
            "9         \n"
            "1 9       \n"
            "2 5 9     \n"
            "3 6 8 9   \n"
            "4 7 9 1 9 \n"
        )
        lower_diag_row_fitness_func = PathLength.fromTSPLib(
            io.StringIO(lower_diag_row_buffer)
        )
        self.assertEqual(
            lower_diag_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            lower_diag_row_fitness_func.distance.shape,
            (expected_num_nodes, expected_num_nodes)
        )
        self.assertTrue(
            np.all(
                upper_diag_row_fitness_func.distance ==
                lower_diag_row_fitness_func.distance
            )
        )

    def test_copy(self):
        """Test the __copy__ method."""
        num_nodes = 5
        func1 = PathLength.fromPath(np.random.permutation(num_nodes))
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.distance), id(func2.distance))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        num_nodes = 5
        func1 = PathLength.fromPath(np.random.permutation(num_nodes))
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        num_nodes = 5
        func1 = PathLength.fromPath(np.random.permutation(num_nodes))

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        func1.dump(serialized_filename)
        func2 = PathLength.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the serialized file
        remove(serialized_filename)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1: ~culebra.fitness_function.tsp.PathLength
        :param func2: The second fitness function
        :type func2: ~culebra.fitness_function.tsp.PathLength
        """
        # Copies all the levels
        self.assertNotEqual(id(func1), id(func2))
        for dist1, dist2 in zip(func1.distance, func2.distance):
            self.assertNotEqual(id(dist1), id(dist2))
            self.assertTrue((dist1 == dist2).all())

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_nodes = 5
        fitness_func = PathLength.fromPath(np.random.permutation(num_nodes))
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class MultiObjectivePathLengthTester(unittest.TestCase):
    """Test the MultiObjectivePathLength class."""

    def test_init(self):
        """Test the constructor."""
        num_nodes = 5

        # Try an invalid objective. Should fail ...
        obj1 = PathLength.fromPath(np.random.permutation(num_nodes))
        obj2 = NumFeats()
        with self.assertRaises(ValueError):
            MultiObjectivePathLength(obj1, obj2)

        # Try objectives with different number of nodes. Should fail ...
        obj1 = PathLength.fromPath(np.random.permutation(num_nodes))
        obj2 = PathLength.fromPath(np.random.permutation(num_nodes + 1))
        with self.assertRaises(ValueError):
            MultiObjectivePathLength(obj1, obj2)

        # Try a function without objectives
        func = MultiObjectivePathLength()
        self.assertEqual(func.num_obj, 0)

        # Try a function with  two objectives
        obj1 = PathLength.fromPath(np.random.permutation(num_nodes))
        obj2 = PathLength.fromPath(np.random.permutation(num_nodes))
        func = MultiObjectivePathLength(obj1, obj2)
        self.assertEqual(func.num_obj, 2)
        self.assertEqual(obj1, func.objectives[0])
        self.assertEqual(obj2, func.objectives[1])

    def test_obj_names(self):
        """Test the obj_names property."""
        num_nodes = 5
        obj1 = PathLength.fromPath(np.random.permutation(num_nodes))
        obj2 = PathLength.fromPath(np.random.permutation(num_nodes))
        func = MultiObjectivePathLength(obj1, obj2)
        self.assertEqual(func.obj_names, ('Len_0', 'Len_1'))

    def test_heuristic(self):
        """Test the heuristic method."""
        num_nodes = 5
        obj1 = PathLength.fromPath(np.random.permutation(num_nodes))
        obj2 = PathLength.fromPath(np.random.permutation(num_nodes))
        func = MultiObjectivePathLength(obj1, obj2)

        heur1 = obj1.heuristic
        heur2 = obj2.heuristic
        heur = func.heuristic

        self.assertEqual(len(heur), 2)
        self.assertTrue((heur[0] == heur1[0]).all())
        self.assertTrue((heur[1] == heur2[0]).all())


if __name__ == '__main__':
    unittest.main()
