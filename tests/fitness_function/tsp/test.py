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

from culebra.abc import Species as BaseSpecies
from culebra.solution.tsp import Species, Solution
from culebra.fitness_function.tsp import SinglePathLength, DoublePathLength


class PathLengthTester(unittest.TestCase):
    """Test the :py:class:`~culebra.fitness_function.tsp.PathLength` class."""

    def test_init(self):
        """Test the constructor."""
        # Try an invalid number of distance matrices. Should fail
        invalid_number_of_matrices = [
            (),
            (
                [
                    [0, 1],
                    [1, 0]
                ],
            ),
            (
                [
                    [0, 1],
                    [1, 0]
                ],
                [
                    [0, 2],
                    [2, 0]
                ],
                [
                    [0, 3],
                    [3, 0]
                ],
            )
        ]
        for matrices in invalid_number_of_matrices:
            with self.assertRaises(ValueError):
                DoublePathLength(*matrices)

        # Try an invalid values within a matrix. Should fail
        invalid_matrix_values = [
            (
                [
                    [max, 1],
                    [1, 0]
                ],
            ),
            (
                [
                    [0, 'a'],
                    [1, 0]
                ],
            ),
            (
                [
                    [0, 1],
                    [1, int]
                ],
            )
        ]
        for matrices in invalid_matrix_values:
            with self.assertRaises(ValueError):
                SinglePathLength(matrices)

        # Try non homogeneous matrices. Should fail
        non_homogeneous_matrices = [
            (
                [
                    [0],
                    [1, 0]
                ],
            ),
            (
                [
                    [0, 1],
                    [1]
                ],
            )
        ]
        for matrices in non_homogeneous_matrices:
            with self.assertRaises(ValueError):
                SinglePathLength(*matrices)

        # Try matrices ith an invalid number of dimensions. Should fail
        invalid_dimension_matrices = [
            (
                [1, 0],
            ),
            (
                [
                    [
                        [0, 1],
                        [1, 0]
                    ],
                    [
                        [0, 1],
                        [1, 0]
                    ]
                ],
            )
        ]
        for matrices in invalid_dimension_matrices:
            with self.assertRaises(ValueError):
                SinglePathLength(*matrices)

        # Try non square matrices. Should fail
        non_square_matrices = [
            [
                [
                    [0, 1, 2],
                    [1, 0, 4]
                ]
            ],
            [
                [
                    [0, 1],
                    [1, 2],
                    [3, 1],
                    [2, 5]
                ]
            ]
        ]
        for matrices in non_square_matrices:
            with self.assertRaises(ValueError):
                SinglePathLength(*matrices)

        # Try matrices with negative values. Should fail
        negative_values = [
            [
                [
                    [-1, 1],
                    [1, 0]
                ]
            ],
            [
                [
                    [0, -1],
                    [1, 0]
                ]
            ],
            [
                [
                    [0, 1],
                    [-1, 0]
                ]
            ],
            [
                [
                    [0, 1],
                    [1, -1]
                ]
            ]
        ]
        for matrices in negative_values:
            with self.assertRaises(ValueError):
                SinglePathLength(*matrices)

        # Try matrices with different shapes. Should fail
        different_shapes_matrices = [
            [
                [0, 1],
                [1, 0]
            ],
            [
                [0, 1, 2],
                [1, 0, 3],
                [2, 3, 0]
            ]
        ]
        with self.assertRaises(ValueError):
            DoublePathLength(*different_shapes_matrices)

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
                SinglePathLength(*matrices)

        # Try valid distance matrices
        valid_matrices = [
            [
                [
                    [0, 1],
                    [1, 0]
                ],
                [
                    [0, 2],
                    [2, 0]
                ]
            ],
            [
                [
                    [0, 2, 3],
                    [2, 0, 6],
                    [3, 6, 0]
                ],
                [
                    [0, 4, 7],
                    [4, 0, 8],
                    [7, 8, 0]
                ]
            ]
        ]
        for matrices in valid_matrices:
            fitness_func = DoublePathLength(*matrices)
            for dist1, dist2 in zip(matrices, fitness_func.distance):
                self.assertTrue(
                    np.all(
                        np.asarray(dist1, dtype=float) == dist2
                    )
                )

    def test_num_nodes(self):
        """Test the num_nodes property."""
        max_num_nodes = 10
        for num_nodes in range(2, max_num_nodes):
            fitness_func = SinglePathLength(np.ones((num_nodes, num_nodes)))
            self.assertEqual(fitness_func.num_nodes, num_nodes)

    def test_heuristic(self):
        """Test the heuristic method."""
        distance_matrices = (
            [
                [0, 1, 2, 3, 4, 5],
                [1, 0, 6, 7, 8, 9],
                [2, 6, 0, 1, 2, 3],
                [3, 7, 1, 0, 4, 5],
                [4, 8, 2, 4, 0, 6],
                [5, 9, 3, 5, 6, 0]
            ],
            [
                [0, 4, 1, 5, 8, 9],
                [4, 0, 2, 6, 3, 1],
                [1, 2, 0, 4, 5, 7],
                [5, 6, 4, 0, 2, 6],
                [8, 3, 5, 2, 0, 4],
                [9, 1, 7, 6, 4, 0]
            ]
        )

        fitness_func = DoublePathLength(*distance_matrices)

        # Try an invalid species. Should fail
        species = BaseSpecies()
        with self.assertRaises(TypeError):
            fitness_func.heuristic(species)

        banned_nodes = [0, fitness_func.num_nodes-1]
        species = Species(
            fitness_func.num_nodes,
            banned_nodes=banned_nodes
        )

        heuristic = fitness_func.heuristic(species)
        self.assertIsInstance(heuristic, Sequence)

        # Check the heuristic_matrices
        for (
            heur,
            dist
        ) in zip(
            heuristic,
            fitness_func.distance
        ):
            for i in range(species.num_nodes):
                for j in range(species.num_nodes):
                    if i == j or i in banned_nodes or j in banned_nodes:
                        self.assertEqual(heur[i][j], 0)
                    else:
                        self.assertAlmostEqual(
                            heur[i][j],
                            1/dist[i][j]
                        )

    def test_greddy_solution(self):
        """Test the greedy solution method."""
        distance_matrices = (
            [
                [0, 1, 2],
                [1, 0, 6],
                [2, 6, 0]
            ],
            [
                [0, 4, 1],
                [4, 0, 2],
                [1, 2, 0]
            ]
        )

        fitness_func = DoublePathLength(*distance_matrices)
        num_nodes = fitness_func.num_nodes
        banned_nodes = [0, fitness_func.num_nodes-1]
        species = Species(
            fitness_func.num_nodes,
            banned_nodes=banned_nodes
        )

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
        distance_matrices = (
            [
                [0, 1, 2],
                [1, 0, 6],
                [2, 6, 0]
            ],
            [
                [0, 4, 1],
                [4, 0, 2],
                [1, 2, 0]
            ]
        )

        fitness_func = DoublePathLength(*distance_matrices)
        species = Species(fitness_func.num_nodes)

        # Try valid permutations
        times = 100
        for _ in repeat(None, times):
            for _ in repeat(None, times):
                sol = Solution(
                    species,
                    DoublePathLength.Fitness,
                    np.random.permutation(fitness_func.num_nodes)
                )
                fitness = fitness_func.evaluate(sol)
                self.assertIsInstance(fitness, tuple)
                self.assertEqual(len(fitness), 2)
                for i in range(fitness_func.num_obj):
                    self.assertGreater(fitness[i], 0)

        # Try an unfeasible solution
        banned_nodes = list(node for node in range(fitness_func.num_nodes))
        species = Species(fitness_func.num_nodes, banned_nodes=banned_nodes)
        sol = fitness_func.greedy_solution(species)
        self.assertGreaterEqual(
            fitness_func.evaluate(sol), (0,)
        )

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
        for paths in invalid_path_sequence_lengths:
            with self.assertRaises(ValueError):
                DoublePathLength.fromPath(*paths)

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
        for paths in invalid_path_types:
            with self.assertRaises(TypeError):
                SinglePathLength.fromPath(*paths)

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
        for paths in invalid_path_values:
            with self.assertRaises(ValueError):
                SinglePathLength.fromPath(*paths)

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
        for paths in invalid_paths:
            with self.assertRaises(ValueError):
                SinglePathLength.fromPath(*paths)

        # Try paths of different lengths. Should fail
        different_length_paths = [
            [0, 1, 2], [0, 1, 2, 3]
        ]

        with self.assertRaises(ValueError):
            DoublePathLength.fromPath(*different_length_paths)

        # Try valid permutations
        times = 100
        num_nodes = 5
        for _ in repeat(None, times):
            paths = [
                np.random.permutation(num_nodes),
                np.random.permutation(num_nodes)
            ]

            fitness_func = DoublePathLength.fromPath(*paths)

            # Check the distance
            for path, dist in zip(paths, fitness_func.distance):
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
        # Try invalid file sequence lengths. Should fail
        invalid_file_sequence_lengths = [
            [],
            [
                "problem1.tsp"
            ],
            [
                "problem1.tsp",
                "problem2.tsp",
                "problem3.tsp"
            ]
        ]
        for files in invalid_file_sequence_lengths:
            with self.assertRaises(ValueError):
                DoublePathLength.fromTSPLib(*files)

        # Try invalid buffer types. Should fail
        invalid_buffer_types = [
            [len],
            [Species],
            [4]
        ]
        for buffers in invalid_buffer_types:
            with self.assertRaises(AttributeError):
                SinglePathLength.fromTSPLib(*buffers)

        # Try invalid file paths. Should fail
        invalid_paths = [
            ["hi"],
            ["/dev/hi"],
            ["https://none.none/none"]
        ]
        for paths in invalid_paths:
            with self.assertRaises(RuntimeError):
                SinglePathLength.fromTSPLib(*paths)

        # Try an invalid problem type. Should fail
        buffer = """TYPE: new"""
        with self.assertRaises(RuntimeError):
            SinglePathLength.fromTSPLib(io.StringIO(buffer))

        # Try invalid dimensions. Should fail
        invalid_buffers = [
            "DIMENSION: new",
            "DIMENSION: 1.4",
            "DIMENSION: -2",
            "DIMENSION: 0"
            ]
        for buffer in invalid_buffers:
            with self.assertRaises(RuntimeError):
                SinglePathLength.fromTSPLib(io.StringIO(buffer))

        # Try an invalid edge weight type. Should fail
        buffer = "EDGE_WEIGHT_TYPE: new"
        with self.assertRaises(RuntimeError):
            SinglePathLength.fromTSPLib(io.StringIO(buffer))

        # Try an invalid edge weight format. Should fail
        buffer = "EDGE_WEIGHT_FORMAT: new"
        with self.assertRaises(RuntimeError):
            SinglePathLength.fromTSPLib(io.StringIO(buffer))

    def test_from_tsplib_node_coord_section(self):
        """Test the node coord section parsing of fromTSPLib."""
        # Try a buffer with missing DIMENSION. Should fail
        buffer = "NODE_COORD_SECTION"
        with self.assertRaises(RuntimeError):
            SinglePathLength.fromTSPLib(io.StringIO(buffer))

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
                SinglePathLength.fromTSPLib(io.StringIO(buffer))

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
        fitness_func = SinglePathLength.fromTSPLib(io.StringIO(buffer))
        self.assertEqual(fitness_func.num_nodes, 3)
        self.assertTrue(
            np.all(
                fitness_func.distance[0] ==
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
        fitness_func = SinglePathLength.fromTSPLib("test.tsp")
        self.assertEqual(fitness_func.num_nodes, 5)
        self.assertTrue(
            np.all(
                fitness_func.distance[0] ==
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
        fitness_func = SinglePathLength.fromTSPLib(
            "https://raw.githubusercontent.com/mastqe/tsplib/master/"
            "kroA100.tsp"
        )
        expected_num_nodes = 100
        self.assertEqual(fitness_func.num_nodes, expected_num_nodes)
        self.assertEqual(
            fitness_func.distance[0].shape,
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
            SinglePathLength.fromTSPLib(io.StringIO(buffer))

        # Try a full matrix with an excessive number of values. Should fail
        buffer += "1 2 3 4"
        with self.assertRaises(RuntimeError):
            SinglePathLength.fromTSPLib(io.StringIO(buffer))

        # Try a correct full matrix
        fitness_func = SinglePathLength.fromTSPLib(
            "https://raw.githubusercontent.com/mastqe/tsplib/master/bays29.tsp"
        )
        expected_num_nodes = 29
        self.assertEqual(fitness_func.num_nodes, expected_num_nodes)
        self.assertEqual(
            fitness_func.distance[0].shape,
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
        upper_row_fitness_func = SinglePathLength.fromTSPLib(
            io.StringIO(upper_row_buffer)
        )
        expected_num_nodes = 5
        self.assertEqual(
            upper_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            upper_row_fitness_func.distance[0].shape,
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
        lower_row_fitness_func = SinglePathLength.fromTSPLib(
            io.StringIO(lower_row_buffer)
        )
        self.assertEqual(
            lower_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            lower_row_fitness_func.distance[0].shape,
            (expected_num_nodes, expected_num_nodes)
        )
        self.assertTrue(
            np.all(
                upper_row_fitness_func.distance[0] ==
                lower_row_fitness_func.distance[0]
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
        upper_diag_row_fitness_func = SinglePathLength.fromTSPLib(
            io.StringIO(upper_diag_row_buffer)
        )
        expected_num_nodes = 5
        self.assertEqual(
            upper_diag_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            upper_diag_row_fitness_func.distance[0].shape,
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
        lower_diag_row_fitness_func = SinglePathLength.fromTSPLib(
            io.StringIO(lower_diag_row_buffer)
        )
        self.assertEqual(
            lower_diag_row_fitness_func.num_nodes,
            expected_num_nodes
        )
        self.assertEqual(
            lower_diag_row_fitness_func.distance[0].shape,
            (expected_num_nodes, expected_num_nodes)
        )
        self.assertTrue(
            np.all(
                upper_diag_row_fitness_func.distance[0] ==
                lower_diag_row_fitness_func.distance[0]
            )
        )

    def test_copy(self):
        """Test the __copy__ method."""
        num_nodes = 5
        func1 = SinglePathLength.fromPath(np.random.permutation(num_nodes))
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.distance), id(func2.distance))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        num_nodes = 5
        func1 = SinglePathLength.fromPath(np.random.permutation(num_nodes))
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        num_nodes = 5
        func1 = DoublePathLength.fromPath(
            np.random.permutation(num_nodes),
            np.random.permutation(num_nodes)
        )

        pickle_filename = "my_pickle.gz"
        func1.save_pickle(pickle_filename)
        func2 = DoublePathLength.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the pickle file
        remove(pickle_filename)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1: :py:class:`~culebra.fitness_function.tsp.PathLength`
        :param func2: The second fitness function
        :type func2: :py:class:`~culebra.fitness_function.tsp.PathLength`
        """
        # Copies all the levels
        self.assertNotEqual(id(func1), id(func2))
        for dist1, dist2 in zip(func1.distance, func2.distance):
            self.assertNotEqual(id(dist1), id(dist2))
            self.assertTrue((dist1 == dist2).all())

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_nodes = 5
        fitness_func = SinglePathLength.fromPath(
            np.random.permutation(num_nodes)
        )
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


if __name__ == '__main__':
    unittest.main()
