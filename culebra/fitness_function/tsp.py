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

"""Traveling salesman problem related fitness functions.

This sub-module provides fitness functions to solve TSP problems. Currently:

  * :py:class:`~culebra.fitness_function.tsp.PathLength`: Abstract base class
    for TSP problem fitness functions.
  * :py:class:`~culebra.fitness_function.tsp.SinglePathLength`: Single
    objective fitness function for TSP problems.
  * :py:class:`~culebra.fitness_function.tsp.DoublePathLength`: Bi-objective
    fitness function for TSP problems.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, Optional, Union, TextIO
from copy import deepcopy
from os import PathLike
import random
from functools import partial
import re
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
from scipy.spatial.distance import pdist, squareform

from culebra.abc import Fitness, FitnessFunction
from culebra.checker import (
    check_sequence,
    check_instance,
    check_matrix,
    check_int
)
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.solution.tsp import Species, Solution

FilePath = Union[str, "PathLike[str]"]
Url = str


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class PathLength(FitnessFunction):
    """Abstract base class for TSP problem fitness functions.

    Evaluate the length(s) of the path encoded by a solution.
    """

    def __init__(
        self, *distance_matrices: Sequence[Sequence[float], ...]
    ) -> None:
        """Construct a fitness function.

        :param distance_matrices: Distance matrices. One per objective
        :type distance_matrices: :py:class:`~collections.abc.Sequence` of
            two-dimensional array-like objects

        :type: A :py:class:`~collections.abc.Sequence` of two-dimensional
            array-like objects
        :raises TypeError: If *distance* is not a
            :py:class:`~collections.abc.Sequence` of array-like objects
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any array-like object has not an homogeneous
            shape
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object is not square
        :raises ValueError: If any element in any array-like object is negative
        :raises ValueError: If the number of array-like objects provided is
            different from
            :py:attr:`~culebra.fitness_function.tsp.PathLength.num_obj`
        :raises ValueError: If the array-like objects have different shapes
        """
        # Init the superclass
        super().__init__()
        self.distance = distance_matrices

    @property
    def num_nodes(self) -> int:
        """Return the problem graph's number of nodes.

        :return: The problem graph's number of nodes
        :rtype: :py:class:`int`
        """
        return self.distance[0].shape[0]

    @property
    def distance(self) -> Sequence[np.ndarray, ...]:
        """Get and set the distance matrices.

        :getter: Return the distance matrices
        :setter: Set new distance matrices. One per objective
        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        :raises TypeError: If a :py:class:`~collections.abc.Sequence` of
            array-like objects is not provided
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any array-like object has not an homogeneous
            shape
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object is not square
        :raises ValueError: If any element in any array-like object is negative
        :raises ValueError: If the number of array-like objects is different
            from :py:attr:`~culebra.fitness_function.tsp.PathLength.num_obj`
        :raises ValueError: If the array-like objects have different shapes
        """
        return self._distance

    @distance.setter
    def distance(
        self,
        values: Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...] | None
    ) -> None:
        """Set new distance matrices.

        :param values: New distance matrices. One per objective
        :type: A :py:class:`~collections.abc.Sequence` of two-dimensional
            array-like objects
        :raises TypeError: If *values* is not a
            :py:class:`~collections.abc.Sequence` of array-like objects
        :raises TypeError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any array-like object has not an homogeneous
            shape
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object is not square
        :raises ValueError: If any element in any array-like object is negative
        :raises ValueError: If *values*'s length is different from
            :py:attr:`~culebra.fitness_function.tsp.PathLength.num_obj`
        :raises ValueError: If the array-like objects have different shapes
        """
        # Check the values
        self._distance = check_sequence(
            values,
            "distance matrices",
            size=self.num_obj,
            item_checker=partial(check_matrix, square=True, ge=0)
        )

        # Check the shape
        the_shape = self._distance[0].shape
        if the_shape[0] == 0:
            raise ValueError("A distance matrix can not be empty")

        for matrix in self._distance:
            if matrix.shape != the_shape:
                raise ValueError(
                    "All the distance matrices must have the same "
                    "shape"
                )

    def heuristic(self, species: Species) -> Sequence[np.ndarray]:
        """Get the heuristic matrices for ACO-based trainers.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.solution.tsp.Species`
        :raises TypeError: If *species* is not an instance of
            :py:class:`~culebra.solution.tsp.Species`

        :return: A sequence of heuristic matrices. One for each objective.
            Arcs involving any banned node or arcs from a node to itself have
            a heuristic value of 0. For the rest of arcs, the reciprocal of
            their nodes distance is used as heuristic
        :rtype: :py:class:`~collections.abc.Sequence` of
            :py:class:`~numpy.ndarray`
        """
        check_instance(species, "species", cls=Species)

        the_heuristics = []

        for dist in self.distance:
            with np.errstate(divide='ignore'):
                heur = np.where(
                    dist != 0.,
                    1 / dist,
                    0.
                )

            # Ignore banned nodes and arcs from a node to itself
            for node in range(species.num_nodes):
                heur[node][node] = 0
                for ignored in species.banned_nodes:
                    heur[node][ignored] = 0
                    heur[ignored][node] = 0

            the_heuristics.append(heur)

        return the_heuristics

    def greedy_solution(self, species: Species) -> Solution:
        """Return a greddy solution for the problem.

        :param species: Species constraining the problem solutions
        :type species: :py:class:`~culebra.solution.tsp.Species`
        :raises TypeError: If *species* is not an instance of
            :py:class:`~culebra.solution.tsp.Species`

        :return: The greddy solution
        :rtype: :py:class:`~culebra.solution.tsp.Solution`
        """
        # Maximum solution's length
        max_len = species.num_nodes - len(species.banned_nodes)

        # Current path
        current_path = []

        # If the path can be constructed ...
        if max_len > 0:
            the_heuristic = np.prod(
                np.stack(self.heuristic(species)),
                axis=0
            )

            # Start with a feasible node
            current_node = random.randint(0, species.num_nodes-1)
            while (current_node in species.banned_nodes):
                current_node = random.randint(0, species.num_nodes-1)
            current_path.append(current_node)

            # Complete the greedy path
            while len(current_path) < max_len:
                current_heuristic = the_heuristic[current_node]
                current_heuristic[current_path] = 0
                current_node = np.argwhere(
                    current_heuristic == np.max(current_heuristic)
                ).flatten()[0]
                current_path.append(current_node)

        # Construct the solution form the greedy path
        sol = Solution(species, self.Fitness, current_path)
        sol.fitness.values = self.evaluate(sol)
        return sol

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.solution.tsp.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Return the solution's paths lengths
        path_lens = [0] * self.num_obj
        if (len(sol.path) > 0):
            org = sol.path[-1]
            for dest in sol.path:
                for index, dist in enumerate(self.distance):
                    path_lens[index] += dist[org][dest]
                org = dest

        return tuple(path_lens)

    @classmethod
    def fromPath(cls, *paths: Sequence[int, ...]) -> PathLength:
        """Create an instance from a sequence of optimum paths.

        This class method has been designed for testing purposes.

        :param paths: Sequence of optimum paths. Each path is a node
            permutation
        :type paths: :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises TypeError: If any path in *paths* is not a
            :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If the number of paths provided is different from
            :py:attr:`~culebra.fitness_function.tsp.PathLength.num_obj`
        :raises ValueError: If the array-like objects have different shapes
        :raises ValueError: If any path in *paths* has any not :py:class:`int`
            element
        :raises ValueError: If any path in *paths* is empty
        :raises ValueError: If any path in *paths* has negative values
        :raises ValueError: If any path in *paths* has loops
        :raises ValueError: If any path in *paths* has missing nodes
        """
        # Check path's type
        the_paths = check_sequence(
            paths, "paths",
            size=len(cls.Fitness.weights)
        )

        # Check each path
        num_nodes = None
        the_distances = []
        for current_path in the_paths:
            current_path = check_sequence(
                current_path,
                "path",
                item_checker=partial(check_int, ge=0)
            )
            if len(current_path) == 0:
                raise ValueError("Empty paths are not allowed")

            current_path = np.asarray(current_path, dtype=int)

            if len(current_path) != len(np.unique(current_path)):
                raise ValueError(
                    f"Invalid path. Repeated nodes: {current_path}"
                )

            if np.min(current_path) != 0:
                raise ValueError(
                    "Invalid path. The minimum value is not 0: "
                    f"{current_path}"
                )

            if np.max(current_path) != len(current_path) - 1:
                raise ValueError(
                    f"Invalid path. Missing nodes: {current_path}"
                )

            if num_nodes is None:
                num_nodes = len(current_path)
            elif len(current_path) != num_nodes:
                raise ValueError(
                    "All the paths must have the same number of nodes"
                )

            # Create the distance matrix
            distance = np.full((num_nodes, num_nodes), 10)

            # Distance from a node to itself are banned
            for node in range(num_nodes):
                distance[node][node] = 0

            # Fix the path in the distance matrix
            org = current_path[-1]
            for dest in current_path:
                distance[org][dest] = 1
                distance[dest][org] = 1
                org = dest

            the_distances.append(distance)

        # Return the fitness function
        return cls(*the_distances)

    @classmethod
    def fromTSPLib(
        cls, *filepaths_or_buffers: FilePath | Url | TextIO
    ) -> PathLength:
        """Generate a fitness function from a sequence of TSPLib files.

        :param filepaths_or_buffers: Sequence of file paths, urls or buffers
        :type filepaths_or_buffers: :py:class:`~collections.abc.Sequence` of
            path-like objects, urls or file-like objects

        :raises ValueError: If the number of filepaths or buffers provided is
            different from
            :py:attr:`~culebra.fitness_function.tsp.PathLength.num_obj`
        :raises RuntimeError: If any filepath or buffer can not be open
        :raises RuntimeError: If an unsupported or incorrect feature is found
            while parsing any filepath or buffer
        """

        def open_file(filepath_or_buffer: FilePath | Url | TextIO) -> TextIO:
            """Open the TSPLib file.

            :param filepath_or_buffer: A file path, url or buffer
            :type filepath_or_buffer: path-like object, url or file-like
                object
            :return: A file-like object
            :rtype: :py:class:`TextIO`
            :raises RuntimeError: If the filepath or buffer can not be open
            """
            # Check if file is a file-like object
            if (
                hasattr(filepath_or_buffer, "read") and
                hasattr(filepath_or_buffer, "write") and
                hasattr(filepath_or_buffer, "__iter__")
            ):
                return filepath_or_buffer

            # Check if filepath_or_buffer is an URL
            split_url = urlsplit(filepath_or_buffer)
            try:
                if bool(split_url.scheme and split_url.netloc):
                    return urlopen(filepath_or_buffer)
                else:
                    return open(filepath_or_buffer, 'rt')
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to open {filepath_or_buffer}'
                ) from exc

        def parse_type(current_line: str) -> str:
            """Parse the problem type.

            Only 'TSP' problems are supported by the moment.

            :param current_line: Current line
            :type current_line: :py:class:`str`
            :return: The problem type
            :rtype: :py:class:`str`
            :raises RuntimeError: If an unsupported problem type is found
            """
            supported_types = ('TSP', )

            parts = current_line.split(":")
            problem_type = parts[1].strip()
            if problem_type not in supported_types:
                raise RuntimeError(
                    f"Unsupported TYPE: {problem_type}"
                )
            return problem_type

        def parse_dimension(current_line: str) -> int:
            """Parse the dimension (number of nodes).

            :param current_line: Current line
            :type current_line: :py:class:`str`
            :return: The problem's dimension
            :rtype: :py:class:`int`
            """
            parts = current_line.split(":")
            try:
                num_nodes = int(parts[1])
                if num_nodes <= 0:
                    raise RuntimeError(f"Invalid DIMENSION: {num_nodes}")
                return num_nodes
            except Exception as e:
                raise RuntimeError(e.args[0])

        def parse_edge_weight_type(current_line: str) -> str:
            """Parse the edge weight type.

            Only 'EXPLICIT', 'EUC_2D', 'EUC_3D', 'MAN_2D', and 'MAN_3D' are
            supported by the moment.

            :param current_line: Current line
            :type current_line: :py:class:`str`
            :return: The problem's edge weight type
            :rtype: :py:class:`str`
            :raises RuntimeError: If an unsupported edge weight type is found
            """
            supported_edge_weight_types = (
                'EXPLICIT', 'EUC_2D', 'EUC_3D', 'MAN_2D', 'MAN_3D'
            )

            parts = current_line.split(":")
            edge_weight_type = parts[1].strip()
            if edge_weight_type not in supported_edge_weight_types:
                raise RuntimeError(
                    f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}"
                )
            return edge_weight_type

        def parse_edge_weight_format(current_line: str) -> str:
            """Parse the edge weight format.

            Only 'FULL_MATRIX', 'UPPER_ROW', 'LOWER_ROW', 'UPPER_DIAG_ROW',
            'LOWER_DIAG_ROW', 'UPPER_COL', 'LOWER_COL', 'UPPER_DIAG_COL', and
            'LOWER_DIAG_COL' are supported by the moment.

            :param current_line: Current line
            :type current_line: :py:class:`str`
            :return: The problem's edge weight format
            :rtype: :py:class:`str`
            :raises RuntimeError: If an unsupported edge weight format is found
            """
            supported_edge_weight_formats = (
                'FULL_MATRIX', 'UPPER_ROW', 'LOWER_ROW', 'UPPER_DIAG_ROW',
                'LOWER_DIAG_ROW', 'UPPER_COL', 'LOWER_COL', 'UPPER_DIAG_COL',
                'LOWER_DIAG_COL'
            )

            parts = current_line.split(":")
            edge_weight_format = parts[1].strip()
            if edge_weight_format not in supported_edge_weight_formats:
                raise RuntimeError(
                    f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}"
                )
            return edge_weight_format

        def parse_node_coord_section(
            problem_params: dict, buffer: TextIO
        ) -> np.ndarray:
            """Parse the node coordinates section.

            :param problem_params: Problem parameters
            :type problem_params: :py:class:`dict`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            supported_distance_metrics = {
                "EUC_2D": "euclidean",
                "EUC_3D": "euclidean",
                "MAN_2D": "cityblock",
                "MAN_3D": "cityblock"
            }
            dimension = problem_params['DIMENSION']
            edge_weight_type = problem_params['EDGE_WEIGHT_TYPE']
            if edge_weight_type in supported_distance_metrics:
                distance_metric = supported_distance_metrics[edge_weight_type]
            else:
                raise RuntimeError(
                    "Unsupported EDGE_WEIGHT_TYPE for the "
                    f"NODE_COORD_SECTION: {edge_weight_type}"
                )

            node_index = 1
            node_coords = []
            for line in buffer:
                # Convert the line to a string if necesary
                if isinstance(line, bytes):
                    line = line.decode('UTF-8')
                parts = re.split("\\s+", line.strip())
                expected_index = parts[0]
                if len(parts[0]) > 0:
                    if int(expected_index) == node_index:
                        node_coords.append(
                            [float(coord) for coord in parts[1:]]
                        )
                        node_index += 1
                    else:
                        raise RuntimeError(
                            "Missing node index within NODE_COORD_SECTION:"
                            f" {node_index}"
                        )
                # Check in finished
                if node_index > dimension:
                    break

            if node_index <= dimension:
                raise RuntimeError(
                    "Missing node index within NODE_COORD_SECTION:"
                    f" {node_index}"
                )

            # Return the distance matrix
            return (
                squareform(
                    pdist(np.asarray(node_coords), distance_metric)
                )
            )

        def parse_matrix_values(expected_num_values: int, buffer: TextIO):
            """Parse the values of a distane matrix.

            :param expected_num_values: Expected number of values
            :type expected_num_values: :py:class:`int`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: The values
            :rtype: :py:class:`numpy.ndarray`
            """
            num_values = 0
            matrix_values = []

            for line in buffer:
                # Convert the line to a string if necesary
                if isinstance(line, bytes):
                    line = line.decode('UTF-8')
                values = [
                    float(item) for item in re.split("\\s+", line.strip())
                ]
                matrix_values.extend(values)
                num_values += len(values)

                # Check in finished
                if num_values >= expected_num_values:
                    break

            if num_values < expected_num_values:
                raise RuntimeError(
                    "Missing values in EDGE_WEIGHT_SECTION"
                )
            if num_values > expected_num_values:
                raise RuntimeError(
                    "Too many values in EDGE_WEIGHT_SECTION"
                )
            return np.asarray(matrix_values, dtype=float)

        def parse_full_matrix(
            dimension: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse a full matrix.

            :param dimension: Matrix dimension
            :type dimesion: :py:class:`int`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            expected_num_values = dimension * dimension
            matrix_values = parse_matrix_values(expected_num_values, buffer)
            return np.reshape(matrix_values, (dimension, dimension))

        def parse_upper_row_matrix(
            dimension: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse an upper row matrix.

            :param dimension: Matrix dimension
            :type dimesion: :py:class:`int`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            the_distances = np.zeros((dimension, dimension))
            expected_num_values = dimension * (dimension - 1) / 2
            matrix_values = parse_matrix_values(expected_num_values, buffer)
            the_distances[np.triu_indices(dimension, 1)] = matrix_values

            return the_distances + the_distances.T

        def parse_upper_diag_row_matrix(
            dimension: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse an upper row matrix with diagonal.

            :param dimension: Matrix dimension
            :type dimesion: :py:class:`int`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            the_distances = np.zeros((dimension, dimension))
            expected_num_values = dimension * (dimension + 1) / 2
            matrix_values = parse_matrix_values(expected_num_values, buffer)
            the_distances[np.triu_indices(dimension)] = matrix_values

            return (
                the_distances + the_distances.T
                - np.diag(np.diag(the_distances))
            )

        def parse_lower_row_matrix(
            dimension: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse a lower row matrix.

            :param dimension: Matrix dimension
            :type dimesion: :py:class:`int`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            the_distances = np.zeros((dimension, dimension))
            expected_num_values = dimension * (dimension - 1) / 2
            matrix_values = parse_matrix_values(expected_num_values, buffer)
            the_distances[np.tril_indices(dimension, -1)] = matrix_values

            return the_distances + the_distances.T

        def parse_lower_diag_row_matrix(
            dimension: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse a lower row matrix with diagonal.

            :param dimension: Matrix dimension
            :type dimesion: :py:class:`int`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            the_distances = np.zeros((dimension, dimension))
            expected_num_values = dimension * (dimension + 1) / 2
            matrix_values = parse_matrix_values(expected_num_values, buffer)
            the_distances[np.tril_indices(dimension)] = matrix_values

            return (
                the_distances + the_distances.T
                - np.diag(np.diag(the_distances))
            )

        def parse_edge_weight_section(
            problem_params: dict, buffer: TextIO
        ) -> np.ndarray:
            """Parse the edge weight section.

            :param problem_params: Problem parameters
            :type problem_params: :py:class:`dict`
            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            supported_matrix_formats = {
                'FULL_MATRIX': parse_full_matrix,
                'UPPER_ROW': parse_upper_row_matrix,
                'LOWER_ROW': parse_lower_row_matrix,
                'UPPER_DIAG_ROW': parse_upper_diag_row_matrix,
                'LOWER_DIAG_ROW': parse_lower_diag_row_matrix,
                'UPPER_COL': parse_lower_row_matrix,
                'LOWER_COL': parse_upper_row_matrix,
                'UPPER_DIAG_COL': parse_lower_diag_row_matrix,
                'LOWER_DIAG_COL': parse_upper_diag_row_matrix
            }

            edge_weight_format = problem_params['EDGE_WEIGHT_FORMAT']
            dimension = problem_params['DIMENSION']
            if edge_weight_format in supported_matrix_formats:
                return supported_matrix_formats[edge_weight_format](
                    dimension,
                    buffer
                )
            else:
                raise RuntimeError(
                    f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}"
                )

        def parse_file(buffer: TextIO) -> dict:
            """Parse a TSPLib file.

            :param buffer: A file-like object
            :type buffer: :py:class:`TextIO`
            :return: A distance matrix
            :rtype: :py:class:`numpy.ndarray`
            """
            parameter_subparsers = {
                "TYPE": parse_type,
                "DIMENSION": parse_dimension,
                "EDGE_WEIGHT_TYPE": parse_edge_weight_type,
                "EDGE_WEIGHT_FORMAT": parse_edge_weight_format
                }
            section_subparsers = {
                "NODE_COORD_SECTION": parse_node_coord_section,
                "EDGE_WEIGHT_SECTION": parse_edge_weight_section
                }

            problem_params = {
                "NAME": None,
                "TYPE": None,
                "DIMENSION": None,
                "EDGE_WEIGHT_TYPE": None,
                "EDGE_WEIGHT_FORMAT": None
                }
            section_name = None

            # Parse the parameters section
            try:
                while True:
                    # Read the next line
                    line = buffer.__next__()

                    # Convert the line to a string if necesary
                    if isinstance(line, bytes):
                        line = line.decode('UTF-8')

                    # Strip the line
                    line = line.strip()
                    param_processed = False
                    if len(line) > 0:
                        for (
                            param_name,
                            param_subparser
                        ) in parameter_subparsers.items():
                            if line.startswith(param_name):
                                problem_params[param_name] = param_subparser(
                                    line
                                )
                                param_processed = True
                                break
                        # Detect a starting section
                        if (
                            param_processed is False and
                            line.endswith("SECTION")
                        ):
                            raise StopIteration
            except StopIteration:
                pass

            # Check the parameters
            if problem_params['TYPE'] is None:
                raise RuntimeError(
                    "Missing TYPE parameter"
                )
            if problem_params['DIMENSION'] is None:
                raise RuntimeError(
                    "Missing DIMENSION parameter"
                )
            if problem_params['EDGE_WEIGHT_TYPE'] is None:
                raise RuntimeError(
                    "Missing EDGE_WEIGHT_TYPE parameter"
                )
            if problem_params['EDGE_WEIGHT_TYPE'] == "EXPLICIT":
                section_name = "EDGE_WEIGHT_SECTION"
                if problem_params['EDGE_WEIGHT_FORMAT'] is None:
                    raise RuntimeError(
                        "Missing EDGE_WEIGHT_FORMAT parameter"
                    )
            else:
                section_name = "NODE_COORD_SECTION"

            # Look for the right section ...
            try:
                while True:
                    section_found = False
                    if line.startswith(section_name):
                        section_found = True
                        raise StopIteration

                    # Read the next line
                    line = buffer.__next__()

                    # Convert the line to a string if necesary
                    if isinstance(line, bytes):
                        line = line.decode('UTF-8')

                    # Strip the line
                    line = line.strip()
            except StopIteration:
                if section_found is False:
                    raise RuntimeError(
                        f"Missing {section_name} section"
                    )

            # Parse the section
            return section_subparsers[section_name](problem_params, buffer)

        # Check filepaths_or_buffers' type
        the_filepaths_or_buffers = check_sequence(
            filepaths_or_buffers, "filepaths or buffers",
            size=len(cls.Fitness.weights)
        )

        # Parse all the provided filepaths or buffers
        the_distances = []
        for filepath_or_buffer in the_filepaths_or_buffers:
            with open_file(filepath_or_buffer) as buffer:
                the_distances.append(parse_file(buffer))

                return cls(*the_distances)

    def __copy__(self) -> PathLength:
        """Shallow copy the fitness function."""
        cls = self.__class__
        result = cls(*self.distance)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> PathLength:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the fitness function
        :rtype:
            :py:class::py:class:`~culebra.fitness_function.tsp.PathLength`
        """
        cls = self.__class__
        result = cls(*self.distance)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the fitness function.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, tuple(self.distance), self.__dict__)


class SinglePathLength(PathLength):
    """Single objective fitness function for TSP problems.

    Evaluate the length of the path encoded by a solution.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.tsp.SinglePathLength.evaluate`
        method within a :py:class:`~culebra.solution.tsp.Solution`.
        """

        weights = (-1.0,)
        """Minimize the path length of a solution."""

        names = ("Len",)
        """Name of the objective."""

        thresholds = (DEFAULT_THRESHOLD,)
        """Similarity threshold for fitness comparisons."""


class DoublePathLength(PathLength):
    """Bi-objective fitness function for TSP problems.

    Evaluate two different lengths of the path encoded by a solution.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.fitness_function.tsp.DoublePathLength.evaluate`
        method within a :py:class:`~culebra.solution.tsp.Solution`.
        """

        weights = (-1.0,) * 2
        """Minimize the path lengths of a solution."""

        names = ("Len0", "Len1")
        """Name of the objectives."""

        thresholds = (DEFAULT_THRESHOLD,) * 2
        """Similarity thresholds for fitness comparisons."""


# Exported symbols for this module
__all__ = [
    'PathLength',
    'SinglePathLength',
    'DoublePathLength'
]
