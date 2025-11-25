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

"""Traveling salesman problem related fitness functions."""

from __future__ import annotations

from typing import Tuple, Optional, Union, TextIO
from collections.abc import Sequence
from copy import deepcopy
from os import PathLike
import random
from functools import partial
import re
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
from scipy.spatial.distance import pdist, squareform

from culebra.abc import Fitness
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction
from culebra.fitness_function import MultiObjectiveFitnessFunction
from culebra.fitness_function.tsp.abc import TSPFitnessFunction
from culebra.checker import (
    check_instance,
    check_sequence,
    check_matrix,
    check_int
)
from culebra.solution.tsp import Species, Solution

FilePath = Union[str, "PathLike[str]"]
Url = str


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class PathLength(SingleObjectiveFitnessFunction, TSPFitnessFunction):
    """Base fitness function for TSP problems.

    Evaluate the length of the path encoded by a solution.
    """

    def __init__(
        self, distance_matrix: Sequence[Sequence[float]],
        index: Optional[int] = None
    ) -> None:
        """Construct a fitness function.

        :param distance_matrix: Distance matrix.
        :type distance_matrix:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]]
        :param index: Index of this objective when it is used for
            multi-objective fitness functions, optional
        :type index: int

        :raises ValueError: If any element in *distance_matrix* is not a
            float number
        :raises ValueError: If *distance_matrix* has not an homogeneous
            shape
        :raises ValueError: If *distance_matrix* has not two dimensions
        :raises ValueError: If *distance_matrix* is not square
        :raises ValueError: If any element in *distance_matrix* is negative
        :raises TypeError: If *index* is not an integer number
        :raises ValueError: If *index* is not positive
        """
        # Init the superclass
        super().__init__(index)
        self.distance = distance_matrix

    @property
    def obj_weights(self) -> Tuple[int, ...]:
        """Objective weights.

        Minimize the path length.

        :return: (-1, )
        :rtype: tuple[int]
        """
        return (-1.0, )

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :return: ("Len",)
        :rtype: tuple[str]
        """
        return ("Len",)

    @property
    def distance(self) -> np.ndarray:
        """Distance matrix.
        
        :rtype: ~numpy.ndarray

        :setter: Set an array-like object as the new distance matrix
        :param value: The new distance matrix
        :type value:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]]
        :raises ValueError: If any element in *value* is not a float number
        :raises ValueError: If *value* has not an homogeneous shape
        :raises ValueError: If *value* has not two dimensions
        :raises ValueError: If *value* is not square
        :raises ValueError: If any element in *value* is negative
        """
        return self._distance

    @distance.setter
    def distance(
        self,
        value: Sequence[Sequence[float], ...] | None
    ) -> None:
        """Set new distance matrix.

        :param value: New distance matrix
        :type value:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]]
        :raises ValueError: If any element in *value* is not a float number
        :raises ValueError: If *value* has not an homogeneous shape
        :raises ValueError: If *value* has not two dimensions
        :raises ValueError: If *value* is not square
        :raises ValueError: If any element in *value* is negative
        """
        # Check the values
        self._distance = check_matrix(
            value, "distance matrix", square=True, ge=0
        )

        # Check the shape
        the_shape = self._distance.shape
        if the_shape[0] == 0:
            raise ValueError("A distance matrix can not be empty")

    @property
    def num_nodes(self) -> int:
        """Number of nodes of the problem graph.

        :rtype: int
        """
        return self.distance.shape[0]

    @property
    def heuristic(self) -> Sequence[np.ndarray, ...]:
        """Heuristic matrices.

        :return: A sequence of heuristic matrices. One for each objective.
            Arcs from a node to itself have a heuristic value of 0. For the
            rest of arcs, the reciprocal of their nodes distance is used as
            heuristic
        :rtype: ~collections.abc.Sequence[~numpy.ndarray]
        """
        with np.errstate(divide='ignore'):
            heur = np.where(
                self.distance != 0.,
                1 / self.distance,
                0.
            )

        for node in range(self.num_nodes):
            heur[node][node] = 0

        return (heur,)

    def greedy_solution(self, species: Species) -> Solution:
        """Return a greddy solution for the problem.

        :param species: Species constraining the problem solutions
        :type species: ~culebra.solution.tsp.Species
        :raises TypeError: If *species* is not an instance of
            :class:`~culebra.solution.tsp.Species`

        :return: The greddy solution
        :rtype: ~culebra.solution.tsp.Solution
        """
        # Check the species
        species = check_instance(species, "species", Species)
        
        # Maximum solution's length
        max_len = species.num_nodes - len(species.banned_nodes)

        # Current path
        current_path = []

        # If the path can be constructed ...
        if max_len > 0:
            # Get the heuristic matrix
            the_heuristic = self.heuristic[0]

           # Ignore banned nodes and arcs from a node to itself
            for node in range(species.num_nodes):
                the_heuristic[node][node] = 0
                for ignored in species.banned_nodes:
                    the_heuristic[node][ignored] = 0
                    the_heuristic[ignored][node] = 0

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
        sol = Solution(species, self.fitness_cls, current_path)
        self.evaluate(sol)
        return sol

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.tsp.Solution
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: int
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        """
        path_len = 0
        if (len(sol.path) > 0):
            org = sol.path[-1]
            for dest in sol.path:
                path_len += self.distance[org][dest]
                org = dest

        # Set the path length
        sol.fitness.update_value(path_len, self.index)

        return sol.fitness

    @classmethod
    def fromPath(cls, path: Sequence[int, ...]) -> PathLength:
        """Create an instance from an optimum path.

        This class method has been designed for testing purposes.

        :param path: An optimum path (a node permutation)
        :type path: ~collections.abc.Sequence[int]
        :return: The fitness function
        :rtype: ~culebra.fitness_function.tsp.PathLength
        :raises ValueError: If *path* is not a
            :class:`~collections.abc.Sequence` of :class:`int`
        :raises ValueError: If *path* has any not int element
        :raises ValueError: If *path* is empty
        :raises ValueError: If *path* has negative values
        :raises ValueError: If *path* has loops
        :raises ValueError: If *path* has missing nodes
        """
        # Check each path
        path = check_sequence(
            path,
            "path",
            item_checker=partial(check_int, ge=0)
        )
        if len(path) == 0:
            raise ValueError("Empty paths are not allowed")

        path = np.asarray(path, dtype=int)

        if len(path) != len(np.unique(path)):
            raise ValueError(
                f"Invalid path. Repeated nodes: {path}"
            )

        if np.min(path) != 0:
            raise ValueError(
                "Invalid path. The minimum value is not 0: {path}"
            )

        if np.max(path) != len(path) - 1:
            raise ValueError(
                f"Invalid path. Missing nodes: {path}"
            )

        num_nodes = len(path)

        # Create the distance matrix
        distance = np.full((num_nodes, num_nodes), 10)

        for i in range(num_nodes):
            distance[i][i] = 0

        # Fill the distance matrix with the path
        org = path[-1]
        for dest in path:
            distance[org][dest] = 1
            distance[dest][org] = 1
            org = dest

        # Return the fitness function
        return cls(distance)

    @classmethod
    def fromTSPLib(
        cls, filepath_or_buffer: FilePath | Url | TextIO
    ) -> PathLength:
        """Generate a fitness function from a TSPLib file.

        :param filepath_or_buffer: File path, url or buffer
        :type filepath_or_buffer: str | ~os.PathLike[str] | ~typing.TextIO
        :return: The fitness function
        :rtype: ~culebra.fitness_function.tsp.PathLength
        :raises RuntimeError: If the filepath or buffer can not be open
        :raises RuntimeError: If an unsupported or incorrect feature is found
            while parsing the filepath or buffer
        """

        def open_file(filepath_or_buffer: FilePath | Url | TextIO) -> TextIO:
            """Open the TSPLib file.

            :param filepath_or_buffer: A file path, url or buffer
            :type filepath_or_buffer: str | ~os.PathLike[str] | ~typing.TextIO
            :return: A file-like object
            :rtype: ~typing.TextIO
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
            :type current_line: str
            :return: The problem type
            :rtype: str
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
            :type current_line: str
            :return: The problem's dimension
            :rtype: int
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
            :type current_line: str
            :return: The problem's edge weight type
            :rtype: str
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
            :type current_line: str
            :return: The problem's edge weight format
            :rtype: str
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
            :type problem_params: dict
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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

        def parse_matrix_values(
            expected_num_values: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse the values of a distane matrix.

            :param expected_num_values: Expected number of values
            :type expected_num_values: int
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: The values
            :rtype: ~numpy.ndarray
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
            :type dimesion: int
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
            """
            expected_num_values = dimension * dimension
            matrix_values = parse_matrix_values(expected_num_values, buffer)
            return np.reshape(matrix_values, (dimension, dimension))

        def parse_upper_row_matrix(
            dimension: int, buffer: TextIO
        ) -> np.ndarray:
            """Parse an upper row matrix.

            :param dimension: Matrix dimension
            :type dimesion: int
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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
            :type dimesion: int
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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
            :type dimesion: int
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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
            :type dimesion: int
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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
            :type problem_params: dict
            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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

        def parse_file(buffer: TextIO) -> np.ndarray:
            """Parse a TSPLib file.

            :param buffer: A file-like object
            :type buffer: ~typing.TextIO
            :return: A distance matrix
            :rtype: ~numpy.ndarray
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

        # Parse all the provided filepaths or buffers
        with open_file(filepath_or_buffer) as buffer:
            distance = parse_file(buffer)

        return cls(distance)

    def __copy__(self) -> PathLength:
        """Shallow copy the fitness function.
        :return:  The copied fitness function
        :rtype: ~culebra.fitness_function.tsp.PathLength
        """
        cls = self.__class__
        result = cls(self.distance)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> PathLength:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: dict
        :return:  The copied fitness function
        :rtype: ~culebra.fitness_function.tsp.PathLength
        """
        cls = self.__class__
        result = cls(self.distance)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the fitness function.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, (self.distance, ), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> PathLength:
        """Return a path length from a state.

        :param state: The state
        :type state: ~dict
        :return: The object
        :rtype: ~culebra.fitness_function.tsp.PathLength
        """
        obj = cls(state['_distance'])
        obj.__setstate__(state)
        return obj


class MultiObjectivePathLength(
    MultiObjectiveFitnessFunction,
    TSPFitnessFunction
):
    """Base fitness function for multi-objctive TSP problems.

    Evaluate multiple lengths of the path encoded by a solution.
    """

    def __init__(
        self,
        *objectives: Tuple[PathLength, ...]
    ) -> None:
        """Construct a multi-objective TSP fitness function.

        :param objectives: Different objectives for this fitness
            function
        :type objectives: tuple[~culebra.fitness_function.tsp.PathLength]
        """
        for obj in objectives:
            if not isinstance(obj, PathLength):
                raise ValueError(
                    "All the objectives must be PathLength instances"
                )
        if len(objectives) > 0:
            num_nodes = objectives[0].num_nodes
            for obj in objectives[1:]:
                if obj.num_nodes != num_nodes:
                    raise ValueError(
                        "All the objectives must have the same number of nodes"
                    )

        super().__init__(*objectives)

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :rtype: tuple[str]
        """
        suffix_len = len(str(self.num_obj-1))

        return tuple(
            f"{obj.obj_names[0]}_{i:0{suffix_len}d}"
            for i, obj in enumerate(self.objectives)
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes of the problem graph.

        :rtype: int
        """
        return self.objectives[0].num_nodes

    @property
    def heuristic(self) -> Sequence[np.ndarray, ...]:
        """Heuristic matrices.

        :return: A sequence of heuristic matrices. One for each objective.
            Arcs from a node to itself have a heuristic value of 0. For the
            rest of arcs, the reciprocal of their nodes distance is used as
            heuristic
        :rtype: ~collections.abc.Sequence[~numpy.ndarray]
        """
        heuristics = ()

        for obj in self.objectives:
            heuristics += obj.heuristic

        return heuristics


# Exported symbols for this module
__all__ = [
    'PathLength',
    'MultiObjectivePathLength'
]
