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
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Unit test for :class:`culebra.trainer.aco.abc.SingleObjACO`."""

import unittest

import numpy as np

from culebra.trainer.aco.abc import SingleObjACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func_multi = MultiObjectivePathLength(
    PathLength.from_path(optimum_paths[0]),
    PathLength.from_path(optimum_paths[1])
)
fitness_func_single = fitness_func_multi.objectives[0]
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.abc.SingleObjACO`."""

    def test_fitness_function(self):
        """Test the fitness_function property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1

        # Try invalid types for fitness function. Should fail
        invalid_fitness_functions = (type, 'a')
        for invalid_fitness_func in invalid_fitness_functions:
            with self.assertRaises(TypeError):
                SingleObjACO(
                    ant_cls,
                    species,
                    invalid_fitness_func,
                    initial_pheromone
                )

        # Try invalid values for fitness function. Should fail
        invalid_fitness_func = fitness_func_multi
        with self.assertRaises(ValueError):
            SingleObjACO(
                ant_cls,
                species,
                invalid_fitness_func,
                initial_pheromone
            )



if __name__ == '__main__':
    unittest.main()
