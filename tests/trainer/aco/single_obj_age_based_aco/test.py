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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for :py:class:`culebra.trainer.aco.SingleObjAgeBasedPACO`."""

import unittest

import numpy as np

from culebra.trainer.aco import SingleObjAgeBasedPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength

num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.SingleObjAgeBasedPACO`."""

    def test_init(self):
        """Test __init__`."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromones": [1],
            "max_pheromones": [3],
            "heuristics": [np.ones((num_nodes, num_nodes))],
            "pheromones_influence": [2],
            "heuristics_influence": [5],
            "max_num_iters": 123,
            "custom_termination_func": max,
            "col_size": 6,
            "pop_size": 5,
            "checkpoint_enable": False,
            "checkpoint_freq": 13,
            "checkpoint_filename": "my_check.gz",
            "verbose": False,
            "random_seed": 15
        }

        # Create the trainer
        trainer = SingleObjAgeBasedPACO(**params)

        # Check the parameters
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, params["species"])
        self.assertEqual(trainer.fitness_function, params["fitness_function"])
        self.assertEqual(
            trainer.initial_pheromones, params["initial_pheromones"]
        )
        self.assertEqual(trainer.max_pheromones, params["max_pheromones"])
        self.assertEqual(trainer.heuristics, params["heuristics"])
        self.assertEqual(
            trainer.pheromones_influence, params["pheromones_influence"]
        )
        self.assertEqual(
            trainer.heuristics_influence, params["heuristics_influence"]
        )
        self.assertEqual(trainer.max_num_iters, params["max_num_iters"])
        self.assertEqual(
            trainer.custom_termination_func, params["custom_termination_func"]
        )
        self.assertEqual(trainer.col_size, params["col_size"])
        self.assertEqual(trainer.pop_size, params["pop_size"])
        self.assertEqual(
            trainer.checkpoint_enable, params["checkpoint_enable"]
        )
        self.assertEqual(trainer.checkpoint_freq, params["checkpoint_freq"])
        self.assertEqual(
            trainer.checkpoint_filename, params["checkpoint_filename"]
        )
        self.assertEqual(trainer.verbose, params["verbose"])
        self.assertEqual(trainer.random_seed, params["random_seed"])


if __name__ == '__main__':
    unittest.main()