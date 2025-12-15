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

"""Unit test for :class:`culebra.trainer.aco.AgeBasedPACO`."""

import unittest

import numpy as np

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco.abc import ACOTSP
from culebra.trainer.aco import AgeBasedPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.from_path(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class AgeBasedPACOTSP(ACOTSP, AgeBasedPACO):
    """Age Based PACO for TSP."""


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.AgeBasedPACO`."""

    def test_init(self):
        """Test __init__`."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3,
            "heuristic": np.ones((num_nodes, num_nodes)),
            "pheromone_influence": 2,
            "heuristic_influence": 5,
            "max_num_iters": 123,
            "custom_termination_func": max,
            "col_size": 6,
            "pop_size": 5,
            "checkpoint_activation": False,
            "checkpoint_freq": 13,
            "checkpoint_filename": "my_check" + SERIALIZED_FILE_EXTENSION,
            "verbosity": False,
            "random_seed": 15
        }

        # Create the trainer
        trainer = AgeBasedPACOTSP(**params)

        # Check the parameters
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, params["species"])
        self.assertEqual(trainer.fitness_function, params["fitness_function"])
        self.assertEqual(
            trainer.initial_pheromone[0], params["initial_pheromone"]
        )
        self.assertEqual(trainer.max_pheromone[0], params["max_pheromone"])
        self.assertTrue(np.all(trainer.heuristic[0] == params["heuristic"]))
        self.assertEqual(
            trainer.pheromone_influence[0], params["pheromone_influence"]
        )
        self.assertEqual(
            trainer.heuristic_influence[0], params["heuristic_influence"]
        )
        self.assertEqual(trainer.max_num_iters, params["max_num_iters"])
        self.assertEqual(
            trainer.custom_termination_func, params["custom_termination_func"]
        )
        self.assertEqual(trainer.col_size, params["col_size"])
        self.assertEqual(trainer.pop_size, params["pop_size"])
        self.assertEqual(
            trainer.checkpoint_activation, params["checkpoint_activation"]
        )
        self.assertEqual(trainer.checkpoint_freq, params["checkpoint_freq"])
        self.assertEqual(
            trainer.checkpoint_filename, params["checkpoint_filename"]
        )
        self.assertEqual(trainer.verbosity, params["verbosity"])
        self.assertEqual(trainer.random_seed, params["random_seed"])

    def test_internals(self):
        """Test _init_internals."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone
        }

        # Create the trainer
        trainer = AgeBasedPACOTSP(**params)

        # Create new internal structures
        trainer._init_internals()
        self.assertEqual(trainer._youngest_index, None)

        # Reset the internal structures
        trainer._reset_internals()
        self.assertEqual(trainer._youngest_index, None)

    def test_update_pop(self):
        """Test the _update_pop method."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1,
            "pop_size": 2
        }

        # Create the trainer
        trainer = AgeBasedPACOTSP(**params)
        trainer._init_search()

        # The initial population should be empty
        trainer._start_iteration()
        self.assertEqual(len(trainer.pop), 0)

        # Try several colonies
        for col_index in range(5):
            # Generate the colony
            trainer._start_iteration()
            trainer._generate_col()

            # Index where the colony's ant will be inserted in the population
            pop_index = col_index % trainer.pop_size

            # Get the outgoing ant, ig any
            if col_index < trainer.pop_size:
                outgoing_ant = None
            else:
                outgoing_ant = trainer.pop[pop_index]

            # Update the population
            trainer._update_pop()

            # Check the population size
            if col_index < trainer.pop_size:
                self.assertEqual(len(trainer.pop), col_index + 1)
            else:
                self.assertEqual(len(trainer.pop), trainer.pop_size)

            # Check that the ant has been inserted in the population
            self.assertEqual(trainer.pop[pop_index], trainer.col[0])

            # The ant should also be in the ingoing list
            self.assertEqual(len(trainer.pop_ingoing), 1)
            self.assertEqual(trainer.pop_ingoing[0], trainer.col[0])

            # Check the outgoing ant, if any
            if col_index < trainer.pop_size:
                self.assertEqual(len(trainer.pop_outgoing), 0)
            else:
                self.assertEqual(len(trainer._pop_outgoing), 1)
                self.assertEqual(trainer.pop_outgoing[0], outgoing_ant)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone
        }

        # Create the trainer
        trainer = AgeBasedPACOTSP(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
