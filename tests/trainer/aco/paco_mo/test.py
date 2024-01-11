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

"""Unit test for :py:class:`culebra.trainer.aco.PACO_MO`."""

import unittest

import numpy as np

from deap.tools import ParetoFront

from culebra.trainer.aco import PACO_MO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import DoublePathLength

num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func = DoublePathLength.fromPath(*optimum_paths)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.PACO_MO`."""

    def test_init(self):
        """Test __init__."""
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
            "checkpoint_enable": False,
            "checkpoint_freq": 13,
            "checkpoint_filename": "my_check.gz",
            "verbose": False,
            "random_seed": 15
        }

        # Create the trainer
        trainer = PACO_MO(**params)

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
            trainer.checkpoint_enable, params["checkpoint_enable"]
        )
        self.assertEqual(trainer.checkpoint_freq, params["checkpoint_freq"])
        self.assertEqual(
            trainer.checkpoint_filename, params["checkpoint_filename"]
        )
        self.assertEqual(trainer.verbose, params["verbose"])
        self.assertEqual(trainer.random_seed, params["random_seed"])

    def test_num_pheromone_matrices(self):
        """Test the num_pheromone_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        self.assertEqual(
            trainer.num_pheromone_matrices, fitness_func.num_obj
        )

    def test_num_heuristic_matrices(self):
        """Test the num_heuristic_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        self.assertEqual(
            trainer.num_heuristic_matrices, fitness_func.num_obj
        )

    def test_state(self):
        """Test the get_state and _set_state methods."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["elite"], trainer._elite)

        # Check that pop is not in state
        with self.assertRaises(KeyError):
            state["pop"]

        elite = ParetoFront()
        elite.update([trainer._generate_ant()])
        # Change the state
        state["num_evals"] = 100
        state["elite"] = elite

        # Set the new state
        trainer._set_state(state)

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertTrue(
            np.all(state["elite"] == trainer._elite)
        )

    def test_new_state(self):
        """Test _new_state."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the elite
        self.assertIsInstance(trainer._elite, ParetoFront)
        self.assertEqual(len(trainer._elite), 0)

    def test_reset_state(self):
        """Test _reset_state."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the elite
        self.assertEqual(trainer._elite, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Try before any colony has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Update the elite
        trainer._init_search()
        trainer._start_iteration()
        ant = trainer._generate_ant()
        trainer._elite.update([ant])

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one solution
        self.assertEqual(len(best_ones[0]), 1)

        # Check that the solution in hof is sol1
        self.assertTrue(ant in best_ones[0])

    def test_init_internals(self):
        """Test _init_internals."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Create a new state
        trainer._init_internals()

        # Check the population
        self.assertIsInstance(trainer.pop, list)
        self.assertEqual(len(trainer.pop), 0)

    def test_reset_internals(self):
        """Test _reset_internals."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Create a new state
        trainer._init_internals()

        # Reset the state
        trainer._reset_internals()

        # Check the population
        self.assertEqual(trainer.pop, None)

    def test_calculate_choice_info(self):
        """Test _test_calculate_choice_info."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3,
            "pop_size": 5
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Init the search
        trainer._init_search()

        # TODO ...


    def test_update_pop(self):
        """Test _update_pop."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3,
            "pop_size": 5
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Init the search
        trainer._init_search()

        # Try to update the population with an empty elite
        trainer._start_iteration()
        trainer._update_pop()
        # The elite should be empty and
        # the population should be filled with random ants
        self.assertEqual(len(trainer._elite), 0)
        self.assertEqual(len(trainer.pop), trainer.pop_size)
        # Check the infoing and outgoing lists
        self.assertEqual(trainer._pop_ingoing, trainer.pop)
        self.assertTrue(len(trainer._pop_ingoing), 0)

        # Try with an elite size lower than the population size
        trainer._start_iteration()
        elite_size = trainer.pop_size // 2
        while len(trainer._elite) < elite_size:
            trainer._elite.update([trainer._generate_ant()])
        trainer._update_pop()
        # All the elite ants should be in the population.
        # The remaining place should be filled with random ants
        for ant1, ant2 in zip(trainer.pop[:elite_size], trainer._elite):
            self.assertEqual(ant1, ant2)
        self.assertEqual(len(trainer.pop), trainer.pop_size)
        # Check the infoing and outgoing lists
        self.assertEqual(trainer._pop_ingoing, trainer.pop)
        self.assertTrue(len(trainer._pop_ingoing), 0)

        # Try with an elite size higher than the population size
        trainer._start_iteration()
        elite_size = trainer.pop_size + 1
        while len(trainer._elite) < elite_size:
            trainer._elite.update([trainer._generate_ant()])
        trainer._update_pop()
        # All the ants in the population should be elite ants
        for ant in trainer.pop:
            self.assertTrue(ant in trainer._elite)
        self.assertEqual(len(trainer.pop), trainer.pop_size)
        # Check the infoing and outgoing lists
        self.assertEqual(trainer._pop_ingoing, trainer.pop)
        self.assertTrue(len(trainer._pop_ingoing), 0)

    def test_update_pheromone(self):
        """Test _update_pheromone."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "max_pheromone": 3,
            "pop_size": 5
        }

        # Create the trainer
        trainer = PACO_MO(**params)

        # Init the search
        trainer._init_search()

        # Generate a new population and update the pheromone matrices
        trainer._start_iteration()
        trainer._update_pop()
        trainer._update_pheromone()

        # Check the pheromone matrices
        for matrix, init_val in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(matrix >= init_val))
            self.assertTrue(np.any(matrix != init_val))
        
        






if __name__ == '__main__':
    unittest.main()
