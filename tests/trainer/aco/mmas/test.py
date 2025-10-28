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

"""Unit test for :py:class:`culebra.trainer.aco.MMAS`."""

import unittest
from math import ceil

import numpy as np
from deap.tools import ParetoFront

from culebra.trainer.aco import (
    MMAS,
    DEFAULT_AS_EXPLOITATION_PROB,
    DEFAULT_MMAS_ITER_BEST_USE_LIMIT,
    DEFAULT_CONVERGENCE_CHECK_FREQ
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.MMAS`."""

    def test_init(self):
        """Test __init__`."""
        ant_cls = Ant
        species = Species(num_nodes)
        initial_pheromone = 1

        # Try invalid types for iter_best_use_limit. Should fail
        invalid_iter_best_use_limit = (type, 'a', 1.5)
        for iter_best_use_limit in invalid_iter_best_use_limit:
            with self.assertRaises(TypeError):
                MMAS(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
                    iter_best_use_limit=iter_best_use_limit
                )

        # Try invalid values for iter_best_use_limit. Should fail
        invalid_iter_best_use_limit = (-1, 0)
        for iter_best_use_limit in invalid_iter_best_use_limit:
            with self.assertRaises(ValueError):
                MMAS(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
                    iter_best_use_limit=iter_best_use_limit
                )

        # Try valid values for iter_best_use_limit
        valid_iter_best_use_limit = (1, 10)
        for iter_best_use_limit in valid_iter_best_use_limit:
            trainer = MMAS(
                ant_cls,
                species,
                fitness_func,
                initial_pheromone,
                iter_best_use_limit=iter_best_use_limit
            )
            self.assertEqual(
                iter_best_use_limit,
                trainer.iter_best_use_limit
            )

        # Try invalid types for convergence_check_freq. Should fail
        invalid_convergence_check_freq = (type, 'a', 1.5)
        for convergence_check_freq in invalid_convergence_check_freq:
            with self.assertRaises(TypeError):
                MMAS(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
                    convergence_check_freq=convergence_check_freq
                )

        # Try invalid values for convergence_check_freq. Should fail
        invalid_convergence_check_freq = (-1, 0)
        for convergence_check_freq in invalid_convergence_check_freq:
            with self.assertRaises(ValueError):
                MMAS(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
                    convergence_check_freq=convergence_check_freq
                )

        # Try valid values for convergence_check_freq
        valid_convergence_check_freq = (1, 10)
        for convergence_check_freq in valid_convergence_check_freq:
            trainer = MMAS(
                ant_cls,
                species,
                fitness_func,
                initial_pheromone,
                convergence_check_freq=convergence_check_freq
            )
            self.assertEqual(
                convergence_check_freq,
                trainer.convergence_check_freq
            )

        # Test default params
        trainer = MMAS(
            ant_cls,
            species,
            fitness_func,
            initial_pheromone
        )
        self.assertEqual(
            trainer.exploitation_prob, DEFAULT_AS_EXPLOITATION_PROB
        )
        self.assertEqual(
            trainer.iter_best_use_limit,
            DEFAULT_MMAS_ITER_BEST_USE_LIMIT
        )
        self.assertEqual(
            trainer.convergence_check_freq,
            DEFAULT_CONVERGENCE_CHECK_FREQ
        )

        self.assertEqual(trainer._max_pheromone, None)
        self.assertEqual(trainer._min_pheromone, None)
        self.assertEqual(
            trainer._global_best_freq,
            DEFAULT_MMAS_ITER_BEST_USE_LIMIT
        )

    def test_global_best_freq(self):
        """Test the _global_best_freq property."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)

        for i in range(300):
            trainer._current_iter = i
            self.assertEqual(
                trainer._global_best_freq,
                ceil(trainer.iter_best_use_limit / (trainer._current_iter + 1))
            )

    def test_state(self):
        """Test the get_state and _set_state methods."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)

        # Check before initialization
        # Save the trainer's state
        state = trainer._get_state()
        self.assertEqual(state["num_evals"], None)
        self.assertEqual(state["pheromone"], None)
        self.assertEqual(state["elite"], None)
        self.assertEqual(state["max_pheromone"], None)
        self.assertEqual(state["min_pheromone"], None)
        self.assertEqual(state["last_elite_iter"], None)

        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["pheromone"], trainer.pheromone)
        self.assertEqual(state["elite"], trainer._elite)
        self.assertEqual(state["max_pheromone"], trainer._max_pheromone)
        self.assertEqual(state["min_pheromone"], trainer._min_pheromone)
        self.assertEqual(state["last_elite_iter"], trainer._last_elite_iter)

        # Change the state
        elite = ParetoFront()
        elite.update([trainer._generate_ant()])
        state["num_evals"] = 100
        state["pheromone"] = [np.full((num_nodes, num_nodes), 8, dtype=float)]
        state["elite"] = elite
        state["max_pheromone"] = -1
        state["min_pheromone"] = -2
        state["last_elite_iter"] = -3

        # Set the new state
        trainer._set_state(state)

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertTrue(
            np.all(state["pheromone"] == trainer.pheromone)
        )
        self.assertTrue(
            np.all(state["elite"] == trainer._elite)
        )
        self.assertEqual(state["max_pheromone"], trainer._max_pheromone)
        self.assertEqual(state["min_pheromone"], trainer._min_pheromone)
        self.assertEqual(state["last_elite_iter"], trainer._last_elite_iter)

    def test_reset_state(self):
        """Test _reset_state."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the pheromone
        self.assertEqual(trainer.pheromone, None)

        # Check the elite
        self.assertEqual(trainer._elite, None)
        self.assertEqual(trainer._max_pheromone, None)
        self.assertEqual(trainer._min_pheromone, None)
        self.assertEqual(trainer._last_elite_iter, None)

    def test_new_state(self):
        """Test _new_state."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the pheromone matrices
        self.assertIsInstance(trainer.pheromone, list)
        for (
            initial_pheromone,
            pheromone_matrix
        ) in zip(
            trainer.initial_pheromone,
            trainer.pheromone
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))

        # Check the elite
        self.assertIsInstance(trainer._elite, ParetoFront)
        self.assertEqual(len(trainer._elite), 0)
        self.assertEqual(trainer._max_pheromone, trainer.initial_pheromone[0])

        self.assertEqual(
            trainer._min_pheromone,
            (
                trainer._max_pheromone /
                (2 * trainer.fitness_function.num_nodes)
            )
        )
        self.assertEqual(trainer._last_elite_iter, None)

    def test_increase_pheromone(self):
        """Test the _increase_pheromone method."""

        def assert_path_pheromone_increment(trainer, ant):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are amount of pheromone.
            """
            pheromone_value = (
                trainer.initial_pheromone[0] +
                ant.fitness.pheromone_amount[0]
            )
            org = ant.path[-1]
            for dest in ant.path:
                self.assertAlmostEqual(
                    trainer.pheromone[0][org][dest],
                    pheromone_value
                )
                self.assertAlmostEqual(
                    trainer.pheromone[0][dest][org],
                    pheromone_value
                )
                org = dest

        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "col_size": 1
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromone
        self.assertTrue(
            np.all(trainer.pheromone[0] == trainer.initial_pheromone[0])
        )

        # In the first iteration the iteration-best ant should deposit
        # pheromone
        trainer._generate_col()
        trainer._increase_pheromone()
        assert_path_pheromone_increment(trainer, trainer.col[0])

        # In iterations above MMAS.iter_best_use_limit only the global-best
        # ant should deposit the pheromone
        trainer._init_search()
        trainer._start_iteration()
        trainer._current_iter = trainer.iter_best_use_limit

        # Generate a new elite
        optimum_ant = Ant(species, fitness_func.fitness_cls, optimum_path)
        trainer.evaluate(optimum_ant)
        trainer._elite.update([optimum_ant])
        trainer._increase_pheromone()
        assert_path_pheromone_increment(trainer, trainer._elite[0])

    def test_update_pheromone(self):
        """Test the _update_pheromone method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "pheromone_evaporation_rate": 0.5
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()
        trainer._max_pheromone = 4
        trainer._update_pheromone()
        self.assertTrue(
            np.all(trainer.pheromone[0] <= trainer._max_pheromone)
        )
        trainer._min_pheromone = 3
        trainer._update_pheromone()
        self.assertTrue(
            np.all(trainer.pheromone[0] >= trainer._min_pheromone)
        )

    def test_update_elite(self):
        """Test the _update_elite method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the limits
        self.assertEqual(trainer._max_pheromone, trainer.initial_pheromone[0])
        self.assertEqual(
            trainer._min_pheromone,
            trainer._max_pheromone / (2 * fitness_func.num_nodes)
        )
        self.assertEqual(trainer._last_elite_iter, None)

        trainer._generate_col()
        trainer._current_iter = 25

        # Update the elite
        trainer._update_elite()

        # Check the limits
        self.assertAlmostEqual(
            trainer._max_pheromone, (
                1 / (
                    trainer._elite[0].fitness.values[0] *
                    trainer.pheromone_evaporation_rate
                )
            )
        )
        self.assertAlmostEqual(
            trainer._min_pheromone,
            trainer._max_pheromone / (2 * fitness_func.num_nodes)
        )
        self.assertEqual(trainer._last_elite_iter, trainer._current_iter)

    def test_has_converged(self):
        """Test the _has_converged method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

        trainer._current_iter = trainer.convergence_check_freq

        # Check convergence
        self.assertFalse(trainer._has_converged())

        # Simulate convergence with all the nodes banned
        trainer.pheromone[0] = np.full(
            (num_nodes, num_nodes), trainer._min_pheromone)
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over one arc only
        trainer.pheromone[0][0][0] = trainer._max_pheromone
        self.assertFalse(trainer._has_converged())

        # Deposit the maximum pheremone amount over two arcs
        trainer.pheromone[0][0][1] = trainer._max_pheromone
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over more than two arcs
        trainer.pheromone[0][0][2] = trainer._max_pheromone
        self.assertFalse(trainer._has_converged())

        trainer._current_iter = 1
        # Deposit the maximum pheremone amount over more than two arcs
        trainer.pheromone[0][0][2] = trainer._min_pheromone
        self.assertFalse(trainer._has_converged())

    def test_init_pheromone(self):
        """Test the init_pheromone method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()

        # Simulate convergence
        heuristic_shape = trainer._heuristic[0].shape
        trainer._pheromone = [
            np.zeros(
                heuristic_shape,
                dtype=float
            ) for initial_pheromone in trainer.initial_pheromone
        ]

        # Check the pheromone
        for pher in trainer.pheromone:
            self.assertTrue(np.all(pher == 0))

        # Reset the pheromone
        trainer._init_pheromone()

        # Check the pheromone
        for pher in trainer.pheromone:
            self.assertTrue(np.all(pher == trainer._max_pheromone))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 10
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
