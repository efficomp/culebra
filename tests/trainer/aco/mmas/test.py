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

"""Unit test for :py:class:`culebra.trainer.aco.MMAS`."""

import unittest
from math import ceil

import numpy as np

from culebra.trainer.aco import (
    MMAS,
    DEFAULT_MMAS_ITER_BEST_USE_LIMIT,
    DEFAULT_CONVERGENCE_CHECK_FREQ
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength

num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.MMAS`."""

    def test_init(self):
        """Test __init__`."""
        ant_cls = Ant
        species = Species(num_nodes)
        initial_pheromones = [1]

        # Try invalid types for iter_best_use_limit. Should fail
        invalid_iter_best_use_limit = (type, 'a', 1.5)
        for iter_best_use_limit in invalid_iter_best_use_limit:
            with self.assertRaises(TypeError):
                MMAS(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
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
                    initial_pheromones,
                    iter_best_use_limit=iter_best_use_limit
                )

        # Try valid values for iter_best_use_limit
        valid_iter_best_use_limit = (1, 10)
        for iter_best_use_limit in valid_iter_best_use_limit:
            trainer = MMAS(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
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
                    initial_pheromones,
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
                    initial_pheromones,
                    convergence_check_freq=convergence_check_freq
                )

        # Try valid values for convergence_check_freq
        valid_convergence_check_freq = (1, 10)
        for convergence_check_freq in valid_convergence_check_freq:
            trainer = MMAS(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
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
            initial_pheromones
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
        initial_pheromones = [10]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
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
        """Test the trainer state."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Check the limits before initialization
        trainer = MMAS(**params)
        self.assertEqual(trainer._max_pheromone, None)
        self.assertEqual(trainer._min_pheromone, None)
        self.assertEqual(trainer._last_elite_iter, None)

        # Check the limits after initialization
        trainer._init_search()
        self.assertEqual(trainer._max_pheromone, initial_pheromones[0])
        self.assertEqual(
            trainer._min_pheromone,
            trainer._max_pheromone / (2 * fitness_func.num_nodes)
        )
        self.assertEqual(trainer._last_elite_iter, None)

        # Check after reset
        trainer.reset()
        self.assertEqual(trainer._max_pheromone, None)
        self.assertEqual(trainer._min_pheromone, None)
        self.assertEqual(trainer._last_elite_iter, None)

    def test_deposit_pheromones(self):
        """Test the _deposit_pheromones method."""

        def assert_path_pheromones_increment(trainer, ant):
            """Check the pheromones in all the arcs of a path.

            All the arcs should have the same are ammount of pheromones.
            """
            pheromones_value = (
                trainer.initial_pheromones[0] +
                ant.fitness.pheromones_amount[0]
            )
            org = ant.path[-1]
            for dest in ant.path:
                self.assertAlmostEqual(
                    trainer.pheromones[0][org][dest],
                    pheromones_value
                )
                self.assertAlmostEqual(
                    trainer.pheromones[0][dest][org],
                    pheromones_value
                )
                org = dest

        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "col_size": 1
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromones
        self.assertTrue(
            np.all(trainer.pheromones[0] == trainer.initial_pheromones[0])
        )

        # In the first iteration the iteration-best ant should deposit
        # pheromones
        trainer._generate_col()
        trainer._deposit_pheromones()
        assert_path_pheromones_increment(trainer, trainer.col[0])

        # In iterations above MMAS.iter_best_use_limit only the global-best
        # ant should deposit the pheromones
        trainer._init_search()
        trainer._start_iteration()
        trainer._current_iter = trainer.iter_best_use_limit

        # Generate a new elite
        optimum_ant = Ant(species, fitness_func.Fitness, optimum_path)
        trainer.evaluate(optimum_ant)
        trainer._elite.update([optimum_ant])
        trainer._deposit_pheromones()
        assert_path_pheromones_increment(trainer, trainer._elite[0])

    def test_update_pheromones(self):
        """Test the _update_pheromones method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromones = [10]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "pheromone_evaporation_rate": 0.5
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()
        trainer._max_pheromone = 4
        trainer._update_pheromones()
        self.assertTrue(
            np.all(trainer.pheromones[0] <= trainer._max_pheromone)
        )
        trainer._min_pheromone = 3
        trainer._update_pheromones()
        self.assertTrue(
            np.all(trainer.pheromones[0] >= trainer._min_pheromone)
        )

    def test_update_elite(self):
        """Test the _update_elite method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromones = [10]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the limits
        self.assertEqual(trainer._max_pheromone, initial_pheromones[0])
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
        initial_pheromones = [10]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
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
        trainer.pheromones[0] = np.full(
            (num_nodes, num_nodes), trainer._min_pheromone)
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over one arc only
        trainer.pheromones[0][0][0] = trainer._max_pheromone
        self.assertFalse(trainer._has_converged())

        # Deposit the maximum pheremone amount over two arcs
        trainer.pheromones[0][0][1] = trainer._max_pheromone
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over more than two arcs
        trainer.pheromones[0][0][2] = trainer._max_pheromone
        self.assertFalse(trainer._has_converged())

        trainer._current_iter = 1
        # Deposit the maximum pheremone amount over more than two arcs
        trainer.pheromones[0][0][2] = trainer._min_pheromone
        self.assertFalse(trainer._has_converged())

    def test_reset_pheromones(self):
        """Test the reset_pheromones method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromones = [10]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()

        # Simulate convergence
        heuristics_shape = trainer._heuristics[0].shape
        trainer._pheromones = [
            np.zeros(
                heuristics_shape,
                dtype=float
            ) for initial_pheromone in initial_pheromones
        ]

        # Check the pheromones
        for pher in trainer.pheromones:
            self.assertTrue(np.all(pher == 0))

        # Reset the pheromones
        trainer._reset_pheromones()

        # Check the pheromones
        for pher in trainer.pheromones:
            self.assertTrue(np.all(pher == trainer._max_pheromone))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromones = [10]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MMAS(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
