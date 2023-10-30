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

"""Unit test for :py:class:`culebra.trainer.ea.abc.ElitistACO`."""

import unittest

import numpy as np

from deap.tools import ParetoFront

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.aco import DEFAULT_CONVERGENCE_CHECK_FREQ
from culebra.trainer.aco.abc import ElitistACO
from culebra.solution.tsp import Species, Solution, Ant
from culebra.fitness_function.tsp import PathLength

num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class MyTrainer(ElitistACO):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromones[0] * self.heuristics[0]

    def _decrease_pheromones(self) -> None:
        """Decrease the amount of pheromones."""

    def _increase_pheromones(self) -> None:
        """Increase the amount of pheromones."""


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.ea.abc.SingleColACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromones = [1]

        # Try invalid ant classes. Should fail
        invalid_ant_classes = (type, None, 1, Solution)
        for solution_cls in invalid_ant_classes:
            with self.assertRaises(TypeError):
                MyTrainer(
                    solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones
                )

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    species,
                    valid_fitness_func,
                    valid_initial_pheromones
                )

        # Try invalid fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    func,
                    valid_initial_pheromones
                )

        # Try invalid types for initial_pheromones. Should fail
        invalid_initial_pheromones = (type, 1)
        for initial_pheromones in invalid_initial_pheromones:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    initial_pheromones
                )

        # Try invalid values for initial_pheromones. Should fail
        invalid_initial_pheromones = [(-1, ), (max, ), (0, ), ()]
        for initial_pheromones in invalid_initial_pheromones:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    initial_pheromones
                )

        # Try invalid types for heuristics. Should fail
        invalid_heuristics = (type, 1)
        for heuristics in invalid_heuristics:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    heuristics=heuristics
                )

        # Try invalid values for heuristics. Should fail
        invalid_heuristics = (
            # Empty
            (),
            # Wrong shape
            (np.ones(shape=(num_nodes, num_nodes + 1), dtype=float), ),
            # Negative values
            (np.ones(shape=(num_nodes, num_nodes), dtype=float) * -1, ),
            # Different shapes
            (
                np.ones(shape=(num_nodes, num_nodes), dtype=float),
                np.ones(shape=(num_nodes+1, num_nodes+1), dtype=float),
            ),
            # Empty matrix
            (np.ones(shape=(0, 0), dtype=float), ),
        )
        for heuristics in invalid_heuristics:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    heuristics=heuristics
                )

        # Try a valid value for heuristics
        heuristics = (np.ones(shape=(num_nodes, num_nodes), dtype=float), )
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones,
            heuristics=heuristics
        )
        for h1, h2 in zip(trainer.heuristics, heuristics):
            self.assertTrue(np.all(h1 == h2))

        # Try invalid types for convergence_check_freq. Should fail
        invalid_convergence_check_freq = (type, 'a', 1.5)
        for convergence_check_freq in invalid_convergence_check_freq:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    convergence_check_freq=convergence_check_freq
                )

        # Try invalid values for convergence_check_freq. Should fail
        invalid_convergence_check_freq = (-1, 0)
        for convergence_check_freq in invalid_convergence_check_freq:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    convergence_check_freq=convergence_check_freq
                )

        # Try valid values for convergence_check_freq
        valid_convergence_check_freq = (1, 10)
        for convergence_check_freq in valid_convergence_check_freq:
            trainer = MyTrainer(
                valid_ant_cls,
                valid_species,
                valid_fitness_func,
                valid_initial_pheromones,
                convergence_check_freq=convergence_check_freq
            )
            self.assertEqual(
                convergence_check_freq,
                trainer.convergence_check_freq
            )

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    max_num_iters=max_num_iters
                )

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    max_num_iters=max_num_iters
                )

        # Try a valid value for max_num_iters
        max_num_iters = 210
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones,
            max_num_iters=max_num_iters
        )
        self.assertEqual(max_num_iters, trainer.max_num_iters)

        # Try invalid types for col_size. Should fail
        invalid_col_size = (type, 'a', 1.5)
        for col_size in invalid_col_size:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    col_size=col_size
                )

        # Try invalid values for col_size. Should fail
        invalid_col_size = (-1, 0)
        for col_size in invalid_col_size:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    col_size=col_size
                )

        # Try a valid value for col_size
        col_size = 233
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones,
            col_size=col_size
        )
        self.assertEqual(col_size, trainer.col_size)

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones
        )
        self.assertEqual(trainer.solution_cls, valid_ant_cls)
        self.assertEqual(trainer.species, valid_species)
        self.assertEqual(trainer.fitness_function, valid_fitness_func)
        self.assertEqual(trainer.initial_pheromones, valid_initial_pheromones)
        self.assertIsInstance(trainer.heuristics, list)
        for matrix in trainer.heuristics:
            self.assertEqual(matrix.shape, (num_nodes, num_nodes))

        # Check the heuristics
        the_heuristics = trainer.heuristics[0]
        for org_idx, org in enumerate(optimum_path):
            dest_1 = optimum_path[org_idx - 1]
            dest_2 = optimum_path[(org_idx + 1) % num_nodes]

            for node in range(num_nodes):
                if (
                    org in banned_nodes or
                    node in banned_nodes or
                    node == org
                ):
                    self.assertEqual(
                        the_heuristics[org][node], 0
                    )
                elif node == dest_1 or node == dest_2:
                    self.assertEqual(
                        the_heuristics[org][node], 1
                    )
                else:
                    self.assertEqual(
                        the_heuristics[org][node], 0.1
                    )

        self.assertEqual(
            trainer.convergence_check_freq,
            DEFAULT_CONVERGENCE_CHECK_FREQ
        )

        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(
            trainer.col_size,
            trainer.fitness_function.num_nodes
        )
        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.col, None)
        self.assertEqual(trainer.pheromones, None)
        self.assertEqual(trainer.choice_info, None)
        self.assertEqual(trainer._node_list, None)

    def test_state(self):
        """Test _state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._state

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["elite"], trainer._elite)

        elite = ParetoFront()
        elite.update([trainer._generate_ant()])
        # Change the state
        state["num_evals"] = 100
        state["elite"] = elite

        # Set the new state
        trainer._state = state

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertTrue(
            np.all(state["elite"] == trainer._elite)
        )

    def test_new_state(self):
        """Test _new_state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2, 3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the pheromones matrix
        self.assertIsInstance(trainer._elite, ParetoFront)
        self.assertEqual(len(trainer._elite), 0)

    def test_reset_state(self):
        """Test _reset_state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = (2, )
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the elite
        self.assertEqual(trainer._elite, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2, 3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

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

    def test_deposit_pheromones(self):
        """Test the _deposit_pheromones method."""

        def assert_path_pheromones_increment(trainer, ant, weight):
            """Check the pheromones in all the arcs of a path.

            All the arcs should have the same are ammount of pheromones.
            """
            pheromones_value = (
                trainer.initial_pheromones[0] +
                ant.fitness.pheromones_amount[0] * weight
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
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromones
        pheromones_value = trainer.initial_pheromones[0]
        self.assertTrue(
            np.all(trainer.pheromones[0] == pheromones_value)
        )

        # Try with an empty elite
        # Only the iteration-best ant should deposit pheromones
        trainer._generate_col()
        weight = 3
        trainer._deposit_pheromones(trainer.col, weight)
        assert_path_pheromones_increment(trainer, trainer.col[0], weight)

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
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check convergence
        self.assertFalse(trainer._has_converged())

        # Simulate convergence with all the nodes banned
        trainer.pheromones[0] = np.full((num_nodes, num_nodes), 0)
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over one arc only
        trainer.pheromones[0][0][0] = initial_pheromones[0]
        self.assertFalse(trainer._has_converged())

        # Deposit the maximum pheremone amount over two arcs
        trainer.pheromones[0][0][1] = initial_pheromones[0]
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over more than two arcs
        trainer.pheromones[0][0][2] = initial_pheromones[0]
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
        trainer = MyTrainer(**params)
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
            self.assertTrue(np.all(pher == initial_pheromones[0]))

    def test_do_iteration(self):
        """Test the _do_iteration method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # The elite should be empty
        self.assertEqual(len(trainer._elite), 0)

        # Generate a new colony
        trainer._do_iteration()

        # The elite should not be empty
        self.assertGreaterEqual(len(trainer._elite), 1)

        # Simulate convergence
        trainer._current_iter = trainer.convergence_check_freq
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

        # Do an interation
        trainer._do_iteration()

        # Check if the pheromones have been reset
        for pher in trainer.pheromones:
            self.assertTrue(np.all(pher == initial_pheromones[0]))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
