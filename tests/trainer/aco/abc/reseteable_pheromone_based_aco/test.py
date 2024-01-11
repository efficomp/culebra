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

"""Unit test for ReseteablePheromoneBasedACO."""

import unittest

import numpy as np

from culebra.trainer.aco import DEFAULT_CONVERGENCE_CHECK_FREQ
from culebra.trainer.aco.abc import (
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    ReseteablePheromoneBasedACO
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import DoublePathLength


class MyTrainer(
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    ReseteablePheromoneBasedACO
):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromone[0] * self.heuristic[0]

    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone."""

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone."""


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func = DoublePathLength.fromPath(*optimum_paths)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.ReseteablePheromoneBasedACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromone = 1

        # Try invalid types for convergence_check_freq. Should fail
        invalid_convergence_check_freq = (type, 'a', 1.5)
        for convergence_check_freq in invalid_convergence_check_freq:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
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
                    valid_initial_pheromone,
                    convergence_check_freq=convergence_check_freq
                )

        # Try valid values for convergence_check_freq
        valid_convergence_check_freq = (1, 10)
        for convergence_check_freq in valid_convergence_check_freq:
            trainer = MyTrainer(
                valid_ant_cls,
                valid_species,
                valid_fitness_func,
                valid_initial_pheromone,
                convergence_check_freq=convergence_check_freq
            )
            self.assertEqual(
                convergence_check_freq,
                trainer.convergence_check_freq
            )

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone
        )

        self.assertEqual(
            trainer.convergence_check_freq,
            DEFAULT_CONVERGENCE_CHECK_FREQ
        )

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
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check convergence
        self.assertFalse(trainer._has_converged())

        # Simulate convergence with all the nodes banned
        for i in range(trainer.num_pheromone_matrices):
            trainer.pheromone[i] = np.full((num_nodes, num_nodes), 0)
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over one arc only
        for pher in trainer.pheromone:
            pher[0][0] = initial_pheromone
        self.assertFalse(trainer._has_converged())

        # Deposit the maximum pheremone amount over two arcs
        for pher in trainer.pheromone:
            pher[0][1] = initial_pheromone
        self.assertTrue(trainer._has_converged())

        # Deposit the maximum pheremone amount over more than two arcs
        for pher in trainer.pheromone:
            pher[0][2] = initial_pheromone
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
        trainer = MyTrainer(**params)
        trainer._init_search()

        # Simulate convergence
        heuristic_shape = trainer._heuristic[0].shape
        trainer._pheromone = [
            np.zeros(
                heuristic_shape,
                dtype=float
            )
        ] * trainer.num_pheromone_matrices

        # Check the pheromone
        for pher in trainer.pheromone:
            self.assertTrue(np.all(pher == 0))

        # Reset the pheromone
        trainer._init_pheromone()

        # Check the pheromone
        for pher in trainer.pheromone:
            self.assertTrue(np.all(pher == initial_pheromone))

    def test_do_iteration(self):
        """Test the _do_iteration method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
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
        heuristic_shape = trainer._heuristic[0].shape
        trainer._pheromone = [
            np.zeros(
                heuristic_shape,
                dtype=float
            )
        ] * trainer.num_pheromone_matrices

        # Check the pheromone
        for pher in trainer.pheromone:
            self.assertTrue(np.all(pher == 0))

        # Do an interation
        trainer._do_iteration()

        # Check if the pheromone have been reset
        for pher in trainer.pheromone:
            self.assertTrue(np.all(pher == initial_pheromone))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [2, 5]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
