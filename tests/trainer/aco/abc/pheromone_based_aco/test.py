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

"""Unit test for :py:class:`culebra.trainer.aco.abc.PheromoneBasedACO`."""

import unittest
from itertools import repeat

import numpy as np

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.abc import Fitness
from culebra.trainer.aco import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE
)
from culebra.trainer.aco.abc import PheromoneBasedACO
from culebra.solution.tsp import Species, Solution, Ant
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.tsp import PathLength


class MyTrainer(PheromoneBasedACO):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromones[0] * self.heuristics[0]

    def _decrease_pheromones(self) -> None:
        """Decrease the amount of pheromones."""

    def _increase_pheromones(self) -> None:
        """Increase the amount of pheromones."""


class MyFitnessFunc(PathLength):
    """Dummy fitness function with two objectives."""

    class Fitness(Fitness):
        """Fitness class."""

        weights = (-1.0, 1.0)
        names = ("Len", "Other")
        thresholds = (DEFAULT_THRESHOLD, DEFAULT_THRESHOLD)

    def heuristics(self, species):
        """Define a dummy heuristics."""
        (the_heuristics, ) = super().heuristics(species)
        return (the_heuristics, the_heuristics * 2)

    def evaluate(self, sol, index=None, representatives=None):
        """Define a dummy evaluation."""
        return super().evaluate(sol) + (3,)


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = MyFitnessFunc.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.PheromoneBasedACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromones = [1, 2]

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones
        )
        self.assertEqual(trainer.pheromones, None)

    def test_state(self):
        """Test the get_state and _set_state methods."""
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

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["pheromones"], trainer.pheromones)

        # Change the state
        state["num_evals"] = 100
        state["pheromones"] = [np.full((num_nodes, num_nodes), 8, dtype=float)]

        # Set the new state
        trainer._set_state(state)

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertTrue(
            np.all(state["pheromones"] == trainer.pheromones)
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

        # Check the pheromones matrices
        self.assertIsInstance(trainer.pheromones, list)
        for (
            initial_pheromone,
            pheromones_matrix
        ) in zip(
            trainer.initial_pheromones,
            trainer.pheromones
        ):
            self.assertTrue(np.all(pheromones_matrix == initial_pheromone))

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

        # Check the pheromones
        self.assertEqual(trainer.pheromones, None)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes)
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
