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

"""Unit test for :py:class:`culebra.trainer.ea.abc.SingleColACO`."""

import unittest

import numpy as np

from deap.tools import ParetoFront

from culebra.trainer.aco.abc import ElitistACO
from culebra.solution.tsp import Species, Ant
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

    def _evaporate_pheromones(self) -> None:
        """Evaporate pheromones."""

    def _deposit_pheromones(self) -> None:
        """Deposit pheromones."""


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.ea.abc.SingleColACO`."""

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

    def test_deposit_ant_weighted_pheromones(self):
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
        elite_weight = 0.8
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
        ant = trainer.col[0]
        weight = 3
        trainer._deposit_ant_weighted_pheromones(ant, weight)
        assert_path_pheromones_increment(trainer, ant, weight)

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
