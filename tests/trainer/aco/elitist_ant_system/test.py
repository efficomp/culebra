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

"""Unit test for :py:class:`culebra.trainer.aco.ElitistAntSystem`."""

import unittest

import numpy as np
from deap.tools import ParetoFront

from culebra.trainer.aco import (
    ElitistAntSystem,
    DEFAULT_AS_EXPLOITATION_PROB,
    DEFAULT_ELITE_WEIGHT
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.ElitistAntSystem`."""

    def test_init(self):
        """Test __init__`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1

        # Try invalid types for elite_weight. Should fail
        invalid_elite_weight_types = (type, 'a')
        for elite_weight in invalid_elite_weight_types:
            with self.assertRaises(TypeError):
                ElitistAntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
                    elite_weight=elite_weight
                )

        # Try invalid values for elite_weight. Should fail
        invalid_elite_weight_values = (-0.5, 1.5)
        for elite_weight in invalid_elite_weight_values:
            with self.assertRaises(ValueError):
                ElitistAntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
                    elite_weight=elite_weight
                )

        # Try a valid value for elite_weight
        valid_elite_weight_values = (0.0, 0.5, 1.0)
        for elite_weight in valid_elite_weight_values:
            trainer = ElitistAntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromone,
                elite_weight=elite_weight
            )
            self.assertEqual(elite_weight, trainer.elite_weight)

        # Test default params
        trainer = ElitistAntSystem(
            ant_cls,
            species,
            fitness_func,
            initial_pheromone
        )
        self.assertEqual(
            trainer.exploitation_prob, DEFAULT_AS_EXPLOITATION_PROB
        )
        self.assertEqual(trainer.elite_weight, DEFAULT_ELITE_WEIGHT)

    def test_state(self):
        """Test the get_state and _set_state methods."""
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
        trainer = ElitistAntSystem(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["pheromone"], trainer.pheromone)
        self.assertEqual(state["elite"], trainer._elite)

        # Change the state
        elite = ParetoFront()
        elite.update([trainer._generate_ant()])
        state["num_evals"] = 100
        state["pheromone"] = [np.full((num_nodes, num_nodes), 8, dtype=float)]
        state["elite"] = elite

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

    def test_new_state(self):
        """Test _new_state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2.5
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = ElitistAntSystem(**params)

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

    def test_reset_state(self):
        """Test _reset_state."""
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
        trainer = ElitistAntSystem(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the pheromone
        self.assertEqual(trainer.pheromone, None)

        # Check the elite
        self.assertEqual(trainer._elite, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 3.7
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = ElitistAntSystem(**params)

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

    def test_increase_pheromone(self):
        """Test the _increase_pheromone method."""

        def assert_path_pheromone_increment(trainer, ant, weight):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are amount of pheromone.
            """
            pheromone_value = (
                trainer.initial_pheromone[0] +
                ant.fitness.pheromone_amount[0] * weight
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
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        elite_weight = 0.8
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "col_size": 1,
            "elite_weight": elite_weight
        }

        # Create the trainer
        trainer = ElitistAntSystem(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromone
        pheromone_value = trainer.initial_pheromone[0]
        self.assertTrue(
            np.all(trainer.pheromone[0] == pheromone_value)
        )

        # Try with an empty elite
        # Only the iteration-best ant should deposit pheromone
        trainer._generate_col()
        ant = trainer.col[0]
        trainer._increase_pheromone()
        assert_path_pheromone_increment(
            trainer,
            ant,
            1 - trainer.elite_weight
        )

        # Update the elite and try with an empty colony
        # Only the elite ant should deposit pheromone
        trainer._init_search()
        trainer._start_iteration()
        trainer._elite.update([ant])
        trainer._col = []
        trainer._increase_pheromone()
        assert_path_pheromone_increment(
            trainer,
            ant,
            trainer.elite_weight
        )

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
        trainer = ElitistAntSystem(**params)
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
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = ElitistAntSystem(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
