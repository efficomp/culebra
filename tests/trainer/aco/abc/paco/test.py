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

"""Unit test for :py:class:`culebra.trainer.aco.abc.PACO`."""

import unittest

import numpy as np

from deap.tools import ParetoFront

from culebra.abc import Fitness
from culebra.trainer.aco.abc import PACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.tsp import PathLength


class MyTrainer(PACO):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromones[0] * self.heuristics[0]

    def _update_pop(self) -> None:
        """Update the population."""
        # Ingoing ants
        best_in_col = ParetoFront()
        best_in_col.update(self.col)

        # Append the best ants to the population
        for ant in best_in_col:
            self._pop.append(ant)


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
    """Test :py:class:`culebra.trainer.aco.abc.PACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromones = [1, 2]

        # Try invalid types for max_pheromones. Should fail
        invalid_max_pheromones = (type, 1)
        for max_pheromones in invalid_max_pheromones:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    max_pheromones=max_pheromones
                )

        # Try invalid values for max_pheromones. Should fail
        invalid_max_pheromones = [
            (-1, ), (max, ), (0, ), (1, 2, 3), [1, 3], (3, 2), (0, 3), (3, 0)
        ]
        for max_pheromones in invalid_max_pheromones:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    max_pheromones=max_pheromones
                )

        # Try valid values for max_pheromones
        valid_max_pheromones = [3, 4]
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones,
            max_pheromones=valid_max_pheromones
        )
        self.assertEqual(trainer.max_pheromones, valid_max_pheromones)

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    valid_max_pheromones,
                    pop_size=pop_size
                )

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromones,
                    valid_max_pheromones,
                    pop_size=pop_size
                )

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromones,
            valid_max_pheromones
        )

        self.assertEqual(
            trainer.pop_size,
            trainer.col_size
        )

    def test_state(self):
        """Test _state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._state

        # Check the state
        self.assertIsInstance(state["pop"], list)
        self.assertEqual(len(state["pop"]), 0)

        # Change the state
        state["pop"].append(trainer._generate_ant())

        # Set the new state
        trainer._state = state

        # Test if the population has been updated
        self.assertEqual(len(trainer.pop), 1)

    def test_new_state(self):
        """Test _new_state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the population
        self.assertIsInstance(trainer.pop, list)
        self.assertEqual(len(trainer.pop), 0)

    def test_reset_state(self):
        """Test _reset_state."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the elite
        self.assertEqual(trainer.pop, None)

    def test_init_internals(self):
        """Test _init_internals."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()

        # Check the internal structures
        self.assertIsInstance(trainer._pop_ingoing, list)
        self.assertIsInstance(trainer._pop_outgoing, list)
        self.assertEqual(len(trainer._pop_ingoing), 0)
        self.assertEqual(len(trainer._pop_outgoing), 0)

    def test_reset_internals(self):
        """Test _reset_internals."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()

        # Reset the internal structures
        trainer._reset_internals()

        # Check the internal strucures
        self.assertEqual(trainer._pop_ingoing, None)
        self.assertEqual(trainer._pop_outgoing, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
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
        trainer.pop.append(ant)

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one solution
        self.assertEqual(len(best_ones[0]), 1)

        # Check that the solution in hof is sol1
        self.assertTrue(ant in best_ones[0])

    def test_do_iteration(self):
        """Test the _do_iteration method."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # The population should be empty
        self.assertEqual(len(trainer.pop), 0)

        # Generate a new colony
        trainer._do_iteration()

        # The population should not be empty
        self.assertGreaterEqual(len(trainer.pop), 1)

    def test_deposit_pheromones(self):
        """Test the _deposit_pheromones method."""

        def assert_path_pheromones_increment(trainer, ant, weight):
            """Check the pheromones in all the arcs of a path.

            All the arcs should have the same are ammount of pheromones.
            """
            for pher_index, (init_pher_val, max_pher_val) in enumerate(
                zip(trainer.initial_pheromones, trainer.max_pheromones)
            ):
                pher_delta = (
                    (max_pher_val - init_pher_val) / trainer.pop_size
                ) * weight
                pheromones_value = init_pher_val + pher_delta
                org = ant.path[-1]
                for dest in ant.path:
                    self.assertAlmostEqual(
                        trainer.pheromones[pher_index][org][dest],
                        pheromones_value
                    )
                    self.assertAlmostEqual(
                        trainer.pheromones[pher_index][dest][org],
                        pheromones_value
                    )
                    org = dest

        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1, 2]
        max_pheromones = [3, 4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromones
        for pher_index, init_pher_val in enumerate(trainer.initial_pheromones):
            self.assertTrue(
                np.all(trainer.pheromones[pher_index] == init_pher_val)
            )

        # Try with an empty elite
        # Only the iteration-best ant should deposit pheromones
        trainer._generate_col()
        weight = 3
        trainer._deposit_pheromones(trainer.col, weight)
        assert_path_pheromones_increment(trainer, trainer.col[0], weight)

    def test_increase_decrease_pheromones(self):
        """Test the _increase_pheromones and _decrease_pheromones methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1, 2]
        max_pheromones = [3, 4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Use the same ant to increase an decrease pheromones
        ant = trainer._generate_ant()
        trainer._pop_ingoing.append(ant)
        trainer._pop_outgoing.append(ant)

        trainer._increase_pheromones()
        trainer._decrease_pheromones()

        # Pheromones should not be altered
        for pher, init_pher_val in zip(
            trainer.pheromones, trainer.initial_pheromones
        ):
            self.assertTrue(np.all(pher == init_pher_val))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
