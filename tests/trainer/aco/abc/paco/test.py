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

"""Unit test for :py:class:`culebra.trainer.aco.abc.PACO`."""

import unittest

import numpy as np

from deap.tools import ParetoFront

from culebra.trainer.aco.abc import (
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    PACO
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import DoublePathLength


class MyTrainer(
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    PACO
):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromone[0] * self.heuristic[0]

    def _update_pop(self) -> None:
        """Update the population."""
        # Ingoing ants
        best_in_col = ParetoFront()
        best_in_col.update(self.col)

        # Append the best ants to the population
        for ant in best_in_col:
            self._pop.append(ant)


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func = DoublePathLength.fromPath(*optimum_paths)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.PACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromone = [1, 2]

        # Try invalid types for max_pheromone. Should fail
        invalid_max_pheromone = (type, None)
        for max_pheromone in invalid_max_pheromone:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    max_pheromone=max_pheromone
                )

        # Try invalid values for max_pheromone. Should fail
        invalid_max_pheromone = [
            (-1, ), (max, ), (0, ), (1, 2, 3), [1, 3], (3, 2), (0, 3), (3, 0)
        ]
        for max_pheromone in invalid_max_pheromone:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    max_pheromone=max_pheromone
                )

        # Try valid values for max_pheromone
        valid_max_pheromone = [3, 4]
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
            max_pheromone=valid_max_pheromone
        )
        self.assertEqual(trainer.max_pheromone, valid_max_pheromone)

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    valid_max_pheromone,
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
                    valid_initial_pheromone,
                    valid_max_pheromone,
                    pop_size=pop_size
                )

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
            valid_max_pheromone
        )

        self.assertEqual(
            trainer.pop_size,
            trainer.col_size
        )

    def test_state(self):
        """Test the get_state and _set_state methods."""
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
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertIsInstance(state["pop"], list)
        self.assertEqual(len(state["pop"]), 0)

        # Change the state
        state["pop"].append(trainer._generate_ant())

        # Set the new state
        trainer._set_state(state)

        # Test if the population has been updated
        self.assertEqual(len(trainer.pop), 1)

    def test_new_state(self):
        """Test _new_state."""
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
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()

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

        # Check the internal structures
        self.assertIsInstance(trainer._pop_ingoing, list)
        self.assertIsInstance(trainer._pop_outgoing, list)
        self.assertEqual(len(trainer._pop_ingoing), 0)
        self.assertEqual(len(trainer._pop_outgoing), 0)

    def test_reset_internals(self):
        """Test _reset_internals."""
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
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()

        # Reset the internal structures
        trainer._reset_internals()

        # Check the internal strucures
        self.assertEqual(trainer.pheromone, None)
        self.assertEqual(trainer._pop_ingoing, None)
        self.assertEqual(trainer._pop_outgoing, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
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
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # The population should be empty
        self.assertEqual(len(trainer.pop), 0)

        # Generate a new colony
        trainer._do_iteration()

        # The population should not be empty
        self.assertGreaterEqual(len(trainer.pop), 1)

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""

        def assert_path_pheromone_increment(trainer, ant, weight):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are ammount of pheromone.
            """
            for pher_index, (init_pher_val, max_pher_val) in enumerate(
                zip(trainer.initial_pheromone, trainer.max_pheromone)
            ):
                pher_delta = (
                    (max_pher_val - init_pher_val) / trainer.pop_size
                ) * weight
                pheromone_value = init_pher_val + pher_delta
                org = ant.path[-1]
                for dest in ant.path:
                    self.assertAlmostEqual(
                        trainer.pheromone[pher_index][org][dest],
                        pheromone_value
                    )
                    self.assertAlmostEqual(
                        trainer.pheromone[pher_index][dest][org],
                        pheromone_value
                    )
                    org = dest

        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [1, 2]
        max_pheromone = [3, 4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromone
        for pher_index, init_pher_val in enumerate(trainer.initial_pheromone):
            self.assertTrue(
                np.all(trainer.pheromone[pher_index] == init_pher_val)
            )

        # Try with an empty elite
        # Only the iteration-best ant should deposit pheromone
        trainer._generate_col()
        weight = 3
        trainer._deposit_pheromone(trainer.col, weight)
        assert_path_pheromone_increment(trainer, trainer.col[0], weight)

    def test_increase_decrease_pheromone(self):
        """Test the _increase_pheromone and _decrease_pheromone methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [1, 2]
        max_pheromone = [3, 4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Use the same ant to increase an decrease pheromone
        ant = trainer._generate_ant()
        trainer._pop_ingoing.append(ant)
        trainer._pop_outgoing.append(ant)

        trainer._increase_pheromone()
        trainer._decrease_pheromone()

        # pheromone should not be altered
        for pher, init_pher_val in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(pher == init_pher_val))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [2]
        max_pheromone = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
