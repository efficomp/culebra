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
from itertools import repeat

import numpy as np

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.aco.abc import SingleColACO
from culebra.solution.tsp import Species, Solution, Ant
from culebra.fitness_function.tsp import PathLength

num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class MyTrainer(SingleColACO):
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

        # Save the trainer's state
        state = trainer._state

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["pheromones"], trainer.pheromones)

        # Change the state
        state["num_evals"] = 100
        state["pheromones"] = [np.full((num_nodes, num_nodes), 8, dtype=float)]

        # Set the new state
        trainer._state = state

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

        # Check the pheromones matrix
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

    def test_init_internals(self):
        """Test the _init_internals method."""
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

        self.assertEqual(trainer.col, [])
        self.assertEqual(trainer.choice_info, None)
        self.assertTrue(
            np.all(
                trainer._node_list == np.arange(0, num_nodes, dtype=int)
            )
        )

    def test_reset_internals(self):
        """Test the _init_internals method."""
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
        trainer._reset_internals()

        self.assertEqual(trainer.col, None)
        self.assertEqual(trainer.choice_info, None)
        self.assertEqual(trainer._node_list, None)

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
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

        # Try to get the choice info before the search initialization
        choice_info = trainer.choice_info
        self.assertEqual(choice_info, None)

        # Try to get the choice_info after initializing the search
        trainer._init_search()
        trainer._start_iteration()
        choice_info = trainer.choice_info

        # Check the probabilities for banned nodes. Should be 0
        for node in banned_nodes:
            self.assertAlmostEqual(np.sum(choice_info[node]), 0)

        for node in feasible_nodes:
            self.assertAlmostEqual(
                np.sum(choice_info[node]),
                np.sum(
                    trainer.pheromones[0][node] * trainer.heuristics[0][node]
                )
            )

    def test_initial_choice(self):
        """Test the _initial_choice method."""
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

        # Try to generate valid first nodes
        times = 1000
        for _ in repeat(None, times):
            self.assertTrue(trainer._initial_choice() in feasible_nodes)

        # Try when all nodes are unfeasible
        trainer.heuristics = [np.zeros((num_nodes, num_nodes))]
        trainer._start_iteration()
        self.assertEqual(trainer._initial_choice(), None)

    def test_feasible_neighborhood_probs(self):
        """Test the _feasible_neighborhood_probs method."""
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

        # Try with an ant with an empty path. Should fail
        ant = Ant(
            species,
            fitness_func.Fitness
        )
        with self.assertRaises(ValueError):
            trainer._feasible_neighborhood_probs(ant)

        # Choose a first node for the ant
        ant.append(trainer._initial_choice())

        # Check the choice probabilities of the ant's current node neighborhood
        node_list = np.arange(0, num_nodes, dtype=int)
        probs = trainer._feasible_neighborhood_probs(ant)
        while np.sum(probs) > 0:
            self.assertAlmostEqual(np.sum(probs), 1)
            next_node = np.random.choice(node_list, p=probs)
            ant.append(next_node)
            probs = trainer._feasible_neighborhood_probs(ant)

    def test_next_choice(self):
        """Test the _next_choice method."""
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

        # Try to generate valid first nodes
        times = 1000
        for _ in repeat(None, times):
            trainer._start_iteration()
            ant = Ant(
                species,
                fitness_func.Fitness
            )
            choice = trainer._next_choice(ant)

            while choice is not None:
                ant.append(choice)
                choice = trainer._next_choice(ant)

            self.assertEqual(len(ant.path), len(feasible_nodes))

    def test_generate_ant(self):
        """Test the _generate_ant method."""
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

        # Try to generate valid ants
        times = 1000
        for _ in repeat(None, times):
            trainer._start_iteration()
            ant = trainer._generate_ant()
            self.assertEqual(len(ant.path), len(feasible_nodes))

    def test_generate_col(self):
        """Test the _generate_col_method."""
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

        # Generate the colony
        trainer._init_search()
        trainer._start_iteration()
        trainer._generate_col()

        # Check the colony
        self.assertEqual(len(trainer.col), trainer.col_size)
        for ant in trainer.col:
            self.assertIsInstance(ant, Ant)

    def test_best_solutions(self):
        """Test the best_solutions method."""
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

        # Try before the colony has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Generate some ants
        ant1 = Ant(
            species,
            fitness_func.Fitness,
            path=optimum_path
        )
        worse_path = np.concatenate(
            (optimum_path[:5], optimum_path[-1:], optimum_path[5:-2])
        )
        ant2 = Ant(
            species,
            fitness_func.Fitness,
            path=worse_path
        )

        # Init the search
        trainer._init_search()

        # Try a colony with different fitnesses
        trainer._col = [ant1, ant2]

        for ant in trainer.col:
            trainer.evaluate(ant)

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one solution
        self.assertEqual(len(best_ones[0]), 1)

        # Check that the solution in hof is ant1
        self.assertTrue(ant1 in best_ones[0])

        # Set the same fitness for both solutions
        for sol in trainer.col:
            sol.fitness.values = (18,)

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has two solutions
        self.assertEqual(len(best_ones[0]), 2)

        # Check that ant1 and ant2 are the solutions in hof
        self.assertTrue(ant1 in best_ones[0])
        self.assertTrue(ant2 in best_ones[0])

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
