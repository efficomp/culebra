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
from copy import copy, deepcopy
from os import remove

import numpy as np

from deap.tools import ParetoFront

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco.abc import (
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    PACO
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


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

        # Remaining room in the population
        remaining_room_in_pop = self.pop_size - len(self.pop)

        # For all the ants in the ingoing list
        for ant in best_in_col:
            # If there is still room in the population, just append it
            if remaining_room_in_pop > 0:
                self._pop.append(ant)
                remaining_room_in_pop -= 1

                # If the population is full, start with ants replacement
                if remaining_room_in_pop == 0:
                    self._youngest_index = 0
            # The eldest ant is replaced
            else:
                self.pop[self._youngest_index] = ant
                self._youngest_index = (
                    (self._youngest_index + 1) % self.pop_size
                )


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func = MultiObjectivePathLength(
    PathLength.fromPath(optimum_paths[0]),
    PathLength.fromPath(optimum_paths[1])
)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.PACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromone = 1

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
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
                    pop_size=pop_size
                )

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
        )

        self.assertEqual(
            trainer.pop_size,
            trainer.col_size
        )
        
        self.assertEqual(
            trainer.num_pheromone_matrices,
            fitness_func.num_obj
        )

        self.assertEqual(
            trainer.num_heuristic_matrices,
            fitness_func.num_obj
        )

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
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertIsInstance(state["pop"], list)
        self.assertGreaterEqual(len(state["pop"]), 0)

        # Get the population and the pheromone matrices
        pop = trainer.pop
        pheromone = trainer.pheromone

        # Reset the trainer
        trainer.reset()

        # Set the new state
        trainer._set_state(state)

        # Test if the pop has been restored
        self.assertEqual(len(pop), len(trainer.pop))
        for ant1, ant2 in zip(pop, trainer.pop):
            self.assertEqual(ant1, ant2)

        # Test if the pheromone has been restored
        for pher1, pher2 in zip(pheromone, trainer.pheromone):
            self.assertTrue(np.all(pher1 == pher2))

    def test_new_state(self):
        """Test _new_state."""
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
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the population
        self.assertEqual(trainer.pop, None)

    def test_init_internals(self):
        """Test _init_internals."""
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

        # Create new internal structures
        trainer._init_internals()

        # Check the pheromone matrices
        self.assertEqual(len(trainer.pheromone), fitness_func.num_obj)
        self.assertIsInstance(trainer.pheromone, list)
        for (
            initial_pheromone,
            pheromone_matrix
        ) in zip(
            trainer.initial_pheromone,
            trainer.pheromone
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))

    def test_reset_internals(self):
        """Test _reset_internals."""
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

        # Create new internal structures
        trainer._init_internals()

        # Reset the internal structures
        trainer._reset_internals()

        # Check the internal strucures
        self.assertEqual(trainer.pheromone, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
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

    def test_pheromone_amount(self):
        """Test the _pheromone_amount method."""

        def assert_path_pheromone_increment(trainer, ant, weight):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are amount of pheromone.
            """
            pheromone_amount = trainer.initial_pheromone
            for pher_index, pher_amount in enumerate(pheromone_amount):
                pher_delta = pher_amount * weight
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
        initial_pheromone = [1]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
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

        # Make the ant tn the colony deposit pheromone
        trainer._generate_col()
        weight = 3
        trainer._deposit_pheromone(trainer.col, weight)
        assert_path_pheromone_increment(trainer, trainer.col[0], weight)

    def test_update_pheromone(self):
        """Test _update_pheromone."""
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

        # Init the search
        trainer._init_search()

        # Generate a new population and update the pheromone matrices
        trainer._start_iteration()
        trainer._pop.append(trainer._generate_ant())
        trainer._update_pheromone()

        # Check the pheromone matrices
        for matrix, init_val in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(matrix >= init_val))
            self.assertTrue(np.any(matrix != init_val))

    def test_do_iteration(self):
        """Test the _do_iteration method."""
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

        # The population should be empty
        self.assertEqual(len(trainer.pop), 0)

        # Generate a new colony
        trainer._do_iteration()

        # The population should not be empty
        self.assertGreaterEqual(len(trainer.pop), 1)

        # Check the pheromone matrices
        for matrix, init_val in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(matrix >= init_val))
            self.assertTrue(np.any(matrix != init_val))

    def test_copy(self):
        """Test the __copy__ method."""
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
        trainer1 = MyTrainer(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertNotEqual(id(trainer1), id(trainer2))

        # The objects attributes are shared
        self.assertEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertEqual(id(trainer1.species), id(trainer2.species))

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
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
        trainer1 = MyTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test."""
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
        trainer1 = MyTrainer(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = MyTrainer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [2]
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

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.aco.abc.PACO`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.aco.abc.PACO`
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )

        self.assertNotEqual(id(trainer1.species), id(trainer2.species))

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
