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

"""Unit test for :py:class:`culebra.trainer.aco.abc.MaxPheromonePACO`."""

import unittest
from copy import copy, deepcopy
from os import remove

import numpy as np

from deap.tools import ParetoFront

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco.abc import (
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    MaxPheromonePACO
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


class MyTrainer(
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    MaxPheromonePACO
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
    """Test :py:class:`culebra.trainer.aco.abc.MaxPheromonePACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromone = 1

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
            (-1, ), (max, ), (0, ), (1, 2, 3), [1, 2, 3]
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
        valid_max_pheromone = [3]
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
            max_pheromone=valid_max_pheromone
        )
        self.assertEqual(
            trainer.max_pheromone,
            valid_max_pheromone * trainer.num_pheromone_matrices
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

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""

        def assert_path_pheromone_increment(trainer, ant, weight):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are amount of pheromone.
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
        initial_pheromone = [1]
        max_pheromone = [3]
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

    def test_copy(self):
        """Test the __copy__ method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [1]
        max_pheromone = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1
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
        self.assertEqual(trainer1.max_pheromone, trainer2.max_pheromone)

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [2]
        max_pheromone = [4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1
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
        initial_pheromone = [2]
        max_pheromone = [4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1
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

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.aco.abc.MaxPheromonePACO`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.aco.abc.MaxPheromonePACO`
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )

        self.assertNotEqual(id(trainer1.species), id(trainer2.species))
        self.assertEqual(trainer1.max_pheromone, trainer2.max_pheromone)

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
