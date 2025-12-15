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

"""Unit test for :class:`culebra.trainer.aco.abc.SingleObjPACO`."""

import unittest

import numpy as np

from culebra.trainer.aco.abc import SingleObjPACO, ACOTSP
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength


class MyTrainer(ACOTSP, SingleObjPACO):
    """Dummy implementation of a trainer method."""


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.from_path(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.abc.SingleObjPACO`."""

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
        valid_max_pheromone = 3.0
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
            max_pheromone=valid_max_pheromone
        )
        self.assertEqual(trainer.max_pheromone, (valid_max_pheromone,))

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

        # Check the internal structures
        self.assertEqual(trainer.pop_ingoing, None)
        self.assertEqual(trainer.pop_outgoing, None)

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
        self.assertIsInstance(trainer.pop_ingoing, list)
        self.assertIsInstance(trainer.pop_outgoing, list)
        self.assertEqual(len(trainer.pop_ingoing), 0)
        self.assertEqual(len(trainer.pop_outgoing), 0)

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
        self.assertEqual(trainer.pop_ingoing, None)
        self.assertEqual(trainer.pop_outgoing, None)

    def test_start_iteration(self):
        """Test the _start_iteration method."""
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

        # Init the search ans start a new iteration
        trainer._init_search()
        trainer._start_iteration()

        # Append an ant in the ingoing and outgoung lists
        trainer.pop_ingoing.append(trainer._generate_ant())
        trainer.pop_outgoing.append(trainer._generate_ant())

        # Check that lists length has increased
        self.assertEqual(len(trainer.pop_ingoing), 1)
        self.assertEqual(len(trainer.pop_outgoing), 1)

        # Start the itreration
        trainer._start_iteration()

        # The lists should be empty
        self.assertEqual(len(trainer.pop_ingoing), 0)
        self.assertEqual(len(trainer.pop_outgoing), 0)

    def test_update_pheromone(self):
        """Test the _update_pheromone method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1
        max_pheromone = 3
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
        trainer.pop_ingoing.append(ant)
        trainer.pop_outgoing.append(ant)

        trainer._update_pheromone()

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
