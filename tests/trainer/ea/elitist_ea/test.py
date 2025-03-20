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

"""Unit test for :py:class:`~culebra.trainer.ea.ElitistEA`."""

import unittest

from deap.tools import HallOfFame

from culebra.trainer.ea import ElitistEA, DEFAULT_ELITE_SIZE
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import KappaIndex as Fitness
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Remove outliers
dataset.remove_outliers()

# Normalize inputs
dataset.robust_scale()


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.ElitistEA`."""

    def test_init(self):
        """Test __init__."""
        # Trainer parameters
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }

        # Create the trainer
        trainer = ElitistEA(**params)

        # Test default parameter values
        self.assertEqual(trainer.elite_size, DEFAULT_ELITE_SIZE)
        self.assertEqual(trainer._elite, None)

    def test_elite_size(self):
        """Test elite_size."""
        # Trainer parameters
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }

        # Create the trainer
        trainer = ElitistEA(**params)

        # Try a valid elite proportion
        valid_size = 3
        trainer.elite_size = valid_size
        self.assertEqual(trainer.elite_size, valid_size)

        # Try not valid elite proportion types, should fail
        invalid_sizes = ['a', len, 1.4]
        for size in invalid_sizes:
            with self.assertRaises(TypeError):
                trainer.elite_size = size

        # Try not valid elite proportion values, should fail
        invalid_sizes = [-1, 0]
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                trainer.elite_size = size

    def test_state(self):
        """Test the get_state and _set_state methods."""
        # Trainer parameters
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 1000,
            "elite_size": 13
        }

        # Create the trainer
        trainer = ElitistEA(**params)

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer._num_evals)
        self.assertEqual(state["elite"], trainer._elite)

        # Change the state
        state["num_evals"] = 100
        state["elite"] = 200

        # Set the new state
        trainer._set_state(state)

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], trainer._num_evals)
        self.assertEqual(state["elite"], trainer._elite)

    def test_new_state(self):
        """Test _new_state."""
        # Trainer parameters
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 100,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = ElitistEA(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the elite
        self.assertIsInstance(trainer._elite, HallOfFame)

        # Check that best_ones contains only one species
        self.assertEqual(len(trainer._elite), max(1, trainer.elite_size))

    def test_reset_state(self):
        """Test _reset_state."""
        # Trainer parameters
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 100,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = ElitistEA(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the elite
        self.assertEqual(trainer._elite, None)

    def test_do_iteration(self):
        """Test _do_iteration."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        trainer = ElitistEA(**params)

        # Init the search process
        trainer._init_search()

        # Do an iteration
        pop_size_before = len(trainer.pop)
        trainer._do_iteration()
        pop_size_after = len(trainer.pop)
        self.assertEqual(pop_size_before, pop_size_after)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        trainer = ElitistEA(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
