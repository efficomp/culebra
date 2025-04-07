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

"""Unit test for :py:class:`culebra.trainer.ea.SimpleEA`."""

import unittest
from copy import copy, deepcopy
from os import remove

from deap.base import Toolbox

from culebra.trainer.ea import SimpleEA
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import KappaIndex as Fitness
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.ea.SimpleEA`."""

    def test_init(self):
        """Test __init__."""
        # Test the superclass initialization
        valid_solution_cls = Individual
        valid_species = Species(dataset.num_feats)
        valid_fitness_func = Fitness(dataset)

        # Try invalid individual classes. Should fail
        invalid_individual_classes = (type, None, 'a', 1)
        for solution_cls in invalid_individual_classes:
            with self.assertRaises(TypeError):
                SimpleEA(solution_cls, valid_species, valid_fitness_func)

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                SimpleEA(valid_solution_cls, species, valid_fitness_func)

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                SimpleEA(valid_solution_cls, valid_species, func)

        # Test initialization
        params = {
            "solution_cls": valid_solution_cls,
            "species": valid_species,
            "fitness_function": valid_fitness_func
        }
        trainer = SimpleEA(**params)
        self.assertEqual(trainer._toolbox, None)

    def test_init_internals(self):
        """Test _init_internals."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = SimpleEA(**params)

        # Init the internals
        trainer._init_internals()
        self.assertIsInstance(trainer._toolbox, Toolbox)
        self.assertEqual(trainer._toolbox.mate.func, trainer.crossover_func)
        self.assertEqual(trainer._toolbox.mutate.func, trainer.mutation_func)
        self.assertEqual(trainer._toolbox.select.func, trainer.selection_func)

    def test_reset_internals(self):
        """Test _reset_internals."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = SimpleEA(**params)
        # Init the internals
        trainer._init_internals()

        # REset the internals
        trainer._reset_internals()
        self.assertEqual(trainer._toolbox, None)

    def test_do_iteration(self):
        """Test _do_iteration."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        trainer = SimpleEA(**params)

        # Init the search process
        trainer._init_search()

        # Do an iteration
        pop_size_before = len(trainer.pop)
        trainer._do_iteration()
        pop_size_after = len(trainer.pop)
        self.assertEqual(pop_size_before, pop_size_after)

    def test_copy(self):
        """Test the __copy__ method."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = SimpleEA(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertNotEqual(id(trainer1), id(trainer2))

        # The objects attributes are shared
        self.assertEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertEqual(
            id(trainer1.species),
            id(trainer2.species)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = SimpleEA(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = SimpleEA(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = SimpleEA.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = SimpleEA(**params)
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.ea.SimpleEA`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.ea.SimpleEA`
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertNotEqual(
            id(trainer1.fitness_function.training_data),
            id(trainer2.fitness_function.training_data)
        )

        self.assertTrue(
            (
                trainer1.fitness_function.training_data.inputs ==
                trainer2.fitness_function.training_data.inputs
            ).all()
        )

        self.assertTrue(
            (
                trainer1.fitness_function.training_data.outputs ==
                trainer2.fitness_function.training_data.outputs
            ).all()
        )

        self.assertNotEqual(id(trainer1.species), id(trainer2.species))
        self.assertEqual(
            id(trainer1.species.num_feats), id(trainer2.species.num_feats)
        )


if __name__ == '__main__':
    unittest.main()
