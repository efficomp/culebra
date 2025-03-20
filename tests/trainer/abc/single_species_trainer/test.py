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

"""Unit test for :py:class:`culebra.trainer.abc.SingleSpeciesTrainer`."""

import unittest
from os import remove
from copy import copy, deepcopy

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.abc import SingleSpeciesTrainer
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.fitness_function.feature_selection import NumFeats
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Remove outliers
dataset.remove_outliers()

# Normalize inputs
dataset.robust_scale()


class MyTrainer(SingleSpeciesTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self._current_iter_evals = 10


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`."""

    def test_init(self):
        """Test the constructor."""
        valid_solution_cls = FeatureSelectionIndividual
        valid_species = FeatureSelectionSpecies(dataset.num_feats)
        valid_fitness_func = NumFeats(dataset)

        # Try invalid solution classes. Should fail
        invalid_solution_classes = (type, None, 'a', 1)
        for solution_cls in invalid_solution_classes:
            with self.assertRaises(TypeError):
                MyTrainer(solution_cls, valid_species, valid_fitness_func)

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyTrainer(valid_solution_cls, species, valid_fitness_func)

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(valid_solution_cls, valid_species, func)

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    max_num_iters=max_num_iters
                )

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    max_num_iters=max_num_iters
                )

        # Test default params
        trainer = MyTrainer(
            valid_solution_cls, valid_species, valid_fitness_func
        )
        self.assertEqual(trainer.solution_cls, valid_solution_cls)
        self.assertEqual(trainer.species, valid_species)
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer._current_iter, None)

    def test_copy(self):
        """Test the __copy__ method."""
        # Set custom params
        params = {
            "solution_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)
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
        # Set custom params
        params = {
            "solution_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Set custom params
        params = {
            "solution_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = MyTrainer.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        params = {
            "solution_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
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
