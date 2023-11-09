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

"""Unit test for :py:class:`culebra.trainer.ea.NSGA`."""

import unittest
import pickle
from copy import copy, deepcopy

from deap.tools import selNSGA3

from culebra.trainer.ea import (
    DEFAULT_POP_SIZE,
    NSGA,
    DEFAULT_NSGA_SELECTION_FUNC,
    DEFAULT_NSGA_SELECTION_FUNC_PARAMS,
    DEFAULT_NSGA3_REFERENCE_POINTS_P
)
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.ea.NSGA`."""

    def test_init(self):
        """Test :py:meth:`~culebra.trainer.ea.NSGA.__init__`."""
        # Test the default parameters
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)

        # The selection function should be DEFAULT_NSGA_SELECTION_FUNC
        self.assertEqual(
            trainer.selection_func, DEFAULT_NSGA_SELECTION_FUNC
        )
        self.assertEqual(
            trainer.selection_func_params, DEFAULT_NSGA_SELECTION_FUNC_PARAMS
        )
        self.assertEqual(
            trainer.nsga3_reference_points_p, DEFAULT_NSGA3_REFERENCE_POINTS_P)
        self.assertEqual(trainer.nsga3_reference_points_scaling, None)

        # Try custom params
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "nsga3_reference_points_p": 1,
            "nsga3_reference_points_scaling": 4
        }
        trainer = NSGA(**params)
        self.assertEqual(
            trainer.nsga3_reference_points_p,
            params["nsga3_reference_points_p"])
        trainer.selection_func = selNSGA3

        self.assertEqual(
            trainer.nsga3_reference_points_scaling,
            params["nsga3_reference_points_scaling"])

    def test_pop_size(self):
        """Test :py:meth:`~culebra.trainer.ea.NSGA.pop_size` getter."""
        # Try with the default pop_size for NSGA2
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)
        # trainer.pop_size should be DEFAULT_POP_SIZE
        self.assertEqual(trainer.pop_size, DEFAULT_POP_SIZE)

        # Try with the default pop_size for NSGA3
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "selection_func": selNSGA3
        }
        trainer = NSGA(**params)
        # trainer.pop_size should be DEFAULT_POP_SIZE
        self.assertEqual(trainer.pop_size, len(trainer.nsga3_reference_points))

        # Set a customized value
        pop_size = 200
        trainer.pop_size = pop_size
        # trainer.pop_size should be the customized value
        self.assertEqual(trainer.pop_size, pop_size)

        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": pop_size
        }
        trainer = NSGA(**params)
        # trainer.pop_size should be the customized value
        self.assertEqual(trainer.pop_size, pop_size)

    def test_selection_func(self):
        """Test :py:meth:`~culebra.trainer.ea.NSGA.selection_func` getter."""
        # Try with the default selection function
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)
        # trainer.selection_func should be DEFAULT_NSGA_SELECTION_FUNC
        self.assertEqual(trainer.selection_func, DEFAULT_NSGA_SELECTION_FUNC)

        # Try with a custom selection function
        params["selection_func"] = selNSGA3
        trainer = NSGA(**params)
        # trainer.selection_func should be selNSGA3
        self.assertEqual(trainer.selection_func, selNSGA3)

    def test_selection_func_params(self):
        """Test :py:meth:`~culebra.trainer.ea.NSGA.selection_func_params`."""
        # Try with the default selection function
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)
        # trainer.selection_func should be DEFAULT_NSGA_SELECTION_FUNC
        self.assertEqual(
            trainer.selection_func_params, DEFAULT_NSGA_SELECTION_FUNC_PARAMS
        )

        # Try with a custom selection function
        params["selection_func"] = selNSGA3
        trainer = NSGA(**params)
        # trainer.selection_func should be selNSGA3
        self.assertEqual(trainer.selection_func, selNSGA3)

    def test_nsga3_reference_points_p(self):
        """Test nsga3_reference_points_p."""
        # Construct the trainer
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)

        # Try invalid types for p
        invalid_p_values = (type, 'a', 1.4)
        for value in invalid_p_values:
            with self.assertRaises(TypeError):
                trainer.nsga3_reference_points_p = value

        # Try invalid values for p
        invalid_p_values = (-3, 0)
        for value in invalid_p_values:
            with self.assertRaises(ValueError):
                trainer.nsga3_reference_points_p = value

    def test_nsga3_reference_points_scaling(self):
        """Test nsga3_reference_points_scaling."""
        # Construct the trainer
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)

        # Init the internals
        trainer._init_internals()

        # Try invalid types for the scaling factor
        invalid_scaling_values = (type, 'a')
        for value in invalid_scaling_values:
            with self.assertRaises(TypeError):
                trainer.nsga3_reference_points_scaling = value

    def test_init_internals(self):
        """Test _init_internals`."""
        # Construct the trainer
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)

        # Init the internals
        trainer._init_internals()

        # Check the current reference points
        self.assertEqual(trainer._nsga3_ref_points, None)

    def test_reset_internals(self):
        """Test _reset_internals`."""
        # Construct the trainer
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        trainer = NSGA(**params)

        # Init the internals
        trainer._init_internals()

        # Reset the internals
        trainer._reset_internals()

        # Check the current reference points
        self.assertEqual(trainer._nsga3_ref_points, None)

    def test_do_iteration(self):
        """Test _do_iteration."""
        params = {
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        trainer = NSGA(**params)

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
        trainer1 = NSGA(**params)
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
        trainer1 = NSGA(**params)
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
        trainer1 = NSGA(**params)

        data = pickle.dumps(trainer1)
        trainer2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.ea.NSGA`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.ea.NSGA`
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
        trainer = NSGA(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
