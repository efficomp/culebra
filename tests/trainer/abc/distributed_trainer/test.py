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

"""Unit test for :py:class:`~culebra.trainer.abc.DistributedTrainer`."""

import unittest
import os
from multiprocessing.queues import Queue
from copy import copy, deepcopy

from deap.tools import Logbook

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer import (
    DEFAULT_NUM_SUBTRAINERS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
)
from culebra.trainer.abc import SingleSpeciesTrainer, DistributedTrainer
from culebra.trainer.topology import ring_destinations
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
)
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Scale inputs
dataset.scale()

# Remove outliers
dataset.remove_outliers()

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MyTrainer(DistributedTrainer):
    """Dummy implementation of a distributed trainer."""

    def _generate_subtrainers(self):
        self._subtrainers = []
        subtrainer_params = {
            "solution_cls": Solution,
            "species": species,
            "fitness_function": self.fitness_function,
            "max_num_iters": self.max_num_iters,
            "checkpoint_enable": self.checkpoint_enable,
            "checkpoint_freq": self.checkpoint_freq,
            "checkpoint_filename": self.checkpoint_filename,
            "verbose": self.verbose,
            "random_seed": self.random_seed
        }

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subtrainer_checkpoint_filenames):
            subtrainer = self.subtrainer_cls(**subtrainer_params)
            subtrainer.checkpoint_filename = checkpoint_filename

            subtrainer.index = index
            subtrainer.container = self
            self._subtrainers.append(subtrainer)

    @property
    def representation_topology_func(self):
        """Get and set the representation topology function."""
        return ring_destinations

    @property
    def representation_topology_func_params(self):
        """Get and set the representation topology function parameters."""
        return {}


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.abc.DistributedTrainer`."""

    def test_init(self):
        """Test :py:meth:`~culebra.trainer.abc.DistributedTrainer.__init__`."""
        valid_fitness_func = Fitness(dataset)
        valid_subtrainer_cls = SingleSpeciesTrainer

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    func,
                    valid_subtrainer_cls
                )

        # Try invalid subtrainer_cls. Should fail
        invalid_trainer_classes = (tuple, str, None, 'a', 1)
        for cls in invalid_trainer_classes:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    cls
                )

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = ('a', 1.5, str)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    max_num_iters=max_num_iters
                )

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    max_num_iters=max_num_iters
                )

        # Try invalid types for num_subtrainers. Should fail
        invalid_num_subtrainers = ('a', 1.5, str)
        for num_subtrainers in invalid_num_subtrainers:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=num_subtrainers
                )

        # Try invalid values for num_subtrainers. Should fail
        invalid_num_subtrainers = (-1, 0)
        for num_subtrainers in invalid_num_subtrainers:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=num_subtrainers
                )

        # Try invalid types for the representation size. Should fail
        invalid_representation_sizes = (str, 'a', -0.001, 1.001)
        for representation_size in invalid_representation_sizes:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    representation_size=representation_size
                )

        # Try invalid values for the representation size. Should fail
        invalid_representation_sizes = (-1, 0)
        for representation_size in invalid_representation_sizes:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    representation_size=representation_size
                )

        # Try invalid types for the representation frequency. Should fail
        invalid_representation_freqs = (str, 'a', 1.5)
        for representation_freq in invalid_representation_freqs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    representation_freq=representation_freq
                )

        # Try invalid values for the representation frequency. Should fail
        invalid_representation_freqs = (-1, 0)
        for representation_freq in invalid_representation_freqs:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    representation_freq=representation_freq
                )

        # Try invalid representation selection function. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    representation_selection_func=func
                )

        # Try invalid types for representation selection function parameters
        # Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    representation_selection_func_params=params
                )

        # Test default params
        trainer = MyTrainer(valid_fitness_func, valid_subtrainer_cls)

        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer.num_subtrainers, DEFAULT_NUM_SUBTRAINERS)

        self.assertEqual(
            trainer.representation_size, DEFAULT_REPRESENTATION_SIZE
        )
        self.assertEqual(
            trainer.representation_freq, DEFAULT_REPRESENTATION_FREQ
        )
        self.assertEqual(
            trainer.representation_selection_func,
            DEFAULT_REPRESENTATION_SELECTION_FUNC
        )
        self.assertEqual(
            trainer.representation_selection_func_params,
            DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
        )

        self.assertEqual(trainer.subtrainer_params, {})
        self.assertEqual(trainer.subtrainers, None)
        self.assertEqual(trainer._communication_queues, None)

        # Test islands trainer extra params
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subtrainer_cls,
            extra="foo")
        self.assertEqual(trainer.subtrainer_params, {"extra": "foo"})

    def test_subtrainer_suffixes(self):
        """Test the _subtrainer_suffixes method."""
        # Parameters for the trainer
        params = {
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": SingleSpeciesTrainer
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Try several number of islands
        max_num_subtrainers = 10
        for num_subtrainers in range(1, max_num_subtrainers+1):
            trainer.num_subtrainers = num_subtrainers

            # Check the suffixes
            for suffix in range(num_subtrainers):
                suffixes = trainer._subtrainer_suffixes
                self.assertTrue(f"{suffix}" in suffixes)

    def test_subtrainer_checkpoint_filenames(self):
        """Test subtrainer_checkpoint_filenames."""
        # Parameters for the trainer
        params = {
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": SingleSpeciesTrainer,
            "num_subtrainers": 2,
            "checkpoint_filename": "my_check.gz"
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Check the file names
        self.assertEqual(
            tuple(trainer.subtrainer_checkpoint_filenames),
            ("my_check_0.gz", "my_check_1.gz")
        )

    def test_init_internals(self):
        """Test _init_internals."""
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_internals()

        # Test the number of subtrainerulations
        self.assertEqual(len(trainer.subtrainers), num_subtrainers)

        # Test each island
        for subtrainer in trainer.subtrainers:
            self.assertIsInstance(subtrainer, subtrainer_cls)
            self.assertEqual(subtrainer.species, species)
            self.assertEqual(subtrainer.solution_cls, Solution)
            self.assertEqual(subtrainer.fitness_function, fitness_func)

        # Test that the communication queues have been created
        self.assertIsInstance(trainer._communication_queues, list)
        self.assertEqual(
            len(trainer._communication_queues), trainer.num_subtrainers
        )
        for queue in trainer._communication_queues:
            self.assertIsInstance(queue, Queue)

        for index1 in range(trainer.num_subtrainers):
            for index2 in range(index1 + 1, trainer.num_subtrainers):
                self.assertNotEqual(
                    id(trainer._communication_queues[index1]),
                    id(trainer._communication_queues[index2])
                )

    def test_reset_internals(self):
        """Test _reset_internals."""
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_internals()

        # Reset the internals
        trainer._reset_internals()

        # Test the number of subtrainerulations
        self.assertEqual(trainer.subtrainers, None)

        # Test the communication queues
        self.assertEqual(trainer._communication_queues, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default trainer
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False
        }

        # Test default params
        trainer1 = MyTrainer(**params)

        # Create the subtrainerulations
        trainer1._init_search()

        # Set state attributes to dummy values
        trainer1._runtime = 10
        trainer1._current_iter = 19

        # Save the state of trainer1
        trainer1._save_state()

        # Create another trainer
        trainer2 = MyTrainer(**params)

        # Trainer2 has no subtrainerulations yet
        self.assertEqual(trainer2.subtrainers, None)

        # Load the state of trainer1 into trainer2
        trainer2._load_state()

        # Check that the state attributes of trainer2 are equal to those of
        # trainer1
        self.assertEqual(trainer1.runtime, trainer2.runtime)
        self.assertEqual(trainer1._current_iter, trainer2._current_iter)

        # Remove the checkpoint files
        os.remove(trainer1.checkpoint_filename)

    def test_new_state(self):
        """Test _new_state."""
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_internals()
        trainer._new_state()

        # Check the current iteration
        self.assertEqual(trainer._current_iter, 0)

        # Test that the logbook is None
        self.assertEqual(trainer._logbook, None)

    def test_logbook(self):
        """Test the logbook property."""
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        self.assertEqual(trainer._logbook, None)

        # Generate the subtrainerulations
        trainer._init_search()
        for subtrainer in trainer.subtrainers:
            subtrainer._init_search()

        # Test the logbook property
        global_logbook_len = 0
        for subtrainer in trainer.subtrainers:
            global_logbook_len += len(subtrainer.logbook)

        self.assertIsInstance(trainer.logbook, Logbook)
        self.assertEqual(global_logbook_len, len(trainer.logbook))

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
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
        self.assertEqual(
            id(trainer1.subtrainer_cls),
            id(trainer2.subtrainer_cls)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Parameters for the trainer
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Set custom params
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyTrainer(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = MyTrainer.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        os.remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        fitness_func = Fitness(dataset)
        num_subtrainers = 2
        subtrainer_cls = SingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.abc.DistributedTrainer`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.abc.DistributedTrainer`
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


if __name__ == '__main__':
    unittest.main()
