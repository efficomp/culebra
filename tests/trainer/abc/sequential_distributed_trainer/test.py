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

"""Test for :py:class:`~culebra.trainer.abc.SequentialDistributedTrainer`."""

import unittest
import os

from culebra.trainer.abc import (
    SingleSpeciesTrainer,
    SequentialDistributedTrainer
)
from culebra.trainer.topology import full_connected_destinations
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
)
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MySingleSpeciesTrainer(SingleSpeciesTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self.sol = Solution(self.species, self.fitness_function.Fitness)
        self.evaluate(self.sol)


class MyDistributedTrainer(SequentialDistributedTrainer):
    """Dummy implementation of a sequential distributed trainer."""

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
        return full_connected_destinations

    @property
    def representation_topology_func_params(self):
        """Get and set the representation topology function parameters."""
        return {}


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.abc.SequentialDistributedTrainer`."""

    def test_init(self):
        """Test the constructor."""
        # Test default params
        trainer = MyDistributedTrainer(
            Fitness(dataset),
            MySingleSpeciesTrainer
        )

        self.assertEqual(trainer._current_iter, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default trainer
        fitness_func = Fitness(dataset)
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False
        }

        # Test default params
        trainer1 = MyDistributedTrainer(**params)

        # Create the subtrainers
        trainer1._init_search()

        # Set state attributes to dummy values
        trainer1._runtime = 10
        trainer1._current_iter = 19

        # Save the state of trainer1
        trainer1._save_state()

        # Create another trainer
        trainer2 = MyDistributedTrainer(**params)

        # Trainer2 has no subtrainers yet
        self.assertEqual(trainer2.subtrainers, None)

        # Load the state of trainer1 into trainer2
        trainer2._init_search()

        # Check that the state attributes of trainer2 are equal to those of
        # trainer1
        self.assertEqual(trainer1.runtime, trainer2.runtime)
        self.assertEqual(trainer1._current_iter, trainer2._current_iter)

        # Remove the checkpoint files
        os.remove(trainer1.checkpoint_filename)
        for file in trainer1.subtrainer_checkpoint_filenames:
            os.remove(file)

    def test_search(self):
        """Test _search and _finish_search."""
        # Create a default trainer
        fitness_func = Fitness(dataset)
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        max_num_iters = 10
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "max_num_iters": max_num_iters,
            "checkpoint_enable": False,
            "verbose": False
        }

        # Test the search method
        trainer = MyDistributedTrainer(**params)
        trainer._init_search()

        self.assertEqual(trainer._current_iter, 0)
        trainer.train()

        self.assertEqual(trainer._current_iter, max_num_iters)
        num_evals = 0
        for island_trainer in trainer.subtrainers:
            self.assertEqual(island_trainer._current_iter, max_num_iters)
            num_evals += island_trainer.num_evals

        self.assertEqual(trainer.num_evals, num_evals)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Create a default trainer
        fitness_func = Fitness(dataset)
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        max_num_iters = 10
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "max_num_iters": max_num_iters,
            "checkpoint_enable": False,
            "verbose": False
        }

        # Test the search method
        trainer = MyDistributedTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
