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

"""Unit test for :py:class:`~culebra.trainer.abc.SequentialMultiPopTrainer`."""

import unittest
import os
from time import sleep

from culebra.trainer.abc import SinglePopTrainer, SequentialMultiPopTrainer
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
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

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MySinglePopTrainerTrainer(SinglePopTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        for _ in range(self.pop_size):
            sol = Solution(self.species, self.fitness_function.Fitness)
            self.evaluate(sol)
            self._pop.append(sol)


class MyMultiPopTrainerTrainer(SequentialMultiPopTrainer):
    """Dummy implementation of a sequential multi-population trainer."""

    def _generate_subpop_trainers(self):
        self._subpop_trainers = []
        subpop_params = {
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
        ) in enumerate(self.subpop_trainer_checkpoint_filenames):
            subpop_trainer = self.subpop_trainer_cls(**subpop_params)
            subpop_trainer.checkpoint_filename = checkpoint_filename

            subpop_trainer.index = index
            subpop_trainer.container = self

            subpop_trainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subpop_trainer.__class__._postprocess_iteration = (
                self.send_representatives
            )
            self._subpop_trainers.append(subpop_trainer)

    @staticmethod
    def receive_representatives(subpop_trainer) -> None:
        """Receive representative solutions.

        :param subpop_trainer: The subpopulation trainer receiving
            representatives
        :type subpop_trainer: :py:class:`culebra.trainer.abc.SinglePopTrainer`
        """
        # Receive all the solutions in the queue
        queue = subpop_trainer.container._communication_queues[
            subpop_trainer.index
        ]
        while not queue.empty():
            subpop_trainer._pop.append(queue.get())

    @staticmethod
    def send_representatives(subpop_trainer) -> None:
        """Send representatives.

        :param subpop_trainer: The sender subpopulation trainer
        :type subpop_trainer: :py:class:`culebra.trainer.abc.SinglePopTrainer`
        """
        container = subpop_trainer.container
        # Check if sending should be performed
        if subpop_trainer._current_iter % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subpop_trainer.index,
                container.num_subpops,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                for _ in range(container.representation_size):
                    # Select one representative each time
                    (sol,) = container.representation_selection_func(
                        subpop_trainer.pop,
                        1,
                        **container.representation_selection_func_params
                    )
                    container._communication_queues[dest].put(sol)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.abc.SequentialMultiPopTrainer`."""

    def test_init(self):
        """Test the constructor."""
        # Test default params
        trainer = MyMultiPopTrainerTrainer(
            Fitness(dataset),
            MySinglePopTrainerTrainer
        )

        self.assertEqual(trainer._current_iter, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default trainer
        fitness_func = Fitness(dataset)
        subpop_trainer_cls = MySinglePopTrainerTrainer
        num_subpops = 2
        params = {
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False
        }

        # Test default params
        trainer1 = MyMultiPopTrainerTrainer(**params)

        # Create the subpopulations
        trainer1._init_search()

        # Set state attributes to dummy values
        trainer1._runtime = 10
        trainer1._current_iter = 19

        # Save the state of trainer1
        trainer1._save_state()

        # Create another trainer
        trainer2 = MyMultiPopTrainerTrainer(**params)

        # Trainer2 has no subpopulations yet
        self.assertEqual(trainer2.subpop_trainers, None)

        # Load the state of trainer1 into trainer2
        trainer2._init_search()

        # Check that the state attributes of trainer2 are equal to those of
        # trainer1
        self.assertEqual(trainer1.runtime, trainer2.runtime)
        self.assertEqual(trainer1._current_iter, trainer2._current_iter)

        # Remove the checkpoint files
        os.remove(trainer1.checkpoint_filename)
        for file in trainer1.subpop_trainer_checkpoint_filenames:
            os.remove(file)

    def test_migrations(self):
        """Test _preprocess_iteration and _postprocess_iteration."""
        # Create a default trainer
        fitness_func = Fitness(dataset)
        subpop_trainer_cls = MySinglePopTrainerTrainer
        num_subpops = 2
        params = {
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False
        }

        # Test default params
        trainer = MyMultiPopTrainerTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

        # All the queues should be empty
        for queue in trainer._communication_queues:
            self.assertTrue(queue.empty())

        # Send migrants
        trainer._postprocess_iteration()

        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should not be empty
        for queue in trainer._communication_queues:
            self.assertFalse(queue.empty())

        # Receive migrants
        trainer._preprocess_iteration()

        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should be empty again
        for queue in trainer._communication_queues:
            self.assertTrue(queue.empty())

    def test_search(self):
        """Test _search and _finish_search."""
        # Create a default trainer
        fitness_func = Fitness(dataset)
        subpop_trainer_cls = MySinglePopTrainerTrainer
        num_subpops = 2
        max_num_iters = 10
        params = {
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "max_num_iters": max_num_iters,
            "checkpoint_enable": False,
            "verbose": False
        }

        # Test the search method
        trainer = MyMultiPopTrainerTrainer(**params)
        trainer._init_search()

        self.assertEqual(trainer._current_iter, 0)
        trainer.train()

        self.assertEqual(trainer._current_iter, max_num_iters)
        num_evals = 0
        for island_trainer in trainer.subpop_trainers:
            self.assertEqual(island_trainer._current_iter, max_num_iters)
            num_evals += island_trainer.num_evals

        self.assertEqual(trainer.num_evals, num_evals)


if __name__ == '__main__':
    unittest.main()
