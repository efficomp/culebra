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

"""Unit test for :py:class:`~culebra.trainer.ea.SequentialCooperativeEA`."""

import os
import unittest
from time import sleep

from culebra.trainer.ea import ElitistEA, SequentialCooperativeEA
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.cooperative import KappaNumFeatsC as Fitness
from culebra.tools import Dataset

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.SequentialCooperativeEA`."""

    def test_init(self):
        """Test the constructor."""
        # Test default params
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA
        }

        # Create the trainer
        trainer = SequentialCooperativeEA(**params)

        self.assertEqual(trainer._current_iter, None)

    def test_new_state(self):
        """Test _new_state."""
        pop_size = 10
        # Create a default trainer
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA,
            "pop_sizes": pop_size,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = SequentialCooperativeEA(**params)
        trainer._init_internals()
        trainer._new_state()

        # Check the current iteration
        self.assertEqual(trainer._current_iter, 0)

        # Test that the logbook is None
        self.assertEqual(trainer._logbook, None)

        # Test that current_iter is 0 and that all the subpopulations trainers
        # have been initializated
        for index in range(trainer.num_subtrainers):
            self.assertEqual(len(trainer.subtrainers[index].pop), pop_size)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default trainer
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False
        }

        # Create the trainer
        trainer1 = SequentialCooperativeEA(**params)

        # Create the subpopulations
        trainer1._init_search()

        # Set state attributes to dummy values
        trainer1._runtime = 10
        trainer1._current_iter = 19

        # Save the state of trainer1
        trainer1._save_state()

        # Create another trainer
        trainer2 = SequentialCooperativeEA(**params)

        # Trainer2 has no subpopulation trainers yet
        self.assertEqual(trainer2.subtrainers, None)

        # Load the state of trainer1 into trainer2
        trainer2._init_search()

        # Check that the state attributes of trainer2 are equal to those of
        # trainer1
        self.assertEqual(trainer1.runtime, trainer2.runtime)
        self.assertEqual(trainer1._current_iter, trainer2._current_iter)
        for (
            subtrainer1,
            subtrainer2
        ) in zip(trainer1.subtrainers, trainer2.subtrainers):
            self.assertEqual(
                len(subtrainer1.pop), len(subtrainer2.pop)
            )

        # Remove the checkpoint files
        os.remove(trainer1.checkpoint_filename)
        for file in trainer1.subtrainer_checkpoint_filenames:
            os.remove(file)

    def test_representatives_exchange(self):
        """Test _preprocess_iteration and _postprocess_iteration."""
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = SequentialCooperativeEA(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check that all the individuals are evaluated
        for subtrainer in trainer.subtrainers:
            for ind in subtrainer.pop:
                self.assertTrue(ind.fitness.valid)

        # All the queues should be empty
        for queue in trainer._communication_queues:
            self.assertTrue(queue.empty())

        # Delete initial representatives
        for subtrainer in trainer.subtrainers:
            subtrainer._representatives = [
                [None] * trainer.num_subtrainers
            ] * trainer.representation_size

        # Send representatives
        trainer._postprocess_iteration()
        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should not be empty
        for queue in trainer._communication_queues:
            self.assertFalse(queue.empty())

        # Receive representatives
        trainer._preprocess_iteration()
        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should be empty again
        for queue in trainer._communication_queues:
            self.assertTrue(queue.empty())

        # Check the received representatives
        for (
            subpop_index,
            subtrainer
        ) in enumerate(trainer.subtrainers):
            for representatives in subtrainer._representatives:
                for ind_index, ind in enumerate(representatives):
                    if ind_index == subpop_index:
                        self.assertEqual(ind, None)
                    else:
                        self.assertTrue(ind.fitness.valid)

    def test_search(self):
        """Test _search."""
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA,
            "max_num_iters": 2,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = SequentialCooperativeEA(**params)
        trainer._init_search()

        self.assertEqual(trainer._current_iter, 0)
        trainer.train()

        max_num_iters = params["max_num_iters"]
        self.assertEqual(trainer._current_iter, max_num_iters)
        num_evals = 0
        for subtrainer in trainer.subtrainers:
            self.assertEqual(subtrainer._current_iter, max_num_iters)
            num_evals += subtrainer.num_evals

        self.assertEqual(trainer.num_evals, num_evals)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA,
            "max_num_iters": 2,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = SequentialCooperativeEA(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
