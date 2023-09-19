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

"""Unit test for :py:class:`culebra.trainer.abc.IslandsTrainer`."""

import unittest
import pickle
from copy import copy, deepcopy
from time import sleep

from culebra.trainer.abc import SinglePopTrainer, IslandsTrainer
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


class MyIslandsTrainer(IslandsTrainer):
    """Dummy implementation of an islands-based trainer."""

    def _init_search(self):
        super()._init_search()
        for island_trainer in self.subpop_trainers:
            island_trainer._init_search()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the metrics before each iteration is run.
        """
        super()._start_iteration()
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            # Fix the current iteration
            subpop_trainer._current_iter = self._current_iter
            # Start the iteration
            subpop_trainer._start_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._do_iteration_stats()

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


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.abc.IslandsTrainer`."""

    def test_init(self):
        """Test :py:meth:`culebra.trainer.abc.IslandsTrainer.__init__`."""
        valid_solution = Solution
        valid_species = Species(dataset.num_feats)
        valid_fitness_func = Fitness(dataset)
        valid_subpop_trainer_cls = MySinglePopTrainerTrainer

        # Try invalid solution classes. Should fail
        invalid_solution_classes = (type, None, 'a', 1)
        for solution_cls in invalid_solution_classes:
            with self.assertRaises(TypeError):
                MyIslandsTrainer(
                    solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls
                )

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for inv_species in invalid_species:
            with self.assertRaises(TypeError):
                MyIslandsTrainer(
                    valid_solution,
                    inv_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls
                )

    def test_best_solutions(self):
        """Test best_solutions."""
        # Parameters for the trainer
        solution_cls = Solution
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_trainer_cls = MySinglePopTrainerTrainer
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsTrainer(**params)

        # Try before the population has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Generate the islands and perform one iteration
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

        # Try again
        best_ones = trainer.best_solutions()

        # Test that a list with only one species is returned
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        for sol in best_ones[0]:
            self.assertIsInstance(sol, solution_cls)

    def test_receive_representatives(self):
        """Test receive_representatives."""
        # Parameters for the trainer
        solution_cls = Solution
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_trainer_cls = MySinglePopTrainerTrainer
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsTrainer(**params)

        # Generate the islands and perform one iteration
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

        for index in range(trainer.num_subpops):
            trainer._communication_queues[index].put([index])

        # Wait for the parallel queue processing
        sleep(1)

        # Call to receive representatives, assigned to
        # island._preprocess_iteration
        # at islands iteration time
        for island_trainer in trainer.subpop_trainers:
            island_trainer._preprocess_iteration()

        # Check the received values
        for index, island_trainer in enumerate(trainer.subpop_trainers):
            self.assertEqual(island_trainer.pop[-1], index)

    def test_send_representatives(self):
        """Test send_representatives."""
        # Parameters for the trainer
        solution_cls = Solution
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_trainer_cls = MySinglePopTrainerTrainer
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsTrainer(**params)

        # Generate the islands
        trainer._init_search()
        for island_trainer in trainer.subpop_trainers:
            island_trainer._init_search()

        trainer._start_iteration()
        trainer._do_iteration()

        # Set an iteration that should not provoke representatives sending
        for island_trainer in trainer.subpop_trainers:
            island_trainer._current_iter = trainer.representation_freq + 1

            # Call to send representatives, assigned to
            # island._postprocess_iteration at islands iteration time
            island_trainer._postprocess_iteration()

        # All the queues should be empty
        for index in range(trainer.num_subpops):
            self.assertTrue(trainer._communication_queues[index].empty())

        # Set an iteration that should provoke representatives sending
        for island_trainer in trainer.subpop_trainers:
            island_trainer._current_iter = trainer.representation_freq

            # Call to send representatives, assigned to
            # island._postprocess_iteration at islands iteration time
            island_trainer._postprocess_iteration()

            # Wait for the parallel queue processing
            sleep(1)

        # All the queues shouldn't be empty
        for index in range(trainer.num_subpops):
            self.assertFalse(trainer._communication_queues[index].empty())
            while not trainer._communication_queues[index].empty():
                trainer._communication_queues[index].get()

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the trainer
        solution_cls = Solution
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_trainer_cls = MySinglePopTrainerTrainer
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyIslandsTrainer(**params)
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
        # Parameters for the trainer
        solution_cls = Solution
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_trainer_cls = MySinglePopTrainerTrainer
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyIslandsTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Set custom params
        # Parameters for the trainer
        solution_cls = Solution
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_trainer_cls = MySinglePopTrainerTrainer
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_trainer_cls": subpop_trainer_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyIslandsTrainer(**params)

        data = pickle.dumps(trainer1)
        trainer2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`culebra.trainer.abc.IslandsTrainer`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`culebra.trainer.abc.IslandsTrainer`
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
