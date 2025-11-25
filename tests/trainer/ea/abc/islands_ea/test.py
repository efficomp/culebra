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

"""Unit test for :class:`~culebra.trainer.ea.abc.IslandsEA`."""

import unittest
from time import sleep

from culebra.solution.feature_selection import BitVector
from culebra.trainer.abc import SingleSpeciesTrainer
from culebra.trainer.ea.abc import SinglePopEA, IslandsEA
from culebra.solution.feature_selection import Species
from culebra.fitness_function import MultiObjectiveFitnessFunction
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats
)
from culebra.tools import Dataset


# Fitness function
def KappaNumFeats(
    training_data,
    test_data=None,
    cv_folds=None,
    classifier=None
):
    """Fitness Function."""
    return MultiObjectiveFitnessFunction(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            cv_folds=cv_folds,
            classifier=classifier
        ),
        NumFeats()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MyIslandsEA(IslandsEA):
    """Dummy implementation of an islands-based EA."""

    def _init_search(self):
        super()._init_search()
        for island_trainer in self.subtrainers:
            island_trainer._init_search()

    def _start_iteration(self):
        """Start an iteration."""
        super()._start_iteration()
        # For all the subpopulation trainers
        for subtrainer in self.subtrainers:
            # Fix the current iteration
            subtrainer._current_iter = self._current_iter
            # Start the iteration
            subtrainer._start_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # For all the subpopulation trainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # For all the subpopulation trainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration_stats()

    def _generate_subtrainers(self) -> None:
        """Generate the subtrainers.

        Also assign an :attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        subpopulation :class:`~culebra.trainer.ea.abc.SinglePopEA` trainer,
        change the subpopulation trainers'
        :attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subpopulation trainers, if necessary
        """

        def subtrainers_properties():
            """Return the subpopulation trainers' properties."""
            # Get the attributes from the container trainer
            cls = self.subtrainer_cls
            properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            properties.update(self.subtrainer_params)

            return properties

        # Get the subpopulations properties
        properties = subtrainers_properties()

        # Generate the subpopulations
        self._subtrainers = []

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subtrainer_checkpoint_filenames):
            subtrainer = self.subtrainer_cls(**properties)
            subtrainer.checkpoint_filename = checkpoint_filename
            subtrainer.index = index
            subtrainer.container = self
            subtrainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subtrainer.__class__._postprocess_iteration = (
                self.send_representatives
            )
            self._subtrainers.append(subtrainer)


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.trainer.ea.abc.IslandsEA`."""

    def test_subtrainer_cls(self):
        """Test the subtrainer_cls property."""
        solution_cls = BitVector
        valid_fitness_func = KappaNumFeats(dataset)
        valid_subtrainer_cls = SinglePopEA

        # Try invalid subtrainer_cls. Should fail
        invalid_trainer_classes = (
            tuple, str, None, 'a', 1, SingleSpeciesTrainer
        )
        for cls in invalid_trainer_classes:
            with self.assertRaises(TypeError):
                IslandsEA(
                    solution_cls,
                    species,
                    valid_fitness_func,
                    cls
                )

        # Test default params
        trainer = MyIslandsEA(
            solution_cls,
            species,
            valid_fitness_func,
            valid_subtrainer_cls
        )
        self.assertEqual(trainer.subtrainer_cls, valid_subtrainer_cls)

    def test_best_solutions(self):
        """Test best_solutions."""
        # Parameters for the trainer
        solution_cls = BitVector
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = SinglePopEA
        num_subtrainers = 2
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsEA(**params)

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
        solution_cls = BitVector
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = SinglePopEA
        num_subtrainers = 2
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsEA(**params)

        # Generate the islands and perform one iteration
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

        for index in range(trainer.num_subtrainers):
            trainer._communication_queues[index].put([index])

        # Wait for the parallel queue processing
        sleep(1)

        # Call to receive representatives, assigned to
        # island._preprocess_iteration
        # at islands iteration time
        for island_trainer in trainer.subtrainers:
            island_trainer._preprocess_iteration()

        # Check the received values
        for index, island_trainer in enumerate(trainer.subtrainers):
            self.assertEqual(island_trainer.pop[-1], index)

    def test_send_representatives(self):
        """Test send_representatives."""
        solution_cls = BitVector
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = SinglePopEA
        num_subtrainers = 2
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsEA(**params)

        # Generate the islands
        trainer._init_search()
        for island_trainer in trainer.subtrainers:
            island_trainer._init_search()

        trainer._start_iteration()
        trainer._do_iteration()

        # Set an iteration that should not provoke representatives sending
        for island_trainer in trainer.subtrainers:
            island_trainer._current_iter = trainer.representation_freq + 1

            # Call to send representatives, assigned to
            # island._postprocess_iteration at islands iteration time
            island_trainer._postprocess_iteration()

        # All the queues should be empty
        for index in range(trainer.num_subtrainers):
            self.assertTrue(trainer._communication_queues[index].empty())

        # Set an iteration that should provoke representatives sending
        for island_trainer in trainer.subtrainers:
            island_trainer._current_iter = trainer.representation_freq

            # Call to send representatives, assigned to
            # island._postprocess_iteration at islands iteration time
            island_trainer._postprocess_iteration()

            # Wait for the parallel queue processing
            sleep(1)

        # All the queues shouldn't be empty
        for index in range(trainer.num_subtrainers):
            self.assertFalse(trainer._communication_queues[index].empty())
            while not trainer._communication_queues[index].empty():
                trainer._communication_queues[index].get()

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        solution_cls = BitVector
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = SinglePopEA
        num_subtrainers = 2
        params = {
            "solution_cls": solution_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyIslandsEA(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
