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

"""Unit test for :class:`culebra.trainer.abc.IslandsTrainer`."""

import unittest
from os import remove
from copy import copy, deepcopy

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer import (
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)
from culebra.trainer.topology import full_connected_destinations
from culebra.trainer.abc import SingleSpeciesTrainer, IslandsTrainer
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
)
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

fitness_func = KappaNumFeats(dataset)


class MySingleSpeciesTrainer(SingleSpeciesTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self.sol = Solution(self.species, self.fitness_function.fitness_cls)
        self.evaluate(self.sol)


class MyIslandsTrainer(IslandsTrainer):
    """Dummy implementation of an islands-based trainer."""

    def _init_search(self):
        super()._init_search()
        for island_trainer in self.subtrainers:
            island_trainer._init_search()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the metrics before each iteration is run.
        """
        super()._start_iteration()
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            # Fix the current iteration
            subtrainer._current_iter = self._current_iter
            # Start the iteration
            subtrainer._start_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration_stats()

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


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.abc.IslandsTrainer`."""

    def test_init(self):
        """Test :meth:`culebra.trainer.abc.IslandsTrainer.__init__`."""
        valid_solution = Solution
        valid_species = Species(dataset.num_feats)
        valid_subtrainer_cls = MySingleSpeciesTrainer

        # Try invalid solution classes. Should fail
        invalid_solution_classes = (type, None, 'a', 1)
        for solution_cls in invalid_solution_classes:
            with self.assertRaises(TypeError):
                MyIslandsTrainer(
                    solution_cls,
                    valid_species,
                    fitness_func,
                    valid_subtrainer_cls
                )

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for inv_species in invalid_species:
            with self.assertRaises(TypeError):
                MyIslandsTrainer(
                    valid_solution,
                    inv_species,
                    fitness_func,
                    valid_subtrainer_cls
                )

        # Try invalid representation topology function. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsTrainer(
                    fitness_func,
                    valid_subtrainer_cls,
                    representation_topology_func=func
                )

        # Try invalid types for representation topology function parameters
        # Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyIslandsTrainer(
                    fitness_func,
                    valid_subtrainer_cls,
                    representation_topology_func_params=params
                )

        # Try a valid representation topology function
        valid_representation_topology_func = full_connected_destinations
        trainer = MyIslandsTrainer(
            valid_solution,
            valid_species,
            fitness_func,
            valid_subtrainer_cls,
            representation_topology_func=valid_representation_topology_func
        )
        self.assertEqual(
            trainer.representation_topology_func,
            valid_representation_topology_func
        )

        # Try valid representation topology function params
        valid_params = {"offset": 2}
        trainer = MyIslandsTrainer(
            valid_solution,
            valid_species,
            fitness_func,
            valid_subtrainer_cls,
            representation_topology_func_params=valid_params
        )
        self.assertEqual(
            trainer.representation_topology_func_params,
            valid_params
        )

        # Test default params
        trainer = MyIslandsTrainer(
            valid_solution,
            valid_species,
            fitness_func,
            valid_subtrainer_cls,
        )

        self.assertEqual(
            trainer.representation_topology_func,
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
        )
        self.assertEqual(
            trainer.representation_topology_func_params,
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
        )

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the trainer
        solution_cls = Solution
        num_subtrainers = 2
        subtrainer_cls = MySingleSpeciesTrainer
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
        num_subtrainers = 2
        subtrainer_cls = MySingleSpeciesTrainer
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
        trainer1 = MyIslandsTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Set custom params
        solution_cls = Solution
        num_subtrainers = 2
        subtrainer_cls = MySingleSpeciesTrainer
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
        trainer1 = MyIslandsTrainer(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = MyIslandsTrainer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        solution_cls = Solution
        num_subtrainers = 2
        subtrainer_cls = MySingleSpeciesTrainer
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
        trainer = MyIslandsTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: culebra.trainer.abc.IslandsTrainer
        :param trainer2: The second trainer
        :type trainer2: culebra.trainer.abc.IslandsTrainer
        """
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.subtrainer_params), id(trainer2.subtrainer_params)
        )
        self.assertEqual(
            trainer1.subtrainer_params, trainer2.subtrainer_params
        )


if __name__ == '__main__':
    unittest.main()
