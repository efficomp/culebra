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

"""Unit test for :py:class:`culebra.trainer.abc.MultiSpeciesTrainer`."""

import unittest
from os import remove
from copy import copy, deepcopy

from culebra.abc import Solution
from culebra.trainer.abc import MultiSpeciesTrainer
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Solution as ClassifierOptimizationSolution
)
from culebra.fitness_function.feature_selection import NumFeats as FitnessFunc
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class MyTrainer(MultiSpeciesTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self._current_iter_evals = 10


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.abc.MultiSpeciesTrainer`."""

    def test_init(self):
        """Test the constructor."""
        valid_solution_classes = [
            Solution,
            ClassifierOptimizationSolution,
            FeatureSelectionSolution
        ]
        invalid_solution_class_types = (1, int, len, None)
        invalid_solution_class_values = (
            [valid_solution_classes[0], 2],
            [None, valid_solution_classes[0]]
        )

        valid_species = [
            # Species to optimize a SVM-based classifier
            ClassifierOptimizationSpecies(
                lower_bounds=[0, 0],
                upper_bounds=[100000, 100000],
                names=["C", "gamma"]
            ),
            # Species for the feature selection problem
            FeatureSelectionSpecies(dataset.num_feats),
        ]
        invalid_species_types = (1, int, len, None)
        invalid_species_values = (
            [valid_species[0], 2],
            [None, valid_species[0]]
        )

        valid_fitness_func = FitnessFunc(dataset)

        # Try invalid types for the individual classes. Should fail
        for solution_cls in invalid_solution_class_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    solution_cls,
                    valid_species,
                    valid_fitness_func
                )

        # Try invalid values for the individual classes. Should fail
        for solution_classes in invalid_solution_class_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    solution_classes,
                    valid_species,
                    valid_fitness_func
                )

        # Try different values of solution_cls for each subtrainer
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func
        )
        for cls1, cls2 in zip(
            trainer.solution_classes, valid_solution_classes
        ):
            self.assertEqual(cls1, cls2)

        # Try invalid types for the species. Should fail
        for species in invalid_species_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_classes,
                    species,
                    valid_fitness_func
                )

        # Try invalid values for the species. Should fail
        for species in invalid_species_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_classes,
                    species,
                    valid_fitness_func
                )

        # Try different values of species for each subtrainer
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func
        )
        for species1, species2 in zip(
            trainer.species, valid_species
        ):
            self.assertEqual(species1, species2)

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "fitness_function": FitnessFunc(dataset),
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
            id(trainer1.species),
            id(trainer2.species)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "fitness_function": FitnessFunc(dataset),
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
        # Parameters for the trainer
        params = {
            "solution_classes": [
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "fitness_function": FitnessFunc(dataset),
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
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        params = {
            "solution_classes": [
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "fitness_function": FitnessFunc(dataset),
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
        :type trainer1: :py:class:`~culebra.trainer.abc.MultiSpeciesTrainer`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.abc.MultiSpeciesTrainer`
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
        for spe1, spe2 in zip(trainer1.species, trainer2.species):
            self.assertNotEqual(id(spe1), id(spe2))


if __name__ == '__main__':
    unittest.main()
