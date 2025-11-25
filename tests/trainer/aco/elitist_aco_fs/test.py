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

"""Unit test for :class:`culebra.trainer.aco.ElitistACOFS`."""

import unittest
from copy import copy, deepcopy
from os import remove

import numpy as np

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco import ElitistACOFS
from culebra.solution.feature_selection import Species, Ant
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

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Species
species = Species(
    num_feats=dataset.num_feats,
    min_size=2,
    min_feat=1,
    max_feat=dataset.num_feats-2
    )

# Training fitness function
training_fitness_function = KappaNumFeats(training_data, cv_folds=5)

# Test fitness function
test_fitness_function = KappaNumFeats(training_data, test_data=test_data)

# Lists of banned and feasible nodes
banned_nodes = [0, dataset.num_feats-1]
feasible_nodes = list(range(1, dataset.num_feats-1))


class ElitistACOFSTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.ElitistACOFS`."""

    def test_state(self):
        """Test the get_state and _set_state methods."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ElitistACOFS(**params)
        trainer._init_search()
        trainer._new_state()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["elite"], trainer._elite)

        # Get the elite and the pheromone matrices
        elite = trainer._elite
        pheromone = trainer.pheromone

        # Reset the trainer
        trainer.reset()

        # Set the new state
        trainer._set_state(state)

        # Test if the elite has been restored
        self.assertEqual(elite, state["elite"])

        # Test if the pheromone has been restored
        for pher1, pher2 in zip(pheromone, trainer.pheromone):
            self.assertTrue(np.all(pher1 == pher2))

    def test_update_elite(self):
        """Test _update_elite."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "col_size": 50,
            "verbose": False
        }

        # Create the trainer
        trainer = ElitistACOFS(**params)

        # Try to update the elite
        trainer._init_search()
        trainer._start_iteration()
        trainer._generate_col()
        trainer._update_elite()

        # Force a non-dominated ants in elite[0] and col[0]
        elite = copy(trainer._elite)
        col = copy(trainer.col)
        good_kappa = 0.9
        bad_kappa = 0.5
        good_nf = 4
        bad_nf = 8

        elite[0].fitness.values = ((good_kappa, bad_nf))
        for elite_ant in elite[1:]:
            elite_ant.fitness.values = ((bad_kappa, bad_nf))

        col[0].fitness.valuesalues = ((bad_kappa, good_nf))
        for col_ant in col[1:]:
            col_ant.fitness.values = ((bad_kappa, bad_nf))

        # Update the elite
        trainer._update_elite()

        # The non-dominated ants should be in the new elite
        self.assertTrue(elite[0] in trainer._elite)
        self.assertTrue(col[0] in trainer._elite)

    def test_pheromone_amount(self):
        """Test _pheromone_amount."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ElitistACOFS(**params)

        # Init the search
        trainer._init_search()

        ant = trainer._generate_ant()
        
        self.assertTrue(
            list(trainer._pheromone_amount(ant)) == trainer.initial_pheromone
        )

    def test_update_pheromone(self):
        """Test _update_pheromone."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ElitistACOFS(**params)

        # Init the search
        trainer._init_search()

        # Reset the pheromone values
        trainer._init_pheromone()

        # Update pheromone according to the elite
        trainer._do_iteration()

        # Check the pheromone matrix
        self.assertTrue(
            np.all(
                trainer.pheromone[0] >= trainer.initial_pheromone[0]
            )
        )
        self.assertTrue(
            np.any(
                trainer.pheromone[0] != trainer.initial_pheromone[0]
            )
        )

    def test_copy(self):
        """Test the __copy__ method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer1 = ElitistACOFS(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertNotEqual(id(trainer1), id(trainer2))

        # The objects attributes are shared
        self.assertEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertEqual(id(trainer1.species), id(trainer2.species))

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer1 = ElitistACOFS(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer1 = ElitistACOFS(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = ElitistACOFS.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ElitistACOFS(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: ~culebra.trainer.aco.ElitistACOFS
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.aco.ElitistACOFS
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(id(trainer1.species), id(trainer2.species))
        self.assertEqual(
            id(trainer1.species.num_feats), id(trainer2.species.num_feats)
        )
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
