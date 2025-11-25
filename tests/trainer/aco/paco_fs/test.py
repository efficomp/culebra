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

"""Unit test for :class:`culebra.trainer.aco.PACOFS`."""

import unittest
from copy import copy, deepcopy
from os import remove

import numpy as np
from deap.tools import sortNondominated

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco import PACOFS
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


class PACOFSTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.PACOFS`."""

    def test_internals(self):
        """Test the _init_internals and _reset_internals methods."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)

        # Init the trainer interal structures
        trainer._init_internals()

        # Check the choice info matrix
        self.assertFalse(trainer._choice_info is None)
        self.assertIsInstance(trainer._choice_info, np.ndarray)
        self.assertEqual(
            trainer._choice_info.shape,
            (
                trainer.species.num_feats,
                trainer.species.num_feats
            )
        )

        # Reset the internals
        trainer._reset_internals()
        # Check the choice info matrix
        self.assertEqual(trainer._choice_info, None)
        # Check the pheromone matrices
        self.assertEqual(trainer.pheromone, None)

    def test_new_state(self):
        """Test _new_state."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the population. Should be empty
        self.assertIsInstance(trainer.pop, list)
        self.assertEqual(len(trainer.pop), 0)

        # Test the pheromone
        for pher, init_pher in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(pher >= init_pher))

    def test_state(self):
        """Test the get_state and _set_state methods."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)
        trainer._init_search()
        trainer._new_state()

        # Fill the pop
        while len(trainer.pop) < trainer.pop_size:
            trainer.pop.append(trainer._generate_ant())
        trainer._update_pheromone()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertIsInstance(state["pop"], list)
        self.assertEqual(len(state["pop"]), trainer.pop_size)

        # Get the population and the pheromone matrices
        pop = trainer.pop
        pheromone = trainer.pheromone

        # Reset the trainer
        trainer.reset()

        # Set the new state
        trainer._set_state(state)

        # Test if the pop has been restored
        self.assertEqual(len(pop), len(trainer.pop))
        for ant1, ant2 in zip(pop, trainer.pop):
            self.assertEqual(ant1, ant2)

        # Test if the pheromone has been restored
        for pher1, pher2 in zip(pheromone, trainer.pheromone):
            self.assertTrue(np.all(pher1 == pher2))

    def test_update_pop(self):
        """Test _update_pop."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "pop_size": 2,
            "col_size": 2,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)

        # Try to update the population with an empty elite
        trainer._init_search()
        trainer._start_iteration()
        trainer._generate_col()

        # Fill the pop
        while len(trainer.pop) < trainer.pop_size:
            trainer.pop.append(trainer._generate_ant())

        # Force a non-dominated ants in pop[0] and col[0]
        pop = copy(trainer.pop)
        col = copy(trainer.col)
        good_kappa = 0.9
        bad_kappa = 0.5
        good_nf = 4
        bad_nf = 8

        pop[0].fitness.values = ((good_kappa, bad_nf))
        pop[1].fitness.values = ((bad_kappa, bad_nf))

        col[0].fitness.values = ((bad_kappa, good_nf))
        col[1].fitness.values = ((bad_kappa, bad_nf))

        # Update the population
        trainer._update_pop()

        # The non-dominated ants should be in the new population
        self.assertTrue(pop[0] in trainer.pop)
        self.assertTrue(col[0] in trainer.pop)

    def test_pheromone_amount(self):
        """Test _pheromone_amount."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)

        # Init the search
        trainer._init_search()

        # Reset the pheromone values
        trainer._init_pheromone()

        # Fill the pop
        while len(trainer.pop) < trainer.pop_size:
            trainer.pop.append(trainer._generate_ant())
        trainer._pareto_fronts = sortNondominated(trainer.pop, trainer.pop_size)
        
        for front_index, front in enumerate(trainer._pareto_fronts):
            for ant in front:
                self.assertEqual(
                    trainer._pheromone_amount(ant), (front_index + 1,)
                )

    def test_update_pheromone(self):
        """Test _update_pheromone."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)

        # Init the search
        trainer._init_search()

        # Reset the pheromone values
        trainer._init_pheromone()

        # Fill the pop
        while len(trainer.pop) < trainer.pop_size:
            trainer.pop.append(trainer._generate_ant())

        # Update pheromone according to the population
        trainer._update_pheromone()

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
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer1 = PACOFS(**params)
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
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer1 = PACOFS(**params)
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
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer1 = PACOFS(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = PACOFS.load(serialized_filename)

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
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = PACOFS(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: ~culebra.trainer.aco.PACOFS
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.aco.PACOFS
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
