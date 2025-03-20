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

"""Unit test for :py:class:`culebra.trainer.aco.PACO_FS`."""

import unittest
from copy import copy, deepcopy
from os import remove

import numpy as np

from culebra.trainer.aco import (
    PACO_FS,
    DEFAULT_ACO_FS_INITIAL_PHEROMONE,
    DEFAULT_ACO_FS_DISCARD_PROB
)
from culebra.solution.feature_selection import Species, Ant
from culebra.fitness_function.feature_selection import KappaNumFeats
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Remove outliers
dataset.remove_outliers()

# Normalize inputs
dataset.robust_scale()
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Species
species = Species(
    num_feats=dataset.num_feats,
    min_size=2,
    min_feat=1,
    max_feat=dataset.num_feats-2
    )

# Training fitness function
training_fitness_function = KappaNumFeats(
    training_data=training_data, cv_folds=5
)

# Test fitness function
test_fitness_function = KappaNumFeats(
    training_data=training_data, test_data=test_data
)

# Lists of banned and feasible nodes
banned_nodes = [0, dataset.num_feats-1]
feasible_nodes = list(range(1, dataset.num_feats-1))


class PACO_FSTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.PACO_FS`."""

    def test_init(self):
        """Test __init__."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = PACO_FS(**params)

        # Check the parameters
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, species)
        self.assertEqual(trainer.fitness_function, training_fitness_function)
        self.assertEqual(
            trainer.initial_pheromone[0], DEFAULT_ACO_FS_INITIAL_PHEROMONE
        )
        self.assertTrue(
            np.all(
                trainer.heuristic[0] ==
                training_fitness_function.heuristic(species)[0]
            )
        )
        self.assertEqual(trainer.col_size, species.num_feats)
        self.assertEqual(trainer.pop_size, species.num_feats)
        self.assertEqual(trainer.discard_prob, DEFAULT_ACO_FS_DISCARD_PROB)

        # Try a custom value for the initial pheromone
        custom_initial_pheromone = 2
        trainer = PACO_FS(**params, initial_pheromone=custom_initial_pheromone)
        self.assertEqual(
            trainer.initial_pheromone[0], custom_initial_pheromone
        )

        # Try invalid types for discard_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                PACO_FS(**params, discard_prob=prob)

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                PACO_FS(**params, discard_prob=prob)

    def test_internals(self):
        """Test the _init_internals and _reset_internals methods."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = PACO_FS(**params)

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
        trainer = PACO_FS(**params)

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
        trainer = PACO_FS(**params)
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
        trainer = PACO_FS(**params)

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

        pop[0].fitness.setValues((good_kappa, bad_nf))
        pop[1].fitness.setValues((bad_kappa, bad_nf))

        col[0].fitness.setValues((bad_kappa, good_nf))
        col[1].fitness.setValues((bad_kappa, bad_nf))

        # Update the population
        trainer._update_pop()

        # The non-dominated ants should be in the new population
        self.assertTrue(pop[0] in trainer.pop)
        self.assertTrue(col[0] in trainer.pop)

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
        trainer = PACO_FS(**params)

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
        trainer1 = PACO_FS(**params)
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
        trainer1 = PACO_FS(**params)
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
        trainer1 = PACO_FS(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = PACO_FS.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        remove(pickle_filename)

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
        trainer = PACO_FS(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.aco.PACO_FS`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.aco.PACO_FS`
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
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
