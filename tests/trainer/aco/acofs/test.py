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

"""Unit test for :py:class:`culebra.trainer.aco.ACOFS`."""

import unittest
from itertools import repeat
from copy import copy

import numpy as np

from culebra.trainer.aco import (
    ACOFS,
    DEFAULT_ACOFS_INITIAL_PHEROMONE,
    DEFAULT_ACOFS_DISCARD_PROB
)
from culebra.solution.feature_selection import Species, Ant
from culebra.fitness_function.feature_selection import KappaNumFeats
from culebra.tools import Dataset


# Dataset
DATASET_PATH = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/australian/australian.dat"
)

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Remove outliers
dataset.remove_outliers()

# Normalize inputs between 0 and 1
dataset.normalize()
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Species
species = Species(
    num_feats=dataset.num_feats,
    min_size=2,
    min_feat=1,
    max_feat=dataset.num_feats-2
    )

# Training fitness function, 50% of samples used for validation
training_fitness_function = KappaNumFeats(
    training_data=training_data, test_prop=0.5
)

# Test fitness function
test_fitness_function = KappaNumFeats(
    training_data=training_data, test_data=test_data
)

# Lists of banned and feasible nodes
banned_nodes = [0, dataset.num_feats-1]
feasible_nodes = list(range(1, dataset.num_feats-1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.ACOFS`."""

    def test_init(self):
        """Test __init__."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        # Check the parameters
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, species)
        self.assertEqual(trainer.fitness_function, training_fitness_function)
        self.assertEqual(
            trainer.initial_pheromone[0], DEFAULT_ACOFS_INITIAL_PHEROMONE
        )
        self.assertTrue(
            np.all(
                trainer.heuristic[0] ==
                training_fitness_function.heuristic(species)[0]
            )
        )
        self.assertEqual(trainer.col_size, species.num_feats)
        self.assertEqual(trainer.pop_size, species.num_feats)
        self.assertEqual(trainer.discard_prob, DEFAULT_ACOFS_DISCARD_PROB)

        # Try a custom value for the initial pheromone
        custom_initial_pheromone = 2
        trainer = ACOFS(**params, initial_pheromone=custom_initial_pheromone)
        self.assertEqual(
            trainer.initial_pheromone[0], custom_initial_pheromone
        )

        # Try invalid types for discard_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                ACOFS(**params, discard_prob=prob)

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                ACOFS(**params, discard_prob=prob)

    def test_num_pheromone_matrices(self):
        """Test the num_pheromone_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        self.assertEqual(trainer.num_pheromone_matrices, 1)

    def test_num_heuristic_matrices(self):
        """Test the num_heuristic_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        self.assertEqual(
            trainer.num_heuristic_matrices, 1
        )

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        # Try to get the choice info before the search initialization
        choice_info = trainer.choice_info
        self.assertEqual(choice_info, None)

        # Try to get the choice_info after initializing the search
        trainer._init_pheromone()
        trainer._calculate_choice_info()
        choice_info = trainer.choice_info

        # Check the probabilities for banned nodes. Should be 0
        for node in banned_nodes:
            self.assertAlmostEqual(np.sum(choice_info[node]), 0)

        for node in feasible_nodes:
            self.assertAlmostEqual(
                np.sum(choice_info[node]),
                np.sum(
                    trainer.pheromone[0][node] * trainer.heuristic[0][node]
                )
            )

    def test_internals(self):
        """Test the _init_internals and _reset_internals methods."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

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

    def test_initial_choice(self):
        """Test the _initial_choice method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        # Number of nodes
        num_nodes = trainer.fitness_function.num_nodes

        # Init the iteration
        trainer._init_search()

        # The ant
        ant = trainer.solution_cls(
            trainer.species, trainer.fitness_function.Fitness
        )

        # Favor a feature
        favored_feature = 3
        scale = 1000
        trainer._choice_info[favored_feature] *= scale
        trainer._choice_info[:, favored_feature] *= scale

        # Make an initial choice
        trainer._initial_choice(ant)

        # Try to generate valid first nodes
        times = 1000
        acc = np.zeros(num_nodes)
        for _ in repeat(None, times):
            node = trainer._initial_choice(ant)
            self.assertTrue(node in feasible_nodes)
            acc[node] += 1

        self.assertEqual(np.argmax(acc), favored_feature)

        # Assess if discarded features are avoided
        for node in range(num_nodes):
            ant.discard(node)
            choice = trainer._initial_choice(ant)
            self.assertFalse(choice in ant.discarded)

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        # Init the iteration
        trainer._init_search()

        species_num_feats = species.max_feat - species.min_feat + 1

        # Generate ants
        for _ in range(1000):
            ant = trainer._generate_ant()
            self.assertEqual(
                species_num_feats,
                len(ant.path) + len(ant.discarded)
            )

    def test_new_state(self):
        """Test _new_state."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbose": False
        }

        # Create the trainer
        trainer = ACOFS(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the population
        self.assertIsInstance(trainer.pop, list)
        self.assertEqual(len(trainer.pop), trainer.pop_size)

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
        trainer = ACOFS(**params)
        trainer._init_search()
        trainer._new_state()

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
        trainer = ACOFS(**params)

        # Try to update the population with an empty elite
        trainer._init_search()
        trainer._start_iteration()
        trainer._generate_col()

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

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "pop_size": 5,
            "verbose": False
        }

        for i in range(1000):
            # Create the trainer
            trainer = ACOFS(**params)

            # Try to update the population with an empty elite
            trainer._init_search()
            trainer._start_iteration()

            # Reset the pheromone
            trainer._init_pheromone()

            # Let only the first ant deposit pheromone
            trainer._deposit_pheromone(trainer.pop[:1], 3)

            ant = trainer.pop[0]
            org = ant.path[-1]
            for dest in ant.path:
                self.assertEqual(trainer.pheromone[0][org][dest], 4)
                self.assertEqual(trainer.pheromone[0][dest][org], 4)
                org = dest

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
        trainer = ACOFS(**params)

        # Init the search
        trainer._init_search()

        # Reset the pheromone values
        trainer._init_pheromone()

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


if __name__ == '__main__':
    unittest.main()
