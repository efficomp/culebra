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

"""Unit test for :class:`culebra.trainer.aco.abc.ACOFS`."""

import unittest

from itertools import combinations
from copy import copy, deepcopy
from os import remove

import numpy as np

from culebra import DEFAULT_MAX_NUM_ITERS, SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_ACOFS_INITIAL_PHEROMONE,
    DEFAULT_ACOFS_HEURISTIC_INFLUENCE,
    DEFAULT_ACOFS_EXPLOITATION_PROB,
    DEFAULT_ACOFS_DISCARD_PROB
)
from culebra.trainer.aco.abc import ACOFS

from culebra.solution.feature_selection import Species, Ant
from culebra.solution.tsp import (
    Species as TSPSpecies,
    Ant as TSPAnt
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

class MyACOFS(ACOFS):
    """Dummy implementation of a trainer method."""

    def _pheromone_amount (self, ant):
        return tuple(self.initial_pheromone)


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

# Lists of banned and feasible features
banned_feats = [0, dataset.num_feats-1]
feasible_feats = list(range(1, dataset.num_feats-1))


class ACOFSTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.abc.ACOFS`."""

    def test_init(self):
        """Test __init__."""
        # Try an invalid ant. Should fail
        with self.assertRaises(TypeError):
            MyACOFS(TSPAnt, species, training_fitness_function)

        # Try an invalid species. Should fail
        tsp_species = TSPSpecies(species.num_feats)
        with self.assertRaises(TypeError):
            MyACOFS(Ant, tsp_species, training_fitness_function)

        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Try invalid fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyACOFS(Ant, tsp_species, func)

        # Create the trainer
        trainer = MyACOFS(**params)

        # Check the parameters
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, species)
        self.assertEqual(trainer.fitness_function, training_fitness_function)
        self.assertEqual(
            trainer.initial_pheromone,
            (DEFAULT_ACOFS_INITIAL_PHEROMONE,) * trainer.num_pheromone_matrices
        )
        default_heuristic = (
            np.ones((species.num_feats, ) * 2) - np.identity(species.num_feats)
        )
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == default_heuristic))
        self.assertEqual(
            trainer.pheromone_influence, (DEFAULT_PHEROMONE_INFLUENCE,)
        )
        self.assertEqual(
            trainer.heuristic_influence, (DEFAULT_ACOFS_HEURISTIC_INFLUENCE,)
        )
        self.assertEqual(
            trainer.exploitation_prob, DEFAULT_ACOFS_EXPLOITATION_PROB
        )
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer.col_size, species.num_feats)
        self.assertEqual(trainer.discard_prob, DEFAULT_ACOFS_DISCARD_PROB)

    def test_num_pheromone_matrices(self):
        """Test the num_pheromone_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        self.assertEqual(trainer.num_pheromone_matrices, 1)

    def test_num_heuristic_matrices(self):
        """Test the num_heuristic_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        self.assertEqual(
            trainer.num_heuristic_matrices, 1
        )

    def test_initial_pheromone(self):
        """Test the initial_pheromone property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Try invalid types for initial_pheromone. Should fail
        invalid_initial_pheromone = (type, max)
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(TypeError):
                MyACOFS(**params, initial_pheromone=initial_pheromone)

        # Try invalid values for initial_pheromone. Should fail
        invalid_initial_pheromone = [
            (-1, ), (max, ), (0, ), (), (1, 2, 3), ('a'), -1, 0, 'a'
            ]
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(ValueError):
                MyACOFS(**params, initial_pheromone=initial_pheromone)

        # Try valid values for initial_pheromone
        initial_pheromone = 3
        trainer = MyACOFS(**params, initial_pheromone=initial_pheromone)
        self.assertEqual(trainer.initial_pheromone, (initial_pheromone,))

        initial_pheromone = [2]
        trainer = MyACOFS(**params, initial_pheromone=initial_pheromone)
        self.assertEqual(trainer.initial_pheromone, tuple(initial_pheromone))

        # Try the default value
        trainer = MyACOFS(**params)
        self.assertEqual(
            trainer.initial_pheromone, (DEFAULT_ACOFS_INITIAL_PHEROMONE,)
        )

    def test_heuristic(self):
        """Test the heuristic property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Try invalid types for heuristic. Should fail
        invalid_heuristic = (type, 1)
        for heuristic in invalid_heuristic:
            with self.assertRaises(TypeError):
                trainer = MyACOFS(**params, heuristic=heuristic)

        # Try invalid values for heuristic. Should fail
        invalid_heuristic = (
            # Empty
            (),
            # Wrong shape
            (
                np.ones(
                    shape=(species.num_feats, species.num_feats + 1),
                    dtype=float
                ),
            ),
            np.ones(
                shape=(species.num_feats, species.num_feats + 1),
                dtype=float
            ),
            np.ones(
                shape=(species.num_feats + 1, species.num_feats + 1),
                dtype=float
            ),
            [[1, 2, 3], [4, 5, 6]],
            ([[1, 2, 3], [4, 5, 6]], ),
            [[1, 2], [3, 4], [5, 6]],
            ([[1, 2], [3, 4], [5, 6]], ),
            # Negative values
            [
                np.ones(
                    shape=(species.num_feats, species.num_feats),
                    dtype=float) * -1
            ],
            np.ones(
                shape=(species.num_feats, species.num_feats), dtype=float
            ) * -1,
            # Empty matrix
            (np.ones(shape=(0, 0), dtype=float), ),
            np.ones(shape=(0, 0), dtype=float),
            # Wrong number of matrices
            (
                np.ones(
                    shape=(species.num_feats, species.num_feats), dtype=float
                ),
            ) * 3,
        )
        for heuristic in invalid_heuristic:
            with self.assertRaises(ValueError):
                MyACOFS(**params, heuristic=heuristic)

        # Try a single two-dimensional array-like object
        valid_heuristic = np.full(
            shape=(species.num_feats, species.num_feats),
            fill_value=4,
            dtype=float
        )
        trainer = MyACOFS(**params, heuristic=valid_heuristic)
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == np.asarray(valid_heuristic)))

        # Try sequences of single two-dimensional array-like objects
        valid_heuristic = [
            np.ones(
                shape=(species.num_feats, species.num_feats),
                dtype=float
            )
        ]
        trainer = MyACOFS(**params, heuristic=valid_heuristic)
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == np.asarray(valid_heuristic[0])))

        # Try the default heuristic
        trainer = MyACOFS(**params)
        default_heuristic = (
            np.ones((species.num_feats, ) * 2) - np.identity(species.num_feats)
        )
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == default_heuristic))

    def test_heuristic_influence(self):
        """Test the heuristic_influence property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Try invalid types for heuristic_influence. Should fail
        invalid_heuristic_influence = (type, max)
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(TypeError):
                MyACOFS(**params, heuristic_influence=heuristic_influence)

        # Try invalid values for heuristic_influence. Should fail
        invalid_heuristic_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(ValueError):
                MyACOFS(**params, heuristic_influence=heuristic_influence)

        # Try valid values for heuristic_influence
        valid_heuristic_influence = [3, 0]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MyACOFS(**params, heuristic_influence=heuristic_influence)
            self.assertEqual(
                trainer.heuristic_influence, (heuristic_influence,)
            )

        valid_heuristic_influence = [(0,), [2]]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MyACOFS(
                **params, heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence,
                tuple(heuristic_influence)
            )

        # Test the default value
        trainer = MyACOFS(**params)
        self.assertEqual(
            trainer.heuristic_influence, (DEFAULT_ACOFS_HEURISTIC_INFLUENCE,)
        )

    def test_exploitation_prob(self):
        """Test the exploitation_prob property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }
        # Try invalid types for exploitation_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyACOFS(**params, exploitation_prob=prob)

        # Try invalid values for exploitation_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyACOFS(**params, exploitation_prob=prob)

        # Try valid values for exploitation_prob
        valid_probs = (0, 0.5, 1)
        for prob in valid_probs:
            trainer = MyACOFS(**params, exploitation_prob=prob)
            self.assertEqual(trainer.exploitation_prob, prob)

        # Test the ddfault value
        trainer = MyACOFS(**params)
        self.assertEqual(
            trainer.exploitation_prob, DEFAULT_ACOFS_EXPLOITATION_PROB
        )

    def test_col_size(self):
        """Test the col_size property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Try invalid types for col_size. Should fail
        invalid_col_size = (type, 'a', 1.5)
        for col_size in invalid_col_size:
            with self.assertRaises(TypeError):
                MyACOFS(**params, col_size=col_size)

        # Try invalid values for col_size. Should fail
        invalid_col_size = (-1, 0)
        for col_size in invalid_col_size:
            with self.assertRaises(ValueError):
                MyACOFS(**params, col_size=col_size)

        # Try a valid value for col_size
        col_size = 233
        trainer = MyACOFS(**params, col_size=col_size)
        self.assertEqual(col_size, trainer.col_size)

        # Try the default value for col_size
        trainer = MyACOFS(**params)
        self.assertEqual(trainer.col_size, trainer.species.num_feats)

    def test_discard_prob(self):
        """Test the discard_prob property."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }
        # Try invalid types for discard_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyACOFS(**params, discard_prob=prob)

        # Try invalid values for discard_prob. Should fail
        invalid_probs = (-1, -0.001, 0, 1, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyACOFS(**params, discard_prob=prob)

        # Try valid values for discard_prob
        valid_probs = (0.1, 0.5, 0.9)
        for prob in valid_probs:
            trainer = MyACOFS(**params, discard_prob=prob)
            self.assertEqual(trainer.discard_prob, prob)

        # Test the ddfault value
        trainer = MyACOFS(**params)
        self.assertEqual(
            trainer.discard_prob, DEFAULT_ACOFS_DISCARD_PROB
        )

    def test_internals(self):
        """Test the _init_internals and _reset_internals methods."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        # Init the trainer internal structures
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

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        # Try to get the choice info before the search initialization
        choice_info = trainer.choice_info
        self.assertEqual(choice_info, None)

        # Try to get the choice_info after initializing the internal
        # structures
        trainer._init_internals()
        choice_info = trainer.choice_info

        the_choice_info = np.ones((species.num_feats,) * 2)
        for (pher, pher_inf, heur, heur_inf) in zip(
            trainer.pheromone,
            trainer.pheromone_influence,
            trainer.heuristic,
            trainer.heuristic_influence
        ):
            the_choice_info *= np.power(pher, pher_inf)
            the_choice_info *= np.power(heur, heur_inf)

        for feat in range(species.num_feats):
            for next_feat in range(species.num_feats):
                self.assertEqual(
                    trainer.choice_info[feat, next_feat],
                    the_choice_info[feat, next_feat]
                )
                self.assertEqual(
                    trainer.choice_info[next_feat, feat],
                    the_choice_info[feat, next_feat]
                )

    def test_ant_choice_info(self):
        """Test the _ant_choice_info method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        # Initialize the internal structures
        trainer._init_internals()

        # The ant
        ant = trainer.solution_cls(
            trainer.species, trainer.fitness_function.fitness_cls
        )

        discard_next_feat = True
        for feat in feasible_feats:
            if discard_next_feat:
                ant.discard(feat)
            else:
                ant.append(feat)
                discard_next_feat = not discard_next_feat

            the_choice_info = trainer._ant_choice_info(ant)
            self.assertTrue((the_choice_info[banned_feats] == 0).all())
            self.assertTrue((the_choice_info[ant.path] == 0).all())
            self.assertTrue((the_choice_info[ant.discarded] == 0).all())
            remaining_feats = np.setdiff1d(
                np.arange(species.num_feats),
                np.concatenate((banned_feats, ant.path, ant.discarded))
            )

            self.assertTrue((the_choice_info[remaining_feats] > 0).all())

        self.assertTrue((the_choice_info[range(species.num_feats)] == 0).all())

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        # Initialize the internal structures
        trainer._init_internals()

        species_num_feats = species.max_feat - species.min_feat + 1

        # Generate ants
        for _ in range(100):
            ant = trainer._generate_ant()
            self.assertEqual(
                species_num_feats,
                len(ant.path) + len(ant.discarded)
            )

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)

        # Init the internal strcutures
        trainer._init_internals()
        for i in range(100):
            ant = trainer._generate_ant()

            # Init the pheromone matrix
            trainer._init_pheromone()

            # Let only the first ant deposit pheromone
            trainer._deposit_pheromone([ant])

            # All the combinations of two features from those in the path
            indices = list(combinations(ant.path, 2))
            for i in range(species.num_feats):
                for j in range(i+1, species.num_feats):
                    if (i, j) in indices or (j, i) in indices:
                        self.assertGreater(
                            trainer.pheromone[0][i][j],
                            trainer.initial_pheromone
                        )
                        self.assertGreater(
                            trainer.pheromone[0][j][i],
                            trainer.initial_pheromone
                        )
                    else:
                        self.assertEqual(
                            trainer.pheromone[0][j][i],
                            trainer.initial_pheromone
                        )

    def test_copy(self):
        """Test the __copy__ method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": training_fitness_function,
            "verbosity": False
        }

        # Create the trainer
        trainer1 = MyACOFS(**params)
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
            "verbosity": False
        }

        # Create the trainer
        trainer1 = MyACOFS(**params)
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
            "verbosity": False
        }

        # Create the trainer
        trainer1 = MyACOFS(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = ACOFS.load(serialized_filename)

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
            "verbosity": False
        }

        # Create the trainer
        trainer = MyACOFS(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: ~culebra.trainer.aco.abc.ACOFS
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.aco.abc.ACOFS
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )

        self.assertNotEqual(id(trainer1.species), id(trainer2.species))
        self.assertEqual(
            id(trainer1.species.num_feats), id(trainer2.species.num_feats)
        )
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
