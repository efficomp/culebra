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

"""Unit test for :class:`culebra.trainer.aco.ACOFS1D`."""

import unittest

import numpy as np

from deap.tools import sortNondominated

from culebra.trainer.aco import ACOFS1D
from culebra.solution.feature_selection import Species, Ant
from culebra.fitness_func import MultiObjectiveFitnessFunction
from culebra.fitness_func.feature_selection import (
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
training_fitness_func = KappaNumFeats(training_data, cv_folds=5)

# Test fitness function
test_fitness_func = KappaNumFeats(training_data, test_data=test_data)

# Lists of banned and feasible features
banned_feats = [0, dataset.num_feats-1]
feasible_feats = list(range(1, dataset.num_feats-1))


class ACOFSTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.ACOFS1D`."""

    def test_heuristic(self):
        """Test the heuristic property."""
        params = {
            "fitness_func": training_fitness_func,
            "solution_cls": Ant,
            "species": species,
            "verbosity": False
        }

        # Try invalid types for heuristic. Should fail
        invalid_heuristic = (type, 1)
        for heuristic in invalid_heuristic:
            with self.assertRaises(TypeError):
                trainer = ACOFS1D(**params, heuristic=heuristic)

        # Try invalid values for heuristic. Should fail
        invalid_heuristic = (
            # Empty
            (),
            # Wrong shape
            (
                np.ones(
                    shape=(species.num_feats, species.num_feats),
                    dtype=float
                ),
            ),
            np.ones(shape=(species.num_feats - 1,), dtype=float),
            np.ones(shape=(species.num_feats + 1,), dtype=float),
            # Negative values
            [
                np.ones(shape=(species.num_feats,),dtype=float) * -1
            ],
            np.ones(shape=(species.num_feats,), dtype=float) * -1,
            # Empty matrix
            (np.ones(shape=(0,), dtype=float), ),
            np.ones(shape=(0,), dtype=float),
            # Wrong number of matrices
            (
                np.ones(
                    shape=(species.num_feats,), dtype=float
                ),
            ) * 3,
        )
        for heuristic in invalid_heuristic:
            with self.assertRaises(ValueError):
                ACOFS1D(**params, heuristic=heuristic)

        # Try a single one-dimensional array-like object
        valid_heuristic = np.full(
            shape=(species.num_feats,), fill_value=4, dtype=float
        )
        trainer = ACOFS1D(**params, heuristic=valid_heuristic)
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == np.asarray(valid_heuristic)))

        # Try sequences of single one-dimensional array-like objects
        valid_heuristic = [np.ones(shape=(species.num_feats), dtype=float)]
        trainer = ACOFS1D(**params, heuristic=valid_heuristic)
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == np.asarray(valid_heuristic[0])))

        # Try the default heuristic
        trainer = ACOFS1D(**params)
        default_heuristic = np.ones((species.num_feats, ))
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == default_heuristic))

    def test_ant_choice_info(self):
        """Test the _ant_choice_info method."""
        params = {
            "fitness_func": training_fitness_func,
            "solution_cls": Ant,
            "species": species,
            "verbosity": False
        }

        # Create the trainer
        trainer = ACOFS1D(**params)

        # Initialize the internal structures
        trainer._init_internals()
        trainer._start_iteration()

        # The ant
        ant = trainer.solution_cls(
            trainer.species, trainer.fitness_func.fitness_cls
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

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""
        params = {
            "fitness_func": training_fitness_func,
            "solution_cls": Ant,
            "species": species,
            "verbosity": False
        }

        # Create the trainer
        trainer = ACOFS1D(**params)

        # Init the the training
        trainer._init_training()
        trainer._start_iteration()

        # Fill the population
        for _ in range(trainer.pop_size):
            trainer.pop.append(trainer._generate_ant())

        for ant in trainer.pop:
            trainer._init_pheromone()

            trainer._pareto_fronts = sortNondominated(
                trainer.pop, trainer.pop_size
            )
            trainer._num_pareto_fronts = len(trainer._pareto_fronts)
            trainer._pheromone_delta = (
                trainer.max_pheromone[0] - trainer.initial_pheromone[0]
                ) / (trainer.pop_size * trainer._num_pareto_fronts)


            # Let the ant deposit pheromone
            trainer._deposit_pheromone([ant])

            for feat in range(species.num_feats):
                if feat in ant.path:
                    self.assertGreater(
                        trainer.pheromone[0][feat],
                        trainer.initial_pheromone
                    )
                else:
                    self.assertEqual(
                        trainer.pheromone[0][feat],
                        trainer.initial_pheromone
                    )


if __name__ == '__main__':
    unittest.main()
