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

"""Unit test for :py:class:`culebra.trainer.aco.AntSystem`."""

import unittest
import math
from itertools import repeat

import numpy as np

from culebra.trainer.aco import (
    AntSystem,
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_PHEROMONE_EVAPORATION_RATE
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength

num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.AntSystem`."""

    def test_init(self):
        """Test __init__`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1]

        # Try invalid types for pheromone_influence. Should fail
        invalid_pheromone_influence = (type, 'a')
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(TypeError):
                AntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    pheromone_influence=pheromone_influence
                )

        # Try invalid values for pheromone_influence. Should fail
        invalid_pheromone_influence = -1
        with self.assertRaises(ValueError):
            AntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                pheromone_influence=invalid_pheromone_influence
            )

        # Try a valid value for pheromone_influence
        valid_pheromone_influence = (0, 0.5, 1, 2)
        for pheromone_influence in valid_pheromone_influence:
            trainer = AntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(pheromone_influence, trainer.pheromone_influence)

        # Try invalid types for heuristic_influence. Should fail
        invalid_heuristic_influence = (type, 'a')
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(TypeError):
                AntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    heuristic_influence=heuristic_influence
                )

        # Try invalid values for heuristic_influence. Should fail
        invalid_heuristic_influence = -1
        with self.assertRaises(ValueError):
            AntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                heuristic_influence=invalid_heuristic_influence
            )

        # Try a valid value for heuristic_influence
        valid_heuristic_influence = (0, 0.5, 1, 2)
        for heuristic_influence in valid_heuristic_influence:
            trainer = AntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(heuristic_influence, trainer.heuristic_influence)

        # Try invalid types for pheromone_evaporation_rate. Should fail
        invalid_pheromone_evaporation_rate = (type, 'a')
        for pheromone_evaporation_rate in invalid_pheromone_evaporation_rate:
            with self.assertRaises(TypeError):
                AntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    pheromone_evaporation_rate=pheromone_evaporation_rate
                )

        # Try invalid values for pheromone_evaporation_rate. Should fail
        invalid_pheromone_evaporation_rate = (-1, 0, 1.5)
        for pheromone_evaporation_rate in invalid_pheromone_evaporation_rate:
            with self.assertRaises(ValueError):
                AntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    pheromone_evaporation_rate=pheromone_evaporation_rate
                )

        # Try a valid value for pheromone_evaporation_rate
        valid_pheromone_evaporation_rate = (0.5, 1)
        for pheromone_evaporation_rate in valid_pheromone_evaporation_rate:
            trainer = AntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                pheromone_evaporation_rate=pheromone_evaporation_rate
            )
            self.assertEqual(
                pheromone_evaporation_rate,
                trainer.pheromone_evaporation_rate
            )

        # Test default params
        trainer = AntSystem(
            ant_cls,
            species,
            fitness_func,
            initial_pheromones
        )
        self.assertEqual(
            trainer.pheromone_influence,
            DEFAULT_PHEROMONE_INFLUENCE
        )
        self.assertEqual(
            trainer.heuristic_influence,
            DEFAULT_HEURISTIC_INFLUENCE
        )
        self.assertEqual(
            trainer.pheromone_evaporation_rate,
            DEFAULT_PHEROMONE_EVAPORATION_RATE
        )

        # Test the initial_pheromones property
        initial_pheromones = [1, 2, 3]
        trainer.initial_pheromones = initial_pheromones
        self.assertEqual(
            trainer.initial_pheromones,
            initial_pheromones[0:1]
            )

        # Test the heuristics property
        heuristics = [
            [
                [1., 2.],
                [3., 4.]
            ],
            [
                [5., 6.],
                [7., 8.]
            ],
        ]
        trainer.heuristics = heuristics
        self.assertEqual(len(trainer.heuristics), 1)
        self.assertTrue(
            np.all(trainer.heuristics[0] == np.asarray(heuristics[0]))
        )

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "pheromone_influence": 2,
            "heuristic_influence": 3,

        }

        # Create the trainer
        trainer = AntSystem(**params)

        # Try to get the choice info before the search initialization
        choice_info = trainer.choice_info
        self.assertEqual(choice_info, None)

        # Try to get the choice_info after initializing the search
        trainer._init_search()
        trainer._start_iteration()
        choice_info = trainer.choice_info

        # Check the probabilities for banned nodes. Should be 0
        for node in banned_nodes:
            self.assertAlmostEqual(np.sum(choice_info[node]), 0)

        for org in feasible_nodes:
            for dest in feasible_nodes:
                if org == dest:
                    self.assertAlmostEqual(choice_info[org][dest], 0)
                else:
                    self.assertAlmostEqual(
                        choice_info[org][dest],
                        math.pow(
                            trainer.pheromones[0][org][dest],
                            trainer.pheromone_influence
                        ) * math.pow(
                            trainer.heuristics[0][org][dest],
                            trainer.heuristic_influence
                        )
                    )

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "pheromone_influence": 2,
            "heuristic_influence": 3,

        }

        # Create the trainer
        trainer = AntSystem(**params)
        trainer._init_search()

        # Try to generate valid ants
        times = 1000
        for _ in repeat(None, times):
            trainer._start_iteration()
            ant = trainer._generate_ant()
            self.assertEqual(len(ant.path), len(feasible_nodes))

    def test_evaporate_pheromones(self):
        """Test the _evaporate_pheromones method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "pheromone_influence": 2,
            "heuristic_influence": 3,

        }

        # Create the trainer
        trainer = AntSystem(**params)
        trainer._init_search()

        # Check the initial pheromones
        pheromones_value = trainer.initial_pheromones[0]
        self.assertTrue(
            np.all(trainer.pheromones[0] == pheromones_value)
        )

        # Evaporate pheromones
        trainer._evaporate_pheromones()

        # Check again
        pheromones_value = (
            trainer.initial_pheromones[0] * (
                1 - trainer.pheromone_evaporation_rate
            )
        )
        self.assertTrue(
            np.all(trainer.pheromones[0] == pheromones_value)
        )

    def test_deposit_pop_pheromones(self):
        """Test the _deposit_pop_pheromones method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "pheromone_influence": 2,
            "heuristic_influence": 3,
            "pop_size": 1
        }

        # Create the trainer
        trainer = AntSystem(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromones
        pheromones_value = trainer.initial_pheromones[0]
        self.assertTrue(
            np.all(trainer.pheromones[0] == pheromones_value)
        )

        # Generate a new colony
        trainer._generate_pop()

        # Evaporate pheromones
        trainer._deposit_pop_pheromones(trainer.pop)

        # Get the ant
        ant = trainer.pop[0]
        pheromones_increment = ant.fitness.pheromones_amount[0]
        pheromones_value += pheromones_increment

        org = ant.path[-1]
        for dest in ant.path:
            self.assertEqual(
                trainer.pheromones[0][org][dest],
                pheromones_value
            )
            self.assertEqual(
                trainer.pheromones[0][dest][org],
                pheromones_value
            )
            org = dest


if __name__ == '__main__':
    unittest.main()
