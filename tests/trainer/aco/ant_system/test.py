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

"""Unit test for :py:class:`culebra.trainer.aco.AntSystem`."""

import unittest
from itertools import repeat

import numpy as np

from culebra.trainer.aco import (
    AntSystem,
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
        initial_pheromone = 1

        # Try invalid types for pheromone_evaporation_rate. Should fail
        invalid_pheromone_evaporation_rate = (type, 'a')
        for pheromone_evaporation_rate in invalid_pheromone_evaporation_rate:
            with self.assertRaises(TypeError):
                AntSystem(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromone,
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
                    initial_pheromone,
                    pheromone_evaporation_rate=pheromone_evaporation_rate
                )

        # Try a valid value for pheromone_evaporation_rate
        valid_pheromone_evaporation_rate = (0.5, 1)
        for pheromone_evaporation_rate in valid_pheromone_evaporation_rate:
            trainer = AntSystem(
                ant_cls,
                species,
                fitness_func,
                initial_pheromone,
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
            initial_pheromone
        )
        self.assertEqual(
            trainer.pheromone_evaporation_rate,
            DEFAULT_PHEROMONE_EVAPORATION_RATE
        )

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 2
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

    def test_decrease_pheromone(self):
        """Test the _decrease_pheromone method."""
        # Trainer parameters
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 2
        }

        # Create the trainer
        trainer = AntSystem(**params)
        trainer._init_search()

        # Check the initial pheromone
        pheromone_value = trainer.initial_pheromone[0]
        self.assertTrue(
            np.all(trainer.pheromone[0] == pheromone_value)
        )

        # Evaporate pheromone
        trainer._decrease_pheromone()

        # Check again
        pheromone_value = (
            trainer.initial_pheromone[0] * (
                1 - trainer.pheromone_evaporation_rate
            )
        )
        self.assertTrue(
            np.all(trainer.pheromone[0] == pheromone_value)
        )

    def test_increase_pheromone(self):
        """Test the _increase_pheromone method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 2,
            "col_size": 1
        }

        # Create the trainer
        trainer = AntSystem(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromone
        pheromone_value = trainer.initial_pheromone[0]
        self.assertTrue(
            np.all(trainer.pheromone[0] == pheromone_value)
        )

        # Generate a new colony
        trainer._generate_col()

        # Evaporate pheromone
        trainer._increase_pheromone()

        # Get the ant
        ant = trainer.col[0]
        pheromone_increment = ant.fitness.pheromone_amount[0]
        pheromone_value += pheromone_increment

        org = ant.path[-1]
        for dest in ant.path:
            self.assertEqual(
                trainer.pheromone[0][org][dest],
                pheromone_value
            )
            self.assertEqual(
                trainer.pheromone[0][dest][org],
                pheromone_value
            )
            org = dest

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = AntSystem(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
