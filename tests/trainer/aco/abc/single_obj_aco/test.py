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

"""Unit test for :py:class:`culebra.trainer.aco.abc.SingleObjACO`."""

import unittest
import math

import numpy as np

from culebra.abc import Fitness
from culebra.trainer.aco.abc import SingleObjACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.tsp import PathLength


class MyTrainer(SingleObjACO):
    """Dummy implementation of a trainer method."""

    def _decrease_pheromones(self) -> None:
        """Decrease the amount of pheromones."""

    def _increase_pheromones(self) -> None:
        """Increase the amount of pheromones."""


class MyFitnessFunc(PathLength):
    """Dummy fitness function with two objectives."""

    class Fitness(Fitness):
        """Fitness class."""

        weights = (-1.0, 1.0)
        names = ("Len", "Other")
        thresholds = (DEFAULT_THRESHOLD, DEFAULT_THRESHOLD)

    def heuristics(self, species):
        """Define a dummy heuristics."""
        (the_heuristics, ) = super().heuristics(species)
        return (the_heuristics, the_heuristics * 2)

    def evaluate(self, sol, index=None, representatives=None):
        """Define a dummy evaluation."""
        return super().evaluate(sol) + (3,)


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.SingleObjACO`."""

    def test_fitness_function(self):
        """Test the fitness_function property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1]

        # Try invalid types for fitness function. Should fail
        invalid_fitness_functions = (type, 'a')
        for invalid_fitness_func in invalid_fitness_functions:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    invalid_fitness_func,
                    initial_pheromones
                )

        # Try invalid values for fitness function. Should fail
        invalid_fitness_func = MyFitnessFunc.fromPath(optimum_path)
        with self.assertRaises(ValueError):
            MyTrainer(
                ant_cls,
                species,
                invalid_fitness_func,
                initial_pheromones
            )

    def test_initial_pheromones(self):
        """Test the initial_pheromones property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)

        # Try invalid types for initial_pheromones. Should fail
        invalid_initial_pheromones = (type, 1)
        for initial_pheromones in invalid_initial_pheromones:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones
                )

        # Try invalid values for initial_pheromones. Should fail
        invalid_initial_pheromones = [
            (-1, ), (max, ), (0, ), (1, 2), (1, 2, 3)
        ]
        for initial_pheromones in invalid_initial_pheromones:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones
                )

        # Try valid values for initial_pheromones
        initial_pheromones = ([2], [3])
        for initial_pher in initial_pheromones:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func,
                initial_pher
            )
            self.assertEqual(trainer.initial_pheromones, initial_pher)

    def test_heuristics(self):
        """Test the heuristics property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1]

        # Try invalid types for heuristics. Should fail
        invalid_heuristics = (type, 1)
        for heuristics in invalid_heuristics:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    heuristics=heuristics
                )

        # Try invalid values for heuristics. Should fail
        invalid_heuristics = (
            # Empty
            (),
            # Wrong shape
            (np.ones(shape=(num_nodes, num_nodes + 1), dtype=float), ),
            # Negative values
            (np.ones(shape=(num_nodes, num_nodes), dtype=float) * -1, ),
            # Different shapes
            (
                np.ones(shape=(num_nodes, num_nodes), dtype=float),
                np.ones(shape=(num_nodes+1, num_nodes+1), dtype=float),
            ),
            # Empty matrix
            (np.ones(shape=(0, 0), dtype=float), ),
            # Wrong number of matrices
            (np.ones(shape=(num_nodes, num_nodes), dtype=float), ) * 2,
            (np.ones(shape=(num_nodes, num_nodes), dtype=float), ) * 3,
        )
        for heuristics in invalid_heuristics:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    heuristics=heuristics
                )

        # Try a valid value for heuristics
        heuristics = (
            np.ones(shape=(num_nodes, num_nodes), dtype=float),
        )
        trainer = MyTrainer(
            ant_cls,
            species,
            fitness_func,
            initial_pheromones,
            heuristics=heuristics
        )
        for h1, h2 in zip(trainer.heuristics, heuristics):
            self.assertTrue(np.all(h1 == h2))

    def test_pheromones_influence(self):
        """Test the pheromones_influence property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1]

        # Try invalid types for pheromones_influence. Should fail
        invalid_pheromones_influence = (type, 1)
        for pheromones_influence in invalid_pheromones_influence:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    pheromones_influence=pheromones_influence
                )

        # Try invalid values for pheromones_influence. Should fail
        invalid_pheromones_influence = [(-1, ), (max, ), (), (1, 2, 3)]
        for pheromones_influence in invalid_pheromones_influence:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    pheromones_influence=pheromones_influence
                )

        # Try valid values for pheromones_influence
        valid_pheromones_influence = ([2], [0])
        for pheromones_influence in valid_pheromones_influence:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                pheromones_influence=pheromones_influence
            )
            self.assertIsInstance(trainer.pheromones_influence, list)
            self.assertEqual(
                trainer.pheromones_influence, pheromones_influence
            )

    def test_heuristics_influence(self):
        """Test the heuristics_influence property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1]

        # Try invalid types for heuristics_influence. Should fail
        invalid_heuristics_influence = (type, 1)
        for heuristics_influence in invalid_heuristics_influence:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    heuristics_influence=heuristics_influence
                )

        # Try invalid values for heuristics_influence. Should fail
        invalid_heuristics_influence = [
            (-1, ), (max, ), (), (1, 2), (1, 2, 3)
        ]
        for heuristics_influence in invalid_heuristics_influence:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    heuristics_influence=heuristics_influence
                )

        # Try valid values for heuristics_influence
        valid_heuristics_influence = ([3], [0])
        for heuristics_influence in valid_heuristics_influence:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                heuristics_influence=heuristics_influence
            )
            self.assertIsInstance(trainer.heuristics_influence, list)
            self.assertEqual(
                trainer.heuristics_influence, heuristics_influence
            )

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        pheromones_influence = [2]
        heuristics_influence = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "pheromones_influence": pheromones_influence,
            "heuristics_influence": heuristics_influence

        }

        # Create the trainer
        trainer = MyTrainer(**params)

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
                            trainer.pheromones_influence[0]
                        ) * math.pow(
                            trainer.heuristics[0][org][dest],
                            trainer.heuristics_influence[0]
                        )
                    )


if __name__ == '__main__':
    unittest.main()
