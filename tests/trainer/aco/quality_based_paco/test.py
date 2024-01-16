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

"""Unit test for :py:class:`culebra.trainer.aco.QualityBasedPACO`."""

import unittest

import numpy as np

from culebra.abc import Fitness
from culebra.trainer.aco import QualityBasedPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.tsp import PathLength


class MyMinimizationFitnessFunc(PathLength):
    """Dummy fitness function with two objectives."""

    class Fitness(Fitness):
        """Fitness class."""

        weights = (-1.0, )
        names = ("Min", )
        thresholds = (DEFAULT_THRESHOLD, )


class MyMaximizationFitnessFunc(PathLength):
    """Dummy fitness function with two objectives."""

    class Fitness(Fitness):
        """Fitness class."""

        weights = (1.0, )
        names = ("Max", )
        thresholds = (DEFAULT_THRESHOLD, )


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.QualityBasedPACO`."""

    def test_update_pop(self):
        """Test the _update_pop method."""
        species = Species(num_nodes)
        initial_pheromone = 2
        max_pheromone = 3

        minimization_params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": MyMinimizationFitnessFunc.fromPath(
                optimum_path
            ),
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1,
            "pop_size": 2
        }

        # Create the minimization trainer
        minimization_trainer = QualityBasedPACO(**minimization_params)
        minimization_trainer._init_search()

        # The initial population should be empty
        minimization_trainer._start_iteration()
        self.assertEqual(len(minimization_trainer.pop), 0)

        # Try several colonies with the same fitness
        initial_fit_values = (3.0, )
        for col_index in range(5):
            # Generate the colony
            minimization_trainer._start_iteration()
            minimization_trainer._generate_col()
            minimization_trainer.col[0].fitness.values = initial_fit_values
            minimization_trainer._update_pop()

        # The population should be filled with ants having the same fitness
        for ant in minimization_trainer.pop:
            self.assertEqual(ant.fitness.values, initial_fit_values)

        # Try a colony with a better ant
        fit_better_values = ((0.0, ), (1.0, ))
        for fit_values in fit_better_values:
            minimization_trainer._start_iteration()
            minimization_trainer._generate_col()
            minimization_trainer.col[0].fitness.values = fit_values
            minimization_trainer._update_pop()

        # Try more colonies with worse ants
        for col_index in range(5):
            # Generate the colony
            minimization_trainer._start_iteration()
            minimization_trainer._generate_col()
            minimization_trainer.col[0].fitness.values = initial_fit_values
            minimization_trainer._update_pop()

        # The worst ants should not be in the population
        # but the best should
        for ant in minimization_trainer.pop:
            self.assertNotEqual(ant.fitness.values, initial_fit_values)
            self.assertTrue(ant.fitness.values in fit_better_values)

        maximization_params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": MyMaximizationFitnessFunc.fromPath(
                optimum_path
            ),
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1,
            "pop_size": 2
        }

        # Create the minimization trainer
        maximization_trainer = QualityBasedPACO(**maximization_params)
        maximization_trainer._init_search()

        # The initial population should be empty
        maximization_trainer._start_iteration()
        self.assertEqual(len(maximization_trainer.pop), 0)

        # Try several colonies with the same fitness
        initial_fit_values = (3.0, )
        for col_index in range(5):
            # Generate the colony
            maximization_trainer._start_iteration()
            maximization_trainer._generate_col()
            maximization_trainer.col[0].fitness.values = initial_fit_values
            maximization_trainer._update_pop()

        # The population should be filled with ants having the same fitness
        for ant in maximization_trainer.pop:
            self.assertEqual(ant.fitness.values, initial_fit_values)

        # Try a colony with a better ant
        fit_better_values = ((4.0, ), (5.0, ))
        for fit_values in fit_better_values:
            maximization_trainer._start_iteration()
            maximization_trainer._generate_col()
            maximization_trainer.col[0].fitness.values = fit_values
            maximization_trainer._update_pop()

        # Try more colonies with worse ants
        for col_index in range(5):
            # Generate the colony
            maximization_trainer._start_iteration()
            maximization_trainer._generate_col()
            maximization_trainer.col[0].fitness.values = initial_fit_values
            maximization_trainer._update_pop()

        # The worst ants should not be in the population
        # but the best should
        for ant in maximization_trainer.pop:
            self.assertNotEqual(ant.fitness.values, initial_fit_values)
            self.assertTrue(ant.fitness.values in fit_better_values)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = 2
        max_pheromone = 3

        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": MyMinimizationFitnessFunc.fromPath(
                optimum_path
            ),
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1,
            "pop_size": 2
        }

        # Create the trainer
        trainer = QualityBasedPACO(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
