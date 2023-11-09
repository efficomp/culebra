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

"""Unit test for :py:class:`culebra.trainer.aco.abc.QualityBasedPACO`."""

import unittest

import numpy as np

from culebra.abc import Fitness
from culebra.trainer.aco.abc import QualityBasedPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.tsp import PathLength


class MyTrainer(QualityBasedPACO):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromones[0] * self.heuristics[0]


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
fitness_func = MyFitnessFunc.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.QualityBasedPACO`."""

    def test_update_pop(self):
        """Test the _update_pop method."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones,
            "col_size": 1,
            "pop_size": 2
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()

        # The initial population should be empty
        trainer._start_iteration()
        self.assertEqual(len(trainer.pop), 0)

        # Try several colonies with the same fitness
        initial_fit_values = (1.0, 2.0)
        for col_index in range(5):
            # Generate the colony
            trainer._start_iteration()
            trainer._generate_col()
            trainer.col[0].fitness.values = initial_fit_values
            trainer._update_pop()

        # The population should be filled with ants having the same fitness
        for ant in trainer.pop:
            self.assertEqual(ant.fitness.values, initial_fit_values)

        # Try a colony with a better ant
        fit_better_values = ((0.0, 2.0), (1.0, 3.0))
        for fit_values in fit_better_values:
            trainer._start_iteration()
            trainer._generate_col()
            trainer.col[0].fitness.values = fit_values
            trainer._update_pop()

        # Try more colonies with worse ants
        for col_index in range(5):
            # Generate the colony
            trainer._start_iteration()
            trainer._generate_col()
            trainer.col[0].fitness.values = initial_fit_values
            trainer._update_pop()

        # The worst ants should not be in the population
        # but the best should
        for ant in trainer.pop:
            self.assertNotEqual(ant.fitness.values, initial_fit_values)
            self.assertTrue(ant.fitness.values in fit_better_values)


if __name__ == '__main__':
    unittest.main()
