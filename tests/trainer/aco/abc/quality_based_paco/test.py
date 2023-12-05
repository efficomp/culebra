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

"""Unit test for :py:class:`culebra.trainer.aco.abc.QualityBasedPACO`."""

import unittest

import numpy as np

from culebra.trainer.aco.abc import (
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    QualityBasedPACO
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import DoublePathLength


class MyTrainer(
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    QualityBasedPACO
):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromone[0] * self.heuristic[0]


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func = DoublePathLength.fromPath(*optimum_paths)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.QualityBasedPACO`."""

    def test_update_pop(self):
        """Test the _update_pop method."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
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
        initial_fit_values = (3.0, 3.0)
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
        fit_better_values = ((1.0, 3.0), (3.0, 2.0))
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
