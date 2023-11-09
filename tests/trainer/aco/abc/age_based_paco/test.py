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

"""Unit test for :py:class:`culebra.trainer.aco.abc.AgeBasedPACO`."""

import unittest

import numpy as np

from culebra.abc import Fitness
from culebra.trainer.aco.abc import AgeBasedPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.tsp import PathLength


class MyTrainer(AgeBasedPACO):
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
    """Test :py:class:`culebra.trainer.aco.abc.AgeBasedPACO`."""

    def test_internals(self):
        """Test _init_internals."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()
        self.assertEqual(trainer._youngest_index, None)

        # Reset the internal structures
        trainer._reset_internals()
        self.assertEqual(trainer._youngest_index, None)

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

        # Try several colonies
        for col_index in range(5):
            # Generate the colony
            trainer._start_iteration()
            trainer._generate_col()

            # Index where the colony's ant will be inserted in the population
            pop_index = col_index % trainer.pop_size

            # Get the outgoing ant, ig any
            if col_index < trainer.pop_size:
                outgoing_ant = None
            else:
                outgoing_ant = trainer.pop[pop_index]

            # Update the population
            trainer._update_pop()

            # Check the population size
            if col_index < trainer.pop_size:
                self.assertEqual(len(trainer.pop), col_index + 1)
            else:
                self.assertEqual(len(trainer.pop), trainer.pop_size)

            # Check that the ant has been inserted in the population
            self.assertEqual(trainer.pop[pop_index], trainer.col[0])

            # The ant should also be in the ingoing list
            self.assertEqual(len(trainer._pop_ingoing), 1)
            self.assertEqual(trainer._pop_ingoing[0], trainer.col[0])

            # Check the outgoing ant, if any
            if col_index < trainer.pop_size:
                self.assertEqual(len(trainer._pop_outgoing), 0)
            else:
                self.assertEqual(len(trainer._pop_outgoing), 1)
                self.assertEqual(trainer._pop_outgoing[0], outgoing_ant)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [2]
        max_pheromones = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromones": initial_pheromones,
            "max_pheromones": max_pheromones
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
