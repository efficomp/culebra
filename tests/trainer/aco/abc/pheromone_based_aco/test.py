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

"""Unit test for :py:class:`culebra.trainer.aco.abc.PheromoneBasedACO`."""

import unittest

import numpy as np

from culebra.trainer.aco.abc import (
    PheromoneBasedACO,
    SinglePheromoneMatrixACO,
    SingleHeuristicMatrixACO,
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO
)
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


class MyTrainer(PheromoneBasedACO):
    """Dummy implementation of a pheromone-based trainer."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromone[0] * self.heuristic[0]

    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone."""

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone."""


class MySinglePheromoneTrainer(
        SinglePheromoneMatrixACO,
        SingleHeuristicMatrixACO,
        MyTrainer):
    """Dummy implementation a trainer with a single pheromone matrix."""


class MyMultiplePheromoneTrainer(
        MultiplePheromoneMatricesACO,
        MultipleHeuristicMatricesACO,
        MyTrainer):
    """Dummy implementation a trainer with multiple pheromone matrices."""


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
multiple_obj_fitness_func = MultiObjectivePathLength(
    PathLength.fromPath(optimum_paths[0]),
    PathLength.fromPath(optimum_paths[1])
)
single_obj_fitness_func = multiple_obj_fitness_func.objectives[0]
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.PheromoneBasedACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = multiple_obj_fitness_func
        valid_initial_pheromone = [1, 2]

        # Test default params
        trainer = MyMultiplePheromoneTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone
        )
        self.assertEqual(trainer.pheromone, None)

    def test_state(self):
        """Test the get_state and _set_state methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": single_obj_fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySinglePheromoneTrainer(**params)

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["pheromone"], trainer.pheromone)

        # Change the state
        state["num_evals"] = 100
        state["pheromone"] = [np.full((num_nodes, num_nodes), 8, dtype=float)]

        # Set the new state
        trainer._set_state(state)

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertTrue(
            np.all(state["pheromone"] == trainer.pheromone)
        )

    def test_new_state(self):
        """Test _new_state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": multiple_obj_fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiplePheromoneTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the pheromone matrices
        self.assertIsInstance(trainer.pheromone, list)
        for (
            initial_pheromone,
            pheromone_matrix
        ) in zip(
            trainer.initial_pheromone,
            trainer.pheromone
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))

    def test_reset_state(self):
        """Test _reset_state."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = (2, )
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": single_obj_fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySinglePheromoneTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the pheromone
        self.assertEqual(trainer.pheromone, None)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = (2, )
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": single_obj_fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySinglePheromoneTrainer(**params)

        # Try before the colony has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Generate some ants
        ant1 = Ant(
            species,
            single_obj_fitness_func.fitness_cls,
            path=optimum_paths[0]
        )
        worse_path = np.concatenate(
            (
                optimum_paths[0][:5],
                optimum_paths[0][-1:],
                optimum_paths[0][5:-2])
        )
        ant2 = Ant(
            species,
            single_obj_fitness_func.fitness_cls,
            path=worse_path
        )

        # Init the search
        trainer._init_search()

        # Try a colony with different fitnesses
        trainer._col = [ant1, ant2]

        for ant in trainer.col:
            trainer.evaluate(ant)

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one solution
        self.assertEqual(len(best_ones[0]), 1)

        # Check that the solution in hof is ant1
        self.assertTrue(ant1 in best_ones[0])

        # Set the same fitness for both solutions
        for sol in trainer.col:
            sol.fitness.values = (18, )

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has two solutions
        self.assertEqual(len(best_ones[0]), 2)

        # Check that ant1 and ant2 are the solutions in hof
        self.assertTrue(ant1 in best_ones[0])
        self.assertTrue(ant2 in best_ones[0])

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": single_obj_fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySinglePheromoneTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
