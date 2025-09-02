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

from culebra.trainer.aco.abc import SingleObjACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


class MyTrainer(SingleObjACO):
    """Dummy implementation of a trainer method."""

    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone."""

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone."""

    def _get_state(self):
        """Return the state of this trainer."""
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["pheromone"] = self._pheromone

        return state

    def _set_state(self, state):
        """Set the state of this trainer."""
        # Set the state of the superclass
        super()._set_state(state)

        # Set the state of this class
        self._pheromone = state["pheromone"]

    def _new_state(self):
        """Generate a new trainer state."""
        super()._new_state()
        heuristic_shape = self._heuristic[0].shape
        self._pheromone = [
            np.full(
                heuristic_shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromone
        ]

    def _reset_state(self):
        """Reset the trainer state."""
        super()._reset_state()
        self._pheromone = None


num_nodes = 25
optimum_paths = [
    np.random.permutation(num_nodes),
    np.random.permutation(num_nodes)
]
fitness_func_multi = MultiObjectivePathLength(
    PathLength.fromPath(optimum_paths[0]),
    PathLength.fromPath(optimum_paths[1])
)
fitness_func_single = fitness_func_multi.objectives[0]
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.SingleObjACO`."""

    def test_fitness_function(self):
        """Test the fitness_function property`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1

        # Try invalid types for fitness function. Should fail
        invalid_fitness_functions = (type, 'a')
        for invalid_fitness_func in invalid_fitness_functions:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    invalid_fitness_func,
                    initial_pheromone
                )

        # Try invalid values for fitness function. Should fail
        invalid_fitness_func = fitness_func_multi
        with self.assertRaises(ValueError):
            MyTrainer(
                ant_cls,
                species,
                invalid_fitness_func,
                initial_pheromone
            )

    def test_initial_pheromone(self):
        """Test the initial_pheromone property."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1

        # Try invalid types for initial_pheromone. Should fail
        invalid_initial_pheromone = (type, None, max)
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone
                )

        # Try invalid values for initial_pheromone. Should fail
        invalid_initial_pheromone = [
            (-1, ), (max, ), (0, ), (), (1, 2, 3), ('a'), -1, 0, 'a'
            ]
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone
                )

        # Try valid values for initial_pheromone
        initial_pheromone = 3
        trainer = MyTrainer(
            ant_cls,
            species,
            fitness_func_single,
            initial_pheromone
        )
        self.assertEqual(trainer.initial_pheromone, [initial_pheromone])

        initial_pheromone = [2]
        trainer = MyTrainer(
            ant_cls,
            species,
            fitness_func_single,
            initial_pheromone
        )
        self.assertEqual(trainer.initial_pheromone, initial_pheromone)

    def test_heuristic(self):
        """Test the heuristic property."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1
        # Try invalid types for heuristic. Should fail
        invalid_heuristic = (type, 1)
        for heuristic in invalid_heuristic:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try invalid values for heuristic. Should fail
        invalid_heuristic = (
            # Empty
            (),
            # Wrong shape
            (np.ones(shape=(num_nodes, num_nodes + 1), dtype=float), ),
            np.ones(shape=(num_nodes, num_nodes + 1), dtype=float),
            [[1, 2, 3], [4, 5, 6]],
            ([[1, 2, 3], [4, 5, 6]], ),
            [[1, 2], [3, 4], [5, 6]],
            ([[1, 2], [3, 4], [5, 6]], ),
            # Negative values
            [np.ones(shape=(num_nodes, num_nodes), dtype=float) * -1],
            np.ones(shape=(num_nodes, num_nodes), dtype=float) * -1,
            # Empty matrix
            (np.ones(shape=(0, 0), dtype=float), ),
            np.ones(shape=(0, 0), dtype=float),
            # Wrong number of matrices
            (np.ones(shape=(num_nodes, num_nodes), dtype=float), ) * 3,
        )
        for heuristic in invalid_heuristic:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try single two-dimensional array-like objects
        valid_heuristic = (
            np.full(shape=(num_nodes, num_nodes), fill_value=4, dtype=float),
            [[1, 2], [3, 4]]
        )
        for heuristic in valid_heuristic:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func_single,
                initial_pheromone,
                heuristic=heuristic
            )
            for heur in trainer.heuristic:
                self.assertTrue(np.all(heur == np.asarray(heuristic)))

        # Try sequences of single two-dimensional array-like objects
        valid_heuristic = (
            [np.ones(shape=(num_nodes, num_nodes), dtype=float)],
            [[[1, 2], [3, 4]]],
            ([[1, 2], [3, 4]], )
        )
        for heuristic in valid_heuristic:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func_single,
                initial_pheromone,
                heuristic=heuristic
            )
            for heur in trainer.heuristic:
                self.assertTrue(np.all(heur == np.asarray(heuristic[0])))

    def test_pheromone_influence(self):
        """Test the pheromone_influence property."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1

        # Try invalid types for pheromone_influence. Should fail
        invalid_pheromone_influence = (type, max)
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone,
                    pheromone_influence=pheromone_influence
                )

        # Try invalid values for pheromone_influence. Should fail
        invalid_pheromone_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone,
                    pheromone_influence=pheromone_influence
                )

        # Try valid values for pheromone_influence
        valid_pheromone_influence = [3, 0]
        for pheromone_influence in valid_pheromone_influence:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func_single,
                initial_pheromone,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(
                trainer.pheromone_influence, [pheromone_influence]
            )

        valid_pheromone_influence = [(0,), [2]]
        for pheromone_influence in valid_pheromone_influence:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func_single,
                initial_pheromone,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(
                trainer.pheromone_influence,
                list(pheromone_influence)
            )

    def test_heuristic_influence(self):
        """Test the heuristic_influence property."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1

        # Try invalid types for heuristic_influence. Should fail
        invalid_heuristic_influence = (type, max)
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(TypeError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone,
                    heuristic_influence=heuristic_influence
                )

        # Try invalid values for heuristic_influence. Should fail
        invalid_heuristic_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(ValueError):
                MyTrainer(
                    ant_cls,
                    species,
                    fitness_func_single,
                    initial_pheromone,
                    heuristic_influence=heuristic_influence
                )

        # Try valid values for heuristic_influence
        valid_heuristic_influence = [3, 0]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func_single,
                initial_pheromone,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence, [heuristic_influence]
            )

        valid_heuristic_influence = [(0,), [2]]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MyTrainer(
                ant_cls,
                species,
                fitness_func_single,
                initial_pheromone,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence,
                list(heuristic_influence)
            )

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        pheromone_influence = 2
        heuristic_influence = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func_single,
            "initial_pheromone": initial_pheromone,
            "pheromone_influence": pheromone_influence,
            "heuristic_influence": heuristic_influence

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
                            trainer.pheromone[0][org][dest],
                            trainer.pheromone_influence[0]
                        ) * math.pow(
                            trainer.heuristic[0][org][dest],
                            trainer.heuristic_influence[0]
                        )
                    )


if __name__ == '__main__':
    unittest.main()
