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

"""Unit test for :py:class:`culebra.trainer.aco.abc.SingleColACO`."""

import unittest
from itertools import repeat
from copy import copy, deepcopy
from os import remove

import numpy as np

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.aco import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE
)
from culebra.trainer.aco.abc import SingleColACO
from culebra.solution.tsp import (
    Species as TSPSpecies,
    Solution as TSPSolution,
    Ant as TSPAnt
)
from culebra.fitness_function.tsp import SinglePathLength, DoublePathLength

from culebra.solution.feature_selection import (
    Species as FSSpecies,
    Ant as FSAnt
)
from culebra.fitness_function.feature_selection import KappaNumFeats
from culebra.tools import Dataset


class MySingleObjTrainer(SingleColACO):
    """Dummy implementation of a trainer method."""

    @property
    def num_pheromone_matrices(self) -> int:
        """Get the number of pheromone matrices used by this trainer."""
        return 1

    @property
    def num_heuristic_matrices(self) -> int:
        """Get the number of heuristic matrices used by this trainer."""
        return 1

    def _calculate_choice_info(self):
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromone[0] * self.heuristic[0]

    def _decrease_pheromone(self):
        """Decrease the amount of pheromone."""

    def _increase_pheromone(self):
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


class MyMultiObjTrainer(MySingleObjTrainer):
    """Dummy implementation of a trainer method."""

    @property
    def num_pheromone_matrices(self) -> int:
        """Get the number of pheromone matrices used by this trainer."""
        return self.fitness_function.num_obj

    @property
    def num_heuristic_matrices(self) -> int:
        """Get the number of heuristic matrices used by this trainer."""
        return self.fitness_function.num_obj


# TSP related stuff
tsp_num_nodes = 25
tsp_optimum_paths = [
    np.random.permutation(tsp_num_nodes),
    np.random.permutation(tsp_num_nodes)
]
tsp_fitness_func_single = SinglePathLength.fromPath(tsp_optimum_paths[0])
tsp_fitness_func_multi = DoublePathLength.fromPath(*tsp_optimum_paths)
tsp_banned_nodes = [0, tsp_num_nodes-1]
tsp_feasible_nodes = list(range(1, tsp_num_nodes - 1))

fs_dataset = Dataset.load_from_uci(name="Wine")

fs_fitness_func = KappaNumFeats(training_data=fs_dataset)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.SingleColACO`."""

    def test_init(self):
        """Test __init__."""
        valid_ant_cls = TSPAnt
        valid_species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        valid_tsp_fitness_func_single = tsp_fitness_func_single
        valid_tsp_fitness_func_multi = tsp_fitness_func_multi
        valid_initial_pheromone = 1

        # Try invalid ant classes. Should fail
        invalid_ant_classes = (type, None, 1, TSPSolution)
        for solution_cls in invalid_ant_classes:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    solution_cls,
                    valid_species,
                    valid_tsp_fitness_func_single,
                    valid_initial_pheromone
                )

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    valid_ant_cls,
                    species,
                    valid_tsp_fitness_func_multi,
                    valid_initial_pheromone
                )

        # Try invalid fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    valid_ant_cls,
                    valid_species,
                    func,
                    valid_initial_pheromone
                )

        # Test default params
        singleObjTrainer = MySingleObjTrainer(
            valid_ant_cls,
            valid_species,
            valid_tsp_fitness_func_single,
            valid_initial_pheromone
        )
        multiObjTrainer = MyMultiObjTrainer(
            valid_ant_cls,
            valid_species,
            valid_tsp_fitness_func_multi,
            valid_initial_pheromone
        )

        self.assertEqual(singleObjTrainer.solution_cls, valid_ant_cls)
        self.assertEqual(multiObjTrainer.solution_cls, valid_ant_cls)

        self.assertEqual(singleObjTrainer.species, valid_species)
        self.assertEqual(multiObjTrainer.species, valid_species)

        self.assertEqual(
            singleObjTrainer.fitness_function,
            valid_tsp_fitness_func_single
        )
        self.assertEqual(
            multiObjTrainer.fitness_function,
            valid_tsp_fitness_func_multi
        )

        self.assertEqual(
            singleObjTrainer.initial_pheromone, [valid_initial_pheromone]
        )
        self.assertEqual(
            multiObjTrainer.initial_pheromone,
            [valid_initial_pheromone] * valid_tsp_fitness_func_multi.num_obj
        )

        self.assertIsInstance(singleObjTrainer.heuristic, list)
        self.assertIsInstance(multiObjTrainer.heuristic, list)

        self.assertEqual(len(singleObjTrainer.heuristic), 1)
        self.assertEqual(
            len(multiObjTrainer.heuristic),
            valid_tsp_fitness_func_multi.num_obj
        )

        for matrix in singleObjTrainer.heuristic:
            self.assertEqual(
                matrix.shape, (tsp_num_nodes, tsp_num_nodes)
            )
        for matrix in multiObjTrainer.heuristic:
            self.assertEqual(
                matrix.shape, (tsp_num_nodes, tsp_num_nodes)
            )

        # Check the heuristic
        for (
            optimum_path,
            heuristic
        ) in zip(
            tsp_optimum_paths[:1] + tsp_optimum_paths,
            singleObjTrainer.heuristic + multiObjTrainer.heuristic
        ):
            for org_idx, org in enumerate(optimum_path):
                dest_1 = optimum_path[org_idx - 1]
                dest_2 = optimum_path[(org_idx + 1) % tsp_num_nodes]

                for node in range(tsp_num_nodes):
                    if (
                        org in tsp_banned_nodes or
                        node in tsp_banned_nodes or
                        node == org
                    ):
                        self.assertEqual(
                            heuristic[org][node], 0
                        )
                    elif node == dest_1 or node == dest_2:
                        self.assertGreaterEqual(
                            heuristic[org][node], 1
                        )
                    else:
                        self.assertGreaterEqual(
                            heuristic[org][node], 0.1
                        )
                        self.assertLess(
                            heuristic[org][node], 1
                        )

        # Check the pheromone influence
        self.assertIsInstance(singleObjTrainer.pheromone_influence, list)
        self.assertIsInstance(multiObjTrainer.pheromone_influence, list)

        self.assertEqual(len(singleObjTrainer.pheromone_influence), 1)
        self.assertEqual(
            len(multiObjTrainer.pheromone_influence),
            valid_tsp_fitness_func_multi.num_obj
        )

        for pher_infl in (
            singleObjTrainer.pheromone_influence +
            multiObjTrainer.pheromone_influence
        ):
            self.assertEqual(pher_infl, DEFAULT_PHEROMONE_INFLUENCE)

        # Check the heuristic influence
        self.assertIsInstance(singleObjTrainer.heuristic_influence, list)
        self.assertIsInstance(multiObjTrainer.heuristic_influence, list)

        self.assertEqual(len(singleObjTrainer.heuristic_influence), 1)
        self.assertEqual(
            len(multiObjTrainer.heuristic_influence),
            valid_tsp_fitness_func_multi.num_obj
        )

        for heur_infl in (
            singleObjTrainer.heuristic_influence +
            multiObjTrainer.heuristic_influence
        ):
            self.assertEqual(heur_infl, DEFAULT_HEURISTIC_INFLUENCE)

        # Check the default parameters
        self.assertEqual(singleObjTrainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(
            singleObjTrainer.col_size,
            singleObjTrainer.fitness_function.num_nodes
        )
        self.assertEqual(singleObjTrainer.current_iter, None)
        self.assertEqual(multiObjTrainer.col, None)
        self.assertEqual(singleObjTrainer.pheromone, None)
        self.assertEqual(multiObjTrainer.choice_info, None)
        self.assertEqual(singleObjTrainer._node_list, None)

    def test_initial_pheromone_single(self):
        """Test the initial_pheromone property for single-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for initial_pheromone. Should fail
        invalid_initial_pheromone = (type, None, max)
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone
                )

        # Try invalid values for initial_pheromone. Should fail
        invalid_initial_pheromone = [
            (-1, ), (max, ), (0, ), (), (1, 2, 3), ('a'), -1, 0, 'a'
            ]
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(ValueError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone
                )

        # Try valid values for initial_pheromone
        initial_pheromone = 3
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone
        )
        self.assertEqual(trainer.initial_pheromone, [initial_pheromone])

        initial_pheromone = [2]
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone
        )
        self.assertEqual(trainer.initial_pheromone, initial_pheromone)

    def test_initial_pheromone_multi(self):
        """Test the initial_pheromone property for multi-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for initial_pheromone. Should fail
        invalid_initial_pheromone = (type, None, max)
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone
                )

        # Try invalid values for initial_pheromone. Should fail
        invalid_initial_pheromone = [
            (-1, ), (max, ), (0, ), (), (1, 2, 3), ('a'), -1, 0, 'a', (1, 0)
            ]
        for initial_pheromone in invalid_initial_pheromone:
            with self.assertRaises(ValueError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone
                )

        # Try valid values for initial_pheromone
        initial_pheromone = 3
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone
        )
        self.assertEqual(
            trainer.initial_pheromone,
            [initial_pheromone] * trainer.fitness_function.num_obj
        )

        initial_pheromone = [2]
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone
        )
        self.assertEqual(
            trainer.initial_pheromone,
            initial_pheromone * trainer.fitness_function.num_obj
        )

        initial_pheromone = [2, 3]
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone
        )
        self.assertEqual(
            trainer.initial_pheromone, initial_pheromone)

    def test_heuristic_single(self):
        """Test the heuristic property for single-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1
        # Try invalid types for heuristic. Should fail
        invalid_heuristic = (type, 1)
        for heuristic in invalid_heuristic:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try invalid values for heuristic. Should fail
        invalid_heuristic = (
            # Empty
            (),
            # Wrong shape
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes + 1),
                    dtype=float
                ),
            ),
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes + 1),
                dtype=float
            ),
            [[1, 2, 3], [4, 5, 6]],
            ([[1, 2, 3], [4, 5, 6]], ),
            [[1, 2], [3, 4], [5, 6]],
            ([[1, 2], [3, 4], [5, 6]], ),
            # Negative values
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float) * -1
            ],
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes), dtype=float
            ) * -1,
            # Empty matrix
            (np.ones(shape=(0, 0), dtype=float), ),
            np.ones(shape=(0, 0), dtype=float),
            # Wrong number of matrices
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes), dtype=float
                ),
            ) * 3,
        )
        for heuristic in invalid_heuristic:
            with self.assertRaises(ValueError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try single two-dimensional array-like objects
        valid_heuristic = (
            np.full(
                shape=(tsp_num_nodes, tsp_num_nodes),
                fill_value=4,
                dtype=float
            ),
            [[1, 2], [3, 4]]
        )
        for heuristic in valid_heuristic:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                heuristic=heuristic
            )
            for heur in trainer.heuristic:
                self.assertTrue(np.all(heur == np.asarray(heuristic)))

        # Try sequences of single two-dimensional array-like objects
        valid_heuristic = (
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                )
            ],
            [[[1, 2], [3, 4]]],
            ([[1, 2], [3, 4]], )
        )
        for heuristic in valid_heuristic:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                heuristic=heuristic
            )
            for heur in trainer.heuristic:
                self.assertTrue(np.all(heur == np.asarray(heuristic[0])))

    def test_heuristic_multi(self):
        """Test the heuristic property for multi-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1
        # Try invalid types for heuristic. Should fail
        invalid_heuristic = (type, 1)
        for heuristic in invalid_heuristic:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try invalid values for heuristic. Should fail
        invalid_heuristic = (
            # Empty
            (),
            # Wrong shape
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes + 1),
                    dtype=float
                ),
            ),
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes + 1),
                dtype=float
            ),
            [[1, 2, 3], [4, 5, 6]],
            ([[1, 2, 3], [4, 5, 6]], ),
            [[1, 2], [3, 4], [5, 6]],
            ([[1, 2], [3, 4], [5, 6]], ),
            # Negative values
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ) * -1
            ],
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes),
                dtype=float
            ) * -1,
            # Different shapes
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ),
                np.ones(
                    shape=(tsp_num_nodes+1, tsp_num_nodes+1),
                    dtype=float
                ),
            ),
            # Empty matrix
            (np.ones(shape=(0, 0), dtype=float), ),
            np.ones(shape=(0, 0), dtype=float),
            # Wrong number of matrices
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ),
            ) * 3,
        )
        for heuristic in invalid_heuristic:
            with self.assertRaises(ValueError):
                # print(heuristic)
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try single two-dimensional array-like objects
        valid_heuristic = (
            np.full(
                shape=(
                    tsp_num_nodes, tsp_num_nodes
                ),
                fill_value=4,
                dtype=float
            ),
            [[1, 2], [3, 4]]
        )
        for heuristic in valid_heuristic:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                heuristic=heuristic
            )
            for heur in trainer.heuristic:
                self.assertTrue(np.all(heur == np.asarray(heuristic)))

        # Try sequences of one single two-dimensional array-like object
        valid_heuristic = (
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                )
            ],
            [[[1, 2], [3, 4]]],
            ([[1, 2], [3, 4]], )
        )
        for heuristic in valid_heuristic:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                heuristic=heuristic
            )
            for heur in trainer.heuristic:
                self.assertTrue(np.all(heur == np.asarray(heuristic[0])))

        # Try sequences of various single two-dimensional array-like objects
        valid_heuristic = (
            [
                np.ones(shape=(tsp_num_nodes, tsp_num_nodes), dtype=float),
                np.full(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    fill_value=4,
                    dtype=float
                )
            ],
            [[[1, 2], [3, 4]], np.ones(shape=(2, 2), dtype=float)],
            ([[1, 2], [3, 4]], ([5, 6], [7, 8]))
        )
        for heuristic in valid_heuristic:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                heuristic=heuristic
            )
            for h1, h2 in zip(trainer.heuristic, heuristic):
                self.assertTrue(np.all(h1 == np.asarray(h2)))

    def test_pheromone_influence_single(self):
        """Test the pheromone_influence property for single-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for pheromone_influence. Should fail
        invalid_pheromone_influence = (type, max)
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    pheromone_influence=pheromone_influence
                )

        # Try invalid values for pheromone_influence. Should fail
        invalid_pheromone_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(ValueError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    pheromone_influence=pheromone_influence
                )

        # Try valid values for pheromone_influence
        valid_pheromone_influence = [3, 0]
        for pheromone_influence in valid_pheromone_influence:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(
                trainer.pheromone_influence, [pheromone_influence]
            )

        valid_pheromone_influence = [(0,), [2]]
        for pheromone_influence in valid_pheromone_influence:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(
                trainer.pheromone_influence,
                list(pheromone_influence)
            )

    def test_pheromone_influence_multi(self):
        """Test the pheromone_influence property for multi-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for pheromone_influence. Should fail
        invalid_pheromone_influence = (type, max)
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    pheromone_influence=pheromone_influence
                )

        # Try invalid values for pheromone_influence. Should fail
        invalid_pheromone_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for pheromone_influence in invalid_pheromone_influence:
            with self.assertRaises(ValueError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    pheromone_influence=pheromone_influence
                )

        # Try valid values for pheromone_influence
        valid_pheromone_influence = [3, 0]
        for pheromone_influence in valid_pheromone_influence:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(
                trainer.pheromone_influence,
                [pheromone_influence] * trainer.fitness_function.num_obj
            )

        valid_pheromone_influence = [(0,), [2]]
        for pheromone_influence in valid_pheromone_influence:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                pheromone_influence=pheromone_influence
            )
            self.assertEqual(
                trainer.pheromone_influence,
                list(pheromone_influence) * trainer.fitness_function.num_obj
            )

        pheromone_influence = [2, 3]
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone,
            pheromone_influence=pheromone_influence
        )
        self.assertEqual(
            trainer.pheromone_influence, pheromone_influence)

    def test_heuristic_influence_single(self):
        """Test the heuristic_influence property for single-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for heuristic_influence. Should fail
        invalid_heuristic_influence = (type, max)
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    heuristic_influence=heuristic_influence
                )

        # Try invalid values for heuristic_influence. Should fail
        invalid_heuristic_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(ValueError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    heuristic_influence=heuristic_influence
                )

        # Try valid values for heuristic_influence
        valid_heuristic_influence = [3, 0]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence, [heuristic_influence]
            )

        valid_heuristic_influence = [(0,), [2]]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence,
                list(heuristic_influence)
            )

    def test_heuristic_influence_multi(self):
        """Test the heuristic_influence property for multi-matrix trainers."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for heuristic_influence. Should fail
        invalid_heuristic_influence = (type, max)
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    heuristic_influence=heuristic_influence
                )

        # Try invalid values for heuristic_influence. Should fail
        invalid_heuristic_influence = [
            (-1, ), (max, ), (), (1, 2, 3), ('a'), -1, 'a'
            ]
        for heuristic_influence in invalid_heuristic_influence:
            with self.assertRaises(ValueError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    heuristic_influence=heuristic_influence
                )

        # Try valid values for heuristic_influence
        valid_heuristic_influence = [3, 0]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence,
                [heuristic_influence] * trainer.fitness_function.num_obj
            )

        valid_heuristic_influence = [(0,), [2]]
        for heuristic_influence in valid_heuristic_influence:
            trainer = MyMultiObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_multi,
                initial_pheromone,
                heuristic_influence=heuristic_influence
            )
            self.assertEqual(
                trainer.heuristic_influence,
                list(heuristic_influence) * trainer.fitness_function.num_obj
            )

        heuristic_influence = [2, 3]
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone,
            heuristic_influence=heuristic_influence
        )
        self.assertEqual(
            trainer.heuristic_influence, heuristic_influence)

    def test_max_num_iters(self):
        """Test the max_num_iters property."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    max_num_iters=max_num_iters
                )

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    max_num_iters=max_num_iters
                )

        # Try a valid value for max_num_iters
        max_num_iters = 210
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone,
            max_num_iters=max_num_iters
        )
        self.assertEqual(max_num_iters, trainer.max_num_iters)

    def test_col_size(self):
        """Test the col_size property."""
        ant_cls = TSPAnt
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for col_size. Should fail
        invalid_col_size = (type, 'a', 1.5)
        for col_size in invalid_col_size:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    col_size=col_size
                )

        # Try invalid values for col_size. Should fail
        invalid_col_size = (-1, 0)
        for col_size in invalid_col_size:
            with self.assertRaises(ValueError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    col_size=col_size
                )

        # Try a valid value for col_size
        col_size = 233
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone,
            col_size=col_size
        )
        self.assertEqual(col_size, trainer.col_size)

    def test_state(self):
        """Test the get_state and _set_state methods."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertEqual(state["num_evals"], trainer.num_evals)
        self.assertEqual(state["pheromone"], trainer.pheromone)

        # Change the state
        state["num_evals"] = 100
        state["pheromone"] = [
            np.full((tsp_num_nodes, tsp_num_nodes), 8, dtype=float)
        ]

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
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)

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
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = (2, )
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Reset the state
        trainer._reset_state()

        # Check the pheromone
        self.assertEqual(trainer.pheromone, None)

    def test_init_internals(self):
        """Test the _init_internals method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()

        self.assertEqual(trainer.col, [])
        self.assertEqual(trainer.choice_info, None)
        self.assertTrue(
            np.all(
                trainer._node_list == np.arange(0, tsp_num_nodes, dtype=int)
            )
        )

    def test_reset_internals(self):
        """Test the _init_internals method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 4]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)
        trainer._init_search()
        trainer._reset_internals()

        self.assertEqual(trainer.col, None)
        self.assertEqual(trainer.choice_info, None)
        self.assertEqual(trainer._node_list, None)

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)

        # Try to get the choice info before the search initialization
        choice_info = trainer.choice_info
        self.assertEqual(choice_info, None)

        # Try to get the choice_info after initializing the search
        trainer._init_search()
        trainer._start_iteration()
        choice_info = trainer.choice_info

        # Check the probabilities for banned nodes. Should be 0
        for node in tsp_banned_nodes:
            self.assertAlmostEqual(np.sum(choice_info[node]), 0)

        for node in tsp_feasible_nodes:
            self.assertAlmostEqual(
                np.sum(choice_info[node]),
                np.sum(
                    trainer.pheromone[0][node] * trainer.heuristic[0][node]
                )
            )

    def test_initial_choice(self):
        """Test the _initial_choice method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Try to generate valid first nodes
        times = 1000
        ant = trainer.solution_cls(
            trainer.species, trainer.fitness_function.Fitness
        )
        for _ in repeat(None, times):
            self.assertTrue(trainer._initial_choice(ant) in tsp_feasible_nodes)

        # Try when all nodes are unfeasible
        trainer.heuristic = [np.zeros((tsp_num_nodes, tsp_num_nodes))]
        trainer._start_iteration()
        self.assertEqual(trainer._initial_choice(ant), None)

    def test_feasible_neighborhood_probs(self):
        """Test the _feasible_neighborhood_probs method."""
        # Trainer parameters
        species = FSSpecies(
            num_feats=fs_dataset.num_feats,
            min_size=1,
            min_feat=1,
            max_feat=fs_dataset.num_feats-2
        )
        initial_pheromone = [2]
        params = {
            "solution_cls": FSAnt,
            "species": species,
            "fitness_function": fs_fitness_func,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Try with an ant with an empty path. Should fail
        ant = FSAnt(
            species,
            fs_fitness_func.Fitness
        )
        with self.assertRaises(ValueError):
            trainer._feasible_neighborhood_probs(ant)

        # Choose a first node for the ant
        ant.append(trainer._initial_choice(ant))

        # Controls if the next node will be discarded or appended
        discardNextNode = True

        # Check the choice probabilities of the ant's current node neighborhood
        node_list = np.arange(0, species.num_feats, dtype=int)
        probs = trainer._feasible_neighborhood_probs(ant)
        while np.sum(probs) > 0:
            self.assertAlmostEqual(np.sum(probs), 1)
            next_node = np.random.choice(node_list, p=probs)

            if discardNextNode:
                ant.discard(next_node)
            else:
                ant.append(next_node)
            discardNextNode = not discardNextNode

            probs = trainer._feasible_neighborhood_probs(ant)

    def test_next_choice(self):
        """Test the _next_choice method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)
        trainer._init_search()

        # Try to generate valid first nodes
        times = 1000
        for _ in repeat(None, times):
            trainer._start_iteration()
            ant = TSPAnt(
                species,
                tsp_fitness_func_multi.Fitness
            )
            choice = trainer._next_choice(ant)

            while choice is not None:
                ant.append(choice)
                choice = trainer._next_choice(ant)

            self.assertEqual(len(ant.path), len(tsp_feasible_nodes))

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        # Trainer parameters
        params = {
            "solution_cls": TSPAnt,
            "species": TSPSpecies(tsp_num_nodes, tsp_banned_nodes),
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": [2]
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()

        # Try to generate valid ants
        times = 1000
        for _ in repeat(None, times):
            trainer._start_iteration()
            ant = trainer._generate_ant()
            self.assertEqual(len(ant.path), len(tsp_feasible_nodes))

        # Try an ant with all the nodes banned
        params["species"] = TSPSpecies(tsp_num_nodes, range(tsp_num_nodes))

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()
        ant = trainer._generate_ant()

    def test_generate_col(self):
        """Test the _generate_col_method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)

        # Generate the colony
        trainer._init_search()
        trainer._start_iteration()
        trainer._generate_col()

        # Check the colony
        self.assertEqual(len(trainer.col), trainer.col_size)
        for ant in trainer.col:
            self.assertIsInstance(ant, TSPAnt)

    def test_init_pheromone(self):
        """Test the _init_pheromone method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        single_obj_params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }
        multi_obj_params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        single_obj_trainer = MySingleObjTrainer(**single_obj_params)
        multi_obj_trainer = MyMultiObjTrainer(**multi_obj_params)

        single_obj_trainer._init_pheromone()
        multi_obj_trainer._init_pheromone()

        # Check the pheromone matrices
        self.assertIsInstance(single_obj_trainer.pheromone, list)
        self.assertIsInstance(multi_obj_trainer.pheromone, list)
        for (
            initial_pheromone,
            pheromone_matrix
        ) in zip(
            single_obj_trainer.initial_pheromone,
            single_obj_trainer.pheromone
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))

        for (
            initial_pheromone,
            pheromone_matrix
        ) in zip(
            multi_obj_trainer.initial_pheromone,
            multi_obj_trainer.pheromone
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""

        def assert_path_pheromone_increment(trainer, ant, weight):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are amount of pheromone.
            """
            for pher_index, init_pher_val in enumerate(
                trainer.initial_pheromone
            ):
                pheromone_value = (
                    init_pher_val +
                    ant.fitness.pheromone_amount[pher_index] * weight
                )
                org = ant.path[-1]
                for dest in ant.path:
                    self.assertAlmostEqual(
                        trainer.pheromone[pher_index][org][dest],
                        pheromone_value
                    )
                    self.assertAlmostEqual(
                        trainer.pheromone[pher_index][dest][org],
                        pheromone_value
                    )
                    org = dest

        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Check the initial pheromone
        for pher_index, init_pher_val in enumerate(trainer.initial_pheromone):
            self.assertTrue(
                np.all(trainer.pheromone[pher_index] == init_pher_val)
            )

        # Try with the current colony
        # Only the iteration-best ant should deposit pheromone
        trainer._generate_col()
        weight = 3
        trainer._deposit_pheromone(trainer.col, weight)
        assert_path_pheromone_increment(trainer, trainer.col[0], weight)

    def test_copy(self):
        """Test the __copy__ method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone,
            "col_size": 13,
            "max_num_iters": 11
        }

        # Create the trainer
        trainer1 = MyMultiObjTrainer(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertNotEqual(id(trainer1), id(trainer2))

        # The objects attributes are shared
        self.assertEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertEqual(id(trainer1.species), id(trainer2.species))

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone,
            "col_size": 13,
            "max_num_iters": 11
        }

        # Create the trainer
        trainer1 = MyMultiObjTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test."""
        # Trainer parameters
        params = {
            "solution_cls": TSPAnt,
            "species": TSPSpecies(tsp_num_nodes, tsp_banned_nodes),
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": [2]
        }

        # Create the trainer
        trainer1 = MySingleObjTrainer(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = MySingleObjTrainer.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = TSPSpecies(tsp_num_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": TSPAnt,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.aco.abc.SingleColACO`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.aco.abc.SingleColACO`
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertNotEqual(
            id(trainer1.fitness_function.distance),
            id(trainer2.fitness_function.distance)
        )

        for dist1, dist2 in zip(
            trainer1.fitness_function.distance,
            trainer2.fitness_function.distance
        ):
            self.assertTrue((dist1 == dist2).all())

        self.assertNotEqual(id(trainer1.species), id(trainer2.species))

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
