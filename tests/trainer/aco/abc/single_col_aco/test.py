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

"""Unit test for :class:`culebra.trainer.aco.abc.SingleColACO`."""

import unittest
from itertools import repeat
from copy import copy, deepcopy
from os import remove

import numpy as np

from culebra import DEFAULT_MAX_NUM_ITERS, SERIALIZED_FILE_EXTENSION
from culebra.abc import Fitness
from culebra.trainer.aco import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_EXPLOITATION_PROB
)
from culebra.trainer.aco.abc import SingleColACO
from culebra.solution.tsp import Species, Solution, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


class MySingleObjTrainer(SingleColACO):
    """Dummy implementation of a trainer method."""

    @property
    def num_pheromone_matrices(self):
        """Number of pheromone matrices used by this trainer."""
        return 1

    @property
    def num_heuristic_matrices(self):
        """Number of heuristic matrices used by this trainer."""
        return 1

    @property
    def _default_heuristic(self):
        """Default heuristic matrices."""
        return self.fitness_function.heuristic

    @property
    def pheromone_shapes(self):
        """Shape of the pheromone matrices."""
        return ((self.species.num_nodes, ) * 2,) * self.num_pheromone_matrices

    @property
    def heuristic_shapes(self):
        """Shape of the heuristic matrices."""
        return ((self.species.num_nodes, ) * 2,) * self.num_heuristic_matrices

    @property
    def _default_col_size(self):
        """Default colony size."""
        return self.species.num_nodes

    def _calculate_choice_info(self):
        """Calculate the choice info matrix."""
        self._choice_info = np.ones((self.species.num_nodes,) * 2)

        for (
            pheromone,
            pheromone_influence
        ) in zip(
            self.pheromone,
            self.pheromone_influence
        ):
            self._choice_info *= np.power(pheromone, pheromone_influence)

        for (
            heuristic,
            heuristic_influence
        ) in zip(
            self.heuristic,
            self.heuristic_influence
        ):
            self._choice_info *= np.power(heuristic, heuristic_influence)

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

    def _ant_choice_info(self, ant):
        """Return the choice info to obtain the next node."""
        ant_choice_info = super()._ant_choice_info(ant)
        ant_choice_info[self.species.banned_nodes] = 0

        return ant_choice_info


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
tsp_fitness_func_multi = MultiObjectivePathLength(
    PathLength.from_path(tsp_optimum_paths[0]),
    PathLength.from_path(tsp_optimum_paths[1])
)
tsp_fitness_func_single = tsp_fitness_func_multi.objectives[0]
tsp_banned_nodes = [0, tsp_num_nodes-1]
tsp_feasible_nodes = list(range(1, tsp_num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.abc.SingleColACO`."""

    def test_init(self):
        """Test __init__."""
        valid_ant_cls = Ant
        valid_species = Species(tsp_num_nodes, tsp_banned_nodes)
        valid_tsp_fitness_func_single = tsp_fitness_func_single
        valid_tsp_fitness_func_multi = tsp_fitness_func_multi
        valid_initial_pheromone = 1

        # Try invalid ant classes. Should fail
        invalid_ant_classes = (type, None, 1, Solution)
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
        single_obj_trainer = MySingleObjTrainer(
            valid_ant_cls,
            valid_species,
            valid_tsp_fitness_func_single,
            valid_initial_pheromone
        )
        multi_obj_trainer = MyMultiObjTrainer(
            valid_ant_cls,
            valid_species,
            valid_tsp_fitness_func_multi,
            valid_initial_pheromone
        )

        self.assertEqual(single_obj_trainer.solution_cls, valid_ant_cls)
        self.assertEqual(multi_obj_trainer.solution_cls, valid_ant_cls)

        self.assertEqual(single_obj_trainer.species, valid_species)
        self.assertEqual(multi_obj_trainer.species, valid_species)

        self.assertEqual(
            single_obj_trainer.fitness_function,
            valid_tsp_fitness_func_single
        )
        self.assertEqual(
            multi_obj_trainer.fitness_function,
            valid_tsp_fitness_func_multi
        )

        self.assertEqual(
            single_obj_trainer.initial_pheromone, (valid_initial_pheromone,)
        )
        self.assertEqual(
            multi_obj_trainer.initial_pheromone,
            (valid_initial_pheromone,) * valid_tsp_fitness_func_multi.num_obj
        )

        self.assertIsInstance(single_obj_trainer.heuristic, tuple)
        self.assertIsInstance(multi_obj_trainer.heuristic, tuple)

        self.assertEqual(len(single_obj_trainer.heuristic), 1)
        self.assertEqual(
            len(multi_obj_trainer.heuristic),
            valid_tsp_fitness_func_multi.num_obj
        )

        for matrix in single_obj_trainer.heuristic:
            self.assertEqual(
                matrix.shape, (tsp_num_nodes, tsp_num_nodes)
            )
        for matrix in multi_obj_trainer.heuristic:
            self.assertEqual(
                matrix.shape, (tsp_num_nodes, tsp_num_nodes)
            )

        # Check the pheromone influence
        self.assertIsInstance(single_obj_trainer.pheromone_influence, tuple)
        self.assertIsInstance(multi_obj_trainer.pheromone_influence, tuple)

        self.assertEqual(len(single_obj_trainer.pheromone_influence), 1)
        self.assertEqual(
            len(multi_obj_trainer.pheromone_influence),
            valid_tsp_fitness_func_multi.num_obj
        )

        for pher_infl in (
            single_obj_trainer.pheromone_influence +
            multi_obj_trainer.pheromone_influence
        ):
            self.assertEqual(pher_infl, DEFAULT_PHEROMONE_INFLUENCE)

        # Check the heuristic influence
        self.assertIsInstance(single_obj_trainer.heuristic_influence, tuple)
        self.assertIsInstance(multi_obj_trainer.heuristic_influence, tuple)

        self.assertEqual(len(single_obj_trainer.heuristic_influence), 1)
        self.assertEqual(
            len(multi_obj_trainer.heuristic_influence),
            valid_tsp_fitness_func_multi.num_obj
        )

        for heur_infl in (
            single_obj_trainer.heuristic_influence +
            multi_obj_trainer.heuristic_influence
        ):
            self.assertEqual(heur_infl, DEFAULT_HEURISTIC_INFLUENCE)

        # Check the default parameters
        self.assertEqual(
            single_obj_trainer.exploitation_prob, DEFAULT_EXPLOITATION_PROB
        )
        self.assertEqual(single_obj_trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(single_obj_trainer.current_iter, None)
        self.assertEqual(multi_obj_trainer.col, None)
        self.assertEqual(single_obj_trainer.pheromone, None)
        self.assertEqual(multi_obj_trainer.choice_info, None)

    def test_initial_pheromone_single(self):
        """Test the initial_pheromone property for single-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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
        self.assertEqual(trainer.initial_pheromone, (initial_pheromone,))

        initial_pheromone = [2]
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone
        )
        self.assertEqual(trainer.initial_pheromone, tuple(initial_pheromone))

    def test_initial_pheromone_multi(self):
        """Test the initial_pheromone property for multi-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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
            (initial_pheromone,) * trainer.fitness_function.num_obj
        )

        initial_pheromone = [2, 3]
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone
        )
        self.assertEqual(
            trainer.initial_pheromone, tuple(initial_pheromone))

    def test_heuristic_single(self):
        """Test the heuristic property for single-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1
        # Try invalid types for heuristic. Should fail
        invalid_heuristic_types = (type, 1, Ant)
        for heuristic in invalid_heuristic_types:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try invalid values for heuristic. Should fail
        invalid_heuristic_values = (
            # Wrong type
            (Ant,),
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
            # Negative values
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ) * -1
            ],
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes), dtype=float
            ) * -1,
            # Wrong number of matrices
            (),
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes), dtype=float
                ),
            ) * 3
        )
        for heuristic in invalid_heuristic_values:
            with self.assertRaises(ValueError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try a single array
        valid_heuristic = np.full(
            shape=(tsp_num_nodes, tsp_num_nodes),
            fill_value=4,
            dtype=float
        )
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone,
            heuristic=valid_heuristic
        )
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == valid_heuristic))

        # Try a sequence of a single arrays
        valid_heuristic = [
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes),
                dtype=float
            )
        ]
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone,
            heuristic=valid_heuristic
        )
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == valid_heuristic[0]))

        # Try the default heuristic
        trainer = MySingleObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_single,
            initial_pheromone
        )
        for heur1, heur2 in zip(
            tsp_fitness_func_single.heuristic, trainer.heuristic
        ):
            self.assertTrue(np.all(heur1 == heur2))

    def test_heuristic_multi(self):
        """Test the heuristic property for multi-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1
        # Try invalid types for heuristic. Should fail
        invalid_heuristic_types = (type, 1, Ant)
        for heuristic in invalid_heuristic_types:
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
            # Wrong type
            (Ant,),
            # Wrong shape
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes + 1),
                    dtype=float
                ),
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ),
            ),
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ),
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes + 1),
                    dtype=float
                ),
            ),
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes + 1),
                dtype=float
            ),
            # Negative values
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ),
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                ) * -1
            ],
            [
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                )* -1,
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes),
                    dtype=float
                )
            ],
            np.ones(
                shape=(tsp_num_nodes, tsp_num_nodes),
                dtype=float
            ) * -1,
            # Wrong number of matrices
            (),
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes), dtype=float
                ),
            ),
            (
                np.ones(
                    shape=(tsp_num_nodes, tsp_num_nodes), dtype=float
                ),
            ) * 3
        )
        for heuristic in invalid_heuristic:
            with self.assertRaises(ValueError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    heuristic=heuristic
                )

        # Try a single array
        valid_heuristic = np.full(
            shape=(
                tsp_num_nodes, tsp_num_nodes
            ),
            fill_value=4,
            dtype=float
        )
        trainer =MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone,
            heuristic=valid_heuristic
        )
        for heur in trainer.heuristic:
            self.assertTrue(np.all(heur == np.asarray(valid_heuristic)))

        # Try asequence of arrays
        valid_heuristic = [
            np.ones(shape=(tsp_num_nodes, tsp_num_nodes), dtype=float),
            np.full(
                shape=(tsp_num_nodes, tsp_num_nodes),
                fill_value=4,
                dtype=float
            )
        ]
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone,
            heuristic=valid_heuristic
        )
        for h1, h2 in zip(trainer.heuristic, valid_heuristic):
            self.assertTrue(np.all(h1 == np.asarray(h2)))

        # Try the default heuristic
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone
        )
        for heur1, heur2 in zip(
            tsp_fitness_func_single.heuristic, trainer.heuristic
        ):
            self.assertTrue(np.all(heur1 == heur2))

    def test_pheromone_influence_single(self):
        """Test the pheromone_influence property for single-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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
                trainer.pheromone_influence, (pheromone_influence,)
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
                tuple(pheromone_influence)
            )

    def test_pheromone_influence_multi(self):
        """Test the pheromone_influence property for multi-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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
                (pheromone_influence,) * trainer.fitness_function.num_obj
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
            trainer.pheromone_influence, tuple(pheromone_influence))

    def test_heuristic_influence_single(self):
        """Test the heuristic_influence property for single-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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
                trainer.heuristic_influence, (heuristic_influence,)
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
                tuple(heuristic_influence)
            )

    def test_heuristic_influence_multi(self):
        """Test the heuristic_influence property for multi-matrix trainers."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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
                (heuristic_influence,) * trainer.fitness_function.num_obj
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
            trainer.heuristic_influence, tuple(heuristic_influence))

    def test_exploitation_prob(self):
        """Test the exploitation_prob property."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for exploitation_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    exploitation_prob=prob
                )

        # Try invalid values for exploitation_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MySingleObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_single,
                    initial_pheromone,
                    exploitation_prob=prob
                )

        # Try valid values for exploitation_prob
        valid_probs = (0, 0.5, 1)
        for prob in valid_probs:
            trainer = MySingleObjTrainer(
                ant_cls,
                species,
                tsp_fitness_func_single,
                initial_pheromone,
                exploitation_prob=prob
            )
            self.assertEqual(trainer.exploitation_prob, prob)

    def test_col_size(self):
        """Test the col_size property."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 1

        # Try invalid types for col_size. Should fail
        invalid_col_size = (type, 'a', 1.5)
        for col_size in invalid_col_size:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
                    initial_pheromone,
                    col_size=col_size
                )

        # Try invalid types for col_size. Should fail
        invalid_col_size = (type, 'a', 1.5)
        for col_size in invalid_col_size:
            with self.assertRaises(TypeError):
                MyMultiObjTrainer(
                    ant_cls,
                    species,
                    tsp_fitness_func_multi,
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
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone,
            col_size=col_size
        )
        self.assertEqual(col_size, trainer.col_size)

        # Try the default value for col_size
        trainer = MyMultiObjTrainer(
            ant_cls,
            species,
            tsp_fitness_func_multi,
            initial_pheromone
        )
        self.assertEqual(trainer.col_size, trainer.species.num_nodes)

    def test_max_num_iters(self):
        """Test the max_num_iters property."""
        ant_cls = Ant
        species = Species(tsp_num_nodes, tsp_banned_nodes)
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

    def test_state(self):
        """Test the get_state and _set_state methods."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": Ant,
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
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": Ant,
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
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = (2, )
        params = {
            "solution_cls": Ant,
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
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()

        self.assertEqual(trainer.col, [])
        self.assertEqual(trainer.choice_info, None)

    def test_reset_internals(self):
        """Test the _init_internals method."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 4]
        params = {
            "solution_cls": Ant,
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

    def test_ant_choice_info(self):
        """Test the _ant_choice_info method."""
        # Trainer parameters
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(tsp_num_nodes, tsp_banned_nodes),
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": [2]
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Try to generate valid first nodes
        ant = trainer.solution_cls(
            trainer.species, trainer.fitness_function.fitness_cls
        )
        ant_choice_info = trainer._ant_choice_info(ant)
        self.assertTrue((ant_choice_info[tsp_banned_nodes] == 0).all())
        self.assertTrue((
            ant_choice_info[tsp_feasible_nodes] ==
            np.sum(trainer.choice_info, axis=1)[tsp_feasible_nodes]
        ).all())

    def test_next_choice(self):
        """Test the _next_choice method."""
        def test(trainer):
        # Try to generate valid first nodes
            times = 1000
            for _ in repeat(None, times):
                trainer._start_iteration()
                ant = Ant(
                    trainer.species,
                    trainer.fitness_function.fitness_cls
                )
                choice = trainer._next_choice(ant)

                while choice is not None:
                    ant.append(choice)
                    choice = trainer._next_choice(ant)

        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySingleObjTrainer(**params)

        # Test the exploitation ...
        trainer.exploitation_prob = 1
        trainer._init_search()
        test(trainer)

        # Test the exploration ...
        trainer.exploitation_prob = 0
        trainer._init_search()
        test(trainer)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 4]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)

        # Try before any colony has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, tuple)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Update the elite
        trainer._init_search()
        trainer._start_iteration()
        ant = trainer._generate_ant()
        trainer.col.append(ant)

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one solution
        self.assertEqual(len(best_ones[0]), 1)

        # Check that the solution in hof is sol1
        self.assertTrue(ant in best_ones[0])

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(tsp_num_nodes, tsp_banned_nodes),
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
        params["species"] = Species(tsp_num_nodes, range(tsp_num_nodes))

        # Create the trainer
        trainer = MySingleObjTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()
        ant = trainer._generate_ant()

    def test_generate_col(self):
        """Test the _generate_col_method."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
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
            self.assertIsInstance(ant, Ant)

    def test_init_pheromone(self):
        """Test the _init_pheromone method."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2]
        single_obj_params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }
        multi_obj_params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": (
                initial_pheromone * tsp_fitness_func_multi.num_obj
            )
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
            pheromone_matrix,
            shape
        ) in zip(
            single_obj_trainer.initial_pheromone,
            single_obj_trainer.pheromone,
            single_obj_trainer.pheromone_shapes
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))
            self.assertEqual(pheromone_matrix.shape, shape)

        for (
            initial_pheromone,
            pheromone_matrix,
            shape
        ) in zip(
            multi_obj_trainer.initial_pheromone,
            multi_obj_trainer.pheromone,
            multi_obj_trainer.pheromone_shapes
        ):
            self.assertTrue(np.all(pheromone_matrix == initial_pheromone))
            self.assertEqual(pheromone_matrix.shape, shape)

    def test_pheromone_amount(self):
        """Test the _pheromone_amount method."""
        class MyFitness(Fitness):
            """Dummy fitness class."""

            weights = (1, -1)
            names = ("obj1", "obj2")
            thresholds = [0.1, 0.2]

        species = Species(tsp_num_nodes, tsp_banned_nodes)
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": 1
        }

        # Create the trainer
        trainer = MyMultiObjTrainer(**params)

        ant = Ant(species, MyFitness)
        ant.fitness.values = (2, 3)

        self.assertEqual(trainer._pheromone_amount(ant), (2, 1/3))

    def test_copy(self):
        """Test the __copy__ method."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": Ant,
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
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": Ant,
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
            "solution_cls": Ant,
            "species": Species(tsp_num_nodes, tsp_banned_nodes),
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": [2]
        }

        # Create the trainer
        trainer1 = MySingleObjTrainer(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = MySingleObjTrainer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(tsp_num_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
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
        :type trainer1: ~culebra.trainer.aco.abc.SingleColACO
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.aco.abc.SingleColACO
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )

        self.assertNotEqual(id(trainer1.species), id(trainer2.species))

        # Check some non mandatory parameters
        self.assertEqual(trainer1.col_size, trainer2.col_size)
        self.assertEqual(trainer1.max_num_iters, trainer2.max_num_iters)


if __name__ == '__main__':
    unittest.main()
