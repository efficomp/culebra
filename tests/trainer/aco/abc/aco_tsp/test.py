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

import numpy as np

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.aco import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_EXPLOITATION_PROB
)
from culebra.trainer.aco.abc import ACOTSP
from culebra.solution.tsp import Species, Ant
from culebra.abc import Species as GenericSpecies
from culebra.solution.abc import Ant as GenericAnt
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)


class MySinglePathTrainer(ACOTSP):
    """Dummy implementation of a trainer method."""

    @property
    def num_pheromone_matrices(self) -> int:
        """Get the number of pheromone matrices used by this trainer."""
        return 1

    @property
    def num_heuristic_matrices(self) -> int:
        """Get the number of heuristic matrices used by this trainer."""
        return 1

    def _new_state(self):
        """Generate a new trainer state."""
        super()._new_state()
        heuristic_shape = self.heuristic[0].shape
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


class MyMultiPathTrainer(MySinglePathTrainer):
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
        invalid_ant_classes = (type, None, 1, GenericAnt)
        for solution_cls in invalid_ant_classes:
            with self.assertRaises(TypeError):
                MySinglePathTrainer(
                    solution_cls,
                    valid_species,
                    valid_tsp_fitness_func_single,
                    valid_initial_pheromone
                )

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1, GenericSpecies)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyMultiPathTrainer(
                    valid_ant_cls,
                    species,
                    valid_tsp_fitness_func_multi,
                    valid_initial_pheromone
                )

        # Try invalid fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MySinglePathTrainer(
                    valid_ant_cls,
                    valid_species,
                    func,
                    valid_initial_pheromone
                )

        # Test default params
        single_path_trainer = MySinglePathTrainer(
            valid_ant_cls,
            valid_species,
            valid_tsp_fitness_func_single,
            valid_initial_pheromone
        )
        multi_path_trainer = MyMultiPathTrainer(
            valid_ant_cls,
            valid_species,
            valid_tsp_fitness_func_multi,
            valid_initial_pheromone
        )

        self.assertEqual(single_path_trainer.solution_cls, valid_ant_cls)
        self.assertEqual(multi_path_trainer.solution_cls, valid_ant_cls)

        self.assertEqual(single_path_trainer.species, valid_species)
        self.assertEqual(multi_path_trainer.species, valid_species)

        self.assertEqual(
            single_path_trainer.fitness_function,
            valid_tsp_fitness_func_single
        )
        self.assertEqual(
            multi_path_trainer.fitness_function,
            valid_tsp_fitness_func_multi
        )

        self.assertEqual(
            single_path_trainer.initial_pheromone, (valid_initial_pheromone,)
        )
        self.assertEqual(
            multi_path_trainer.initial_pheromone,
            (valid_initial_pheromone,) * valid_tsp_fitness_func_multi.num_obj
        )

        self.assertIsInstance(single_path_trainer.heuristic, tuple)
        self.assertIsInstance(multi_path_trainer.heuristic, tuple)

        self.assertEqual(len(single_path_trainer.heuristic), 1)
        self.assertEqual(
            len(multi_path_trainer.heuristic),
            valid_tsp_fitness_func_multi.num_obj
        )

        for matrix in single_path_trainer.heuristic:
            self.assertEqual(
                matrix.shape, (tsp_num_nodes, tsp_num_nodes)
            )
        for matrix in multi_path_trainer.heuristic:
            self.assertEqual(
                matrix.shape, (tsp_num_nodes, tsp_num_nodes)
            )

        # Check the heuristic
        for (
            heuristic1,
            heuristic2
        ) in zip(
            tsp_fitness_func_single.heuristic +
            tsp_fitness_func_multi.heuristic,
            single_path_trainer.heuristic + multi_path_trainer.heuristic
        ):
            self.assertTrue((heuristic1==heuristic2).all())

        # Check the pheromone influence
        self.assertIsInstance(single_path_trainer.pheromone_influence, tuple)
        self.assertIsInstance(multi_path_trainer.pheromone_influence, tuple)

        self.assertEqual(len(single_path_trainer.pheromone_influence), 1)
        self.assertEqual(
            len(multi_path_trainer.pheromone_influence),
            valid_tsp_fitness_func_multi.num_obj
        )

        for pher_infl in (
            single_path_trainer.pheromone_influence +
            multi_path_trainer.pheromone_influence
        ):
            self.assertEqual(pher_infl, DEFAULT_PHEROMONE_INFLUENCE)

        # Check the heuristic influence
        self.assertIsInstance(single_path_trainer.heuristic_influence, tuple)
        self.assertIsInstance(multi_path_trainer.heuristic_influence, tuple)

        self.assertEqual(len(single_path_trainer.heuristic_influence), 1)
        self.assertEqual(
            len(multi_path_trainer.heuristic_influence),
            valid_tsp_fitness_func_multi.num_obj
        )

        for heur_infl in (
            single_path_trainer.heuristic_influence +
            multi_path_trainer.heuristic_influence
        ):
            self.assertEqual(heur_infl, DEFAULT_HEURISTIC_INFLUENCE)

        # Check the default parameters
        self.assertEqual(
            single_path_trainer.exploitation_prob, DEFAULT_EXPLOITATION_PROB
        )
        self.assertEqual(
            single_path_trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS
        )
        self.assertEqual(single_path_trainer.current_iter, None)
        self.assertEqual(multi_path_trainer.col, None)
        self.assertEqual(single_path_trainer.pheromone, None)
        self.assertEqual(multi_path_trainer.choice_info, None)

    def test_calculate_choice_info(self):
        """Test the _calculate_choice_info method."""
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
        trainer = MyMultiPathTrainer(**params)

        # Try to get the choice info before the search initialization
        self.assertEqual(trainer.choice_info, None)

        # Try to get the choice_info after initializing the search
        trainer._init_search()
        trainer._start_iteration()

        the_choice_info = np.ones((species.num_nodes,) * 2)
        for (pher, pher_inf, heur, heur_inf) in zip(
            trainer.pheromone,
            trainer.pheromone_influence,
            trainer.heuristic,
            trainer.heuristic_influence
        ):
            the_choice_info *= np.power(pher, pher_inf)
            the_choice_info *= np.power(heur, heur_inf)

        for node in range(tsp_num_nodes):
            for next_node in range(tsp_num_nodes):
                self.assertEqual(
                    trainer.choice_info[node, next_node],
                    the_choice_info[node, next_node]
                )
                self.assertEqual(
                    trainer.choice_info[next_node, node],
                    the_choice_info[node, next_node]
                )

    def test_ant_choice_info(self):
        """Test the _ant_choice_info method."""
        # Trainer parameters
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = 2
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_single,
            "initial_pheromone": initial_pheromone
        }

        # Create the trainer
        trainer = MySinglePathTrainer(**params)
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

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""

        def assert_path_pheromone_increment(trainer, ant, weight):
            """Check the pheromone in all the arcs of a path.

            All the arcs should have the same are amount of pheromone.
            """
            pheromone_amount = trainer._pheromone_amount(ant)

            for pher_index, init_pher_val in enumerate(
                trainer.initial_pheromone
            ):
                pheromone_value = (
                    init_pher_val +
                    pheromone_amount[pher_index] * weight
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
        species = Species(tsp_num_nodes, tsp_banned_nodes)
        initial_pheromone = [2, 3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": tsp_fitness_func_multi,
            "initial_pheromone": initial_pheromone,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyMultiPathTrainer(**params)
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


if __name__ == '__main__':
    unittest.main()
