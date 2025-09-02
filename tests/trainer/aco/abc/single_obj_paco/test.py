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

"""Unit test for :py:class:`culebra.trainer.aco.abc.SingleObjPACO`."""

import unittest

import numpy as np

from deap.tools import ParetoFront

from culebra.trainer.aco.abc import SingleObjPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength


class MyTrainer(SingleObjPACO):
    """Dummy implementation of a trainer method."""

    def _calculate_choice_info(self) -> None:
        """Calculate a dummy choice info matrix."""
        self._choice_info = self.pheromone[0] * self.heuristic[0]

    def _update_pop(self) -> None:
        """Update the population.

        The population is updated with the current iteration's colony. The best
        ants in the current colony, which are put in the *_pop_ingoing* list,
        will replace the eldest ants in the population, put in the
        *_pop_outgoing* list.

        These lists will be used later within the
        :py:meth:`~culebra.trainer.aco.abc.SingleObjPACO._increase_pheromone`
        and
        :py:meth:`~culebra.trainer.aco.abc.SingleObjPACO._decrease_pheromone`
        methods, respectively.
        """
        # Ingoing ants
        self._pop_ingoing = ParetoFront()
        self._pop_ingoing.update(self.col)

        # Outgoing ants
        self._pop_outgoing = []

        # Remaining room in the population
        remaining_room_in_pop = self.pop_size - len(self.pop)

        # For all the ants in the ingoing list
        for ant in self._pop_ingoing:
            # If there is still room in the population, just append it
            if remaining_room_in_pop > 0:
                self._pop.append(ant)
                remaining_room_in_pop -= 1

                # If the population is full, start with ants replacement
                if remaining_room_in_pop == 0:
                    self._youngest_index = 0
            # The eldest ant is replaced
            else:
                self._pop_outgoing.append(self.pop[self._youngest_index])
                self.pop[self._youngest_index] = ant
                self._youngest_index = (
                    (self._youngest_index + 1) % self.pop_size
                )


num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.aco.abc.SingleObjPACO`."""

    def test_init(self):
        """Test __init__`."""
        valid_ant_cls = Ant
        valid_species = Species(num_nodes, banned_nodes)
        valid_fitness_func = fitness_func
        valid_initial_pheromone = 1

        # Try invalid types for max_pheromone. Should fail
        invalid_max_pheromone = (type, None)
        for max_pheromone in invalid_max_pheromone:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    max_pheromone=max_pheromone
                )

        # Try invalid values for max_pheromone. Should fail
        invalid_max_pheromone = [
            (-1, ), (max, ), (0, ), (1, 2, 3), [1, 3], (3, 2), (0, 3), (3, 0)
        ]
        for max_pheromone in invalid_max_pheromone:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    max_pheromone=max_pheromone
                )

        # Try valid values for max_pheromone
        valid_max_pheromone = 3.0
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
            max_pheromone=valid_max_pheromone
        )
        self.assertEqual(trainer.max_pheromone, [valid_max_pheromone])

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    valid_max_pheromone,
                    pop_size=pop_size
                )

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_ant_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_initial_pheromone,
                    valid_max_pheromone,
                    pop_size=pop_size
                )

        # Test default params
        trainer = MyTrainer(
            valid_ant_cls,
            valid_species,
            valid_fitness_func,
            valid_initial_pheromone,
            valid_max_pheromone
        )

        self.assertEqual(
            trainer.pop_size,
            trainer.col_size
        )

    def test_init_internals(self):
        """Test _init_internals."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()

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

        # Check the internal structures
        self.assertIsInstance(trainer._pop_ingoing, list)
        self.assertIsInstance(trainer._pop_outgoing, list)
        self.assertEqual(len(trainer._pop_ingoing), 0)
        self.assertEqual(len(trainer._pop_outgoing), 0)

    def test_reset_internals(self):
        """Test _reset_internals."""
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 2
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Create new internal structures
        trainer._init_internals()

        # Reset the internal structures
        trainer._reset_internals()

        # Check the internal strucures
        self.assertEqual(trainer.pheromone, None)
        self.assertEqual(trainer._pop_ingoing, None)
        self.assertEqual(trainer._pop_outgoing, None)

    def test_increase_decrease_pheromone(self):
        """Test the _increase_pheromone and _decrease_pheromone methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = 1
        max_pheromone = 3
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone,
            "col_size": 1
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Use the same ant to increase an decrease pheromone
        ant = trainer._generate_ant()
        trainer._pop_ingoing.append(ant)
        trainer._pop_outgoing.append(ant)

        trainer._increase_pheromone()
        trainer._decrease_pheromone()

        # pheromone should not be altered
        for pher, init_pher_val in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(pher == init_pher_val))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        species = Species(num_nodes, banned_nodes)
        initial_pheromone = [2]
        max_pheromone = [3]
        params = {
            "solution_cls": Ant,
            "species": species,
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pheromone,
            "max_pheromone": max_pheromone
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
