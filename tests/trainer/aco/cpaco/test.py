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

"""Unit test for :class:`culebra.trainer.aco.CPACO`."""

import unittest
from random import randrange
from copy import copy, deepcopy
from os import remove

import numpy as np

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer.aco.abc import ACOTSP
from culebra.trainer.aco import CPACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import (
    PathLength,
    MultiObjectivePathLength
)

num_nodes = 25
fitness_func = MultiObjectivePathLength(
    PathLength.fromPath(np.random.permutation(num_nodes)),
    PathLength.fromPath(np.random.permutation(num_nodes))
)
banned_nodes = [0, num_nodes-1]
feasible_nodes = list(range(1, num_nodes - 1))


class CPACOTSP(CPACO, ACOTSP):
    """CPACO for TSP."""

class InvalidCPACO(CPACO):
    """Invalid CPACO subclass."""

    @property
    def pheromone_shapes(self):
        """Return the shape of the pheromone matrices."""
        return [(3, ) * 2] * self.num_pheromone_matrices

    @property
    def heuristic_shapes(self):
        """Return the shape of the heuristic matrices."""
        return [(2, ) * 2] * self.num_heuristic_matrices


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.aco.CPACO`."""

    def test_init(self):
        """Test __init__."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "heuristic": np.ones((num_nodes, num_nodes)),
            "pheromone_influence": 2,
            "heuristic_influence": 5,
            "max_num_iters": 123,
            "custom_termination_func": max,
            "col_size": 6,
            "pop_size": 5,
            "checkpoint_enable": False,
            "checkpoint_freq": 13,
            "checkpoint_filename": "my_check" + SERIALIZED_FILE_EXTENSION,
            "verbose": False,
            "random_seed": 15
        }

        # Try an invalid subclass. Should fail...
        with self.assertRaises(RuntimeError):
            InvalidCPACO(**params)

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Check the parameters
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, params["species"])
        self.assertEqual(trainer.fitness_function, params["fitness_function"])

        self.assertEqual(trainer.pheromone, None)
        self.assertEqual(
            len(trainer.initial_pheromone), trainer.num_pheromone_matrices
        )
        for pher_idx in range(trainer.num_pheromone_matrices):
            self.assertEqual(
                trainer.initial_pheromone[pher_idx],
                params["initial_pheromone"]
            )
        self.assertEqual(
            len(trainer.pheromone_influence), trainer.num_pheromone_matrices
        )
        for pher_idx in range(trainer.num_pheromone_matrices):
            self.assertEqual(
                trainer.pheromone_influence[pher_idx],
                params["pheromone_influence"]
            )
        self.assertEqual(
            len(trainer.heuristic), trainer.num_heuristic_matrices
        )
        for heur_idx in range(trainer.num_heuristic_matrices):
            self.assertTrue(
                np.all(trainer.heuristic[heur_idx] == params["heuristic"])
            )
            
            
        self.assertEqual(
            len(trainer.heuristic_influence), trainer.num_heuristic_matrices
        )
        for heur_idx in range(trainer.num_heuristic_matrices):
            self.assertEqual(
                trainer.heuristic_influence[heur_idx],
                params["heuristic_influence"]
        )
        
        self.assertEqual(trainer.max_num_iters, params["max_num_iters"])
        self.assertEqual(
            trainer.custom_termination_func, params["custom_termination_func"]
        )
        self.assertEqual(trainer.col_size, params["col_size"])
        self.assertEqual(trainer.pop_size, params["pop_size"])
        self.assertEqual(
            trainer.checkpoint_enable, params["checkpoint_enable"]
        )
        self.assertEqual(trainer.checkpoint_freq, params["checkpoint_freq"])
        self.assertEqual(
            trainer.checkpoint_filename, params["checkpoint_filename"]
        )
        self.assertEqual(trainer.verbose, params["verbose"])
        self.assertEqual(trainer.random_seed, params["random_seed"])

    def test_num_pheromone_matrices(self):
        """Test the num_pheromone_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        self.assertEqual(trainer.num_pheromone_matrices, 1)

    def test_num_heuristic_matrices(self):
        """Test the num_heuristic_matrices property."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        self.assertEqual(
            trainer.num_heuristic_matrices, fitness_func.num_obj
        )

    def test_state(self):
        """Test the get_state and _set_state methods."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)
        trainer._init_search()
        trainer._start_iteration()

        # Save the trainer's state
        state = trainer._get_state()

        # Check the state
        self.assertIsInstance(state["pop"], list)
        self.assertGreaterEqual(len(state["pop"]), trainer.pop_size)

        # Get the population and the pheromone matrices
        pop = trainer.pop
        pheromone = trainer.pheromone

        # Reset the trainer
        trainer.reset()

        # Set the new state
        trainer._set_state(state)

        # Test if the pop has been restored
        self.assertEqual(len(pop), len(trainer.pop))
        for ant1, ant2 in zip(pop, trainer.pop):
            self.assertEqual(ant1, ant2)

        # Test if the pheromone has been restored
        for pher1, pher2 in zip(pheromone, trainer.pheromone):
            self.assertTrue(np.all(pher1 == pher2))

    def test_new_state(self):
        """Test _new_state."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the population
        self.assertIsInstance(trainer.pop, list)
        self.assertGreaterEqual(len(trainer.pop), trainer.pop_size)

        # Test the pheromone
        for pher, init_pher in zip(
            trainer.pheromone, trainer.initial_pheromone
        ):
            self.assertTrue(np.all(pher >= init_pher))

    def test_internals(self):
        """Test the _init_internals and _reset_internals methods."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Create new internals
        trainer._init_internals()

        # Test if the heuristic influence correction factors are created
        self.assertEqual(
            len(trainer._heuristic_influence_correction),
            trainer.fitness_function.num_obj
        )

        # Reset the internals
        trainer._reset_internals()
        # Test if the heuristic influence correction factors are created
        self.assertEqual(
            trainer._heuristic_influence_correction,
            None
        )

    def test_calculate_choice_info(self):
        """Test _test_calculate_choice_info."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 4,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Init the internals
        trainer._init_internals()
        # trainer._start_iteration()

        # Calculate the choice_info for the next iteration
        trainer._calculate_choice_info()

        # Check that the coince info has been calculated
        self.assertEqual(
            trainer.choice_info.shape, trainer.heuristic_shapes[0]
        )

    def test_generate_ant(self):
        """Test the _generate_ant method."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 4,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Init the internals
        trainer._init_internals()

        # Get the previous heuristic influence correction factors and
        # choice info
        prev_heur_correction = trainer._heuristic_influence_correction
        prev_choice_info = trainer._choice_info

        # Generate a new ant
        ant = trainer._generate_ant()

        # Check that the ant has set its own heuristic influence correction
        # factors and choice info
        self.assertTrue(
            np.any(
                prev_heur_correction !=
                trainer._heuristic_influence_correction
            )
        )
        self.assertTrue(
            np.any(
                prev_choice_info != trainer._choice_info
            )
        )

        # Check that a correct ant has been returned
        self.assertIsInstance(ant, Ant)

    def test_update_pop(self):
        """Test _update_pop."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Try to update the population with an empty elite
        trainer._init_search()
        trainer._start_iteration()

        # Get a copy of a random ant within the population
        ant_index = randrange(trainer.pop_size)
        ant = deepcopy(trainer.pop[ant_index])

        # Swap two nodes in its path
        ant.path[0], ant.path[1] = ant.path[1], ant.path[0]

        # Update the population
        trainer._update_pop()

        # Check that ant hasn't replaced the previous ant in ant_index
        self.assertFalse(trainer.pop[ant_index] is ant)

        # Improve the ant fitness
        ant.fitness.values = (0, 0)

        # Append this ant to the current colony
        trainer._col.append(ant)

        # Update the population
        trainer._update_pop()

        # Check that ant has replaced the previous ant in ant_index
        self.assertTrue(trainer.pop[ant_index] is ant)

    def test_deposit_pheromone(self):
        """Test the _deposit_pheromone method."""
        initial_pher = 1
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": initial_pher,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Try to update the population with an empty elite
        trainer._init_search()
        trainer._start_iteration()

        # Reset the pheromone
        trainer._init_pheromone()

        # Let only the first ant deposit pheromone
        trainer._deposit_pheromone(trainer.pop[:1])

        ant = trainer.pop[0]
        org = ant.path[-1]
        for dest in ant.path:            
            self.assertEqual(
                trainer.pheromone[0][org][dest],
                initial_pher + trainer._pheromone_amount(ant)[0]
            )
            self.assertEqual(
                trainer.pheromone[0][dest][org],
                initial_pher + trainer._pheromone_amount(ant)[0]
            )
            org = dest

    def test_update_pheromone(self):
        """Test _update_pheromone."""
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)

        # Init the search
        trainer._init_search()

        # Reset the pheromone values
        trainer._init_pheromone()

        # Update pheromone according to the population
        trainer._update_pheromone()

        # Check the pheromone matrix
        self.assertTrue(
            np.all(
                trainer.pheromone[0] >= trainer.initial_pheromone[0]
            )
        )
        self.assertTrue(
            np.any(
                trainer.pheromone[0] != trainer.initial_pheromone[0]
            )
        )

    def test_copy(self):
        """Test the __copy__ method."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer1 = CPACOTSP(**params)
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
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer1 = CPACOTSP(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer1 = CPACOTSP(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = CPACOTSP.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Trainer parameters
        params = {
            "solution_cls": Ant,
            "species": Species(num_nodes, banned_nodes),
            "fitness_function": fitness_func,
            "initial_pheromone": 1,
            "pop_size": 5,
            "verbose": False
        }

        # Create the trainer
        trainer = CPACOTSP(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: ~culebra.trainer.aco.CPACO
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.aco.CPACO
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
