#!/usr/bin/env python3
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

"""Unit test for :py:class:`~culebra.abc.Trainer`."""

import unittest
import os
import random
import pickle
from copy import copy, deepcopy
from multiprocessing import Manager
from functools import partial

import numpy as np
from deap.tools import Logbook, Statistics, HallOfFame

from culebra import (
    DEFAULT_MAX_NUM_ITERS,
    DEFAULT_CHECKPOINT_ENABLE,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_VERBOSITY,
    DEFAULT_INDEX
)
from culebra.abc import (
    Fitness,
    FitnessFunction,
    Solution,
    Species,
    Trainer
)


class MySolution(Solution):
    """Dummy subclass to test the :py:class:`~culebra.abc.Solution` class."""

    def __init__(
        self,
        species,
        fitness_cls,
        val=0
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: Any subclass of :py:class:`~culebra.abc.Species`
        :param fitness: The solution's fitness class
        :type fitness: Any subclass of :py:class:`~culebra.abc.Fitness`
        :param val: A value
        :type val: :py:class:`int`
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)
        self.val = val


class MySpecies(Species):
    """Dummy subclass to test the :py:class:`~culebra.abc.Species` class."""

    def check(self, _):
        """Check a solution."""
        return True


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("max",)

    def evaluate(self, sol, index=None, representatives=None):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.

        Return the maximum of the values stored by *sol* and
        *representatives* (if provided)
        """
        max_val = sol.val
        if representatives is not None:
            for other in representatives:
                if other is not None:
                    if other.val > max_val:
                        max_val = other.val

        return (max_val,)


class MyOtherFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("doublemax",)

    def evaluate(self, sol, index=None, representatives=None):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.

        Return the double of the maximum value stored by *sol* and
        *representatives* (if provided)
        """
        max_val = sol.val
        if representatives is not None:
            for other in representatives:
                if other is not None:
                    if other.val > max_val:
                        max_val = other.val

        return (max_val*2,)


class MyTrainer(Trainer):
    """Dummy implementation of a trainer method."""

    def best_solutions(self):
        """Get the best solutions found for each species.

        Dummy implementation.
        """
        species = MySpecies()
        solution = MySolution(species, MyFitnessFunction.Fitness)
        solution.fitness.values = self.fitness_function.evaluate(solution)
        population = (solution,)

        hof = HallOfFame(population)
        hof.update(population)
        return [hof]

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self._current_iter_evals = 10


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.abc.Trainer`."""

    def test_init(self):
        """Test the :py:meth:`~culebra.abc.Trainer.__init__` constructor."""
        valid_fitness_func = MyFitnessFunction()

        # Construct a default trainer
        invalid_fitness_functions = (None, 'a', 1)
        for func in invalid_fitness_functions:
            with self.assertRaises(TypeError):
                MyTrainer(func)

        trainer = MyTrainer(valid_fitness_func)

        # Check the default attributes
        self.assertIsInstance(trainer.fitness_function, FitnessFunction)
        self.assertEqual(trainer.fitness_function, valid_fitness_func)

        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer._current_iter_evals, None)

        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer._stats, None)

        self.assertEqual(
            trainer.checkpoint_enable, DEFAULT_CHECKPOINT_ENABLE
        )
        self.assertEqual(
            trainer.checkpoint_freq, DEFAULT_CHECKPOINT_FREQ)
        self.assertEqual(
            trainer.checkpoint_filename, DEFAULT_CHECKPOINT_FILENAME)
        self.assertEqual(trainer.random_seed, None)
        self.assertEqual(trainer.verbose, DEFAULT_VERBOSITY)
        self.assertEqual(trainer.index, DEFAULT_INDEX)
        self.assertEqual(trainer.container, None)
        self.assertEqual(trainer.representatives, None)

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MyTrainer(valid_fitness_func, max_num_iters=max_num_iters)

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyTrainer(valid_fitness_func, max_num_iters=max_num_iters)

        # Set custom params
        params = {
            "fitness_function": MyFitnessFunction(),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)

        # Check the attributes
        self.assertTrue(trainer.fitness_function is params["fitness_function"])

        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer._stats, None)

        self.assertTrue(
            trainer.checkpoint_enable is params["checkpoint_enable"])
        self.assertTrue(trainer.checkpoint_freq is params["checkpoint_freq"])
        self.assertTrue(
            trainer.checkpoint_filename is params["checkpoint_filename"])
        self.assertTrue(trainer.random_seed is params["random_seed"])
        self.assertTrue(trainer.verbose is params["verbose"])

    def test_index(self):
        """Test the index property."""
        trainer = MyTrainer(MyFitnessFunction())

        # Check the default index
        self.assertEqual(trainer.index, DEFAULT_INDEX)

        # Try a valid index
        valid_indices = (0, 1, 18)
        for val in valid_indices:
            trainer.index = val
            self.assertEqual(trainer.index, val)

        # Try an invalid index class
        invalid_index = 'a'
        with self.assertRaises(TypeError):
            trainer.index = invalid_index

        # Try an invalid index value
        invalid_indices = (-1, -18)
        for val in invalid_indices:
            with self.assertRaises(ValueError):
                trainer.index = val

    def test_container(self):
        """Test the container property."""
        trainer = MyTrainer(MyFitnessFunction())

        # Check the default container
        self.assertEqual(trainer.container, None)

        # Try a valid container
        trainer.container = trainer
        self.assertEqual(trainer.container, trainer)

        # Try an invalid container
        invalid_container = 'a'
        with self.assertRaises(TypeError):
            trainer.container = invalid_container

    def test_checkpointing_management(self):
        """Test the checkpointing management methods."""
        trainer = MyTrainer(MyFitnessFunction())

        trainer.checkpoint_enable = False
        self.assertFalse(trainer.checkpoint_enable)
        trainer.checkpoint_enable = True
        self.assertTrue(trainer.checkpoint_enable)

        with self.assertRaises(TypeError):
            trainer.checkpoint_freq = 'a'

        with self.assertRaises(ValueError):
            trainer.checkpoint_freq = 0

        trainer.checkpoint_freq = 14
        self.assertEqual(trainer.checkpoint_freq, 14)

        with self.assertRaises(TypeError):
            trainer.checkpoint_filename = ['a']

        trainer.checkpoint_filename = "my_check.gz"
        self.assertEqual(trainer.checkpoint_filename, "my_check.gz")

    def test_random_seed(self):
        """Test :py:attr:`~culebra.abc.Trainer.random_seed`."""
        trainer = MyTrainer(MyFitnessFunction())
        trainer.random_seed = 18
        self.assertEqual(trainer.random_seed, 18)

    def test_verbose(self):
        """Test :py:attr:`~culebra.abc.Trainer.verbose`."""
        trainer = MyTrainer(MyFitnessFunction())

        with self.assertRaises(TypeError):
            trainer.verbose = "hello"
        trainer.verbose = False
        self.assertFalse(trainer.verbose)
        trainer.verbose = True
        self.assertTrue(trainer.verbose)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default trainer
        trainer1 = MyTrainer(MyFitnessFunction())

        # Set state attributes to dummy values
        trainer1._logbook = Logbook()
        trainer1._runtime = 10
        trainer1._num_evals = 20
        trainer1._current_iter = 10
        trainer1._representatives = 'a'
        trainer1._search_finished = 'b'

        # Create another trainer
        trainer2 = MyTrainer(MyFitnessFunction())

        # Check that state attributes in trainer1 are not None
        self.assertNotEqual(trainer1.logbook, None)
        self.assertNotEqual(trainer1.runtime, None)
        self.assertNotEqual(trainer1.num_evals, None)
        self.assertNotEqual(trainer1._current_iter, None)
        self.assertNotEqual(trainer1._representatives, None)

        # Check that state attributes in trainer2 are None
        self.assertEqual(trainer2.logbook, None)
        self.assertEqual(trainer2.runtime, None)
        self.assertEqual(trainer2.num_evals, None)
        self.assertEqual(trainer2._current_iter, None)
        self.assertEqual(trainer2._representatives, None)

        # Save the state of trainer1
        trainer1._save_state()

        # Change the random state
        rnd_int_before = random.randint(0, 10)
        rnd_array_before = np.arange(3)

        # Load the state of trainer1 into trainer2
        trainer2._load_state()

        # Check that the state attributes of trainer2 are equal to those of
        # trainer1
        self.assertEqual(trainer1.logbook, trainer2.logbook)
        self.assertEqual(trainer1.runtime, trainer2.runtime)
        self.assertEqual(trainer1.num_evals, trainer2.num_evals)
        self.assertEqual(trainer1._current_iter, trainer2._current_iter)
        self.assertEqual(trainer1._representatives, trainer2._representatives)
        self.assertEqual(
            trainer1._search_finished, trainer2._search_finished
        )

        # Check that the random state has been restored
        rnd_int_after = random.randint(0, 10)
        rnd_array_after = np.arange(3)
        self.assertEqual(rnd_int_before, rnd_int_after)
        self.assertTrue((rnd_array_before == rnd_array_after).all())

        # Remove the checkpoint file
        os.remove(trainer1.checkpoint_filename)

    def test_evaluate(self):
        """Test the solution evaluation."""
        # Create the species
        species = MySpecies()

        # Create one solution
        sol1 = MySolution(species, MyFitnessFunction.Fitness, 1)
        sol2 = MySolution(species, MyFitnessFunction.Fitness, 2)
        sol3 = MySolution(species, MyFitnessFunction.Fitness, 3)

        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Omit the fitness function.
        # The default training funtion should be used
        trainer.evaluate(sol1)
        self.assertEqual(
            sol1.fitness.values, (sol1.val,) * sol1.fitness.num_obj
        )
        self.assertEqual(
            sol1.fitness.names,
            MyFitnessFunction.Fitness.names
        )

        # Provide a different fitness function.
        # The default training function should be used
        trainer.evaluate(sol1, MyOtherFitnessFunction())
        self.assertEqual(
            sol1.fitness.values, (sol1.val*2,) * sol1.fitness.num_obj
        )
        self.assertEqual(
            sol1.fitness.names,
            MyOtherFitnessFunction.Fitness.names
        )

        # Provide representatives
        trainer.evaluate(sol2, representatives=[[sol1], [sol3]])
        self.assertEqual(
            sol2.fitness.values,
            (sol3.val,) * sol2.fitness.num_obj
        )

        trainer.evaluate(
            sol2,
            fitness_func=MyOtherFitnessFunction(),
            representatives=[[sol1], [sol3]]
        )
        self.assertEqual(
            sol2.fitness.values,
            (sol3.val * 2,) * sol2.fitness.num_obj
        )

    def test_new_state(self):
        """Test :py:meth:`~culebra.abc.Trainer._new_state`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Try initialization
        trainer._new_state()
        self.assertIsInstance(trainer.logbook, Logbook)
        self.assertEqual(trainer.num_evals, 0)
        self.assertEqual(trainer.runtime, 0)
        self.assertEqual(trainer._search_finished, False)
        self.assertEqual(trainer._current_iter, 0)
        self.assertEqual(trainer._representatives, None)

    def test_reset_state(self):
        """Test :py:meth:`~culebra.abc.Trainer._reset_state`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Try reset
        trainer._reset_state()
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer.representatives, None)
        self.assertEqual(trainer._current_iter, None)
        self.assertEqual(trainer._search_finished, None)

    def test_init_internals(self):
        """Test :py:meth:`~culebra.abc.Trainer._init_internals`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Create the internals
        trainer._init_internals()

        # Check the trainer stats
        self.assertIsInstance(trainer._stats, Statistics)

        # Check the current iteration number of evaluations
        self.assertEqual(trainer._current_iter_evals, None)

        # Check the current iteration start time
        self.assertEqual(trainer._current_iter_start_time, None)

    def test_reset_internals(self):
        """Test :py:meth:`~culebra.abc.Trainer._reset_internals`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Try reset
        trainer._reset_internals()
        self.assertEqual(trainer._stats, None)

        # Check the current iteration number of evaluations
        self.assertEqual(trainer._current_iter_evals, None)

        # Check the current iteration start time
        self.assertEqual(trainer._current_iter_start_time, None)

    def test_reset(self):
        """Test :py:meth:`~culebra.abc.Trainer.reset`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Try reset
        trainer.reset()
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.representatives, None)
        self.assertEqual(trainer._search_finished, None)
        self.assertEqual(trainer._stats, None)

    def test_init_search(self):
        """Test :py:meth:`~culebra.abc.Trainer._init_search`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Try initialization
        trainer._init_search()
        self.assertEqual(trainer.runtime, 0)
        self.assertEqual(trainer.num_evals, 0)
        self.assertIsInstance(trainer.logbook, Logbook)

        # Change the current iteration
        trainer._current_iter = 10

        # Save the context
        trainer._save_state()

        # Create another trainer
        trainer2 = MyTrainer(MyFitnessFunction())

        # Try initialization from the other trainer
        trainer2._init_search()
        self.assertEqual(trainer._current_iter, 10)

        # Remove the checkpoint file
        os.remove(trainer.checkpoint_filename)

    def test_start_iteration(self):
        """Test _start_iteration`."""
        # Construct a trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Init the search process
        trainer._init_search()

        # Start an iteration
        trainer._start_iteration()

        # Check the current iteration number of evaluations
        self.assertEqual(trainer._current_iter_evals, 0)

        # Check the current iteration start time
        self.assertGreater(trainer._current_iter_start_time, 0)

    def test_finish_iteration(self):
        """Test _finish_iteration`."""
        # Construct a trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Init the search process
        trainer._init_search()

        # The runtime and num_evals should be 0
        self.assertEqual(trainer.runtime, 0)
        self.assertEqual(trainer.num_evals, 0)

        # Start an iteration
        trainer._start_iteration()

        # Perform the iteration
        trainer._do_iteration()

        # Finish the iteration
        trainer._finish_iteration()

        # The runtime and num_evals should be greater than 0
        self.assertGreater(trainer.runtime, 0)
        self.assertGreater(trainer.num_evals, 0)

        # The checkpoint file should exit
        trainer._load_state()

        # Remove the checkpoint file
        os.remove(trainer.checkpoint_filename)

        # Fix an iteration number that should not save data
        trainer._current_iter = trainer.checkpoint_freq + 1

        # Finish the iteration
        trainer._finish_iteration()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            trainer._load_state()

        # Fix an iteration number that should save data
        trainer._current_iter = trainer.checkpoint_freq

        # Disable checkpoining
        trainer.checkpoint_enable = False

        # Start an iteration
        trainer._start_iteration()

        # Finish the iteration
        trainer._finish_iteration()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            trainer._load_state()

    def test_finish_search(self):
        """Test _finish_search`."""
        # Construct a trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Init the search process
        trainer._init_search()

        # Finish the search
        trainer._finish_search()

        # The checkpoint file should exit
        trainer._load_state()

        # Remove the checkpoint file
        os.remove(trainer.checkpoint_filename)

        # Disable checkpoining
        trainer.checkpoint_enable = False

        # Finish the search
        trainer._finish_search()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            trainer._load_state()

    def test_default_termination_func(self):
        """Test the default termination criterion."""
        # Construct a trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            checkpoint_enable=False,
            max_num_iters=10
        )

        # Init the search process
        trainer._init_search()

        # Current iteration is 0, so, the default termination criterion should
        # return false
        self.assertFalse(trainer._default_termination_func())

        # Try the last iteration. Should return false
        trainer._current_iter = trainer.max_num_iters-1
        self.assertFalse(trainer._default_termination_func())

        # Try one more iteration. Should return true
        trainer._current_iter += 1
        self.assertTrue(trainer._default_termination_func())

        # Try another one more. Should return true again
        trainer._current_iter += 1
        self.assertTrue(trainer._default_termination_func())

    def test_custom_termination_criterion(self):
        """Test the custom termination criterion."""
        def my_criterion(trainer, my_max_num_iters):
            """Define a dummy termination criterion."""
            if trainer.current_iter >= my_max_num_iters:
                return True
            return False

        # Try to assign an invalid function. Should fail
        with self.assertRaises(TypeError):
            trainer = MyTrainer(
                MyFitnessFunction(),
                checkpoint_enable=False,
                custom_termination_func=1
            )

        # Construct a trainer
        my_max_num_iters = 4
        trainer = MyTrainer(
            MyFitnessFunction(),
            checkpoint_enable=False,
            custom_termination_func=(
                partial(my_criterion, my_max_num_iters=my_max_num_iters)
            )
        )

        # Init the search process
        trainer._init_search()

        # Search
        trainer._search()

        # The trainer should have executed only my_max_num_iters iterations
        self.assertEqual(trainer.current_iter, my_max_num_iters)

    def test_search(self):
        """Test :py:meth:`~culebra.abc.Trainer._search`."""
        # Construct a trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            checkpoint_enable=False
        )

        # Init the search process
        trainer._init_search()

        # Check the current iteration and the runtime
        self.assertEqual(trainer._current_iter, 0)
        self.assertEqual(trainer.runtime, 0)

        trainer._search()

        # Check the current iteration and the runtime
        self.assertEqual(trainer._current_iter, trainer.max_num_iters)
        self.assertGreater(trainer.runtime, 0)
        self.assertEqual(trainer._current_iter_evals, 10)
        self.assertEqual(trainer.num_evals, trainer.max_num_iters * 10)

    def test_train(self):
        """Test :py:meth:`~culebra.abc.Trainer.train`."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction())

        # Try some invalid proxies. It should fail
        invalid_proxies = (1, 'a')
        for proxy in invalid_proxies:
            with self.assertRaises(TypeError):
                trainer.train(state_proxy=proxy)

        # Try a valid proxy
        manager = Manager()
        state_proxy = manager.dict()
        trainer.train(state_proxy=state_proxy)
        self.assertGreater(state_proxy["runtime"], 0)
        self.assertEqual(state_proxy["num_evals"], 1000)
        self.assertIsInstance(state_proxy["logbook"], Logbook)
        self.assertEqual(trainer._search_finished, True)

        # Remove the checkpoint file
        os.remove(trainer.checkpoint_filename)

    def test_test(self):
        """Test :py:meth:`~culebra.abc.Trainer.test`."""
        # Trainer parameters
        params = {
            "fitness_function": MyFitnessFunction(),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Not a valid sequence of hofs
        hofs = None
        with self.assertRaises(TypeError):
            trainer.test(hofs)

        # Not a valid sequence of hofs
        hofs = ["a"]
        with self.assertRaises(ValueError):
            trainer.test(hofs)

        # Train
        trainer.train()
        hofs = trainer.best_solutions()

        # Not a valid fitness function
        with self.assertRaises(TypeError):
            trainer.test(hofs, fitness_function='a')

        # Not a valid sequence of representative solutions
        with self.assertRaises(TypeError):
            trainer.test(hofs, representatives=1)

        # Not a valid sequence of representative solutions
        with self.assertRaises(ValueError):
            trainer.test(hofs, representatives=['a'])

        # representatives and hofs must have the same size
        # (the number of species)
        representatives = (hofs[0][0],) * (len(hofs) + 1)
        with self.assertRaises(ValueError):
            trainer.test(hofs, representatives=representatives)

        trainer.test(hofs, MyOtherFitnessFunction())
        # Check the test fitness values
        for hof in hofs:
            for sol in hof:
                self.assertEqual(sol.fitness.values, (sol.val * 2,))

    def test_copy(self):
        """Test the :py:meth:`~culebra.abc.Trainer.__copy__` method."""
        # Set custom params
        params = {
            "fitness_function": MyFitnessFunction(),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertNotEqual(id(trainer1), id(trainer2))

        # The objects attributes are shared
        self.assertEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertEqual(
            id(trainer1.checkpoint_filename),
            id(trainer2.checkpoint_filename)
        )

    def test_deepcopy(self):
        """Test the :py:meth:`~culebra.abc.Trainer.__deepcopy__` method."""
        # Set custom params
        params = {
            "fitness_function": MyFitnessFunction(),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.abc.Trainer.__setstate__` and
        :py:meth:`~culebra.abc.Trainer.__reduce__` methods.
        """
        params = {
            "fitness_function": MyFitnessFunction(),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)

        data = pickle.dumps(trainer1)
        trainer2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.abc.Trainer`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.abc.Trainer`
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertTrue(
            trainer1.fitness_function.Fitness.weights ==
            trainer2.fitness_function.Fitness.weights
        )

        self.assertTrue(
            (
                trainer1.fitness_function.Fitness.names ==
                trainer2.fitness_function.Fitness.names
            )
        )

    def test_repr(self):
        """Test the repr and str dunder methods."""
        params = {
            "fitness_function": MyFitnessFunction(),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
