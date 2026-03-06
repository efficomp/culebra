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

"""Unit test for :class:`~culebra.trainer.abc.CentralizedTrainer`."""

import unittest
import os
from copy import copy, deepcopy
from functools import partial
import random

import numpy as np
from multiprocess import Manager
from deap.tools import ParetoFront, Statistics, Logbook

from culebra import (
    DEFAULT_INDEX,
    SERIALIZED_FILE_EXTENSION
)
from culebra.abc import (
    FitnessFunction,
    Solution,
    Species
)
from culebra.trainer import (
    DEFAULT_MAX_NUM_ITERS,
    DEFAULT_CHECKPOINT_ACTIVATION,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_BASENAME,
    DEFAULT_VERBOSITY
)
from culebra.trainer.abc import (
    CentralizedTrainer,
    SequentialDistributedTrainer,
    ParallelDistributedTrainer,
    IslandsTrainer,
    CooperativeTrainer
)


class MySolution(Solution):
    """Dummy subclass to test the :class:`~culebra.abc.Solution` class."""

    def __init__(
        self,
        species,
        fitness_cls,
        val=0
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: Any subclass of :class:`~culebra.abc.Species`
        :param fitness: The solution's fitness class
        :type fitness: Any subclass of :class:`~culebra.abc.Fitness`
        :param val: A value
        :type val: int
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)
        self.val = val

class MyOtherSolution(MySolution):
    """Dummy subclass to test the :class:`~culebra.abc.Solution` class."""


class MySpecies(Species):
    """Dummy subclass to test the :class:`~culebra.abc.Species` class."""

    def is_member(self, sol):
        """Check if a solution meets the constraints imposed by the species."""
        return True

class MyOtherSpecies(MySpecies):
    """Dummy subclass to test the :class:`~culebra.abc.Species` class."""


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("max",)

    def evaluate(self, sol, index=None, cooperators=None):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.

        Obtain the maximum of the values stored by *sol* and *cooperators*
        (if provided)
        """
        max_val = sol.val
        if cooperators is not None:
            for other in cooperators:
                if other is not None:
                    max_val = max(other.val, max_val)

        sol.fitness.values = (max_val,)

        return sol.fitness


class MyOtherFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("doublemax",)

    def evaluate(self, sol, index=None, cooperators=None):
        """Evaluate one solution.

        Dummy implementation of the evaluation function.

        Obtain the double of the maximum value stored by *sol* and
        *cooperators* (if provided)
        """
        max_val = sol.val
        if cooperators is not None:
            for other in cooperators:
                if other is not None:
                    max_val = max(other.val, max_val)

        sol.fitness.values = (max_val*2,)

        return sol.fitness


class MyTrainer(CentralizedTrainer):
    """Dummy implementation of a trainer."""

    def __init__(
        self,
        fitness_func,
        solution_cls,
        species,
        custom_termination_func=None,
        max_num_iters=None,
        checkpoint_activation=None,
        checkpoint_freq=None,
        checkpoint_basename=None,
        verbosity=None,
        random_seed=None
    ):
        """Constructor."""
        super().__init__(
            fitness_func=fitness_func,
            solution_cls=solution_cls,
            species=species,
            custom_termination_func=custom_termination_func,
            max_num_iters=max_num_iters,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_basename=checkpoint_basename,
            verbosity=verbosity,
            random_seed=random_seed
        )
        self.pop = None
        self.sent = False
        self.received = False

    def _do_iteration(self):
        """Implement an iteration of the training process."""
        self.pop = [
            self.solution_cls(
                self.species, self.fitness_func.fitness_cls, 1
            ),
            self.solution_cls(
                self.species, self.fitness_func.fitness_cls, 3
            )
        ]
        for sol in self.pop:
            self._current_iter_evals += self.evaluate(
                sol,
                self.fitness_func,
                self.index,
                self.cooperators
            )

    def _get_objective_stats(self) -> dict:
        """Gather the objective stats."""
        return self._stats.compile(self.pop) if self._stats else {}

    def select_representatives(self):
        """Select the representatives."""
        if self.container:
            return self.container.representatives_selection_func(
                self.pop, self.container.num_representatives
                )

        return []

    def integrate_representatives(self, representatives):
        """Integrate the representatives."""
        self.pop.extend(representatives)

    def best_solutions(self):
        """Get the best solutions found for each species."""
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return (hof,)


class MyIslandsTrainer(SequentialDistributedTrainer, IslandsTrainer):
    """Dummy implementation of an islands-based distributed trainer."""


class MyCooperativeTrainer(SequentialDistributedTrainer, CooperativeTrainer):
    """Dummy implementation of an islands-based distributed trainer."""


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.trainer.abc.CentralizedTrainer`."""

    def test_init(self):
        """Test the constructor."""
        valid_fitness_func = MyFitnessFunction()
        valid_solution_cls = MySolution
        valid_species = MySpecies()

        # Try invalid fitness functions. Shoud fail ...
        invalid_fitness_funcs = (None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(func, valid_solution_cls, valid_species)

        # Try invalid solution classes. Shoud fail ...
        invalid_solution_classes = (None, 1)
        for solution_cls in invalid_solution_classes:
            with self.assertRaises(TypeError):
                MyTrainer(valid_fitness_func, solution_cls, valid_species)

        # Try invalid species. Shoud fail ...
        invalid_species = (None, 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyTrainer(valid_fitness_func, solution_cls, species)

        # Try an invalid custom termination function. Should fail ...
        with self.assertRaises(TypeError):
            trainer = MyTrainer(
            valid_fitness_func,
            valid_solution_cls,
            valid_species,
            custom_termination_func=1
        )

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_solution_cls,
                    valid_species,
                    max_num_iters=max_num_iters
                )

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_solution_cls,
                    valid_species,
                    max_num_iters=max_num_iters
                )

        # Check the default attributes
        trainer = MyTrainer(
            valid_fitness_func,
            valid_solution_cls,
            valid_species
        )

        self.assertEqual(trainer.custom_termination_func, None)
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(
            trainer.checkpoint_activation, DEFAULT_CHECKPOINT_ACTIVATION
        )
        self.assertEqual(trainer.checkpoint_freq, DEFAULT_CHECKPOINT_FREQ)
        self.assertEqual(
            trainer.checkpoint_basename, DEFAULT_CHECKPOINT_BASENAME)
        self.assertEqual(trainer.verbosity, DEFAULT_VERBOSITY)
        self.assertEqual(trainer.random_seed, None)
        self.assertEqual(trainer.index, DEFAULT_INDEX)
        self.assertEqual(trainer.container, None)
        self.assertEqual(trainer.state_proxy, None)
        self.assertEqual(trainer.cooperators, None)
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.num_iters, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer.training_finished, False)
        self.assertEqual(trainer._current_iter_evals, None)

        # Set custom params
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": valid_solution_cls,
            "species": valid_species,
            "custom_termination_func": max,
            "max_num_iters": 100,
            "checkpoint_activation": False,
            "checkpoint_freq": 25,
            "checkpoint_basename": "my_check",
            "verbosity": False,
            "random_seed": 18
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)

        # Check the attributes
        self.assertTrue(trainer.fitness_func is params["fitness_func"])
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, params["species"])
        self.assertEqual(
            trainer.custom_termination_func, params["custom_termination_func"]
        )
        self.assertEqual(trainer.max_num_iters, params["max_num_iters"])
        self.assertEqual(
            trainer.checkpoint_activation, params["checkpoint_activation"])
        self.assertEqual(trainer.checkpoint_freq, params["checkpoint_freq"])
        self.assertEqual(
            trainer.checkpoint_basename, params["checkpoint_basename"])
        self.assertEqual(trainer.verbosity, params["verbosity"])
        self.assertEqual(trainer.random_seed, params["random_seed"])

        self.assertEqual(trainer.index, DEFAULT_INDEX)
        self.assertEqual(trainer.container, None)
        self.assertEqual(trainer.state_proxy, None)
        self.assertEqual(trainer.cooperators, None)
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.num_iters, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer.training_finished, False)
        self.assertEqual(trainer._current_iter_evals, None)

    def test_checkpointing_management(self):
        """Test the checkpointing management methods."""
        # Construct the trainer
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
        }
        trainer = MyTrainer(**params)

        trainer.checkpoint_activation = False
        self.assertFalse(trainer.checkpoint_activation)
        trainer.checkpoint_activation = True
        self.assertTrue(trainer.checkpoint_activation)

        with self.assertRaises(TypeError):
            trainer.checkpoint_freq = 'a'

        with self.assertRaises(ValueError):
            trainer.checkpoint_freq = 0

        trainer.checkpoint_freq = 14
        self.assertEqual(trainer.checkpoint_freq, 14)

        with self.assertRaises(TypeError):
            trainer.checkpoint_basename = ['a']

        checkpoint_basename = "my_check"
        trainer.checkpoint_basename = checkpoint_basename
        self.assertEqual(trainer.checkpoint_basename, checkpoint_basename)

    def test_verbosity(self):
        """Test the verbosity property."""
        # Construct the trainer
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
        }
        trainer = MyTrainer(**params)

        with self.assertRaises(TypeError):
            trainer.verbosity = "hello"
        trainer.verbosity = False
        self.assertFalse(trainer.verbosity)
        trainer.verbosity = True
        self.assertTrue(trainer.verbosity)

    def test_random_seed(self):
        """Test the random_seed property."""
        # Construct the trainer
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
        }
        trainer = MyTrainer(**params)
        trainer.random_seed = 18
        self.assertEqual(trainer.random_seed, 18)

    def test_checkpoint_filename(self):
        """Test the checkpoint_filename property."""
        # Construct the trainer
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_basename": "my_check"
        }

        trainer = MyTrainer(**params)
        self.assertEqual(
            trainer.checkpoint_filename,
            trainer.checkpoint_basename + SERIALIZED_FILE_EXTENSION
        )

        num_subtrainers = 11
        subtrainers = tuple(
            MyTrainer(**params) for _ in range(num_subtrainers)
        )

        parallel_trainer = ParallelDistributedTrainer(*subtrainers)

        for idx, subtr in enumerate(parallel_trainer.subtrainers):
            self.assertEqual(
                subtr.checkpoint_filename,
                (
                    subtr.checkpoint_basename +
                    ("_0" if idx < 10 else "_") +
                    f"{subtr.index}" +
                    SERIALIZED_FILE_EXTENSION
                )
            )

    def test_index(self):
        """Test the index property."""
        # Construct the trainer
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_basename": "my_check"
        }

        trainer = MyTrainer(**params)

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
        num_subtrainers = 3
        subtrainer_params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies()
        }
        subtrainers = tuple(
            MyTrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        for subtrainer in subtrainers:
            # Check the default container
            self.assertEqual(subtrainer.container, None)

        container = MyIslandsTrainer(*subtrainers)
        for subtrainer in subtrainers:
            # Try a valid container
            subtrainer.container = container
            self.assertEqual(subtrainer.container, container)

            # Try an invalid container
            invalid_container = 'a'
            with self.assertRaises(TypeError):
                subtrainer.container = invalid_container

    def test_state_proxy(self):
        """Test the state_proxy property."""
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies()
        }

        trainer = MyTrainer(**params)

        # Try some invalid proxies. It should fail
        invalid_proxies = (1, 'a')
        for proxy in invalid_proxies:
            with self.assertRaises(TypeError):
                trainer.state_proxy = proxy

        # Try a valid proxy
        manager = Manager()
        proxy = manager.dict()
        trainer.state_proxy = proxy

    def test_receive_representatives_func(self):
        """Test the receive_representatives_func property."""
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Try an invalid value. Should fail ...
        with self.assertRaises(TypeError):
            trainer.receive_representatives_func = "hello"

        # Try the default value
        trainer.receive_representatives_func = None
        trainer.receive_representatives_func(trainer)

        # Try an arbitrary function
        trainer.receive_representatives_func = max
        self.assertEqual(trainer.receive_representatives_func, max)

    def test_send_representatives_func(self):
        """Test the send_representatives_func property."""
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Try an invalid value. Should fail ...
        with self.assertRaises(TypeError):
            trainer.send_representatives_func = "hello"

        # Try the default value
        trainer.send_representatives_func = None
        trainer.send_representatives_func(trainer)

        # Try an arbitrary function
        trainer.send_representatives_func = max
        self.assertEqual(trainer.send_representatives_func, max)

    def test_generate_cooperators(self):
        """Test _generate_cooperators`."""
        num_subtrainers = 3
        subtrainer_params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies()
        }
        subtrainers = tuple(
            MyTrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Test the generation when the trainer is isolated
        trainer = subtrainers[0]
        self.assertEqual(trainer._generate_cooperators(), None)

        # Test the generation when the trainer is within a cooparative
        # distributed trainer
        cooperative_trainer = MyCooperativeTrainer(*subtrainers)
        for subtr_idx, subtr in enumerate(cooperative_trainer.subtrainers):
            cooperators = subtr._generate_cooperators()
            self.assertIsInstance(cooperators, list)
            self.assertEqual(
                len(cooperators), cooperative_trainer.num_representatives
            )
            for context in cooperators:
                for sol_idx, sol in enumerate(context):
                    if sol_idx == subtr_idx:
                        self.assertEqual(sol, None)
                    else:
                        self.assertIsInstance(
                            sol, subtr.solution_cls
                        )

        # Test the generation when the trainer is within a non-cooparative
        # distributed trainer
        islands_trainer = MyIslandsTrainer(*subtrainers)
        for subtr in islands_trainer.subtrainers:
            self.assertEqual(subtr._generate_cooperators(), None)

    def test_select_representatives(self):
        """Test the select_representatives method."""
        num_subtrainers = 3
        subtrainer_params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_activation": False,
            "verbosity": False
        }

        # Test the selecction when the trainer is isolated
        trainer = MyTrainer(**subtrainer_params)
        trainer.train()
        representatives = trainer.select_representatives()
        self.assertEqual(representatives, [])

        # Try when the trainer is within a distributed trainer
        subtrainers = tuple(
            MyTrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        cooperative_trainer = MyCooperativeTrainer(*subtrainers)
        trainer = subtrainers[0]
        cooperative_trainer._init_training()
        trainer.train()
        representatives = trainer.select_representatives()
        self.assertEqual(
            len(representatives),
            cooperative_trainer.num_representatives
        )
        for sol in representatives:
            self.assertTrue(sol in trainer.pop)

    def test_integrate_representatives(self):
        """Test the integrate_representatives method."""
        # Create the trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            MySolution,
            MySpecies(),
            checkpoint_activation=False,
            verbosity=False
        )
        trainer.train()

        # Generate a representative list
        representatives = [
            trainer.solution_cls(
                trainer.species, trainer.fitness_func.fitness_cls, 2
            ),
            trainer.solution_cls(
                trainer.species, trainer.fitness_func.fitness_cls, 4
            )
        ]
        for sol in representatives:
            trainer.evaluate(
                sol,
                trainer.fitness_func,
                trainer.index,
                trainer.cooperators
            )

        # Integrate the representatives
        trainer.integrate_representatives(representatives)

        # Check the representatives list
        for sol in representatives:
            self.assertTrue(sol in trainer.pop)

    def test_init_internals(self):
        """Test the _init_internals method."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Create the internals
        trainer._init_internals()

        # Check the trainer stats
        self.assertIsInstance(trainer._stats, Statistics)

        # Check the current iteration number of evaluations
        self.assertEqual(trainer._current_iter_evals, None)

        # Check the current iteration start time
        self.assertEqual(trainer._current_iter_start_time, None)

    def test_reset_internals(self):
        """Test the _reset_internals method."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Reset the internals
        trainer._reset_internals()

        # Check the trainer stats
        self.assertEqual(trainer._stats, None)

        # Check the current iteration number of evaluations
        self.assertEqual(trainer._current_iter_evals, None)

        # Check the current iteration start time
        self.assertEqual(trainer._current_iter_start_time, None)

    def test_new_state(self):
        """Test the _new_state method."""
        # Create the trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            MySolution,
            MySpecies()
        )

        # Try initialization
        trainer._new_state()
        self.assertFalse(trainer.training_finished)
        self.assertEqual(trainer.cooperators, None)
        self.assertIsInstance(trainer.logbook, Logbook)
        self.assertEqual(trainer.num_evals, 0)
        self.assertEqual(trainer.runtime, 0)
        self.assertEqual(trainer.current_iter, 0)
        self.assertEqual(trainer.num_iters, 0)

        # Check the logbook header
        self.assertFalse('SubTr' in trainer.logbook.header)

        # Test when the trainder is within a cooperative container
        num_subtrainers = 3
        subtrainer_params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies()
        }
        subtrainers = tuple(
            MyTrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer = MyCooperativeTrainer(*subtrainers)
        trainer._init_training()
        self.assertTrue('SubTr' in trainer.logbook.header)
        for subtr in trainer.subtrainers:
            self.assertIsInstance(subtr.cooperators, list)

        # Test when the trainer is within a non-cooperative container
        trainer = MyIslandsTrainer(*subtrainers)
        trainer._init_training()
        self.assertTrue('SubTr' in trainer.logbook.header)
        for subtr in trainer.subtrainers:
            self.assertEqual(subtr.cooperators, None)

    def test_reset_state(self):
        """Test the _reset_state method."""
        # Create the trainer
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Try reset
        trainer._reset_state()
        self.assertFalse(trainer.training_finished)
        self.assertEqual(trainer.cooperators, None)
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.num_iters, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        num_subtrainers = 3
        subtrainer_params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies()
        }
        subtrainers = tuple(
            MyTrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        container = MyCooperativeTrainer(*subtrainers)

        # Create a default trainer
        trainer = container.subtrainers[0]

        # Set state attributes to dummy values
        container._init_training()
        trainer._init_training()
        trainer._training_finished = True
        trainer._num_evals = 45
        trainer._runtime = 101
        trainer._current_iter = 13

        # Save the state of trainer1
        trainer._save_state()
        state = trainer._get_state()

        # Change the random state
        rnd_int_before = random.randint(0, 10)
        rnd_array_before = np.arange(3)

        # REset the trainer
        trainer.reset()
        self.assertFalse(trainer.training_finished)
        self.assertEqual(trainer.cooperators, None)
        self.assertEqual(trainer.logbook, None)
        self.assertEqual(trainer.num_evals, None)
        self.assertEqual(trainer.runtime, None)
        self.assertEqual(trainer.current_iter, None)
        self.assertEqual(trainer.num_iters, None)

        # Load the state of trainer
        trainer._load_state()

        # Check the state
        self.assertEqual(trainer.training_finished, state["training_finished"])
        self.assertEqual(trainer.cooperators, state["cooperators"])
        self.assertEqual(trainer.logbook, state["logbook"])
        self.assertEqual(trainer.num_evals, state["num_evals"])
        self.assertEqual(trainer.runtime, state["runtime"])
        self.assertEqual(trainer.current_iter, state["current_iter"])

        # Check that the random state has been restored
        rnd_int_after = random.randint(0, 10)
        rnd_array_after = np.arange(3)
        self.assertEqual(rnd_int_before, rnd_int_after)
        self.assertTrue((rnd_array_before == rnd_array_after).all())

        # Remove the checkpoint file
        os.remove(trainer.checkpoint_filename)

    def test_init_training(self):
        """Test the _init_training method."""
        # Create the trainer
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies()
        }
        trainer = MyTrainer(**params)

        # Check internals and state before
        self.assertIsNone(trainer._stats)
        self.assertIsNone(trainer.logbook)

        # Try initialization
        trainer._init_training()

        # Check internals and state after
        self.assertIsNotNone(trainer._stats)
        self.assertIsNotNone(trainer.logbook)

    def test_start_iteration(self):
        """Test the _start_iteration method."""
        # Construct a trainer
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Init the training process
        trainer._init_training()

        # Start an iteration
        trainer._start_iteration()

        # Check the current iteration number of evaluations
        self.assertEqual(trainer._current_iter_evals, 0)

        # Check the current iteration start time
        self.assertGreater(trainer._current_iter_start_time, 0)

    def test_default_termination_func(self):
        """Test the default termination criterion."""
        # Construct a trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            MySolution,
            MySpecies(),
            max_num_iters=10,
            checkpoint_activation=False
        )

        # Init the training process
        trainer._init_training()

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

        # Construct a trainer
        my_max_num_iters = 4
        trainer = MyTrainer(
            MyFitnessFunction(),
            MySolution,
            MySpecies(),
            custom_termination_func=(
                partial(my_criterion, my_max_num_iters=my_max_num_iters)
            ),
            checkpoint_activation=False,
            verbosity=False
        )

        # Init the training process
        trainer._init_training()

        # Training
        trainer._do_training()

        # The trainer should have executed only my_max_num_iters iterations
        self.assertEqual(trainer.current_iter, my_max_num_iters)

    def test_finish_iteration(self):
        """Test the _finish_iteration method."""
        # Construct a trainer
        trainer = MyTrainer(MyFitnessFunction(), MySolution, MySpecies())

        # Init the training process
        trainer._init_training()

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
        trainer.checkpoint_activation = False

        # Start an iteration
        trainer._start_iteration()

        # Finish the iteration
        trainer._finish_iteration()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            trainer._load_state()

    def test_do_training(self):
        """Test the _do_training method."""
        def my_receive_representatives_func(trainer):
            """Define a dummy receive representatives function."""
            trainer.received = True

        def my_send_representatives_func(trainer):
            """Define a dummy send representatives function."""
            trainer.sent = True

        # Construct a trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            MySolution,
            MySpecies(),
            checkpoint_activation=False,
            verbosity=False
        )

        trainer.receive_representatives_func = my_receive_representatives_func
        trainer.send_representatives_func = my_send_representatives_func

        # Init the training process
        trainer._init_training()

        # Check the current iteration and the runtime
        self.assertEqual(trainer._current_iter, 0)
        self.assertEqual(trainer.runtime, 0)

        trainer._do_training()

        # Check the current iteration and the runtime
        self.assertEqual(trainer._current_iter, trainer.max_num_iters)
        self.assertGreater(trainer.runtime, 0)
        self.assertEqual(trainer._current_iter_evals, 2)
        self.assertEqual(trainer.num_evals, trainer.max_num_iters * 2)
        self.assertTrue(trainer.received)
        self.assertTrue(trainer.sent)

    def test_finish_training(self):
        """Test _finish_training`."""
        # Construct a trainer
        trainer = MyTrainer(
            MyFitnessFunction(),
            MySolution,
            MySpecies(),
            verbosity=False
        )

        # Init the training process
        trainer._init_training()
        trainer._do_training()

        # Finish the training
        self.assertFalse(trainer.training_finished)
        trainer._finish_training()
        self.assertTrue(trainer.training_finished)

        # The checkpoint file should exit
        trainer._load_state()

        # Remove the checkpoint file
        os.remove(trainer.checkpoint_filename)

        # Disable checkpoining
        trainer.checkpoint_activation = False

        # Finish the training
        trainer._finish_training()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            trainer._load_state()

        # Check the state proxy
        manager = Manager()
        trainer.state_proxy = manager.dict()

        # Check the state before and after finishing the training
        self.assertNotIn("runtime", trainer.state_proxy)
        trainer._finish_training()
        self.assertIn("runtime", trainer.state_proxy)
        self.assertEqual(trainer.state_proxy["runtime"], trainer.runtime)

    def test_copy(self):
        """Test the __copy__ method."""
        # Set custom params
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_activation": False,
            "checkpoint_freq": 25,
            "verbosity": False,
            "random_seed": 18
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertFalse(trainer1 is trainer2)

        # The objects attributes are shared
        self.assertTrue(trainer1.fitness_func is trainer2.fitness_func)
        self.assertTrue(trainer1.species is trainer2.species)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Set custom params
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_activation": False,
            "checkpoint_freq": 25,
            "verbosity": False,
            "random_seed": 18
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the :meth:`~culebra.trainer.abc.CentralizedTrainer.__setstate__`
        and :meth:`~culebra.trainer.abc.CentralizedTrainer.__reduce__` methods,
        :meth:`~culebra.trainer.abc.CentralizedTrainer.dump` and
        :meth:`~culebra.trainer.abc.CentralizedTrainer.load` methods.
        """
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_activation": False,
            "checkpoint_freq": 25,
            "verbosity": False,
            "random_seed": 18
        }

        # Construct a parameterized trainer
        trainer1 = MyTrainer(**params)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = MyTrainer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        os.remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        params = {
            "fitness_func": MyFitnessFunction(),
            "solution_cls": MySolution,
            "species": MySpecies(),
            "checkpoint_activation": False,
            "checkpoint_freq": 25,
            "verbosity": False,
            "random_seed": 18
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)
        trainer._init_training()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: ~culebra.trainer.abc.CentralizedTrainer
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.abc.CentralizedTrainer
        """
        # Copies all the levels
        self.assertFalse(trainer1 is trainer2)
        self.assertFalse(trainer1.fitness_func is trainer2.fitness_func)
        self.assertTrue(
            trainer1.fitness_func.obj_weights ==
            trainer2.fitness_func.obj_weights
        )

        self.assertTrue(
            (
                trainer1.fitness_func.obj_names ==
                trainer2.fitness_func.obj_names
            )
        )


if __name__ == '__main__':
    unittest.main()
