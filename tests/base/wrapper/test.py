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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for :py:class:`base.Wrapper`."""

import unittest
import os
import random
from copy import copy, deepcopy
import pickle
from multiprocessing import Manager
import numpy as np
from deap.tools import Logbook, Statistics, HallOfFame
from culebra.base import (
    Dataset,
    Fitness,
    FitnessFunction,
    Individual,
    Species,
    Wrapper,
    DEFAULT_CHECKPOINT_ENABLE,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_VERBOSITY,
    DEFAULT_INDEX)


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyIndividual(Individual):
    """Dummy subclass to test the :py:class:`~base.Individual` class."""

    def __init__(
        self,
        species,
        fitness_cls,
        val=0
    ) -> None:
        """Construct a default individual.

        :param species: The species the individual will belong to
        :type species: Any subclass of :py:class:`~base.Species`
        :param fitness: The individual's fitness class
        :type fitness: Any subclass of :py:class:`~base.Fitness`
        :param val: A value
        :type val: :py:class:`int`
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)
        self.val = val

    def crossover(self, other):
        """Cross this individual with another one."""
        return (self, other)

    def mutate(self, indpb):
        """Mutate the individual."""
        return (self,)


class MySpecies(Species):
    """Dummy subclass to test the :py:class:`~base.Species` class."""

    def check(self, ind):
        """Check an individual."""
        return True


class MyFitnessFunction(FitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0,)
        names = ("max",)

    def evaluate(self, ind, index=None, representatives=None):
        """Evaluate one individual.

        Dummy implementation of the evaluation function.

        Return the maximum of the values stored by *ind* and
        *representatives* (if provided)
        """
        max_val = ind.val
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

    def evaluate(self, ind, index=None, representatives=None):
        """Evaluate one individual.

        Dummy implementation of the evaluation function.

        Return the double of the maximum value stored by *ind* and
        *representatives* (if provided)
        """
        max_val = ind.val
        if representatives is not None:
            for other in representatives:
                if other is not None:
                    if other.val > max_val:
                        max_val = other.val

        return (max_val*2,)


class MyWrapper(Wrapper):
    """Dummy implementation of a wrapper method."""

    def best_solutions(self):
        """Get the best individuals found for each species.

        Dummy implementation.
        """
        species = MySpecies()
        individual = MyIndividual(species, MyFitnessFunction.Fitness)
        individual.fitness.values = self.fitness_function.evaluate(individual)
        population = (individual,)

        hof = HallOfFame(population)
        hof.update(population)
        return [hof]

    def _search(self):
        """Apply the search algorithm."""
        pass


class WrapperTester(unittest.TestCase):
    """Test :py:class:`base.Wrapper`."""

    def test_init(self):
        """Test the :py:meth:`~base.Wrapper.__init__` constructor."""
        # Construct a default wrapper
        invalid_fitness_functions = (None, 'a', 1)
        for func in invalid_fitness_functions:
            with self.assertRaises(TypeError):
                MyWrapper(func)

        fitness_function = MyFitnessFunction(Dataset())
        wrapper = MyWrapper(fitness_function)

        # Check the default attributes
        self.assertIsInstance(wrapper.fitness_function, FitnessFunction)
        self.assertEqual(wrapper.fitness_function, fitness_function)

        self.assertEqual(wrapper.num_evals, None)
        self.assertEqual(wrapper.logbook, None)
        self.assertEqual(wrapper.runtime, None)
        self.assertEqual(wrapper._stats, None)

        self.assertEqual(
            wrapper.checkpoint_enable, DEFAULT_CHECKPOINT_ENABLE
        )
        self.assertEqual(
            wrapper.checkpoint_freq, DEFAULT_CHECKPOINT_FREQ)
        self.assertEqual(
            wrapper.checkpoint_filename, DEFAULT_CHECKPOINT_FILENAME)
        self.assertEqual(wrapper.random_seed, None)
        self.assertEqual(wrapper.verbose, DEFAULT_VERBOSITY)
        self.assertEqual(wrapper.index, DEFAULT_INDEX)
        self.assertEqual(wrapper.container, None)
        self.assertEqual(wrapper.representatives, None)

        # Set custom params
        params = {
            "fitness_function": MyFitnessFunction(dataset),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = MyWrapper(**params)

        # Check the attributes
        self.assertTrue(wrapper.fitness_function is params["fitness_function"])

        self.assertEqual(wrapper.num_evals, None)
        self.assertEqual(wrapper.logbook, None)
        self.assertEqual(wrapper.runtime, None)
        self.assertEqual(wrapper._stats, None)

        self.assertTrue(
            wrapper.checkpoint_enable is params["checkpoint_enable"])
        self.assertTrue(wrapper.checkpoint_freq is params["checkpoint_freq"])
        self.assertTrue(
            wrapper.checkpoint_filename is params["checkpoint_filename"])
        self.assertTrue(wrapper.random_seed is params["random_seed"])
        self.assertTrue(wrapper.verbose is params["verbose"])

    def test_index(self):
        """Test the index property."""
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Check the default index
        self.assertEqual(wrapper.index, DEFAULT_INDEX)

        # Try a valid index
        valid_indices = (0, 1, 18)
        for val in valid_indices:
            wrapper.index = val
            self.assertEqual(wrapper.index, val)

        # Try an invalid index class
        invalid_index = 'a'
        with self.assertRaises(TypeError):
            wrapper.index = invalid_index

        # Try an invalid index value
        invalid_indices = (-1, -18)
        for val in invalid_indices:
            with self.assertRaises(ValueError):
                wrapper.index = val

    def test_container(self):
        """Test the container property."""
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Check the default container
        self.assertEqual(wrapper.container, None)

        # Try a valid container
        wrapper.container = wrapper
        self.assertEqual(wrapper.container, wrapper)

        # Try an invalid container
        invalid_container = 'a'
        with self.assertRaises(TypeError):
            wrapper.container = invalid_container

    def test_checkpointing_management(self):
        """Test the checkpointing management methods."""
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        wrapper.checkpoint_enable = False
        self.assertFalse(wrapper.checkpoint_enable)
        wrapper.checkpoint_enable = True
        self.assertTrue(wrapper.checkpoint_enable)

        with self.assertRaises(TypeError):
            wrapper.checkpoint_freq = 'a'

        with self.assertRaises(ValueError):
            wrapper.checkpoint_freq = 0

        wrapper.checkpoint_freq = 14
        self.assertEqual(wrapper.checkpoint_freq, 14)

        with self.assertRaises(TypeError):
            wrapper.checkpoint_filename = ['a']

        wrapper.checkpoint_filename = "my_check.gz"
        self.assertEqual(wrapper.checkpoint_filename, "my_check.gz")

    def test_random_seed(self):
        """Test :py:attr:`~base.Wrapper.random_seed`."""
        wrapper = MyWrapper(MyFitnessFunction(dataset))
        wrapper.random_seed = 18
        self.assertEqual(wrapper.random_seed, 18)

    def test_verbose(self):
        """Test :py:attr:`~base.Wrapper.verbose`."""
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        with self.assertRaises(TypeError):
            wrapper.verbose = "hello"
        wrapper.verbose = False
        self.assertFalse(wrapper.verbose)
        wrapper.verbose = True
        self.assertTrue(wrapper.verbose)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default wrapper
        wrapper1 = MyWrapper(MyFitnessFunction(dataset))

        # Set state attributes to dummy values
        wrapper1._logbook = Logbook()
        wrapper1._runtime = 10
        wrapper1._num_evals = 20
        wrapper1._representatives = 'a'
        wrapper1._search_finished = 'b'

        # Create another wrapper
        wrapper2 = MyWrapper(MyFitnessFunction(dataset))

        # Check that state attributes in wrapper1 are not None
        self.assertNotEqual(wrapper1.logbook, None)
        self.assertNotEqual(wrapper1.runtime, None)
        self.assertNotEqual(wrapper1.num_evals, None)
        self.assertNotEqual(wrapper1._representatives, None)

        # Check that state attributes in wrapper2 are None
        self.assertEqual(wrapper2.logbook, None)
        self.assertEqual(wrapper2.runtime, None)
        self.assertEqual(wrapper2.num_evals, None)
        self.assertEqual(wrapper2._representatives, None)

        # Save the state of wrapper1
        wrapper1._save_state()

        # Change the random state
        rnd_int_before = random.randint(0, 10)
        rnd_array_before = np.arange(3)

        # Load the state of wrapper1 into wrapper2
        wrapper2._load_state()

        # Check that the state attributes of wrapper2 are equal to those of
        # wrapper1
        self.assertEqual(wrapper1.logbook, wrapper2.logbook)
        self.assertEqual(wrapper1.runtime, wrapper2.runtime)
        self.assertEqual(wrapper1.num_evals, wrapper2.num_evals)
        self.assertEqual(wrapper1._representatives, wrapper2._representatives)
        self.assertEqual(
            wrapper1._search_finished, wrapper2._search_finished
        )

        # Check that the random state has been restored
        rnd_int_after = random.randint(0, 10)
        rnd_array_after = np.arange(3)
        self.assertEqual(rnd_int_before, rnd_int_after)
        self.assertTrue((rnd_array_before == rnd_array_after).all())

        # Remove the checkpoint file
        os.remove(wrapper1.checkpoint_filename)

    def test_evaluate(self):
        """Test the individual evaluation."""
        # Create the species
        species = MySpecies()

        # Create one individual
        ind1 = MyIndividual(species, MyFitnessFunction.Fitness, 1)
        ind2 = MyIndividual(species, MyFitnessFunction.Fitness, 2)
        ind3 = MyIndividual(species, MyFitnessFunction.Fitness, 3)

        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Omit the fitness function.
        # The default training funtion should be used
        wrapper.evaluate(ind1)
        self.assertEqual(
            ind1.fitness.values, (ind1.val,) * ind1.fitness.num_obj
        )
        self.assertEqual(
            ind1.fitness.names,
            MyFitnessFunction.Fitness.names
        )

        # Provide a different fitness function.
        # The default training function should be used
        wrapper.evaluate(ind1, MyOtherFitnessFunction(dataset))
        self.assertEqual(
            ind1.fitness.values, (ind1.val*2,) * ind1.fitness.num_obj
        )
        self.assertEqual(
            ind1.fitness.names,
            MyOtherFitnessFunction.Fitness.names
        )

        # Provide representatives
        wrapper.evaluate(ind2, representatives=[[ind1], [ind3]])
        self.assertEqual(
            ind2.fitness.values,
            (ind3.val,) * ind2.fitness.num_obj
        )

        wrapper.evaluate(
            ind2,
            fitness_func=MyOtherFitnessFunction(dataset),
            representatives=[[ind1], [ind3]]
        )
        self.assertEqual(
            ind2.fitness.values,
            (ind3.val * 2,) * ind2.fitness.num_obj
        )

    def test_new_state(self):
        """Test :py:meth:`~base.wrapper._new_state`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Try initialization
        wrapper._new_state()
        self.assertIsInstance(wrapper.logbook, Logbook)
        self.assertEqual(wrapper.num_evals, 0)
        self.assertEqual(wrapper.runtime, 0)
        self.assertEqual(wrapper._search_finished, False)
        self.assertEqual(wrapper._representatives, None)

    def test_reset_state(self):
        """Test :py:meth:`~base.wrapper._reset_state`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Try reset
        wrapper._reset_state()
        self.assertEqual(wrapper.logbook, None)
        self.assertEqual(wrapper.num_evals, None)
        self.assertEqual(wrapper.runtime, None)
        self.assertEqual(wrapper.representatives, None)
        self.assertEqual(wrapper._search_finished, None)

    def test_init_internals(self):
        """Test :py:meth:`~base.wrapper._init_internals`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Create the internals
        wrapper._init_internals()

        # Check it
        self.assertIsInstance(wrapper._stats, Statistics)

    def test_reset_internals(self):
        """Test :py:meth:`~base.wrapper._reset_internals`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Try reset
        wrapper._reset_internals()
        self.assertEqual(wrapper._stats, None)

    def test_reset(self):
        """Test :py:meth:`~base.wrapper.reset`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Try reset
        wrapper.reset()
        self.assertEqual(wrapper.logbook, None)
        self.assertEqual(wrapper.num_evals, None)
        self.assertEqual(wrapper.runtime, None)
        self.assertEqual(wrapper.representatives, None)
        self.assertEqual(wrapper._search_finished, None)
        self.assertEqual(wrapper._stats, None)

    def test_init_search(self):
        """Test :py:meth:`~base.wrapper._init_search`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Try initialization
        wrapper._init_search()
        self.assertEqual(wrapper.runtime, 0)
        self.assertEqual(wrapper.num_evals, 0)
        self.assertIsInstance(wrapper.logbook, Logbook)

        # Change the current generation
        wrapper._current_gen = 10

        # Save the context
        wrapper._save_state()

        # Create another wrapper
        wrapper2 = MyWrapper(MyFitnessFunction(dataset))

        # Try initialization from the other wrapper
        wrapper2._init_search()
        self.assertEqual(wrapper._current_gen, 10)

        # Remove the checkpoint file
        os.remove(wrapper.checkpoint_filename)

    def test_train(self):
        """Test :py:meth:`~base.wrapper.train`."""
        # Create the wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Try some invalid proxies. It should fail
        invalid_proxies = (1, 'a')
        for proxy in invalid_proxies:
            with self.assertRaises(TypeError):
                wrapper.train(state_proxy=proxy)

        # Try a valid proxy
        manager = Manager()
        state_proxy = manager.dict()
        wrapper.train(state_proxy=state_proxy)
        self.assertEqual(state_proxy["runtime"], 0)
        self.assertEqual(state_proxy["num_evals"], 0)
        self.assertIsInstance(state_proxy["logbook"], Logbook)
        self.assertEqual(wrapper._search_finished, True)

        # Remove the checkpoint file
        os.remove(wrapper.checkpoint_filename)

    def test_test(self):
        """Test :py:meth:`~base.Wrapper.test`."""
        # Wrapper parameters
        params = {
            "fitness_function": MyFitnessFunction(dataset),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Not a valid sequence of hofs
        hofs = None
        with self.assertRaises(TypeError):
            wrapper.test(hofs)

        # Not a valid sequence of hofs
        hofs = ["a"]
        with self.assertRaises(ValueError):
            wrapper.test(hofs)

        # Train
        wrapper.train()
        hofs = wrapper.best_solutions()

        # Not a valid fitness function
        with self.assertRaises(TypeError):
            wrapper.test(hofs, fitness_function='a')

        # Not a valid sequence of representative individuals
        with self.assertRaises(TypeError):
            wrapper.test(hofs, representatives=1)

        # Not a valid sequence of representative individuals
        with self.assertRaises(ValueError):
            wrapper.test(hofs, representatives=['a'])

        # representatives and hofs must have the same size
        # (the number of species)
        representatives = (hofs[0][0],) * (len(hofs) + 1)
        with self.assertRaises(ValueError):
            wrapper.test(hofs, representatives=representatives)

        wrapper.test(hofs, MyOtherFitnessFunction(dataset))
        # Check the test fitness values
        for hof in hofs:
            for ind in hof:
                self.assertEqual(ind.fitness.values, (ind.val * 2,))

    def test_copy(self):
        """Test the :py:meth:`~base.Wrapper.__copy__` method."""
        # Set custom params
        params = {
            "fitness_function": MyFitnessFunction(dataset),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)
        wrapper2 = copy(wrapper1)

        # Copy only copies the first level (wrapper1 != wrapperl2)
        self.assertNotEqual(id(wrapper1), id(wrapper2))

        # The objects attributes are shared
        self.assertEqual(
            id(wrapper1.fitness_function),
            id(wrapper2.fitness_function)
        )
        self.assertEqual(
            id(wrapper1.checkpoint_filename),
            id(wrapper2.checkpoint_filename)
        )

    def test_deepcopy(self):
        """Test the :py:meth:`~base.Wrapper.__deepcopy__` method."""
        # Set custom params
        params = {
            "fitness_function": MyFitnessFunction(dataset),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)
        wrapper2 = deepcopy(wrapper1)

        # Check the copy
        self._check_deepcopy(wrapper1, wrapper2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~base.Individual.__setstate__` and
        :py:meth:`~base.Individual.__reduce__` methods.
        """
        params = {
            "fitness_function": MyFitnessFunction(dataset),
            "checkpoint_enable": False,
            "checkpoint_freq": 25,
            "checkpoint_filename": "my_check.gz",
            "random_seed": 18,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)

        data = pickle.dumps(wrapper1)
        wrapper2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(wrapper1, wrapper2)

    def _check_deepcopy(self, wrapper1, wrapper2):
        """Check if *wrapper1* is a deepcopy of *wrapper2*.

        :param wrapper1: The first wrapper
        :type wrapper1: :py:class:`~base.Wrapper`
        :param wrapper2: The second wrapper
        :type wrapper2: :py:class:`~base.Wrapper`
        """
        # Copies all the levels
        self.assertNotEqual(id(wrapper1), id(wrapper2))
        self.assertNotEqual(
            id(wrapper1.fitness_function),
            id(wrapper2.fitness_function)
        )
        self.assertNotEqual(
            id(wrapper1.fitness_function.training_data),
            id(wrapper2.fitness_function.training_data)
        )

        self.assertTrue(
            (
                wrapper1.fitness_function.training_data.inputs ==
                wrapper2.fitness_function.training_data.inputs
            ).all()
        )

        self.assertTrue(
            (
                wrapper1.fitness_function.training_data.outputs ==
                wrapper2.fitness_function.training_data.outputs
            ).all()
        )


if __name__ == '__main__':
    unittest.main()
