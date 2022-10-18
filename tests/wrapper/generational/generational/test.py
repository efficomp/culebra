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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for :py:class:`wrapper.Generational`."""

import os
import unittest
from culebra.base import (
    Dataset,
    Fitness,
    FitnessFunction,
    Individual,
    Species)
from culebra.wrapper import (
    Generational,
    DEFAULT_NUM_GENS
)


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyIndividual(Individual):
    """Dummy subclass to test the :py:class:`~base.Individual` class."""

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

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")

    def evaluate(self, ind, index=None, representatives=None):
        """Evaluate one individual.

        Dummy implementation of the evaluation function.
        """
        return (ind.fitness.num_obj, ) * ind.fitness.num_obj


class MyWrapper(Generational):
    """Dummy implementation of a wrapper method."""

    def _do_generation(self):
        """Implement a generation of the search process."""
        self._current_gen_evals = 10


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.generational.Generational`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.Generational.__init__`."""
        valid_fitness_func = MyFitnessFunction(dataset)

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(func)

        # Try invalid types for num_gens. Should fail
        invalid_num_gens = (type, 'a', 1.5)
        for num_gens in invalid_num_gens:
            with self.assertRaises(TypeError):
                MyWrapper(valid_fitness_func, num_gens=num_gens)

        # Try invalid values for num_gens. Should fail
        invalid_num_gens = (-1, 0)
        for num_gens in invalid_num_gens:
            with self.assertRaises(ValueError):
                MyWrapper(valid_fitness_func, num_gens=num_gens)

        # Test default params
        wrapper = MyWrapper(valid_fitness_func)
        self.assertEqual(wrapper.num_gens, DEFAULT_NUM_GENS)
        self.assertEqual(wrapper._current_gen, None)
        self.assertEqual(wrapper._current_gen_evals, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(MyFitnessFunction(dataset))

        # Set state attributes to dummy values
        wrapper1._current_gen = 10

        # Create another wrapper
        wrapper2 = MyWrapper(MyFitnessFunction(dataset))

        # Check that state attributes in wrapper1 are not default values
        self.assertNotEqual(wrapper1._current_gen, None)

        # Check that state attributes in wrapper2 are defaults
        self.assertEqual(wrapper2._current_gen, None)

        # Save the state of wrapper1
        wrapper1._save_state()

        # Load the state of wrapper1 into wrapper2
        wrapper2._load_state()

        # Check that the state attributes of wrapper2 are equal to those of
        # wrapper1
        self.assertEqual(wrapper1._current_gen, wrapper2._current_gen)

        # Remove the checkpoint file
        os.remove(wrapper1.checkpoint_filename)

    def test_new_state(self):
        """Test :py:meth:`~wrapper.generational.Generational._new_state`.

        Also test
        :py:meth:`~wrapper.generational.Generational._generate_initial_pop`.
        """
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Init the state
        wrapper._new_state()

        # Check the current generation
        self.assertEqual(wrapper._current_gen, 0)

    def test_reset_state(self):
        """Test _reset_state."""
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Reset the state
        wrapper._reset_state()

        # Check the current generation
        self.assertEqual(wrapper._current_gen, None)

    def test_init_internals(self):
        """Test _init_internals`."""
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Init the internals
        wrapper._init_internals()

        # Check the current generation number of evaluations
        self.assertEqual(wrapper._current_gen_evals, None)

        # Check the current generation start time
        self.assertEqual(wrapper._current_gen_start_time, None)

    def test_reset_internals(self):
        """Test _reset_internals`."""
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Reset the internals
        wrapper._reset_internals()

        # Check the current generation number of evaluations
        self.assertEqual(wrapper._current_gen_evals, None)

        # Check the current generation start time
        self.assertEqual(wrapper._current_gen_start_time, None)

    def test_start_generation(self):
        """Test _start_generation`."""
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Init the search process
        wrapper._init_search()

        # Start a generation
        wrapper._start_generation()

        # Check the current generation number of evaluations
        self.assertEqual(wrapper._current_gen_evals, 0)

        # Check the current generation start time
        self.assertGreater(wrapper._current_gen_start_time, 0)

    def test_finish_generation(self):
        """Test _finish_generation`."""
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Init the search process
        wrapper._init_search()

        # The runtime and num_evals should be 0
        self.assertEqual(wrapper.runtime, 0)
        self.assertEqual(wrapper.num_evals, 0)

        # Start a generation
        wrapper._start_generation()

        # Perform the generation
        wrapper._do_generation()

        # Finish the generation
        wrapper._finish_generation()

        # The runtime and num_evals should be greater than 0
        self.assertGreater(wrapper.runtime, 0)
        self.assertGreater(wrapper.num_evals, 0)

        # The checkpoint file should exit
        wrapper._load_state()

        # Remove the checkpoint file
        os.remove(wrapper.checkpoint_filename)

        # Fix a generation number that should not save data
        wrapper._current_gen = wrapper.checkpoint_freq + 1

        # Finish the generation
        wrapper._finish_generation()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            wrapper._load_state()

        # Fix a generation number that should save data
        wrapper._current_gen = wrapper.checkpoint_freq

        # Disable checkpoining
        wrapper.checkpoint_enable = False

        # Start a generation
        wrapper._start_generation()

        # Finish the generation
        wrapper._finish_generation()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            wrapper._load_state()

    def test_finish_search(self):
        """Test _finish_generation`."""
        # Construct a wrapper
        wrapper = MyWrapper(MyFitnessFunction(dataset))

        # Init the search process
        wrapper._init_search()

        # Finish the search
        wrapper._finish_search()

        # The checkpoint file should exit
        wrapper._load_state()

        # Remove the checkpoint file
        os.remove(wrapper.checkpoint_filename)

        # Disable checkpoining
        wrapper.checkpoint_enable = False

        # Finish the search
        wrapper._finish_search()

        # The checkpoint file should not exit
        with self.assertRaises(FileNotFoundError):
            wrapper._load_state()

    def test_search(self):
        """Test :py:meth:`~wrapper.generational.Generational._search`.

        Also test
        :py:meth:`~wrapper.generational.Generational._generate_initial_pop`.
        """
        # Construct a wrapper
        wrapper = MyWrapper(
            MyFitnessFunction(dataset),
            checkpoint_enable=False
        )

        # Init the state
        wrapper._init_search()

        # Check the current generation and the runtime
        self.assertEqual(wrapper._current_gen, 0)
        self.assertEqual(wrapper.runtime, 0)

        wrapper._search()

        # Check the current generation and the runtime
        self.assertEqual(wrapper._current_gen, wrapper.num_gens)
        self.assertGreater(wrapper.runtime, 0)
        self.assertEqual(wrapper._current_gen_evals, 10)
        self.assertEqual(wrapper.num_evals, wrapper.num_gens * 10)


if __name__ == '__main__':
    unittest.main()
