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

"""Unit test for :py:class:`wrapper.multi_pop.MultiPop`."""

import unittest
import pickle
import os
from multiprocessing.queues import Queue
from copy import copy, deepcopy
from deap.tools import Logbook
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper import DEFAULT_NUM_GENS
from culebra.wrapper.single_pop import (
    NSGA as SinglePopWrapper, DEFAULT_POP_SIZE
)
from culebra.wrapper.multi_pop import (
    MultiPop,
    DEFAULT_NUM_SUBPOPS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
)


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MyWrapper(MultiPop):
    """Dummy implementation of a wrapper method."""

    def _generate_subpop_wrappers(self):
        self._subpop_wrappers = []
        subpop_params = {
            "individual_cls": Individual,
            "species": species,
            "fitness_function": self.fitness_function,
            "num_gens": self.num_gens,
            "checkpoint_enable": self.checkpoint_enable,
            "checkpoint_freq": self.checkpoint_freq,
            "checkpoint_filename": self.checkpoint_filename,
            "verbose": self.verbose,
            "random_seed": self.random_seed
        }

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subpop_wrapper_checkpoint_filenames):
            subpop_wrapper = self.subpop_wrapper_cls(**subpop_params)
            subpop_wrapper.checkpoint_filename = checkpoint_filename

            subpop_wrapper.index = index
            subpop_wrapper.container = self
            self._subpop_wrappers.append(subpop_wrapper)


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.MultiPop`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.multi_pop.MultiPop.__init__`."""
        valid_fitness_func = Fitness(dataset)
        valid_subpop_wrapper_cls = SinglePopWrapper

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    func,
                    valid_subpop_wrapper_cls
                )

        # Try invalid subpop_wrapper_cls. Should fail
        invalid_wrapper_classes = (tuple, str, None, 'a', 1)
        for cls in invalid_wrapper_classes:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    cls
                )

        # Try invalid types for num_gens. Should fail
        invalid_num_gens = ('a', 1.5, str)
        for num_gens in invalid_num_gens:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_gens=num_gens
                )

        # Try invalid values for num_gens. Should fail
        invalid_num_gens = (-1, 0)
        for num_gens in invalid_num_gens:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_gens=num_gens
                )

        # Try invalid types for num_subpops. Should fail
        invalid_num_subpops = ('a', 1.5, str)
        for num_subpops in invalid_num_subpops:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=num_subpops
                )

        # Try invalid values for num_subpops. Should fail
        invalid_num_subpops = (-1, 0)
        for num_subpops in invalid_num_subpops:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=num_subpops
                )

        # Try invalid types for the representation size. Should fail
        invalid_representation_sizes = (str, 'a', -0.001, 1.001)
        for representation_size in invalid_representation_sizes:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_size=representation_size
                )

        # Try invalid values for the representation size. Should fail
        invalid_representation_sizes = (-1, 0)
        for representation_size in invalid_representation_sizes:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_size=representation_size
                )

        # Try invalid types for the representation frequency. Should fail
        invalid_representation_freqs = (str, 'a', 1.5)
        for representation_freq in invalid_representation_freqs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_freq=representation_freq
                )

        # Try invalid values for the representation frequency. Should fail
        invalid_representation_freqs = (-1, 0)
        for representation_freq in invalid_representation_freqs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_freq=representation_freq
                )

        # Try invalid representation topology function. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_topology_func=func
                )

        # Try invalid types for representation topology function parameters
        # Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_topology_func_params=params
                )

        # Try invalid representation selection function. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_selection_func=func
                )

        # Try invalid types for representation selection function parameters
        # Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_selection_func_params=params
                )

        # Test default params
        wrapper = MyWrapper(valid_fitness_func, valid_subpop_wrapper_cls)

        self.assertEqual(wrapper.num_gens, DEFAULT_NUM_GENS)
        self.assertEqual(wrapper.num_subpops, DEFAULT_NUM_SUBPOPS)

        self.assertEqual(
            wrapper.representation_size, DEFAULT_REPRESENTATION_SIZE
        )
        self.assertEqual(
            wrapper.representation_freq, DEFAULT_REPRESENTATION_FREQ
        )
        self.assertEqual(
            wrapper.representation_topology_func,
            DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
        )
        self.assertEqual(
            wrapper.representation_topology_func_params,
            DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
        )
        self.assertEqual(
            wrapper.representation_selection_func,
            DEFAULT_REPRESENTATION_SELECTION_FUNC
        )
        self.assertEqual(
            wrapper.representation_selection_func_params,
            DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
        )

        self.assertEqual(wrapper.subpop_wrapper_params, {})
        self.assertEqual(wrapper.subpop_wrappers, None)
        self.assertEqual(wrapper._communication_queues, None)

        # Test islands wrapper extra params
        wrapper = MyWrapper(
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            extra="foo")
        self.assertEqual(wrapper.subpop_wrapper_params, {"extra": "foo"})

    def test_subpop_suffixes(self):
        """Test :py:attr:`~wrapper.multi_pop.MultiPop._subpop_suffixes`."""
        # Parameters for the wrapper
        params = {
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": SinglePopWrapper
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Try several number of islands
        max_num_subpops = 10
        for num_subpops in range(1, max_num_subpops+1):
            wrapper.num_subpops = num_subpops

            # Check the suffixes
            for suffix in range(num_subpops):
                suffixes = wrapper._subpop_suffixes
                self.assertTrue(f"{suffix}" in suffixes)

    def test_subpop_checkpoint_filenames(self):
        """Test subpop_checkpoint_filenames."""
        # Parameters for the wrapper
        params = {
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": SinglePopWrapper,
            "num_subpops": 2,
            "checkpoint_filename": "my_check.gz"
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Check the file names
        self.assertEqual(
            tuple(wrapper.subpop_wrapper_checkpoint_filenames),
            ("my_check_0.gz", "my_check_1.gz")
        )

    def test_init_internals(self):
        """Test _init_internals."""
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)
        wrapper._init_internals()

        # Test the number of subpopulations
        self.assertEqual(len(wrapper.subpop_wrappers), num_subpops)

        # Test each island
        for subpop_wrapper in wrapper.subpop_wrappers:
            self.assertIsInstance(subpop_wrapper, subpop_wrapper_cls)
            self.assertEqual(subpop_wrapper.species, species)
            self.assertEqual(subpop_wrapper.individual_cls, Individual)
            self.assertEqual(subpop_wrapper.fitness_function, fitness_func)
            self.assertEqual(subpop_wrapper.pop_size, DEFAULT_POP_SIZE)

        # Test that the communication queues have been created
        self.assertIsInstance(wrapper._communication_queues, list)
        self.assertEqual(
            len(wrapper._communication_queues), wrapper.num_subpops
        )
        for queue in wrapper._communication_queues:
            self.assertIsInstance(queue, Queue)

        for index1 in range(wrapper.num_subpops):
            for index2 in range(index1 + 1, wrapper.num_subpops):
                self.assertNotEqual(
                    id(wrapper._communication_queues[index1]),
                    id(wrapper._communication_queues[index2])
                )

    def test_reset_internals(self):
        """Test _reset_internals."""
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)
        wrapper._init_internals()

        # Reset the internals
        wrapper._reset_internals()

        # Test the number of subpopulations
        self.assertEqual(wrapper.subpop_wrappers, None)

        # Test the communication queues
        self.assertEqual(wrapper._communication_queues, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default wrapper
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False
        }

        # Test default params
        wrapper1 = MyWrapper(**params)

        # Create the subpopulations
        wrapper1._init_search()

        # Set state attributes to dummy values
        wrapper1._runtime = 10
        wrapper1._current_gen = 19

        # Save the state of wrapper1
        wrapper1._save_state()

        # Create another wrapper
        wrapper2 = MyWrapper(**params)

        # Wrapper2 has no subpopulations yet
        self.assertEqual(wrapper2.subpop_wrappers, None)

        # Load the state of wrapper1 into wrapper2
        wrapper2._load_state()

        # Check that the state attributes of wrapper2 are equal to those of
        # wrapper1
        self.assertEqual(wrapper1.runtime, wrapper2.runtime)
        self.assertEqual(wrapper1._current_gen, wrapper2._current_gen)

        # Remove the checkpoint files
        os.remove(wrapper1.checkpoint_filename)

    def test_new_state(self):
        """Test _new_state."""
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)
        wrapper._init_internals()
        wrapper._new_state()

        # Check the current generation
        self.assertEqual(wrapper._current_gen, 0)

        # Test that the logbook is None
        self.assertEqual(wrapper._logbook, None)

    def test_logbook(self):
        """Test the logbook property."""
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        self.assertEqual(wrapper._logbook, None)

        # Generate the subpopulations
        wrapper._init_search()
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._init_search()

        # Test the logbook property
        global_logbook_len = 0
        for subpop_wrapper in wrapper.subpop_wrappers:
            global_logbook_len += len(subpop_wrapper.logbook)

        self.assertIsInstance(wrapper.logbook, Logbook)
        self.assertEqual(global_logbook_len, len(wrapper.logbook))

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
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
            id(wrapper1.subpop_wrapper_cls),
            id(wrapper2.subpop_wrapper_cls)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper1 = MyWrapper(**params)
        wrapper2 = deepcopy(wrapper1)

        # Check the copy
        self._check_deepcopy(wrapper1, wrapper2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Set custom params
        # Parameters for the wrapper
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = SinglePopWrapper
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper1 = MyWrapper(**params)

        data = pickle.dumps(wrapper1)
        wrapper2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(wrapper1, wrapper2)

    def _check_deepcopy(self, wrapper1, wrapper2):
        """Check if *wrapper1* is a deepcopy of *wrapper2*.

        :param wrapper1: The first wrapper
        :type wrapper1: :py:class:`~wrapper.multi_pop.MultiPop`
        :param wrapper2: The second wrapper
        :type wrapper2: :py:class:`~wrapper.multi_pop.MultiPop`
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
