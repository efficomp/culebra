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

"""Unit test for :py:class:`wrapper.multi_pop.Islands`."""

import pickle
from copy import copy, deepcopy
from time import sleep
import unittest
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper.single_pop import NSGA
from culebra.wrapper.multi_pop import Islands


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyWrapper(Islands):
    """Dummy implementation of a wrapper method."""

    def _generate_subpop_wrappers(self):
        self._subpop_wrappers = []
        island_params = {
            "individual_cls": self.individual_cls,
            "species": self.species,
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
            island_wrapper = self.subpop_wrapper_cls(**island_params)
            island_wrapper.checkpoint_filename = checkpoint_filename

            island_wrapper.index = index
            island_wrapper.container = self
            island_wrapper.__class__._preprocess_generation = (
                self.receive_representatives
            )
            island_wrapper.__class__._postprocess_generation = (
                self.send_representatives
            )
            self._subpop_wrappers.append(island_wrapper)


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.Islands`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.multi_pop.Islands.__init__`."""
        valid_individual = Individual
        valid_species = Species(dataset.num_feats)
        valid_fitness_func = Fitness(dataset)
        valid_subpop_wrapper_cls = NSGA

        # Try invalid individual classes. Should fail
        invalid_individual_classes = (type, None, 'a', 1)
        for individual_cls in invalid_individual_classes:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls
                )

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual,
                    species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls
                )

    def test_best_solutions(self):
        """Test best_solutions."""
        # Parameters for the wrapper
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = NSGA
        params = {
            "individual_cls": individual_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Try before the population has been created
        best_ones = wrapper.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Generate the islands
        wrapper._init_search()
        for island_wrapper in wrapper.subpop_wrappers:
            island_wrapper._init_search()
            island_wrapper._evaluate_pop(island_wrapper.pop)

        # Try again
        best_ones = wrapper.best_solutions()

        # Test that a list with only one species is returned
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        for ind in best_ones[0]:
            self.assertIsInstance(ind, individual_cls)

    def test_receive_representatives(self):
        """Test receive_representatives."""
        # Parameters for the wrapper
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = NSGA
        params = {
            "individual_cls": individual_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Generate the islands
        wrapper._init_search()
        for island_wrapper in wrapper.subpop_wrappers:
            island_wrapper._init_search()

        for index in range(wrapper.num_subpops):
            wrapper._communication_queues[index].put([index])

        # Wait for the parallel queue processing
        sleep(1)

        # Call to receive representatives, assigned to
        # island._preprocess_generation
        # at islands generation time
        for island_wrapper in wrapper.subpop_wrappers:
            island_wrapper._preprocess_generation()

        # Check the received values
        for index, island_wrapper in enumerate(wrapper.subpop_wrappers):
            self.assertEqual(island_wrapper.pop[-1], index)

    def test_send_representatives(self):
        """Test send_representatives."""
        # Parameters for the wrapper
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = NSGA
        params = {
            "individual_cls": individual_cls,
            "species": species,
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Generate the islands
        wrapper._init_search()
        for island_wrapper in wrapper.subpop_wrappers:
            island_wrapper._init_search()

        # Set a generation that should not provoke representatives sending
        for island_wrapper in wrapper.subpop_wrappers:
            island_wrapper._current_gen = wrapper.representation_freq + 1

            # Call to send representatives, assigned to
            # island._postprocess_generation at islands generation time
            island_wrapper._postprocess_generation()

        # All the queues should be empty
        for index in range(wrapper.num_subpops):
            self.assertTrue(wrapper._communication_queues[index].empty())

        # Set a generation that should provoke representatives sending
        for island_wrapper in wrapper.subpop_wrappers:
            island_wrapper._current_gen = wrapper.representation_freq

            # Call to send representatives, assigned to
            # island._postprocess_generation at islands generation time
            island_wrapper._postprocess_generation()

            # Wait for the parallel queue processing
            sleep(1)

        # All the queues shouldn't be empty
        for index in range(wrapper.num_subpops):
            self.assertFalse(wrapper._communication_queues[index].empty())
            while not wrapper._communication_queues[index].empty():
                wrapper._communication_queues[index].get()

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the wrapper
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = NSGA
        params = {
            "individual_cls": individual_cls,
            "species": species,
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
            id(wrapper1.species),
            id(wrapper2.species)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Parameters for the wrapper
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = NSGA
        params = {
            "individual_cls": individual_cls,
            "species": species,
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
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        num_subpops = 2
        subpop_wrapper_cls = NSGA
        params = {
            "individual_cls": individual_cls,
            "species": species,
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
        :type wrapper1: :py:class:`~wrapper.multi_pop.Islands`
        :param wrapper2: The second wrapper
        :type wrapper2: :py:class:`~wrapper.multi_pop.Islands`
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

        self.assertNotEqual(id(wrapper1.species), id(wrapper2.species))
        self.assertEqual(
            id(wrapper1.species.num_feats), id(wrapper2.species.num_feats)
        )


if __name__ == '__main__':
    unittest.main()
