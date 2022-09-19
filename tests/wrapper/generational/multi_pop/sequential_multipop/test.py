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

"""Unit test for :py:class:`wrapper.multi_pop.SequentialMultiPop`."""

import unittest
import os
from time import sleep
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper.single_pop import NSGA
from culebra.wrapper.multi_pop import SequentialMultiPop


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class MyWrapper(SequentialMultiPop):
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

            subpop_wrapper.__class__._preprocess_generation = (
                self.receive_representatives
            )
            subpop_wrapper.__class__._postprocess_generation = (
                self.send_representatives
            )
            self._subpop_wrappers.append(subpop_wrapper)

    @staticmethod
    def receive_representatives(subpop_wrapper) -> None:
        """Receive representative individuals.

        :param subpop_wrapper: The subpopulation wrapper receiving
            representatives
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        """
        # Receive all the individuals in the queue
        queue = subpop_wrapper.container._communication_queues[
            subpop_wrapper.index
        ]
        while not queue.empty():
            subpop_wrapper._pop.append(queue.get())

    @staticmethod
    def send_representatives(subpop_wrapper) -> None:
        """Send representatives.

        :param subpop_wrapper: The sender subpopulation wrapper
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        """
        container = subpop_wrapper.container
        # Check if sending should be performed
        if subpop_wrapper._current_gen % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subpop_wrapper.index,
                container.num_subpops,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                for _ in range(container.representation_size):
                    # Select one representative each time
                    (ind,) = container.representation_selection_func(
                        subpop_wrapper.pop,
                        1,
                        **container.representation_selection_func_params
                    )
                    container._communication_queues[dest].put(ind)


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.SequentialMultiPop`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.multi_pop.SequentialMultiPop.__init__`."""
        # Test default params
        wrapper = MyWrapper(
            Fitness(dataset),
            NSGA
        )

        self.assertEqual(wrapper._current_gen, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default wrapper
        fitness_func = Fitness(dataset)
        subpop_wrapper_cls = NSGA
        num_subpops = 2
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
        wrapper2._init_search()

        # Check that the state attributes of wrapper2 are equal to those of
        # wrapper1
        self.assertEqual(wrapper1.runtime, wrapper2.runtime)
        self.assertEqual(wrapper1._current_gen, wrapper2._current_gen)

        # Remove the checkpoint files
        os.remove(wrapper1.checkpoint_filename)
        for file in wrapper1.subpop_wrapper_checkpoint_filenames:
            os.remove(file)

    def test_migrations(self):
        """Test _preprocess_generation and _postprocess_generation."""
        # Create a default wrapper
        fitness_func = Fitness(dataset)
        subpop_wrapper_cls = NSGA
        num_subpops = 2
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "verbose": False
        }

        # Test default params
        wrapper = MyWrapper(**params)
        wrapper._init_search()

        # All the queues should be empty
        for queue in wrapper._communication_queues:
            self.assertTrue(queue.empty())

        # Send migrants
        wrapper._postprocess_generation()

        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should not be empty
        for queue in wrapper._communication_queues:
            self.assertFalse(queue.empty())

        # Receive migrants
        wrapper._preprocess_generation()

        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should be empty again
        for queue in wrapper._communication_queues:
            self.assertTrue(queue.empty())

    def test_search(self):
        """Test _search."""
        # Create a default wrapper
        fitness_func = Fitness(dataset)
        subpop_wrapper_cls = NSGA
        num_subpops = 2
        num_gens = 10
        params = {
            "fitness_function": fitness_func,
            "subpop_wrapper_cls": subpop_wrapper_cls,
            "num_subpops": num_subpops,
            "num_gens": num_gens,
            "checkpoint_enable": False,
            "verbose": False
        }

        # Test the search method
        wrapper = MyWrapper(**params)
        wrapper._init_search()

        self.assertEqual(wrapper._current_gen, 0)
        wrapper._search()

        self.assertEqual(wrapper._current_gen, num_gens)
        num_evals = 0
        for island_wrapper in wrapper.subpop_wrappers:
            self.assertEqual(island_wrapper._current_gen, num_gens)
            num_evals += island_wrapper.num_evals

        self.assertEqual(wrapper.num_evals, num_evals)


if __name__ == '__main__':
    unittest.main()
