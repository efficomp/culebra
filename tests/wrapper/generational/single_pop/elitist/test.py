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

"""Unit test for :py:class:`wrapper.single_pop.Elitist`."""

import unittest
from deap.tools import HallOfFame
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaIndex as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper.single_pop import Elitist, DEFAULT_ELITE_SIZE


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.single_pop.Elitist`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.single_pop.Elitist.__init__`."""
        # Wrapper parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }

        # Create the wrapper
        wrapper = Elitist(**params)

        # Test default parameter values
        self.assertEqual(wrapper.elite_size, DEFAULT_ELITE_SIZE)
        self.assertEqual(wrapper._elite, None)

    def test_elite_size(self):
        """Test :py:attr:`~wrapper.single_pop.Elitist.elite_size`."""
        # Wrapper parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }

        # Create the wrapper
        wrapper = Elitist(**params)

        # Try a valid elite proportion
        valid_size = 3
        wrapper.elite_size = valid_size
        self.assertEqual(wrapper.elite_size, valid_size)

        # Try not valid elite proportion types, should fail
        invalid_sizes = ['a', len, 1.4]
        for size in invalid_sizes:
            with self.assertRaises(TypeError):
                wrapper.elite_size = size

        # Try not valid elite proportion values, should fail
        invalid_sizes = [-1, 0]
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                wrapper.elite_size = size

    def test_state(self):
        """Test :py:attr:`~wrapper.single_pop.Elitist._state`."""
        # Wrapper parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 1000,
            "elite_size": 13
        }

        # Create the wrapper
        wrapper = Elitist(**params)

        # Save the wrapper's state
        state = wrapper._state

        # Check the state
        self.assertEqual(state["num_evals"], wrapper._num_evals)
        self.assertEqual(state["elite"], wrapper._elite)

        # Change the state
        state["num_evals"] = 100
        state["elite"] = 200

        # Set the new state
        wrapper._state = state

        # Test if the new values have been set
        self.assertEqual(state["num_evals"], wrapper._num_evals)
        self.assertEqual(state["elite"], wrapper._elite)

    def test_best_solutions(self):
        """Test :py:meth:`~wrapper.single_pop.Elitist.best_solutions`."""
        # Wrapper parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 100,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = Elitist(**params)

        # Try before the population has been created
        best_ones = wrapper.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Set state attributes to dummy values
        ind = Individual(
            params["species"],
            params["fitness_function"].Fitness
        )

        # Generate ans evaluate the initial population
        wrapper._init_search()
        best_ones = wrapper.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one individual
        self.assertEqual(len(best_ones[0]), 1)

        # Check the fitness of he best on is better the or equel to the
        # fitness fo all tie individuals
        best_one = best_ones[0][0]
        for ind in wrapper.pop:
            self.assertGreaterEqual(best_one.fitness, ind.fitness)

        # Check that best_one is in the elite
        self.assertTrue(
            (best_one.features == wrapper._elite[0].features).all()
        )

    def test_new_state(self):
        """Test :py:meth:`~wrapper.single_pop.Elitist._new_state`."""
        # Wrapper parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 100,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = Elitist(**params)

        # Create a new state
        wrapper._init_internals()
        wrapper._new_state()

        # Check the elite
        self.assertIsInstance(wrapper._elite, HallOfFame)

        # Check that best_ones contains only one species
        self.assertEqual(len(wrapper._elite), max(1, wrapper.elite_size))

    def test_reset_state(self):
        """Test :py:meth:`~wrapper.single_pop.Elitist._reset_state`."""
        # Wrapper parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": 100,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = Elitist(**params)

        # Create a new state
        wrapper._init_internals()
        wrapper._new_state()

        # Reset the state
        wrapper._reset_state()

        # Check the elite
        self.assertEqual(wrapper._elite, None)

    def test_do_generation(self):
        """Test _do_generation."""
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        wrapper = Elitist(**params)

        # Init the search process
        wrapper._init_search()

        # Do a generation
        pop_size_before = len(wrapper.pop)
        wrapper._do_generation()
        pop_size_after = len(wrapper.pop)
        self.assertEqual(pop_size_before, pop_size_after)


if __name__ == '__main__':
    unittest.main()
