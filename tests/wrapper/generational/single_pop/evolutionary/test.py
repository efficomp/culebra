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

"""Unit test for :py:class:`wrapper.single_pop.Evolutionary`."""

import unittest
import pickle
from copy import copy, deepcopy
from deap.base import Toolbox
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaIndex as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper.single_pop import Evolutionary


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.single_pop.Evolutionary`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.single_pop.Evolutionary.__init__`."""
        # Test the superclass initialization
        valid_individual_cls = Individual
        valid_species = Species(dataset.num_feats)
        valid_fitness_func = Fitness(dataset)

        # Try invalid individual classes. Should fail
        invalid_individual_classes = (type, None, 'a', 1)
        for individual_cls in invalid_individual_classes:
            with self.assertRaises(TypeError):
                Evolutionary(individual_cls, valid_species, valid_fitness_func)

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                Evolutionary(valid_individual_cls, species, valid_fitness_func)

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                Evolutionary(valid_individual_cls, valid_species, func)

        # Test initialization
        params = {
            "individual_cls": valid_individual_cls,
            "species": valid_species,
            "fitness_function": valid_fitness_func
        }
        wrapper = Evolutionary(**params)
        self.assertEqual(wrapper._toolbox, None)

    def test_init_internals(self):
        """Test _init_internals."""
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = Evolutionary(**params)

        # Init the internals
        wrapper._init_internals()
        self.assertIsInstance(wrapper._toolbox, Toolbox)
        self.assertEqual(wrapper._toolbox.mate.func, wrapper.crossover_func)
        self.assertEqual(wrapper._toolbox.mutate.func, wrapper.mutation_func)
        self.assertEqual(wrapper._toolbox.select.func, wrapper.selection_func)

    def test_reset_internals(self):
        """Test _reset_internals."""
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = Evolutionary(**params)
        # Init the internals
        wrapper._init_internals()

        # REset the internals
        wrapper._reset_internals()
        self.assertEqual(wrapper._toolbox, None)

    def test_do_generation(self):
        """Test _do_generation."""
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        wrapper = Evolutionary(**params)

        # Init the search process
        wrapper._init_search()

        # Do a generation
        pop_size_before = len(wrapper.pop)
        wrapper._do_generation()
        pop_size_after = len(wrapper.pop)
        self.assertEqual(pop_size_before, pop_size_after)

    def test_copy(self):
        """Test the __copy__ method."""
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = Evolutionary(**params)
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
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = Evolutionary(**params)
        wrapper2 = deepcopy(wrapper1)

        # Check the copy
        self._check_deepcopy(wrapper1, wrapper2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = Evolutionary(**params)

        data = pickle.dumps(wrapper1)
        wrapper2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(wrapper1, wrapper2)

    def _check_deepcopy(self, wrapper1, wrapper2):
        """Check if *wrapper1* is a deepcopy of *wrapper2*.

        :param wrapper1: The first wrapper
        :type wrapper1: :py:class:`~wrapper.single_pop.SinglePop`
        :param wrapper2: The second wrapper
        :type wrapper2: :py:class:`~wrapper.single_pop.SinglePop`
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
