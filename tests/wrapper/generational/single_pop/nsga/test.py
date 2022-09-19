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
from deap.tools import selNSGA3
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper.single_pop import (
    NSGA,
    DEFAULT_POP_SIZE,
    DEFAULT_NSGA_SELECTION_FUNC,
    DEFAULT_NSGA_SELECTION_FUNC_PARAMS,
    DEFAULT_NSGA3_REFERENCE_POINTS_P
)


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.single_pop.NSGA`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.single_pop.NSGA.__init__`."""
        # Test the default parameters
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)

        # The selection function should be DEFAULT_NSGA_SELECTION_FUNC
        self.assertEqual(
            wrapper.selection_func, DEFAULT_NSGA_SELECTION_FUNC
        )
        self.assertEqual(
            wrapper.selection_func_params, DEFAULT_NSGA_SELECTION_FUNC_PARAMS
        )
        self.assertEqual(
            wrapper.nsga3_reference_points_p, DEFAULT_NSGA3_REFERENCE_POINTS_P)
        self.assertEqual(wrapper.nsga3_reference_points_scaling, None)

        # Try custom params
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "nsga3_reference_points_p": 1,
            "nsga3_reference_points_scaling": 4
        }
        wrapper = NSGA(**params)
        self.assertEqual(
            wrapper.nsga3_reference_points_p,
            params["nsga3_reference_points_p"])
        wrapper.selection_func = selNSGA3

        self.assertEqual(
            wrapper.nsga3_reference_points_scaling,
            params["nsga3_reference_points_scaling"])

    def test_pop_size(self):
        """Test :py:meth:`~wrapper.single_pop.NSGA.pop_size` getter."""
        # Try with the default pop_size for NSGA2
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)
        # wrapper.pop_size should be DEFAULT_POP_SIZE
        self.assertEqual(wrapper.pop_size, DEFAULT_POP_SIZE)

        # Try with the default pop_size for NSGA3
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "selection_func": selNSGA3
        }
        wrapper = NSGA(**params)
        # wrapper.pop_size should be DEFAULT_POP_SIZE
        self.assertEqual(wrapper.pop_size, len(wrapper.nsga3_reference_points))

        # Set a customized value
        pop_size = 200
        wrapper.pop_size = pop_size
        # wrapper.pop_size should be the customized value
        self.assertEqual(wrapper.pop_size, pop_size)

        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "pop_size": pop_size
        }
        wrapper = NSGA(**params)
        # wrapper.pop_size should be the customized value
        self.assertEqual(wrapper.pop_size, pop_size)

    def test_selection_func(self):
        """Test :py:meth:`~wrapper.single_pop.NSGA.selection_func` getter."""
        # Try with the default selection function
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)
        # wrapper.selection_func should be DEFAULT_NSGA_SELECTION_FUNC
        self.assertEqual(wrapper.selection_func, DEFAULT_NSGA_SELECTION_FUNC)

        # Try with a custom selection function
        params["selection_func"] = selNSGA3
        wrapper = NSGA(**params)
        # wrapper.selection_func should be selNSGA3
        self.assertEqual(wrapper.selection_func, selNSGA3)

    def test_selection_func_params(self):
        """Test :py:meth:`~wrapper.single_pop.NSGA.selection_func_params`."""
        # Try with the default selection function
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)
        # wrapper.selection_func should be DEFAULT_NSGA_SELECTION_FUNC
        self.assertEqual(
            wrapper.selection_func_params, DEFAULT_NSGA_SELECTION_FUNC_PARAMS
        )

        # Try with a custom selection function
        params["selection_func"] = selNSGA3
        wrapper = NSGA(**params)
        # wrapper.selection_func should be selNSGA3
        self.assertEqual(wrapper.selection_func, selNSGA3)

    def test_nsga3_reference_points_p(self):
        """Test nsga3_reference_points_p."""
        # Construct the wrapper
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)

        # Try invalid types for p
        invalid_p_values = (type, 'a', 1.4)
        for value in invalid_p_values:
            with self.assertRaises(TypeError):
                wrapper.nsga3_reference_points_p = value

        # Try invalid values for p
        invalid_p_values = (-3, 0)
        for value in invalid_p_values:
            with self.assertRaises(ValueError):
                wrapper.nsga3_reference_points_p = value

    def test_nsga3_reference_points_scaling(self):
        """Test nsga3_reference_points_scaling."""
        # Construct the wrapper
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)

        # Init the internals
        wrapper._init_internals()

        # Try invalid types for the scaling factor
        invalid_scaling_values = (type, 'a')
        for value in invalid_scaling_values:
            with self.assertRaises(TypeError):
                wrapper.nsga3_reference_points_scaling = value

    def test_init_internals(self):
        """Test _init_internals`."""
        # Construct the wrapper
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)

        # Init the internals
        wrapper._init_internals()

        # Check the current reference points
        self.assertEqual(wrapper._nsga3_ref_points, None)

    def test_reset_internals(self):
        """Test _reset_internals`."""
        # Construct the wrapper
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset)
        }
        wrapper = NSGA(**params)

        # Init the internals
        wrapper._init_internals()

        # Reset the internals
        wrapper._reset_internals()

        # Check the current reference points
        self.assertEqual(wrapper._nsga3_ref_points, None)

    def test_do_generation(self):
        """Test _do_generation."""
        params = {
            "individual_cls": Individual,
            "species": Species(dataset.num_feats),
            "fitness_function": Fitness(dataset),
            "checkpoint_enable": False,
            "verbose": False
        }
        wrapper = NSGA(**params)

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
        wrapper1 = NSGA(**params)
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
        wrapper1 = NSGA(**params)
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
        wrapper1 = NSGA(**params)

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
