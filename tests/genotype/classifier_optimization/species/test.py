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

"""Unit test for :py:class:`classifier_hyperparameters.Species`."""

import unittest
import pickle
from numbers import Real, Integral
from copy import copy, deepcopy
from culebra.base import Individual
from culebra.fitness_function.classifier_optimization import KappaC
from culebra.genotype.classifier_optimization import (
    Species,
    DEFAULT_HYPERPARAMETER_NAME
)

Fitness = KappaC.Fitness
"""Default fitness class."""


class MyIndividual(Individual):
    """Classifier hyperparameters individuals."""

    def __init__(self, species, fitness_cls, values):
        """Minimal constructor."""
        super().__init__(species, fitness_cls)
        self.values = values


class SpeciesTester(unittest.TestCase):
    """Test :py:class:`classifier_hyperparameters.Species`."""

    def test_init(self):
        """Test the __init__ method."""
        # Try wrong a wrong type for the lower_bounds. Should fail
        lower_bounds = 1
        with self.assertRaises(TypeError):
            Species(lower_bounds, None)

        # Try wrong a wrong type for the upper_bounds. Should fail
        lower_bounds = [1]
        upper_bounds = 2
        with self.assertRaises(TypeError):
            Species(lower_bounds, upper_bounds)

        # Try wrong a wrong type for the types. Should fail
        upper_bounds = [2]
        types = int
        with self.assertRaises(TypeError):
            Species(lower_bounds, upper_bounds, types)

        # Try wrong a wrong type for the names. Should fail
        upper_bounds = [2]
        types = [int]
        names = 4
        with self.assertRaises(TypeError):
            Species(lower_bounds, upper_bounds, types, names)

        # Try sequences of different lengths. Should fail
        names = ["name"]
        with self.assertRaises(ValueError):
            Species(lower_bounds * 2, upper_bounds, types)
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds * 2, types)
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds, types * 2)
        with self.assertRaises(ValueError):
            Species(lower_bounds * 2, upper_bounds, names=names)
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds * 2, names=names)
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds, names=names + ['hyper'])

        # Try empty sequences. Should fail
        types = [int]
        with self.assertRaises(ValueError):
            Species([], upper_bounds, types, names)
        with self.assertRaises(ValueError):
            Species(lower_bounds, [], types, names)
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds, [], names)
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds, types, [])

        # Try wrong types for types and names
        max_len = 10
        for length in range(1, max_len):
            lower_bounds = [1] * length
            upper_bounds = [2] * length
            types = [int] * length
            names = list("name" + str(i) for i in range(length))
            for i in range(0, length):
                # Wrong type
                types[i] = str
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types, names)
                types[i] = float

                # Wrong name
                names[i] = 1
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types, names)
                names[i] = "hyper" + str(i)

        # Try wrong chars
        names[0] = "."
        with self.assertRaises(ValueError):
            Species(lower_bounds, upper_bounds, types, names)
        names[0] = "hyper0"

        # Try wrong types and values for bounds, should fail
        for length in range(1, max_len):
            lower_bounds = [1] * length
            upper_bounds = [2] * length
            types = [int] * length

            for i in range(0, length):
                # Wrong type for a lower bound
                lower_bounds[i] = 'a'
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types)

                lower_bounds[i] = 1.5
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types)
                lower_bounds[i] = 1

                # Wrong type for an upper bound
                upper_bounds[i] = 'a'
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types)

                upper_bounds[i] = 2.5
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types)
                upper_bounds[i] = 2

                # upper < lower
                upper_bounds[i] = 0
                with self.assertRaises(ValueError):
                    Species(lower_bounds, upper_bounds, types)
                upper_bounds[i] = 2

        # Try repeated names, should fail
        for length in range(1, max_len):
            lower_bounds = [1] * length
            upper_bounds = [2] * length
            names = list("name" + str(i) for i in range(length))
            for i in range(length):
                for j in range(i+1, length):
                    names[j] = names[i]
                    with self.assertRaises(ValueError):
                        Species(lower_bounds, upper_bounds, names=names)
                    names[j] = "name" + str(i)

        # Test defaults types
        max_len = 105
        for length in range(1, max_len):
            species = Species([1] * length, [2] * length)

            # Check num_hyperparams
            self.assertEqual(species.num_hyperparams, length)

            # Check types
            self.assertEqual(species.types, (float,) * length)

            # Check names
            names = species.names
            index_len = len((length-1).__str__())
            for index, name in enumerate(names):
                self.assertEqual(
                    name,
                    f"{DEFAULT_HYPERPARAMETER_NAME}_"
                    f"{index:0{index_len}d}"
                )

        # Create a valid species
        species = Species(
            [1, 1, 1, 1],
            [2.0, 2.0, 2.0, 2.0],
            [int, float, Integral, Real],
            ["A", "b", "C", "d"]
        )
        self.assertEqual((1, 1.0, 1, 1.0), species.lower_bounds)
        self.assertEqual((2, 2.0, 2, 2.0), species.upper_bounds)
        self.assertEqual((int, float, int, float), species.types)
        self.assertEqual(("A", "b", "C", "d"), species.names)

    def test_is_member(self):
        """Test the is_member method."""
        # Create a species
        length = 4
        min_val = 1
        max_val = 3
        species = Species(
            [int(min_val)] * length,
            [float(max_val)] * length,
            [int, float, Integral, Real]
        )

        # Check an individual with different number of hyperparameters
        # Should return False
        ind = MyIndividual(species, Fitness, [min_val] * (length - 1))
        self.assertFalse(species.is_member(ind))
        ind.values = [min_val] * (length + 1)
        self.assertFalse(species.is_member(ind))

        # Check integer features. Should return True
        for val in range(min_val, max_val + 1):
            ind.values = [int(val)] * length
            self.assertTrue(species.is_member(ind))

        # Check float features that are also integer. Should return True
        for val in range(min_val, max_val + 1):
            ind.values = [float(val)] * length
            self.assertTrue(species.is_member(ind))

        # Try wrong types and values for the values, should return False
        species = Species(
            [int(min_val)] * length,
            [float(max_val)] * length,
            [int] * length
        )
        ind.values = [min_val] * length
        for i in range(length):
            # Wrong type for a lower bound
            ind. values[i] = 'a'
            self.assertFalse(species.is_member(ind))

            ind. values[i] = 1.5
            self.assertFalse(species.is_member(ind))

            ind. values[i] = min_val - 1
            self.assertFalse(species.is_member(ind))

            ind. values[i] = max_val + 1
            self.assertFalse(species.is_member(ind))

            ind. values[i] = min_val

        species = Species(
            [int(min_val)] * length,
            [float(max_val)] * length,
            [int] * length
        )
        ind.values = [min_val] * length
        for i in range(length):
            # Wrong type for a lower bound
            ind. values[i] = 'a'
            self.assertFalse(species.is_member(ind))

            ind. values[i] = 1.5
            self.assertFalse(species.is_member(ind))

            ind. values[i] = min_val - 1
            self.assertFalse(species.is_member(ind))

            ind. values[i] = max_val + 1
            self.assertFalse(species.is_member(ind))

            ind. values[i] = min_val

        min_val = 1.5
        max_val = 3.5
        species = Species(
            [min_val] * length,
            [max_val] * length,
            [float] * length
        )
        ind.values = [min_val] * length
        for i in range(length):
            # Wrong type for a lower bound
            ind. values[i] = 'a'
            self.assertFalse(species.is_member(ind))

            ind. values[i] = min_val - 1
            self.assertFalse(species.is_member(ind))

            ind. values[i] = max_val + 1
            self.assertFalse(species.is_member(ind))

            ind. values[i] = min_val

    def test_copy(self):
        """Test the __copy__ method."""
        species1 = Species([1], [2])
        species2 = copy(species1)

        # Copy only copies the first level (species1 != species2)
        self.assertNotEqual(id(species1), id(species2))

        # The species attributes are shared
        self.assertEqual(
            id(species1._lower_bounds),
            id(species1._lower_bounds)
        )

    def test_deepcopy(self):
        """Test :py:meth:`~classifier_hyperparameters.Species.__deepcopy__`."""
        species1 = Species([1], [2])
        species2 = deepcopy(species1)

        # Check the copy
        self._check_deepcopy(species1, species2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~classifier_hyperparameters.Species.__setstate__`
        and :py:meth:`~classifier_hyperparameters.Species.__reduce__` methods.
        """
        species1 = Species([1], [2])

        data = pickle.dumps(species1)
        species2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(species1, species2)

    def _check_deepcopy(self, species1, species2):
        """Check if *species1* is a deepcopy of *species2*.

        :param species1: The first species
        :type species1: :py:class:`~classifier_hyperparameters.Species`
        :param species2: The second species
        :type species2: :py:class:`~classifier_hyperparameters.Species`
        """
        # Copies all the levels
        self.assertNotEqual(id(species1), id(species2))
        self.assertEqual(species1._lower_bounds, species2._lower_bounds)


if __name__ == '__main__':
    unittest.main()
