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

"""Unit test for the parameter optimization solutions."""

import unittest
import pickle
import copy
from numbers import Integral, Real
from itertools import repeat

from culebra.abc import Species as BaseSpecies, Fitness
from culebra.solution.parameter_optimization import Solution, Species

DEFAULT_MAX_LENGTH = 50
"""Default maximum number of parameters used to define the Species."""

DEFAULT_MIN_BOUND = 0
"""Default minimum bound for all the parameters."""

DEFAULT_MAX_BOUND = 5
"""Default maximum bound for all the parameters."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an implementation is run."""


class MyFitness(Fitness):
    """Dummy fitness."""

    weights = (1.0, 1.0)
    names = ("obj1", "obj2")
    thresholds = (0.001, 0.001)


class SolutionTester(unittest.TestCase):
    """Tester for the parameter optimization solutions.

    Test extensively the generation and breeding operators.
    """

    max_length = DEFAULT_MAX_LENGTH
    """Default maximum number of parameters used to define the Species."""

    min_bound = DEFAULT_MIN_BOUND
    """Default minimum bound for all the parameters."""

    max_bound = DEFAULT_MAX_BOUND
    """Default maximum bound for all the parameters."""

    times = DEFAULT_TIMES
    """Times each function is executed."""

    def setUp(self):
        """Check that all the parameters are alright.

        :raises TypeError: If any of the parameters is not of the appropriate
            type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        if not isinstance(self.max_length, Integral):
            raise TypeError(
                f"max_length must be an integer value: {self.max_length}")

        if self.max_length <= 0:
            raise ValueError(
                f"max_length must greater than 0: {self.max_length}")

        if not isinstance(self.min_bound, Real):
            raise TypeError(
                f"min_bound must be a real value: {self.min_bound}")

        if not isinstance(self.max_bound, Real):
            raise TypeError(
                f"max_bound must be a real value: {self.max_bound}")

        if self.min_bound >= self.max_bound:
            raise ValueError(
                f"max_bound ({self.max_bound}) must greater than "
                f"min_bound ({self.min_bound})")

        if not isinstance(self.times, Integral):
            raise TypeError(
                f"times must be an integer value: {self.times}")

        if self.times <= 0:
            raise ValueError(
                f"times must greater than 0: {self.times}")

    def test_0_constructor(self):
        """Test the behavior of a parameters optimization constructor.

        The constructor is executed under different combinations of values for
        the number of features, minimum feature value, maximum feature value,
        minimum size and maximum size.
        """
        print('Testing the constructor ...', end=' ')

        # Check the type of arguments
        with self.assertRaises(TypeError):
            Solution(BaseSpecies(), MyFitness)
        with self.assertRaises(TypeError):
            Solution(Species([1], [2]), Species)

        # For each length until max_length ...
        for length in range(1, self.max_length + 1):
            species_i = Species(
                [self.min_bound] * length,
                [self.max_bound] * length,
                [Integral] * length
            )
            species_f = Species(
                [self.min_bound] * length,
                [self.max_bound] * length,
                [Real] * length
            )
            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                # Check that the feature selector meets the species
                # constraints
                self.assertTrue(
                    species_i.is_member(Solution(species_i, MyFitness))
                )
                self.assertTrue(
                    species_f.is_member(Solution(species_f, MyFitness))
                )

        print('Ok')

    def test_1_values(self):
        """Test the values property."""
        print('Testing the values property ...', end=' ')

        # Test the named tuples
        species = Species([0] * 2, [9] * 2, [int, float], ['a', 'b'])
        values = [1, 5]
        sol = Solution(species, MyFitness, values=values)
        self.assertEqual(sol.values.a, values[0])
        self.assertEqual(sol.values.b, values[1])
        new_values = sol.named_values_cls(b=3, a=2)
        sol.values = new_values
        self.assertEqual(sol.values.a, new_values.a)
        self.assertEqual(sol.values.b, new_values.b)

        # For each length until max_length ...
        for length in range(1, self.max_length + 1):
            species_i = Species(
                [self.min_bound] * length,
                [self.max_bound] * length,
                [Integral] * length
            )
            species_f = Species(
                [self.min_bound] * length,
                [self.max_bound] * length,
                [Real] * length
            )
            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                sol1_i = Solution(species_i, MyFitness)
                sol2_i = Solution(species_i, MyFitness)
                sol2_i.values = sol1_i.values
                self.assertTrue(sol1_i.values, sol2_i.values)
                self.assertTrue(sol1_i._values, sol2_i._values)

                sol1_f = Solution(species_f, MyFitness)
                sol2_f = Solution(species_f, MyFitness)
                sol2_f.values = sol1_f.values
                self.assertTrue(sol1_f.values, sol2_f.values)
                self.assertTrue(sol1_f._values, sol2_f._values)

        print('Ok')

    def test_2_get(self):
        """Test the get method."""
        print('Testing the get method ...', end=' ')

        species = Species([0] * 2, [9] * 2, [int, float], ['a', 'b'])
        values = [1, 5]
        sol = Solution(species, MyFitness, values=values)

        # Try to get an invalid parameter name. It should fail
        with self.assertRaises(ValueError):
            sol.get('aa')

        # Try to get an valid parameter name
        sol.get('a')

        print('Ok')

    def test_3_serialization(self):
        """Serialization test."""
        print('Testing serialization ...', end=' ')
        # For each length until max_length ...
        for length in range(1, self.max_length + 1):
            species = Species(
                [self.min_bound] * length,
                [self.max_bound] * length
            )

            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                sol1 = Solution(species, MyFitness)
                data = pickle.dumps(sol1)
                sol2 = pickle.loads(data)
                self.assertTrue(sol1.values, sol2.values)

        print('Ok')

    def test_4_copy(self):
        """Copy test."""
        print('Testing copy and deepcopy ...', end=' ')
        # For each length until max_length ...
        for length in range(1, self.max_length + 1):
            species = Species(
                [self.min_bound] * length,
                [self.max_bound] * length
            )

            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                sol1 = Solution(species, MyFitness)
                sol2 = copy.copy(sol1)
                sol3 = copy.deepcopy(sol1)
                self.assertTrue(sol1.values, sol2.values)
                self.assertTrue(sol1.values, sol3.values)

        print('Ok')


# Tests the classes in this file
if __name__ == '__main__':

    SolutionTester.times = 5
    SolutionTester.max_length = 5

    t = unittest.TestLoader().loadTestsFromTestCase(SolutionTester)
    unittest.TextTestRunner(verbosity=0).run(t)
