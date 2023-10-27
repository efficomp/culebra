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

"""Unit test for the parameter optimization individuals."""

import unittest
from numbers import Integral, Real
from itertools import repeat

from culebra.abc import Fitness
from culebra.solution.parameter_optimization import Species, Individual


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


class IndividualTester(unittest.TestCase):
    """Tester for the parameter optimization individuals.

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

    def test_0_crossover(self):
        """Test the behavior of the crossover operator."""
        print('Testing the crossover ...', end=' ')

        # For each length until max_length ...
        for length in range(1, self.max_length + 1):
            species = Species(
                [self.min_bound] * length,
                [self.max_bound] * length
            )

            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                parent1 = Individual(species, MyFitness)
                parent2 = Individual(species, MyFitness)
                offspring1, offspring2 = parent1.crossover(parent2)

                # Check that the offspring meet the species constraints
                self.assertTrue(species.is_member(offspring1))
                self.assertTrue(species.is_member(offspring2))

        print('Ok')

    def test_1_mutation(self):
        """Test the behavior of the mutation operator."""
        print('Testing the mutation ...', end=' ')

        # For each length until max_length ...
        for length in range(1, self.max_length + 1):
            species = Species(
                [self.min_bound] * length,
                [self.max_bound] * length
            )

            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                ind = Individual(species, MyFitness)
                mutant, = ind.mutate(indpb=1)

                # Check that the offspring meet the species constraints
                self.assertTrue(species.is_member(mutant))

        print('Ok')

    def test_2_repr(self):
        """Test the repr and str dunder methods."""
        print('Testing the __repr__ and __str__ dunder methods ...', end=' ')
        species = Species([self.min_bound], [self.max_bound])
        individual = Individual(species, MyFitness)
        self.assertIsInstance(repr(individual), str)
        self.assertIsInstance(str(individual), str)
        print('Ok')


# Tests the classes in this file
if __name__ == '__main__':

    # Number of times each function is executed
    IndividualTester.times = 5

    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)
