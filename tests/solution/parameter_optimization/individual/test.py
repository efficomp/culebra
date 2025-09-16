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
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Unit test for the parameter optimization individuals."""

import unittest
from numbers import Integral
from itertools import repeat

from culebra.solution.parameter_optimization import Species, Individual
from culebra.fitness_function.svc_optimization import C

DEFAULT_MIN_BOUND = 0
"""Default minimum bound for all the parameters."""

DEFAULT_MAX_BOUND = 100000
"""Default maximum bound for all the parameters."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an implementation is run."""

# Default species
species = Species(
    lower_bounds=[DEFAULT_MIN_BOUND] * 2,
    upper_bounds=[DEFAULT_MAX_BOUND] * 2,
    names=["C", "gamma"]
)

# Default fitness function
fitness_function = C()

# Default fitness class
fitness_cls = fitness_function.fitness_cls


class IndividualTester(unittest.TestCase):
    """Tester for the parameter optimization individuals.

    Test extensively the generation and breeding operators.
    """

    times = DEFAULT_TIMES
    """Times each function is executed."""

    def setUp(self):
        """Check that all the parameters are alright.

        :raises TypeError: If any of the parameters is not of the appropriate
            type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        if not isinstance(self.times, Integral):
            raise TypeError(
                f"times must be an integer value: {self.times}")

        if self.times <= 0:
            raise ValueError(
                f"times must greater than 0: {self.times}")

    def test_0_crossover(self):
        """Test the behavior of the crossover operator."""
        print('Testing the crossover ...', end=' ')

        # Execute the generator function the given number of times
        for _ in repeat(None, self.times):
            # Generate the two parents
            parent1 = Individual(species, fitness_cls)
            parent2 = Individual(species, fitness_cls)

            # Generate the two offspring
            offspring1, offspring2 = parent1.crossover(parent2)

            # Check that the offspring meet the species constraints
            self.assertTrue(species.is_member(offspring1))
            self.assertTrue(species.is_member(offspring2))

        print('Ok')

    def test_1_mutation(self):
        """Test the behavior of the mutation operator."""
        print('Testing the mutation ...', end=' ')

        # Execute the generator function the given number of times
        for _ in repeat(None, self.times):
            # Generate one individual
            ind = Individual(species, fitness_cls)

            # Generate the mutant
            mutant, = ind.mutate(indpb=1)

            # Check that the offspring meet the species constraints
            self.assertTrue(species.is_member(mutant))

        print('Ok')

    def test_2_repr(self):
        """Test the repr and str dunder methods."""
        print('Testing the __repr__ and __str__ dunder methods ...', end=' ')
        individual = Individual(species, fitness_cls)
        self.assertIsInstance(repr(individual), str)
        self.assertIsInstance(str(individual), str)
        print('Ok')


# Tests the classes in this file
if __name__ == '__main__':

    # Number of times each function is executed
    IndividualTester.times = 5

    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)
