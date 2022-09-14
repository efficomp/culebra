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

"""Unit test for the individuals defined in :py:mod:`feature_selector`."""

import unittest
import pickle
import copy

from numbers import Integral, Real
from itertools import repeat
from culebra.base import Species as BaseSpecies
from culebra.fitness_function.classifier_optimization import KappaC
from culebra.genotype.classifier_optimization import Individual, Species

Fitness = KappaC.Fitness
"""Default fitness class."""

DEFAULT_MAX_LENGTH = 50
"""Default maximum number of hyperparameters used to define the
Species."""

DEFAULT_MIN_BOUND = 0
"""Default minimum bound for all the hyperparameters."""

DEFAULT_MAX_BOUND = 5
"""Default maximum bound for all the hyperparameters."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an implementation is run in the
:py:class:`~feature_selector.Tester`."""


class IndividualTester(unittest.TestCase):
    """Tester for the classifier hyperparameter individuals.

    Test extensively the generation and breeding operators.
    """

    max_length = DEFAULT_MAX_LENGTH
    """Default maximum number of hyperparameters used to define the
    Species."""

    min_bound = DEFAULT_MIN_BOUND
    """Default minimum bound for all the hyperparameters."""

    max_bound = DEFAULT_MAX_BOUND
    """Default maximum bound for all the hyperparameters."""

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
        """Test the behavior of a classifier hyperparameters constructor.

        The constructor is executed under different combinations of values for
        the number of features, minimum feature value, maximum feature value,
        minimum size and maximum size.
        """
        print('Testing the constructor ...', end=' ')

        # Check the type of arguments
        with self.assertRaises(TypeError):
            Individual(BaseSpecies(), Fitness)
        with self.assertRaises(TypeError):
            Individual(Species([1], [2]), Species)

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
                    species_i.is_member(Individual(species_i, Fitness))
                )
                self.assertTrue(
                    species_f.is_member(Individual(species_f, Fitness))
                )

        print('Ok')

    def test_1_values(self):
        """Test the values property."""
        print('Testing the values property ...', end=' ')

        # Test the named tuples
        species = Species([0] * 2, [9] * 2, [int, float], ['a', 'b'])
        values = [1, 5]
        ind = Individual(species, Fitness, values=values)
        self.assertEqual(ind.values.a, values[0])
        self.assertEqual(ind.values.b, values[1])
        new_values = ind.named_values_cls(b=3, a=2)
        ind.values = new_values
        self.assertEqual(ind.values.a, new_values.a)
        self.assertEqual(ind.values.b, new_values.b)

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
                ind1_i = Individual(species_i, Fitness)
                ind2_i = Individual(species_i, Fitness)
                ind2_i.values = ind1_i.values
                self.assertTrue(ind1_i.values, ind2_i.values)
                self.assertTrue(ind1_i._values, ind2_i._values)

                ind1_f = Individual(species_f, Fitness)
                ind2_f = Individual(species_f, Fitness)
                ind2_f.values = ind1_f.values
                self.assertTrue(ind1_f.values, ind2_f.values)
                self.assertTrue(ind1_f._values, ind2_f._values)

        print('Ok')

    def test_2_serialization(self):
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
                ind1 = Individual(species, Fitness)
                data = pickle.dumps(ind1)
                ind2 = pickle.loads(data)
                self.assertTrue(ind1.values, ind2.values)

        print('Ok')

    def test_3_copy(self):
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
                ind1 = Individual(species, Fitness)
                ind2 = copy.copy(ind1)
                ind3 = copy.deepcopy(ind1)
                self.assertTrue(ind1.values, ind2.values)
                self.assertTrue(ind1.values, ind3.values)

        print('Ok')

    def test_4_crossover(self):
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
                parent1 = Individual(species, Fitness)
                parent2 = Individual(species, Fitness)
                offspring1, offspring2 = parent1.crossover(parent2)

                # Check that the offspring meet the species constraints
                self.assertTrue(species.is_member(offspring1))
                self.assertTrue(species.is_member(offspring2))

        print('Ok')

    def test_5_mutation(self):
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
                ind = Individual(species, Fitness)
                mutant, = ind.mutate(indpb=1)

                # Check that the offspring meet the species constraints
                self.assertTrue(species.is_member(mutant))

        print('Ok')


# Tests the classes in this file
if __name__ == '__main__':

    IndividualTester.times = 5
    IndividualTester.mas_length = 5

    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)
