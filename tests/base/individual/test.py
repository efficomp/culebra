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

"""Unit test for :py:class:`base.Individual`."""

import unittest
from copy import copy, deepcopy
import pickle
from culebra.base import Individual, Species, Fitness


class MyIndividual(Individual):
    """Dummy subclass to test the :py:class:`~base.Individual` class."""

    def crossover(self, other):
        """Cross this individual with another one."""
        return (self, other)

    def mutate(self, indpb):
        """Mutate the individual."""
        return (self,)


class MySpecies(Species):
    """Dummy subclass to test the :py:class:`~base.Species` class."""

    def check(self, ind):
        """Check an individual."""
        return True


class MyFitness(Fitness):
    """Dummy fitness."""

    weights = (1.0, 1.0)
    names = ("obj1", "obj2")
    thresholds = (0.001, 0.001)


class IndividualTester(unittest.TestCase):
    """Test the :py:class:`~base.Individual` class."""

    def test_init(self):
        """Test the :py:meth:`~base.Individual.__init__` constructor."""
        # Invalid species
        invalid_species = (None, 'a', 1)

        # Try to create an individual with an invalid species
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyIndividual(species, MyFitness)

        # Try a valid species
        species = MySpecies()
        ind = MyIndividual(species, MyFitness)

        # Check the species
        self.assertEqual(ind.species, species)

        # Check the default fitness
        self.assertIsInstance(ind.fitness, MyFitness)

        # Invalid fitness classes
        invalid_fitness_classes = (str, int)

        # Try to create an individual with an invalid fitness
        for fitness in invalid_fitness_classes:
            with self.assertRaises(TypeError):
                MyIndividual(species, fitness)

        # Try a valid fitness
        ind = MyIndividual(species, MyFitness)
        self.assertIsInstance(ind.fitness, MyFitness)

    def test_delete_fitness(self):
        """Test the :py:meth:`~base.Individual.delete_fitness` method."""
        ind = MyIndividual(MySpecies(), MyFitness)
        self.assertEqual(ind.fitness.values, ())

        ind.fitness.values = (2, 2)
        self.assertEqual(ind.fitness.values, (2, 2))

        ind.delete_fitness()
        self.assertIsInstance(ind.fitness, MyFitness)
        self.assertEqual(ind.fitness.values, ())

    def test_eq(self):
        """Test the equality operator."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = MyIndividual(MySpecies(), MyFitness)

        self.assertEqual(ind1, ind2)

    def test_dominates(self):
        """Test the domination operator."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = MyIndividual(MySpecies(), MyFitness)

        ind1.fitness.values = (2, 2)
        ind2.fitness.values = (1, 1)
        self.assertTrue(ind1.dominates(ind2))

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (2, 1)
        self.assertFalse(ind1.dominates(ind2))

    def test_lt(self):
        """Test the less than operator."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = MyIndividual(MySpecies(), MyFitness)

        ind1.fitness.values = (0, 3)
        ind2.fitness.values = (1, 2)
        self.assertTrue(ind1 < ind2)

        ind1.fitness.values = (1, 1)
        ind2.fitness.values = (1, 2)
        self.assertTrue(ind1 < ind2)

    def test_le(self):
        """Test the less than or equal to operator."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = MyIndividual(MySpecies(), MyFitness)

        ind1.fitness.values = (0, 3)
        ind2.fitness.values = (1, 2)
        self.assertTrue(ind1 <= ind2)

        ind1.fitness.values = (1, 1)
        ind2.fitness.values = (1, 2)
        self.assertTrue(ind1 <= ind2)

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (1, 2)
        self.assertTrue(ind1 <= ind2)

    def test_gt(self):
        """Test the greater than operator."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = MyIndividual(MySpecies(), MyFitness)

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (0, 3)
        self.assertTrue(ind1 > ind2)

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (1, 1)
        self.assertTrue(ind1 > ind2)

    def test_ge(self):
        """Test the greater than or equal to operator."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = MyIndividual(MySpecies(), MyFitness)

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (0, 3)
        self.assertTrue(ind1 >= ind2)

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (1, 1)
        self.assertTrue(ind1 >= ind2)

        ind1.fitness.values = (1, 2)
        ind2.fitness.values = (1, 2)
        self.assertTrue(ind1 >= ind2)

    def test_copy(self):
        """Test the :py:meth:`~base.Individual.__copy__` method."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = copy(ind1)

        # Copy only copies the first level (ind1 != ind2)
        self.assertNotEqual(id(ind1), id(ind2))

        # The objects attributes are shared
        self.assertEqual(id(ind1.species), id(ind2.species))
        self.assertEqual(id(ind1.fitness), id(ind2.fitness))

    def test_deepcopy(self):
        """Test the :py:meth:`~base.Individual.__deepcopy__` method."""
        ind1 = MyIndividual(MySpecies(), MyFitness)
        ind2 = deepcopy(ind1)

        # Check the copy
        self._check_deepcopy(ind1, ind2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~base.Individual.__setstate__` and
        :py:meth:`~base.Individual.__reduce__` methods.
        """
        ind1 = MyIndividual(MySpecies(), MyFitness)

        data = pickle.dumps(ind1)
        ind2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(ind1, ind2)

    def _check_deepcopy(self, ind1, ind2):
        """Check if *ind1* is a deepcopy of *ind2*.

        :param ind1: The first individual
        :type ind1: :py:class:`~base.Individual`
        :param ind2: The second individual
        :type ind2: :py:class:`~base.Individual`
        """
        # Copies all the levels
        self.assertNotEqual(id(ind1), id(ind2))
        self.assertNotEqual(id(ind1.species), id(ind2.species))
        self.assertNotEqual(id(ind1.fitness), id(ind2.fitness))
        self.assertEqual(ind1.fitness, ind2.fitness)


if __name__ == '__main__':
    unittest.main()
