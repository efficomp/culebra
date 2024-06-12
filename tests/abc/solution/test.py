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

"""Unit test for :py:class:`culebra.abc.Solution`."""

import unittest
import pickle
from copy import copy, deepcopy

from culebra.abc import Solution, Species, Fitness


class MySolution(Solution):
    """Dummy subclass to test the :py:class:`~culebra.abc.Solution` class."""


class MySpecies(Species):
    """Dummy subclass to test the :py:class:`~culebra.abc.Species` class."""

    def check(self, _):
        """Check a solution."""
        return True


class MyFitness(Fitness):
    """Dummy fitness."""

    weights = (1.0, 1.0)
    names = ("obj1", "obj2")
    thresholds = [0.001, 0.001]


class SolutionTester(unittest.TestCase):
    """Test the :py:class:`~culebra.abc.Solution` class."""

    def test_init(self):
        """Test the :py:meth:`~culebra.abc.Solution.__init__` constructor."""
        # Invalid species
        invalid_species = (None, 'a', 1)

        # Try to create a solution with an invalid species
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MySolution(species, MyFitness)

        # Try a valid species
        species = MySpecies()
        sol = MySolution(species, MyFitness)

        # Check the species
        self.assertEqual(sol.species, species)

        # Check the default fitness
        self.assertIsInstance(sol.fitness, MyFitness)

        # Invalid fitness classes
        invalid_fitness_classes = (str, int)

        # Try to create a solution with an invalid fitness
        for fitness in invalid_fitness_classes:
            with self.assertRaises(TypeError):
                MySolution(species, fitness)

        # Try a valid fitness
        sol = MySolution(species, MyFitness)
        self.assertIsInstance(sol.fitness, MyFitness)

    def test_delete_fitness(self):
        """Test the :py:meth:`~culebra.abc.Solution.delete_fitness` method."""
        sol = MySolution(MySpecies(), MyFitness)
        self.assertEqual(sol.fitness.values, ())

        sol.fitness.values = (2, 2)
        self.assertEqual(sol.fitness.values, (2, 2))

        sol.delete_fitness()
        self.assertIsInstance(sol.fitness, MyFitness)
        self.assertEqual(sol.fitness.values, ())

    def test_eq(self):
        """Test the equality operator."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = MySolution(MySpecies(), MyFitness)

        self.assertEqual(sol1, sol2)

    def test_dominates(self):
        """Test the domination operator."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = MySolution(MySpecies(), MyFitness)

        sol1.fitness.values = (2, 2)
        sol2.fitness.values = (1, 1)
        self.assertTrue(sol1.dominates(sol2))

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (2, 1)
        self.assertFalse(sol1.dominates(sol2))

    def test_lt(self):
        """Test the less than operator."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = MySolution(MySpecies(), MyFitness)

        sol1.fitness.values = (0, 3)
        sol2.fitness.values = (1, 2)
        self.assertTrue(sol1 < sol2)

        sol1.fitness.values = (1, 1)
        sol2.fitness.values = (1, 2)
        self.assertTrue(sol1 < sol2)

    def test_le(self):
        """Test the less than or equal to operator."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = MySolution(MySpecies(), MyFitness)

        sol1.fitness.values = (0, 3)
        sol2.fitness.values = (1, 2)
        self.assertTrue(sol1 <= sol2)

        sol1.fitness.values = (1, 1)
        sol2.fitness.values = (1, 2)
        self.assertTrue(sol1 <= sol2)

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (1, 2)
        self.assertTrue(sol1 <= sol2)

    def test_gt(self):
        """Test the greater than operator."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = MySolution(MySpecies(), MyFitness)

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (0, 3)
        self.assertTrue(sol1 > sol2)

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (1, 1)
        self.assertTrue(sol1 > sol2)

    def test_ge(self):
        """Test the greater than or equal to operator."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = MySolution(MySpecies(), MyFitness)

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (0, 3)
        self.assertTrue(sol1 >= sol2)

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (1, 1)
        self.assertTrue(sol1 >= sol2)

        sol1.fitness.values = (1, 2)
        sol2.fitness.values = (1, 2)
        self.assertTrue(sol1 >= sol2)

    def test_copy(self):
        """Test the :py:meth:`~culebra.abc.Solution.__copy__` method."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = copy(sol1)

        # Copy only copies the first level (sol1 != sol2)
        self.assertNotEqual(id(sol1), id(sol2))

        # The objects attributes are shared
        self.assertEqual(id(sol1.species), id(sol2.species))
        self.assertEqual(id(sol1.fitness), id(sol2.fitness))

    def test_deepcopy(self):
        """Test the :py:meth:`~culebra.abc.Solution.__deepcopy__` method."""
        sol1 = MySolution(MySpecies(), MyFitness)
        sol2 = deepcopy(sol1)

        # Check the copy
        self._check_deepcopy(sol1, sol2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.abc.Solution.__setstate__` and
        :py:meth:`~culebra.abc.Solution.__reduce__` methods.
        """
        sol1 = MySolution(MySpecies(), MyFitness)

        data = pickle.dumps(sol1)
        sol2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(sol1, sol2)

    def _check_deepcopy(self, sol1, sol2):
        """Check if *sol1* is a deepcopy of *sol2*.

        :param sol1: The first solution
        :type sol1: :py:class:`~culebra.abc.Solution`
        :param sol2: The second solution
        :type sol2: :py:class:`~culebra.abc.Solution`
        """
        # Copies all the levels
        self.assertNotEqual(id(sol1), id(sol2))
        self.assertNotEqual(id(sol1.species), id(sol2.species))
        self.assertNotEqual(id(sol1.fitness), id(sol2.fitness))
        self.assertEqual(sol1.fitness, sol2.fitness)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        sol = MySolution(MySpecies(), MyFitness)
        self.assertIsInstance(repr(sol), str)
        self.assertIsInstance(str(sol), str)


if __name__ == '__main__':
    unittest.main()
