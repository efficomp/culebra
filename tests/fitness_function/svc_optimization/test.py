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

"""Test the classifier optimization fitness functions."""

import unittest

from culebra.solution.parameter_optimization import (
    Species,
    Solution
)
from culebra.fitness_function.svc_optimization import (
    C,
    KappaIndex,
    KappaC,
    Accuracy,
    AccuracyC
)
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Species for the solution
species = Species(
    lower_bounds=[0, 0], upper_bounds=[100000, 100000], names=["C", "gamma"]
)


class CTester(unittest.TestCase):
    """Test :py:class:`~culebra.fitness_function.svc_optimization.C`."""

    FitnessFunc = C

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        sol.fitness.values = func.evaluate(sol)

        # Check the fitness function
        self.assertEqual(sol.fitness.values[0], sol.values.C)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        fitness_func = self.FitnessFunc(dataset)
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class KappaIndexTester(unittest.TestCase):
    """Test KappaIndex."""

    FitnessFunc = KappaIndex

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset, test_prop=0.3)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Check that Kappa is in [-1, 1]
        sol.fitness.values = func.evaluate(sol)
        self.assertGreaterEqual(sol.fitness.values[0], -1)
        self.assertLessEqual(sol.fitness.values[0], 1)

    test_repr = CTester.test_repr


class KappaCTester(unittest.TestCase):
    """Test :py:class:`~culebra.fitness_function.svc_optimization.KappaC`."""

    FitnessFunc = KappaC

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset, test_prop=0.3)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Evaluate the solution
        sol.fitness.values = func.evaluate(sol)

        # Check that Kappa is in [-1, 1]
        self.assertGreaterEqual(sol.fitness.values[0], -1)
        self.assertLessEqual(sol.fitness.values[0], 1)

        # Check the number of features
        self.assertEqual(sol.fitness.values[1], sol.values.C)

    test_repr = CTester.test_repr


class AccuracyTester(unittest.TestCase):
    """Test Accuracy."""

    FitnessFunc = Accuracy

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset, test_prop=0.3)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Check that the accuracy is in [0, 1]
        sol.fitness.values = func.evaluate(sol)
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)

    test_repr = CTester.test_repr


class AccuracyCTester(unittest.TestCase):
    """Test AccuracyC."""

    FitnessFunc = AccuracyC

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset, test_prop=0.3)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.Fitness)

        # Evaluate the solution
        sol.fitness.values = func.evaluate(sol)

        # Check that the accuracy is in [0, 1]
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)

        # Check the number of features
        self.assertEqual(sol.fitness.values[1], sol.values.C)

    test_repr = CTester.test_repr


if __name__ == '__main__':
    unittest.main()
