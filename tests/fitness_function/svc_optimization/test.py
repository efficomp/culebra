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
from culebra.solution.feature_selection import (
    Species as FSSpecies,
    BitVector as FSIndividual
)
from culebra.fitness_function.svc_optimization import (
    C,
    KappaIndex,
    Accuracy
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
        func = self.FitnessFunc()

        # Try an invalid solution class. Should fail...
        fs_species = FSSpecies(num_feats=3)
        fs_ind = FSIndividual(fs_species, func.fitness_cls)
        with self.assertRaises(ValueError):
            func.evaluate(fs_ind)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.fitness_cls)

        fit_values = func.evaluate(sol).values

        # Check the fitness function
        self.assertEqual(sol.fitness.values[0], sol.values.C)
        self.assertEqual(fit_values, sol.fitness.values)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        fitness_func = self.FitnessFunc()
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class KappaIndexTester(unittest.TestCase):
    """Test KappaIndex."""

    FitnessFunc = KappaIndex

    def test_evaluate(self):
        """Test the evaluation method."""
        training_data, test_data = dataset.split(0.3)

        # Fitness function to be tested
        func = self.FitnessFunc(training_data)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.fitness_cls)

        # Check that the Kappa index is in [-1, 1]

        # Test kfcv evaluation
        fit_values = func.evaluate(sol).values
        self.assertGreaterEqual(sol.fitness.values[0], -1)
        self.assertLessEqual(sol.fitness.values[0], 1)
        self.assertEqual(fit_values, sol.fitness.values)

        del sol.fitness.values

        # Test train_test evaluation
        func = self.FitnessFunc(training_data, test_data)
        fit_values = func.evaluate(sol).values
        self.assertGreaterEqual(sol.fitness.values[0], -1)
        self.assertLessEqual(sol.fitness.values[0], 1)
        self.assertEqual(fit_values, sol.fitness.values)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        fitness_func = self.FitnessFunc(dataset)
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class AccuracyTester(unittest.TestCase):
    """Test Accuracy."""

    FitnessFunc = Accuracy

    def test_evaluate(self):
        """Test the evaluation method."""
        training_data, test_data = dataset.split(0.3)

        # Fitness function to be tested
        func = self.FitnessFunc(training_data)

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.fitness_cls)

        # Check that the accuracy is in [0, 1]

        # Test kfcv evaluation
        fit_values = func.evaluate(sol).values
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)
        self.assertEqual(fit_values, sol.fitness.values)
        del sol.fitness.values

        # Test train_test evaluation
        func = self.FitnessFunc(training_data, test_data)
        fit_values = func.evaluate(sol).values
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)
        self.assertEqual(fit_values, sol.fitness.values)

    test_repr = KappaIndexTester.test_repr


if __name__ == '__main__':
    unittest.main()
