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

"""Test the classifier optimization fitness functions."""

import unittest
from sklearn.svm import SVC
from culebra.base import Dataset, Fitness
from culebra.fitness_function.classifier_optimization import (
    RBFSVCFitnessFunction,
    C,
    KappaIndex,
    KappaC
)
from culebra.genotype.classifier_optimization import Species, Individual

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

# Species for the individual
species = Species(
    lower_bounds=[0, 0], upper_bounds=[100000, 100000], names=["C", "gamma"]
)


class MyFitnessFunction(RBFSVCFitnessFunction):
    """Dummy implementation of a fitness function."""

    class Fitness(Fitness):
        """Fitness returned by this fitness function."""

        weights = (1.0, 1.0)
        names = ("obj1", "obj2")
        thresholds = (0.001, 0.001)

    def evaluate(self, ind, index, representatives):
        """Evaluate one individual.

        Dummy implementation of the evaluation function.
        """


class RBFSVCFitnessFunctionTester(unittest.TestCase):
    """Test RBFSVCFitnessFunction."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = MyFitnessFunction(dataset)
        self.assertTrue(isinstance(func.classifier, SVC))

    def test_classifier(self):
        """Test the valid_prop property."""
        # Fitness function to be tested
        func = MyFitnessFunction(Dataset())

        # Check a valid value
        func.classifier = SVC(kernel='rbf')
        self.assertIsInstance(func.classifier, SVC)
        self.assertEqual(func.classifier.kernel, 'rbf')

        # Check invalid values
        with self.assertRaises(TypeError):
            func.classifier = 'a'

        with self.assertRaises(ValueError):
            func.classifier = SVC(kernel='linear')


class CTester(unittest.TestCase):
    """Test :py:class:`~fitness_function.classifier_optimization.C`."""

    FitnessFunc = C

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Create the individual
        ind = Individual(species=species, fitness_cls=func.Fitness)

        ind.fitness.values = func.evaluate(ind)

        # Check the fitness function
        self.assertEqual(ind.fitness.values[0], ind.values.C)


class KappaIndexTester(unittest.TestCase):
    """Test KappaIndex."""

    FitnessFunc = KappaIndex

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Create the individual
        ind = Individual(species=species, fitness_cls=func.Fitness)

        # Check that Kappa is in [-1, 1]
        ind.fitness.values = func.evaluate(ind)
        self.assertGreaterEqual(ind.fitness.values[0], -1)
        self.assertLessEqual(ind.fitness.values[0], 1)


class KappaCTester(unittest.TestCase):
    """Test :py:class:`~fitness_function.classifier_optimization.KappaC`."""

    FitnessFunc = KappaC

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc(dataset)

        # Create the individual
        ind = Individual(species=species, fitness_cls=func.Fitness)

        # Evaluate the individual
        ind.fitness.values = func.evaluate(ind)

        # Check that Kappa is in [-1, 1]
        self.assertGreaterEqual(ind.fitness.values[0], -1)
        self.assertLessEqual(ind.fitness.values[0], 1)

        # Check the number of features
        self.assertEqual(ind.fitness.values[1], ind.values.C)


if __name__ == '__main__':
    unittest.main()
