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

"""Test the feature selection fitness functions."""

import unittest

from os import remove
from copy import copy, deepcopy

from sklearn.metrics import cohen_kappa_score

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
)
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction

from culebra.fitness_function.feature_selection.abc import (
    FSClassificationScorer
)


from culebra.fitness_function.feature_selection import (
    NumFeats,
    FeatsProportion,
    KappaIndex,
    Accuracy,
    FSMultiObjectiveDatasetScorer
)
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Species for the solution
species = Species(num_feats=dataset.num_feats)


class MyFSClassificationScorer(FSClassificationScorer):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    @property
    def obj_names(self):
        """Objective names."""
        return ("Kappa",)

    _score = cohen_kappa_score


class AnotherSingleObjectiveFitnessFunction(SingleObjectiveFitnessFunction):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    def evaluate(self, sol, index, representatives):
        """Evaluate a solution."""
        sol.fitness.update_value(2, self.index)
        return sol.fitness


class NumFeatsTester(unittest.TestCase):
    """Test NumFeats."""

    FitnessFunc = NumFeats

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc()

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.fitness_cls)

        fit_values = func.evaluate(sol).values

        # Check the fitness function
        self.assertEqual(sol.fitness.values, (sol.num_feats,))
        self.assertEqual(fit_values, sol.fitness.values)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        fitness_func = self.FitnessFunc()
        self.assertIsInstance(repr(fitness_func), str)
        self.assertIsInstance(str(fitness_func), str)


class FeatsProportionTester(unittest.TestCase):
    """Test FeatsProportion."""

    FitnessFunc = FeatsProportion

    def test_evaluate(self):
        """Test the evaluation method."""
        # Fitness function to be tested
        func = self.FitnessFunc()

        # Create the solution
        sol = Solution(species=species, fitness_cls=func.fitness_cls)

        # Check that the proportion of selected features is in [0, 1]
        fit_values = func.evaluate(sol).values
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)
        self.assertEqual(fit_values, sol.fitness.values)

    test_repr = NumFeatsTester.test_repr


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

        # Test train_test evaluation
        func = self.FitnessFunc(training_data, test_data)
        fit_values = func.evaluate(sol).values
        self.assertGreaterEqual(sol.fitness.values[0], 0)
        self.assertLessEqual(sol.fitness.values[0], 1)
        self.assertEqual(fit_values, sol.fitness.values)

    test_repr = KappaIndexTester.test_repr


class FSMultiObjectiveDatasetScorerTester(unittest.TestCase):
    """Test FSMultiObjectiveDatasetScorer."""

    def test_init(self):
        """Test the constructor."""
        # Objectives
        obj0 = MyFSClassificationScorer(dataset)
        obj1 = AnotherSingleObjectiveFitnessFunction()

        # Try with only an objective, that is not valid
        with self.assertRaises(ValueError):
            FSMultiObjectiveDatasetScorer(obj1)

        # Try with only an objective
        func = FSMultiObjectiveDatasetScorer(obj0)
        self.assertEqual(func.num_obj, 1)
        self.assertEqual(func.objectives, [obj0])
        self.assertEqual(func.obj_weights, obj0.obj_weights)
        self.assertEqual(func.obj_names, obj0.obj_names)
        self.assertEqual(obj0.index, 0)

        # Try a bi-objective fitness function
        func = FSMultiObjectiveDatasetScorer(obj0, obj1)

        self.assertEqual(func.num_obj, 2)
        self.assertEqual(func.objectives, [obj0, obj1])
        self.assertEqual(
            func.obj_weights, obj0.obj_weights + obj1.obj_weights
        )
        self.assertEqual(
            func.obj_names, obj0.obj_names + obj1.obj_names
        )
        self.assertEqual(obj0.index, 0)
        self.assertEqual(obj1.index, 1)

    def test_num_nodes(self):
        """Test the num_nodes property."""
        func = FSMultiObjectiveDatasetScorer(
            MyFSClassificationScorer(dataset),
            AnotherSingleObjectiveFitnessFunction()
        )

        self.assertEqual(func.num_nodes, dataset.num_feats)

    def test_heuristic(self):
        """Test the heuristic method."""
        obj0 = MyFSClassificationScorer(dataset)
        obj1 = AnotherSingleObjectiveFitnessFunction()
        func = FSMultiObjectiveDatasetScorer(obj0, obj1)
        num_feats = 10
        min_feat = 2
        max_feat = 8
        species = Species(
            num_feats=num_feats, min_feat=min_feat, max_feat=max_feat
        )
        self.assertTrue(
            (func.heuristic(species)[0] == obj0.heuristic(species)[0]).all()
        )

    def test_copy(self):
        """Test the __copy__ method."""
        func1 = FSMultiObjectiveDatasetScorer(
            MyFSClassificationScorer(dataset),
            AnotherSingleObjectiveFitnessFunction()
        )
        func2 = copy(func1)

        # Copy only copies the first level (func1 != func2)
        self.assertNotEqual(id(func1), id(func2))

        # The objects attributes are shared
        self.assertEqual(id(func1.objectives), id(func2.objectives))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        func1 = FSMultiObjectiveDatasetScorer(
            MyFSClassificationScorer(dataset),
            AnotherSingleObjectiveFitnessFunction()
        )
        func2 = deepcopy(func1)

        # Check the copy
        self._check_deepcopy(func1, func2)

    def test_serialization(self):
        """Serialization test."""
        func1 = FSMultiObjectiveDatasetScorer(
            MyFSClassificationScorer(dataset),
            AnotherSingleObjectiveFitnessFunction()
        )

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        func1.dump(serialized_filename)
        func2 = FSMultiObjectiveDatasetScorer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(func1, func2)

        # Remove the serialized file
        remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = FSMultiObjectiveDatasetScorer(
            MyFSClassificationScorer(dataset),
            AnotherSingleObjectiveFitnessFunction()
        )
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)

    def _check_deepcopy(self, func1, func2):
        """Check if *func1* is a deepcopy of *func2*.

        :param func1: The first fitness function
        :type func1:
            :py:class:`~culebra.fitness_function.feature_selection.FSMultiObjectiveDatasetScorerTester`
        :param func2: The second fitness function
        :type func2:
            :py:class:`~culebra.fitness_function.feature_selection.FSMultiObjectiveDatasetScorerTester`
        """
        # Copies all the levels
        self.assertNotEqual(id(func1), id(func2))
        self.assertNotEqual(id(func1.objectives), id(func2.objectives))

        self.assertTrue(
            (
                func1.objectives[0].training_data.inputs ==
                func2.objectives[0].training_data.inputs
            ).all()
        )


if __name__ == '__main__':
    unittest.main()
