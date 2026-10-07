# !/usr/bin/env python3
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

"""Unit test for :mod:`culebra.trainer.aco.heuristic`."""

import unittest
from collections.abc import Sequence

import numpy as np

from culebra.fitness_func.tsp import PathLength, MultiObjectivePathLength
from culebra.fitness_func.feature_selection import KappaIndex
from culebra.trainer.aco.heuristic import (
    tsp_default_heuristic,
    fs_rough_set_heuristic,
    fs_accuracy_heuristic,
    fs_pearson_heuristic,
    fs_information_theory_heuristic
)
from culebra.tools import Dataset


class TSPDefaultHeuristicTester(unittest.TestCase):
    """Test tsp_default_heuristic."""

    def test_single_objective(self):
        """Test the heuristic with a single-objective fitness function."""
        distance_matrix = [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 6, 7, 8, 9],
            [2, 6, 0, 1, 2, 3],
            [3, 7, 1, 0, 4, 5],
            [4, 8, 2, 4, 0, 6],
            [5, 9, 3, 5, 6, 0]
        ]

        single_obj_fitness_func = PathLength(distance_matrix)

        heuristic = tsp_default_heuristic(single_obj_fitness_func)
        self.assertIsInstance(heuristic, Sequence)

        # Check the heuristic_matrix
        for i in range(single_obj_fitness_func.num_nodes):
            for j in range(single_obj_fitness_func.num_nodes):
                if i == j:
                    self.assertEqual(heuristic[0][i][j], 0)
                else:
                    self.assertAlmostEqual(
                        heuristic[0][i][j],
                        1/distance_matrix[i][j]
                    )

    def test_multi_objective(self):
        """Test the heuristic with a multi-objective fitness function."""
        num_nodes = 5
        obj1 = PathLength.from_path(np.random.permutation(num_nodes))
        obj2 = PathLength.from_path(np.random.permutation(num_nodes))
        func = MultiObjectivePathLength(obj1, obj2)

        heur1 = tsp_default_heuristic(obj1)
        heur2 = tsp_default_heuristic(obj2)
        heur = tsp_default_heuristic(func)

        self.assertEqual(len(heur), 2)
        self.assertTrue((heur[0] == heur1[0]).all())
        self.assertTrue((heur[1] == heur2[0]).all())


class FSRoughSetHeuristicTester(unittest.TestCase):
    """Test fs_rough_set_heuristic."""

    def test_all_equivalence_classes_pure(self):
        """Test fs_rough_set_heuristic with pure equivalence classes."""
        inputs = np.array(
            [
                [0],
                [0],
                [1],
                [1],
            ]
        )
        outputs = np.array([0, 0, 1, 1])
        expected_heuristic = np.array([np.exp(1)])

        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        self.assertTrue(
            (
                fs_rough_set_heuristic(training_fitness_func)[0] ==
                expected_heuristic
            ).all()
        )

    def test_none_equivalence_classes_pure(self):
        """Test fs_rough_set_heuristic without pure equivalence classes."""
        inputs = np.array(
            [
                [0],
                [0],
                [1],
                [1],
            ]
        )

        outputs = np.array([0, 1, 0, 1])
        expected_heuristic = np.array([1.0])

        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        self.assertTrue(
            (
                fs_rough_set_heuristic(training_fitness_func)[0] ==
                expected_heuristic
            ).all()
        )

    def test_pure_and_impure_equivalence_classes_pure(self):
        """Test fs_rough_set_heuristic with both pure and impure equivalence classes."""
        inputs = np.array(
            [
                [0],
                [0],
                [1],
                [1],
            ]
        )

        outputs = np.array([0, 0, 0, 1])
        expected_heuristic = np.array([np.exp(0.5)])

        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        self.assertTrue(
            (
                fs_rough_set_heuristic(training_fitness_func)[0] ==
                expected_heuristic
            ).all()
        )

    def test_two_feats_with_different_dependencies(self):
        """Test fs_rough_set_heuristic with two features with different dependencies."""
        inputs = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )
        outputs = np.array([0, 0, 1, 1])
        expected_heuristic = np.array(
            [
                np.exp(1),
                1.0
            ]
        )

        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        self.assertTrue(
            (
                fs_rough_set_heuristic(training_fitness_func)[0] ==
                expected_heuristic
            ).all()
        )

    def test_three_different_induced_classes(self):
        """Test fs_rough_set_heuristic with three different induced classes."""
        inputs = np.array(
            [
                [0],
                [0],
                [1],
                [1],
                [2],
                [2],
            ]
        )
        outputs = np.array([0, 0, 1, 1, 0, 1])
        expected_heuristic = np.array([np.exp(4 / 6)])

        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        self.assertTrue(
            (
                fs_rough_set_heuristic(training_fitness_func)[0] ==
                expected_heuristic
            ).all()
        )

    def test_only_one_class(self):
        """Test fs_rough_set_heuristic with only one class."""
        inputs = np.array(
            [
                [0],
                [0],
                [1],
                [1],
            ]
        )

        outputs = np.array([0, 0, 0, 0])
        expected_heuristic = np.array([np.exp(1)])

        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        self.assertTrue(
            (
                fs_rough_set_heuristic(training_fitness_func)[0] ==
                expected_heuristic
            ).all()
        )


class FSAccuracyHeuristicTester(unittest.TestCase):
    """Test fs_accuracy_heuristic."""

    def test(self):
        """Test fs_accuracy_heuristic."""
        # The first feature keeps all the information
        inputs = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [1, 1]
            ]
        )

        outputs = inputs[:,0]
        dataset = Dataset(inputs, outputs)
        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        heuristic = fs_accuracy_heuristic(training_fitness_func)[0]
        self.assertGreater(heuristic[0], heuristic[1])


class PearsonHeuristicTester(unittest.TestCase):
    """Test fs_pearson_heuristic."""

    def test_correlation_heuristic(self):
        """Test Pearson-based heuristic computation."""

        dataset = Dataset(
            inputs=np.array([
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
                [3.0, 3.0, 1.0],
            ]),
            outputs=np.array([0, 0, 1, 1])
        )

        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        (heuristics,) = fs_pearson_heuristic(training_fitness_func)

        self.assertEqual(heuristics.shape, (3, 3))

        # No self-transitions
        np.testing.assert_allclose(
            np.diag(heuristics),
            np.zeros(3)
        )

        # Third feature is constant
        np.testing.assert_allclose(
            heuristics[2, :],
            np.zeros(3)
        )

        np.testing.assert_allclose(
            heuristics[:, 2],
            np.zeros(3)
        )

        # Features 0 and 1 are perfectly correlated
        expected = 1.0 / (1.0 + 1.0)

        self.assertAlmostEqual(
            heuristics[0, 1],
            expected
        )

        self.assertAlmostEqual(
            heuristics[1, 0],
            expected
        )

    def test_single_feature_dataset(self):
        """Test heuristic computation with a single feature."""

        dataset = Dataset(
            inputs=np.array([
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]),
            outputs=np.array([0, 0, 1, 1])
        )

        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        (heuristics,) = fs_pearson_heuristic(training_fitness_func)

        self.assertEqual(
            heuristics.shape,
            (1, 1)
        )

        self.assertEqual(
            heuristics[0, 0],
            0.0
        )

    def test_constant_single_feature(self):
        """Test a dataset containing one constant feature."""

        dataset = Dataset(
            inputs=np.array([
                [1.0],
                [1.0],
                [1.0],
                [1.0],
            ]),
            outputs=np.array([0, 1, 0, 1])
        )

        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        (heuristics,) = fs_pearson_heuristic(training_fitness_func)

        self.assertEqual(
            heuristics.shape,
            (1, 1)
        )

        self.assertEqual(
            heuristics[0, 0],
            0.0
        )


class FSInformationTheoryHeuristicTester(unittest.TestCase):
    """Test the information-theory-based heuristic."""

    def test_heuristic_matrix(self):
        """Test a dataset with relevant, redundant and irrelevant features."""

        inputs = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
        ])

        outputs = np.array([
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1
        ])

        dataset = Dataset(
            inputs=inputs,
            outputs=outputs
        )

        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        (heuristics,) = fs_information_theory_heuristic(training_fitness_func)

        self.assertEqual(
            heuristics.shape,
            (3, 3)
        )

        # No self-transitions
        np.testing.assert_allclose(
            np.diag(heuristics),
            np.zeros(3)
        )

        # Feature 2 is constant and therefore irrelevant
        np.testing.assert_allclose(
            heuristics[:, 2],
            np.zeros(3)
        )

        # Features 0 and 1 are identical
        self.assertAlmostEqual(
            heuristics[0, 1],
            0.0
        )

        self.assertAlmostEqual(
            heuristics[1, 0],
            0.0
        )

    def test_single_feature(self):
        """Test a dataset containing a single feature."""

        inputs = np.array([
            [0],
            [0],
            [1],
            [1],
            [0],
            [0],
            [1],
            [1],
            [0],
            [0],
            [1],
            [1],
        ])

        outputs = np.array([
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1
        ])

        dataset = Dataset(
            inputs=inputs,
            outputs=outputs
        )

        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        (heuristics,) = fs_information_theory_heuristic(training_fitness_func)

        self.assertEqual(
            heuristics.shape,
            (1, 1)
        )

        self.assertEqual(
            heuristics[0, 0],
            0.0
        )

    def test_relevant_non_redundant_feature(self):
        """A relevant non-redundant feature should receive
        higher heuristic values.
        """

        inputs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 0],
        ])

        outputs = np.array([
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1
        ])

        dataset = Dataset(
            inputs=inputs,
            outputs=outputs
        )

        training_fitness_func = KappaIndex(
            training_data=dataset,
            cv_num_folds=5
        )

        (heuristics,) = fs_information_theory_heuristic(training_fitness_func)

        self.assertGreater(
            heuristics[1, 0],
            0.0
        )

        self.assertGreater(
            heuristics[0, 1],
            0.0
        )


if __name__ == '__main__':
    unittest.main()
