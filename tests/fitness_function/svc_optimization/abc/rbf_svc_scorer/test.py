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

"""Test the abstract base feature selection fitness functions."""

import unittest

from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score

from culebra import DEFAULT_SIMILARITY_THRESHOLD
from culebra.fitness_function.dataset_score import DEFAULT_CV_FOLDS
from culebra.fitness_function.svc_optimization.abc import RBFSVCScorer
from culebra.solution.parameter_optimization import (
    Species as ParamOptSpecies,
    Individual as ParamOptIndividual
)
from culebra.solution.feature_selection import (
    Species as FSSpecies,
    BitVector as FSIndividual
)

from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Species for the solution
species = ParamOptSpecies(
    lower_bounds=[0, 0], upper_bounds=[100000, 100000], names=["C", "gamma"]
)


class MyRBFSVCScorer(RBFSVCScorer):
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


class RBFSVCScorerTester(unittest.TestCase):
    """Test RBFSVCScorer."""

    def test_init(self):
        """Test the constructor."""
        # Fitness function to be tested
        func = MyRBFSVCScorer(dataset)
        self.assertTrue(
            (func.training_data.inputs == dataset.inputs).all()
        )
        self.assertTrue(
            (func.training_data.outputs == dataset.outputs).all()
        )
        self.assertEqual(func.test_data, None)
        self.assertEqual(func.test_prop, None)
        self.assertTrue(isinstance(func.classifier, SVC))
        self.assertEqual(func.cv_folds, DEFAULT_CV_FOLDS)
        self.assertEqual(func.index, 0)
        self.assertEqual(func.obj_thresholds, [DEFAULT_SIMILARITY_THRESHOLD])

    def test_classifier(self):
        """Test the classifier property."""
        # Fitness function to be tested
        func = MyRBFSVCScorer(dataset)

        # Check a valid value
        func.classifier = SVC(kernel='rbf')
        self.assertIsInstance(func.classifier, SVC)
        self.assertEqual(func.classifier.kernel, 'rbf')

        # Check invalid values
        with self.assertRaises(TypeError):
            func.classifier = 'a'

        with self.assertRaises(ValueError):
            func.classifier = SVC(kernel='linear')

    def test_evaluate(self):
        """Test the evaluate method."""
        func = MyRBFSVCScorer(dataset)

        # Try an invalid solution class. Should fail...
        fs_species = FSSpecies(num_feats=3)
        fs_ind = FSIndividual(fs_species, func.fitness_cls)
        with self.assertRaises(ValueError):
            func.evaluate(fs_ind)

        # Try a valid solution
        ind = ParamOptIndividual(species, func.fitness_cls)
        fit_values = func.evaluate(ind)
        self.assertEqual(func.classifier.C, ind.values.C)
        self.assertEqual(func.classifier.gamma, ind.values.gamma)
        self.assertTrue(-1 <= fit_values[0] <= 1)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        func = MyRBFSVCScorer(Dataset())
        self.assertIsInstance(repr(func), str)
        self.assertIsInstance(str(func), str)


if __name__ == '__main__':
    unittest.main()
