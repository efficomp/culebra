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

"""Test the FSSVCScorer fitness function."""

import unittest

from numpy import concatenate
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score

from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Solution as ClassifierOptimizationSolution
)

from culebra.fitness_function.feature_selection.abc import FSDatasetScorer
from culebra.fitness_function.feature_selection import (
    NumFeats,
    KappaIndex as FSKappaIndex
)
from culebra.fitness_function.svc_optimization import (
    C,
    KappaIndex as SVCKappaIndex
)

from culebra.fitness_function.cooperative import FSSVCScorer
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Species to optimize a SVM-based classifier
hyperparams_species = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[100000, 100000],
    names=["C", "gamma"]
)

# Species for the feature selection problem
min_feat1 = 0
max_feat1 = dataset.num_feats // 2
features_species1 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=min_feat1,
    max_feat=max_feat1
)

min_feat2 = max_feat1 + 1
max_feat2 = dataset.num_feats - 1
features_species2 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=min_feat2,
    max_feat=max_feat2
)


class MyFSDatasetScorer(FSDatasetScorer):
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


# Objectives
fs_dataset_scorer = MyFSDatasetScorer(dataset)
fs_kappa_index_nb = FSKappaIndex(dataset)
fs_kappa_index_svc_rbf = FSKappaIndex(
    dataset, classifier=SVC(kernel='rbf')
)
fs_kappa_index_svc_linear = FSKappaIndex(
    dataset, classifier=SVC(kernel='linear')
)
svc_kappa_index = SVCKappaIndex(dataset)

fs_num_feats = NumFeats()
svc_c = C()


class FSSVCScorerTester(unittest.TestCase):
    """Test FSSVCScorer."""

    def test_init(self):
        """Test the constructor."""
        # Try with fs_kappa_index_nb, should fail
        # (it does not use an SVC as classifier)
        with self.assertRaises(ValueError):
            FSSVCScorer(fs_kappa_index_svc_rbf, fs_kappa_index_nb)

        # Try with fs_kappa_index_svc_linear, should fail
        # (it uses an SVC as classifier, but not an RBF kernel)
        with self.assertRaises(ValueError):
            FSSVCScorer(fs_kappa_index_svc_linear)

        # Try with fs_dataset_scorer, should fail
        # (it is not an instance of FSClassificationScorer)
        with self.assertRaises(ValueError):
            FSSVCScorer(fs_dataset_scorer)

        # Try with svc_kappa_index, should fail
        # (the classification objs must be instances of FSClassificationScorer)
        with self.assertRaises(ValueError):
            FSSVCScorer(svc_kappa_index)

        # Correct arguments
        func = FSSVCScorer(fs_kappa_index_svc_rbf, fs_num_feats, svc_c)

        self.assertEqual(func.num_obj, 3)
        self.assertEqual(
            func.objectives, [fs_kappa_index_svc_rbf, fs_num_feats, svc_c]
        )
        self.assertEqual(
            func.obj_weights,
            (
                fs_kappa_index_svc_rbf.obj_weights +
                fs_num_feats.obj_weights +
                svc_c.obj_weights
            )
        )
        self.assertEqual(
            func.obj_names,
            (
                fs_kappa_index_svc_rbf.obj_names +
                fs_num_feats.obj_names +
                svc_c.obj_names
            )
        )
        self.assertEqual(fs_kappa_index_svc_rbf.index, 0)
        self.assertEqual(fs_num_feats.index, 1)
        self.assertEqual(svc_c.index, 2)

    def test_construct_solutions(self):
        """Test the construct_solutions method."""
        func = FSSVCScorer(fs_kappa_index_svc_rbf, fs_num_feats, svc_c)

        hyperparams_sol = ClassifierOptimizationSolution(
            hyperparams_species, func.fitness_cls
        )

        features_sol1 = FeatureSelectionSolution(
            features_species1, func.fitness_cls
        )

        features_sol2 = FeatureSelectionSolution(
            features_species2, func.fitness_cls
        )
        all_the_features = concatenate(
            (features_sol1.features, features_sol2.features)
        )

        representatives = [hyperparams_sol, features_sol1, features_sol2]

        (hyperparams, features) = func.construct_solutions(
            hyperparams_sol,
            index=0,
            representatives=representatives
        )
        self.assertEqual(hyperparams_sol, hyperparams)
        self.assertTrue(
            (all_the_features == features.features).all()
        )

        (hyperparams, features) = func.construct_solutions(
            features_sol1,
            index=1,
            representatives=representatives
        )
        self.assertEqual(hyperparams_sol, hyperparams)
        self.assertTrue(
            (all_the_features == features.features).all()
        )

        (hyperparams, features) = func.construct_solutions(
            features_sol2,
            index=2,
            representatives=representatives
        )
        self.assertEqual(hyperparams_sol, hyperparams)
        self.assertTrue(
            (all_the_features == features.features).all()
        )


if __name__ == '__main__':
    unittest.main()
