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

"""Unit test for :py:class:`~culebra.trainer.ea.ParallelCooperativeEA`."""

import unittest

from culebra.trainer.ea import ElitistEA, ParallelCooperativeEA
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.cooperative import KappaNumFeatsC as Fitness
from culebra.tools import Dataset


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Remove outliers
dataset.remove_outliers()

# Normalize inputs
dataset.robust_scale()


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.ParallelCooperativeEA`."""

    def test_init(self):
        """Test __init__."""
        # Test default params
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA
        }

        # Create the trainer
        trainer = ParallelCooperativeEA(**params)

        # Check current_iter
        self.assertEqual(trainer._current_iter, None)

        # Check num_subtrainers
        self.assertEqual(trainer.num_subtrainers, len(trainer.species))

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        params = {
            "solution_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": Fitness(dataset),
            "subtrainer_cls": ElitistEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = ParallelCooperativeEA(**params)
        trainer._init_search()
        for subtrainer in trainer.subtrainers:
            subtrainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
