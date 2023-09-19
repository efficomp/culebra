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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

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
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.ParallelCooperativeEA`."""

    def test_init(self):
        """Test __init__."""
        # Test default params
        params = {
            "solution_classes": [
                FeatureSelectionIndividual,
                ClassifierOptimizationIndividual
            ],
            "species": [
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats),
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                )
            ],
            "fitness_function": Fitness(dataset),
            "subpop_trainer_cls": ElitistEA
        }

        # Create the trainer
        trainer = ParallelCooperativeEA(**params)

        # Check current_iter
        self.assertEqual(trainer._current_iter, None)

        # Check num_subpops
        self.assertEqual(trainer.num_subpops, len(trainer.species))


if __name__ == '__main__':
    unittest.main()
