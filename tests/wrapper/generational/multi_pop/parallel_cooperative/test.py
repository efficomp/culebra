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

"""Unit test for :py:class:`wrapper.multi_pop.ParallelCooperative`."""

import unittest
from culebra.base import Dataset
from culebra.fitness_function.cooperative import KappaNumFeatsC as Fitness
from culebra.genotype.feature_selection import (
    Species as FeatureSelectionSpecies
)
from culebra.genotype.feature_selection.individual import (
    BitVector as FeatureSelectionIndividual
)
from culebra.genotype.classifier_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.wrapper.single_pop import Elitist
from culebra.wrapper.multi_pop import ParallelCooperative


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.ParallelCooperative`."""

    def test_init(self):
        """Test __init__."""
        # Test default params
        params = {
            "individual_classes": [
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
            "subpop_wrapper_cls": Elitist
        }

        # Create the wrapper
        wrapper = ParallelCooperative(**params)

        # Check current_gen
        self.assertEqual(wrapper._current_gen, None)

        # Check num_subpops
        self.assertEqual(wrapper.num_subpops, len(wrapper.species))


if __name__ == '__main__':
    unittest.main()
