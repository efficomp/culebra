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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.IslandsEA`."""

import unittest

from culebra.solution.abc import Individual
from culebra.trainer.abc import SinglePopTrainer
from culebra.trainer.ea.abc import SinglePopEA, IslandsEA
from culebra.solution.feature_selection import Species
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.abc.IslandsEA`."""

    def test_subpop_trainer_cls(self):
        """Test the subpop_trainer_cls property."""
        solution_cls = Individual
        valid_fitness_func = Fitness(dataset)
        valid_subpop_trainer_cls = SinglePopEA

        # Try invalid subpop_trainer_cls. Should fail
        invalid_trainer_classes = (tuple, str, None, 'a', 1, SinglePopTrainer)
        for cls in invalid_trainer_classes:
            with self.assertRaises(TypeError):
                IslandsEA(
                    solution_cls,
                    species,
                    valid_fitness_func,
                    cls
                )

        # Test default params
        trainer = IslandsEA(
            solution_cls,
            species,
            valid_fitness_func,
            valid_subpop_trainer_cls
        )
        self.assertEqual(trainer.subpop_trainer_cls, valid_subpop_trainer_cls)


if __name__ == '__main__':
    unittest.main()
