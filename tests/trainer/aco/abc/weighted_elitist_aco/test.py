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

"""Unit test for :py:class:`culebra.trainer.abc.aco.WeightedElitistACO`."""

import unittest

import numpy as np

from culebra.trainer.aco import DEFAULT_ELITE_WEIGHT
from culebra.trainer.aco.abc import WeightedElitistACO
from culebra.solution.tsp import Species, Ant
from culebra.fitness_function.tsp import PathLength

num_nodes = 25
optimum_path = np.random.permutation(num_nodes)
fitness_func = PathLength.fromPath(optimum_path)
banned_nodes = [0, num_nodes-1]
feasible_nodes = np.setdiff1d(optimum_path, banned_nodes)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.abc.aco.WeightedElitistACO`."""

    def test_init(self):
        """Test __init__`."""
        ant_cls = Ant
        species = Species(num_nodes, banned_nodes)
        initial_pheromones = [1]

        # Try invalid types for elite_weight. Should fail
        invalid_elite_weight_types = (type, 'a')
        for elite_weight in invalid_elite_weight_types:
            with self.assertRaises(TypeError):
                WeightedElitistACO(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    elite_weight=elite_weight
                )

        # Try invalid values for elite_weight. Should fail
        invalid_elite_weight_values = (-0.5, 1.5)
        for elite_weight in invalid_elite_weight_values:
            with self.assertRaises(ValueError):
                WeightedElitistACO(
                    ant_cls,
                    species,
                    fitness_func,
                    initial_pheromones,
                    elite_weight=elite_weight
                )

        # Try a valid value for elite_weight
        valid_elite_weight_values = (0.0, 0.5, 1.0)
        for elite_weight in valid_elite_weight_values:
            trainer = WeightedElitistACO(
                ant_cls,
                species,
                fitness_func,
                initial_pheromones,
                elite_weight=elite_weight
            )
            self.assertEqual(elite_weight, trainer.elite_weight)

        # Test default params
        trainer = WeightedElitistACO(
            ant_cls,
            species,
            fitness_func,
            initial_pheromones
        )
        self.assertEqual(trainer.elite_weight, DEFAULT_ELITE_WEIGHT)


if __name__ == '__main__':
    unittest.main()
