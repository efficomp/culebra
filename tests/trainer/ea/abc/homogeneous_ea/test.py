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

"""Unit test for :py:class:`culebra.trainer.ea.abc.HomogeneousEA`."""

import unittest

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.ea import (
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_SELECTION_FUNC_PARAMS
)
from culebra.trainer.ea.abc import HomogeneousEA
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution,
    BitVector as FeatureSelectionIndividual
)
from culebra.fitness_function.feature_selection import NumFeats
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyTrainer(HomogeneousEA):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self._current_iter_evals = 10


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.ea.abc.HomogeneousEA`."""

    def test_init(self):
        """Test __init__`."""
        valid_solution_cls = FeatureSelectionIndividual
        valid_species = FeatureSelectionSpecies(dataset.num_feats)
        valid_fitness_func = NumFeats(dataset)

        # Try invalid individual classes. Should fail
        invalid_individual_classes = (type, None, 1, FeatureSelectionSolution)
        for solution_cls in invalid_individual_classes:
            with self.assertRaises(TypeError):
                MyTrainer(solution_cls, valid_species, valid_fitness_func)

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyTrainer(valid_solution_cls, species, valid_fitness_func)

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(valid_solution_cls, valid_species, func)

        # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    max_num_iters=max_num_iters
                )

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    max_num_iters=max_num_iters
                )

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    pop_size=pop_size
                )

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    pop_size=pop_size
                )

        # Try invalid types for crossover_func. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    crossover_func=func
                )

        # Try invalid types for mutation_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    mutation_func=func
                )

        # Try invalid types for selection_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    selection_func=func
                )

        # Try invalid types for crossover_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    crossover_prob=prob
                )

        # Try invalid types for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    mutation_prob=prob
                )

        # Try invalid types for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    crossover_prob=prob
                )

        # Try invalid values for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    mutation_prob=prob
                )

        # Try invalid values for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid types for selection_func_params. Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    selection_func_params=params
                )

        # Test default params
        trainer = MyTrainer(
            valid_solution_cls, valid_species, valid_fitness_func
        )
        self.assertEqual(trainer.solution_cls, valid_solution_cls)
        self.assertEqual(trainer.species, valid_species)
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(
            trainer.crossover_func, trainer.solution_cls.crossover)
        self.assertEqual(trainer.mutation_func, trainer.solution_cls.mutate)
        self.assertEqual(trainer.selection_func, DEFAULT_SELECTION_FUNC)
        self.assertEqual(trainer.crossover_prob, DEFAULT_CROSSOVER_PROB)
        self.assertEqual(trainer.mutation_prob, DEFAULT_MUTATION_PROB)
        self.assertEqual(trainer.gene_ind_mutation_prob,
                         DEFAULT_GENE_IND_MUTATION_PROB)
        self.assertEqual(
            trainer.selection_func_params, DEFAULT_SELECTION_FUNC_PARAMS)
        self.assertEqual(trainer._current_iter, None)


if __name__ == '__main__':
    unittest.main()
