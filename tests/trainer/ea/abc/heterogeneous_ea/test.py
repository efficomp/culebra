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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.HeterogeneousEA`."""

import unittest

from culebra.trainer.ea.abc import SinglePopEA, HeterogeneousEA
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyTrainer(HeterogeneousEA):
    """Dummy implementation of an island-based evolutionary algorithm."""

    solution_cls = Individual
    species = Species(dataset.num_feats)

    def _search(self):
        pass


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.abc.HeterogeneousEA`."""

    def test_init(self):
        """Test :py:meth:`~culebra.trainer.ea.abc.HeterogeneousEA.__init__`."""
        valid_fitness_func = Fitness(dataset)
        valid_subpop_trainer_cls = SinglePopEA
        valid_num_subpops = 3

        invalid_pop_size_types = (type, {}, 1.5)
        invalid_pop_size_values = (-1, 0)
        valid_pop_size = 13
        valid_pop_sizes = tuple(
            valid_pop_size + i for i in range(valid_num_subpops)
        )

        invalid_funcs = (1, 1.5, {})
        valid_func = len
        valid_funcs = (
            Individual.crossover1p,
            Individual.crossover2p,
            Individual.mutate
        )

        invalid_prob_types = (type, {}, len)
        invalid_prob_values = (-1, 2)
        valid_prob = 0.33
        valid_probs = tuple(
            valid_prob + i * 0.1 for i in range(valid_num_subpops)
        )

        invalid_params = (1, 1.5, Individual)
        valid_params = {"parameter": 12}
        valid_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
            {"parameter2": 14}
        )

        # Try invalid types for pop_sizes. Should fail
        for pop_size in invalid_pop_size_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    pop_sizes=pop_size
                )

        # Try invalid values for pop_size. Should fail
        for pop_size in invalid_pop_size_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    pop_sizes=pop_size
                )

        # Try a fixed value for pop_sizes,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            pop_sizes=valid_pop_size
        )

        # Check the length of the sequence
        self.assertEqual(len(trainer.pop_sizes), trainer.num_subpops)

        # Check that all the values match
        for island_pop_size in trainer.pop_sizes:
            self.assertEqual(island_pop_size, valid_pop_size)

        # Try different values of pop_size for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            pop_sizes=valid_pop_sizes
        )
        for pop_size1, pop_size2 in zip(trainer.pop_sizes, valid_pop_sizes):
            self.assertEqual(pop_size1, pop_size2)

        # Try invalid types for crossover_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    crossover_funcs=func
                )

        # Try a fixed value for all the crossover functions,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_funcs), trainer.num_subpops)

        # Check that all the values match
        for island_crossover_func in trainer.crossover_funcs:
            self.assertEqual(island_crossover_func, valid_func)

        # Try different values of crossover_func for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            crossover_funcs=valid_funcs
        )
        for crossover_func1, crossover_func2 in zip(
            trainer.crossover_funcs, valid_funcs
        ):
            self.assertEqual(crossover_func1, crossover_func2)

        # Try invalid types for mutation_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    mutation_funcs=func
                )

        # Try a fixed value for all the mutation functions,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            mutation_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.mutation_funcs), trainer.num_subpops)

        # Check that all the values match
        for island_mutation_func in trainer.mutation_funcs:
            self.assertEqual(island_mutation_func, valid_func)

        # Try different values of mutation_func for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            mutation_funcs=valid_funcs
        )
        for mutation_func1, mutation_func2 in zip(
            trainer.mutation_funcs, valid_funcs
        ):
            self.assertEqual(mutation_func1, mutation_func2)

        # Try invalid types for selection_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    selection_funcs=func
                )

        # Try a fixed value for all the selection functions,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            selection_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.selection_funcs), trainer.num_subpops)

        # Check that all the values match
        for island_selection_func in trainer.selection_funcs:
            self.assertEqual(island_selection_func, valid_func)

        # Try different values of selection_func for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            selection_funcs=valid_funcs
        )
        for selection_func1, selection_func2 in zip(
            trainer.selection_funcs, valid_funcs
        ):
            self.assertEqual(selection_func1, selection_func2)

        # Try invalid types for crossover_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    crossover_probs=prob
                )

        # Try invalid values for crossover_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    crossover_probs=prob
                )

        # Try a fixed value for the crossover probability,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            crossover_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_probs), trainer.num_subpops)

        # Check that all the values match
        for island_crossover_prob in trainer.crossover_probs:
            self.assertEqual(island_crossover_prob, valid_prob)

        # Try different values of crossover_prob for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            crossover_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.crossover_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    mutation_probs=prob
                )

        # Try invalid values for mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.mutation_probs), trainer.num_subpops)

        # Check that all the values match
        for island_mutation_prob in trainer.mutation_probs:
            self.assertEqual(island_mutation_prob, valid_prob)

        # Try different values of mutation_prob for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    gene_ind_mutation_probs=prob
                )

        # Try invalid values for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    gene_ind_mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            gene_ind_mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(
            len(trainer.gene_ind_mutation_probs), trainer.num_subpops
        )

        # Check that all the values match
        for island_gene_ind_mutation_prob in trainer.gene_ind_mutation_probs:
            self.assertEqual(island_gene_ind_mutation_prob, valid_prob)

        # Try different values of gene_ind_mutation_prob for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            gene_ind_mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.gene_ind_mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for selection_funcs_params. Should fail
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyTrainer(
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops,
                    selection_funcs_params=params
                )

        # Try a fixed value for the selection function parameters,
        # all islands should have the same value
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            selection_funcs_params=valid_params
        )
        # Check the length of the sequence
        self.assertEqual(
            len(trainer.selection_funcs_params), trainer.num_subpops
        )

        # Check that all the values match
        for island_selection_func_params in trainer.selection_funcs_params:
            self.assertEqual(island_selection_func_params, valid_params)

        # Try different values of selection_funcs_params for each island
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops,
            selection_funcs_params=valid_funcs_params
        )
        for selection_func_params1, selection_func_params2 in zip(
            trainer.selection_funcs_params, valid_funcs_params
        ):
            self.assertEqual(selection_func_params1, selection_func_params2)

        # Test default params
        trainer = MyTrainer(
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops
        )

        # Default values for not initialized subpopulations should be None
        for pop_size in trainer.pop_sizes:
            self.assertEqual(pop_size, None)

        for crossover_func in trainer.crossover_funcs:
            self.assertEqual(crossover_func, None)

        for mutation_func in trainer.mutation_funcs:
            self.assertEqual(mutation_func, None)

        for selection_func in trainer.selection_funcs:
            self.assertEqual(selection_func, None)

        for crossover_prob in trainer.crossover_probs:
            self.assertEqual(crossover_prob, None)

        for mutation_prob in trainer.mutation_probs:
            self.assertEqual(mutation_prob, None)

        for gene_ind_mutation_prob in trainer.gene_ind_mutation_probs:
            self.assertEqual(gene_ind_mutation_prob, None)

        for selection_func_params in trainer.selection_funcs_params:
            self.assertEqual(selection_func_params, None)


if __name__ == '__main__':
    unittest.main()
