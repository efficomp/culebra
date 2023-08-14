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

"""Unit test for :py:class:`culebra.trainer.ea.abc.SinglePopEA`."""

import unittest
from functools import partialmethod

from culebra import DEFAULT_MAX_NUM_ITERS, DEFAULT_POP_SIZE
from culebra.trainer.ea import (
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_SELECTION_FUNC_PARAMS
)
from culebra.trainer.ea.abc import SinglePopEA
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.feature_selection import NumFeats
from culebra.fitness_function.cooperative import KappaNumFeatsC
from culebra.tools import Dataset


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

# KappaNumFeatsC expects that the first species codes hyperparameters and
# the following ones code features
cooperative_individual_classes = [
    ClassifierOptimizationIndividual,
    FeatureSelectionIndividual
]

cooperative_species = [
    ClassifierOptimizationSpecies(
        lower_bounds=[0, 0],
        upper_bounds=[100000, 100000],
        names=["C", "gamma"]
    ),
    FeatureSelectionSpecies(dataset.num_feats)
]


def init_representatives(
    trainer,
    individual_classes,
    species,
    representation_size
):
    """Init the representatives of other species.

    :param individual_clases: The individual class for each species.
    :type individual_classes: A :py:class:`~collections.abc.Sequence`
        of :py:class:`~culebra.solution.abc.Individual` subclasses
    :param species: The species to be evolved
    :type species: A :py:class:`~collections.abc.Sequence` of
        :py:class:`~culebra.abc.Species` instances
    :param representation_size: Number of representative individuals
        from each species
    :type representation_size: :py:class:`int`
    """
    trainer._representatives = []

    for _ in range(representation_size):
        trainer._representatives.append(
            [
                ind_cls(
                    spe, trainer.fitness_function.Fitness
                ) if i != trainer.index else None
                for i, (ind_cls, spe) in enumerate(
                    zip(
                        individual_classes,
                        species)
                )
            ]

        )


class MyTrainer(SinglePopEA):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self._current_iter_evals = 10


class MyCooperativeTrainer(MyTrainer):
    """Dummy implementation of a cooperative trainer method."""

    _init_representatives = partialmethod(
        init_representatives,
        individual_classes=cooperative_individual_classes,
        species=cooperative_species,
        representation_size=2
    )


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.ea.abc.SinglePopEA`."""

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
        self.assertEqual(trainer.pop_size, DEFAULT_POP_SIZE)
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
        self.assertEqual(trainer.pop, None)
        self.assertEqual(trainer._current_iter, None)

    def test_new_state(self):
        """Test _new_state`.

        Also test _fill_initial_pop`.
        """
        params = {
            "solution_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the pop size
        self.assertEqual(len(trainer.pop), trainer.pop_size)

        # Check the individuals in the population
        for ind in trainer.pop:
            self.assertIsInstance(ind, trainer.solution_cls)

    def test_evaluate_pop(self):
        """Test the evaluation of a population."""
        # Check a single population
        params = {
            "solution_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = MyTrainer(**params)

        # Init the search
        trainer._init_search()

        # Check that the individuals in the population are evaluated
        for ind in trainer.pop:
            self.assertTrue(ind.fitness.valid)

        # Check the number of evaluations performed
        self.assertEqual(trainer._current_iter_evals, trainer.pop_size)

        # Check a cooperative problem
        fitness_function = KappaNumFeatsC(dataset)

        # Parameters for the trainer
        params = {
            "solution_cls": cooperative_individual_classes[0],
            "species": cooperative_species[0],
            "fitness_function": fitness_function,
            "pop_size": 5,
            "verbose": False
        }

        # Construct a parameterized trainer
        trainer = MyCooperativeTrainer(**params)

        # Init the search
        trainer._init_search()

        # Check that the individuals in the population are evaluated
        for ind in trainer.pop:
            self.assertTrue(ind.fitness.valid)

        # Check the number of evaluations performed
        self.assertEqual(
            trainer._current_iter_evals,
            trainer.pop_size * len(trainer._representatives)
        )


if __name__ == '__main__':
    unittest.main()
