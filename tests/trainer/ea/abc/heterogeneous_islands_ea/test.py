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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA`."""

import unittest

from culebra.trainer.ea import (
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_SELECTION_FUNC_PARAMS
)
from culebra.trainer.ea.abc import SinglePopEA, HeterogeneousIslandsEA
from culebra.solution.feature_selection import (
    Species,
    BitVector as Individual
)
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats,
    FSMultiObjectiveDatasetScorer
)
from culebra.tools import Dataset


# Fitness function
def KappaNumFeats(
    training_data,
    test_data=None,
    cv_folds=None,
    classifier=None
):
    """Fitness Function."""
    return FSMultiObjectiveDatasetScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            cv_folds=cv_folds,
            classifier=classifier
        ),
        NumFeats()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)


class MySinglePopEA(SinglePopEA):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        for _ in range(self.pop_size):
            sol = self.solution_cls(
                self.species, self.fitness_function.Fitness
            )
            self.evaluate(sol)
            self._pop.append(sol)


class MyIslandsEA(HeterogeneousIslandsEA):
    """Dummy implementation of an island-based evolutionary algorithm."""

    def _search(self):
        pass


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA`."""

    def test_init(self):
        """Test the constructor."""
        valid_solution_cls = Individual
        valid_species = Species(dataset.num_feats)
        valid_fitness_func = KappaNumFeats(dataset)
        valid_subtrainer_cls = MySinglePopEA
        valid_num_subtrainers = 3

        invalid_pop_size_types = (type, {}, 1.5)
        invalid_pop_size_values = (-1, 0)
        valid_pop_size = 13
        valid_pop_sizes = tuple(
            valid_pop_size + i for i in range(valid_num_subtrainers)
        )

        invalid_funcs = (1, 1.5, {})
        valid_func = len
        valid_funcs = (
            valid_solution_cls.crossover1p,
            valid_solution_cls.crossover2p,
            valid_solution_cls.mutate
        )

        invalid_prob_types = (type, {}, len)
        invalid_prob_values = (-1, 2)
        valid_prob = 0.33
        valid_probs = tuple(
            valid_prob + i * 0.1 for i in range(valid_num_subtrainers)
        )

        invalid_params = (1, 1.5, valid_solution_cls)
        valid_params = {"parameter": 12}
        valid_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
            {"parameter2": 14}
        )

        # Try invalid types for pop_sizes. Should fail
        for pop_size in invalid_pop_size_types:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    pop_sizes=pop_size
                )

        # Try invalid values for pop_size. Should fail
        for pop_size in invalid_pop_size_values:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    pop_sizes=pop_size
                )

        # Try a fixed value for pop_sizes,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            pop_sizes=valid_pop_size
        )

        # Check the length of the sequence
        self.assertEqual(len(trainer.pop_sizes), trainer.num_subtrainers)

        # Check that all the values match
        for island_pop_size in trainer.pop_sizes:
            self.assertEqual(island_pop_size, valid_pop_size)

        # Try different values of pop_size for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            pop_sizes=valid_pop_sizes
        )
        for pop_size1, pop_size2 in zip(trainer.pop_sizes, valid_pop_sizes):
            self.assertEqual(pop_size1, pop_size2)

        # Try invalid types for crossover_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    crossover_funcs=func
                )

        # Try a fixed value for all the crossover functions,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for island_crossover_func in trainer.crossover_funcs:
            self.assertEqual(island_crossover_func, valid_func)

        # Try different values of crossover_func for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_funcs=valid_funcs
        )
        for crossover_func1, crossover_func2 in zip(
            trainer.crossover_funcs, valid_funcs
        ):
            self.assertEqual(crossover_func1, crossover_func2)

        # Try invalid types for mutation_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    mutation_funcs=func
                )

        # Try a fixed value for all the mutation functions,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.mutation_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for island_mutation_func in trainer.mutation_funcs:
            self.assertEqual(island_mutation_func, valid_func)

        # Try different values of mutation_func for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_funcs=valid_funcs
        )
        for mutation_func1, mutation_func2 in zip(
            trainer.mutation_funcs, valid_funcs
        ):
            self.assertEqual(mutation_func1, mutation_func2)

        # Try invalid types for selection_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    selection_funcs=func
                )

        # Try a fixed value for all the selection functions,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.selection_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for island_selection_func in trainer.selection_funcs:
            self.assertEqual(island_selection_func, valid_func)

        # Try different values of selection_func for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs=valid_funcs
        )
        for selection_func1, selection_func2 in zip(
            trainer.selection_funcs, valid_funcs
        ):
            self.assertEqual(selection_func1, selection_func2)

        # Try invalid types for crossover_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    crossover_probs=prob
                )

        # Try invalid values for crossover_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    crossover_probs=prob
                )

        # Try a fixed value for the crossover probability,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_probs), trainer.num_subtrainers)

        # Check that all the values match
        for island_crossover_prob in trainer.crossover_probs:
            self.assertEqual(island_crossover_prob, valid_prob)

        # Try different values of crossover_prob for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            crossover_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.crossover_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    mutation_probs=prob
                )

        # Try invalid values for mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.mutation_probs), trainer.num_subtrainers)

        # Check that all the values match
        for island_mutation_prob in trainer.mutation_probs:
            self.assertEqual(island_mutation_prob, valid_prob)

        # Try different values of mutation_prob for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    gene_ind_mutation_probs=prob
                )

        # Try invalid values for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    gene_ind_mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            gene_ind_mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(
            len(trainer.gene_ind_mutation_probs), trainer.num_subtrainers
        )

        # Check that all the values match
        for island_gene_ind_mutation_prob in trainer.gene_ind_mutation_probs:
            self.assertEqual(island_gene_ind_mutation_prob, valid_prob)

        # Try different values of gene_ind_mutation_prob for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            gene_ind_mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(trainer.gene_ind_mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for selection_funcs_params. Should fail
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    valid_solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers,
                    selection_funcs_params=params
                )

        # Try a fixed value for the selection function parameters,
        # all islands should have the same value
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs_params=valid_params
        )
        # Check the length of the sequence
        self.assertEqual(
            len(trainer.selection_funcs_params), trainer.num_subtrainers
        )

        # Check that all the values match
        for island_selection_func_params in trainer.selection_funcs_params:
            self.assertEqual(island_selection_func_params, valid_params)

        # Try different values of selection_funcs_params for each island
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers,
            selection_funcs_params=valid_funcs_params
        )
        for selection_func_params1, selection_func_params2 in zip(
            trainer.selection_funcs_params, valid_funcs_params
        ):
            self.assertEqual(selection_func_params1, selection_func_params2)

        # Test default params
        trainer = MyIslandsEA(
            valid_solution_cls,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers
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

        # Create the islands
        trainer._generate_subtrainers()

        # Check the default values for all the subpopulations
        for pop_size in trainer.pop_sizes:
            self.assertEqual(pop_size, DEFAULT_POP_SIZE)

        for crossover_func in trainer.crossover_funcs:
            self.assertEqual(crossover_func, trainer.solution_cls.crossover)

        for mutation_func in trainer.mutation_funcs:
            self.assertEqual(mutation_func, trainer.solution_cls.mutate)

        for selection_func in trainer.selection_funcs:
            self.assertEqual(selection_func, DEFAULT_SELECTION_FUNC)

        for crossover_prob in trainer.crossover_probs:
            self.assertEqual(crossover_prob, DEFAULT_CROSSOVER_PROB)

        for mutation_prob in trainer.mutation_probs:
            self.assertEqual(mutation_prob, DEFAULT_MUTATION_PROB)

        for gene_ind_mutation_prob in trainer.gene_ind_mutation_probs:
            self.assertEqual(
                gene_ind_mutation_prob, DEFAULT_GENE_IND_MUTATION_PROB
            )

        for selection_func_params in trainer.selection_funcs_params:
            self.assertEqual(
                selection_func_params, DEFAULT_SELECTION_FUNC_PARAMS
            )

    def test_generate_subtrainers(self):
        """Test _generate_subtrainers."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = MySinglePopEA
        num_subtrainers = 2
        pop_sizes = (13, 15)
        crossover_funcs = (
            solution_cls.crossover1p, solution_cls.crossover2p
        )
        mutation_funcs = (solution_cls.mutate,
                          len)
        selection_funcs = (isinstance, issubclass)
        crossover_probs = (0.33, 0.44)
        mutation_probs = (0.133, 0.144)
        gene_ind_mutation_probs = (0.1133, 0.1144)
        selection_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
        )

        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subtrainer_cls,
            num_subtrainers=num_subtrainers,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )

        # Islands have not been created yet
        self.assertEqual(trainer.subtrainers, None)

        # Create the islands
        trainer._generate_subtrainers()

        # Check the islands
        self.assertIsInstance(trainer.subtrainers, list)
        self.assertEqual(len(trainer.subtrainers), num_subtrainers)

        for index1 in range(trainer.num_subtrainers):
            for index2 in range(index1 + 1, trainer.num_subtrainers):
                self.assertNotEqual(
                    id(trainer.subtrainers[index1]),
                    id(trainer.subtrainers[index2])
                )

        # Check the islands common parameters
        for island_trainer in trainer.subtrainers:
            self.assertIsInstance(island_trainer, subtrainer_cls)

            self.assertEqual(
                island_trainer.solution_cls, trainer.solution_cls
            )
            self.assertEqual(island_trainer.species, trainer.species)
            self.assertEqual(
                island_trainer.fitness_function, trainer.fitness_function
            )
            self.assertEqual(
                island_trainer.max_num_iters,
                trainer.max_num_iters
            )
            self.assertEqual(
                island_trainer.checkpoint_enable, trainer.checkpoint_enable
            )
            self.assertEqual(
                island_trainer.checkpoint_freq, trainer.checkpoint_freq
            )
            self.assertEqual(island_trainer.verbose, trainer.verbose)
            self.assertEqual(island_trainer.random_seed, trainer.random_seed)
            self.assertEqual(island_trainer.container, trainer)
            self.assertEqual(
                island_trainer._preprocess_iteration.__name__,
                "receive_representatives"
            )
            self.assertEqual(
                island_trainer._postprocess_iteration.__name__,
                "send_representatives"
            )

        # Check the island specific parameters
        for (
            island_trainer,
            island_index,
            island_checkpoint_filename,
            island_pop_size,
            island_crossover_func,
            island_mutation_func,
            island_selection_func,
            island_crossover_prob,
            island_mutation_prob,
            island_gene_ind_mutation_prob,
            island_selection_func_params
        ) in zip(
            trainer.subtrainers,
            range(trainer.num_subtrainers),
            trainer.subtrainer_checkpoint_filenames,
            trainer.pop_sizes,
            trainer.crossover_funcs,
            trainer.mutation_funcs,
            trainer.selection_funcs,
            trainer.crossover_probs,
            trainer.mutation_probs,
            trainer.gene_ind_mutation_probs,
            trainer.selection_funcs_params
        ):
            self.assertEqual(island_trainer.index, island_index)
            self.assertEqual(
                island_trainer.checkpoint_filename, island_checkpoint_filename
            )
            self.assertEqual(island_trainer.pop_size, island_pop_size)
            self.assertEqual(
                island_trainer.crossover_func, island_crossover_func
            )
            self.assertEqual(
                island_trainer.mutation_func, island_mutation_func
            )
            self.assertEqual(
                island_trainer.selection_func, island_selection_func
            )
            self.assertEqual(
                island_trainer.crossover_prob, island_crossover_prob
            )
            self.assertEqual(
                island_trainer.mutation_prob, island_mutation_prob
            )
            self.assertEqual(
                island_trainer.gene_ind_mutation_prob,
                island_gene_ind_mutation_prob
            )
            self.assertEqual(
                island_trainer.selection_func_params,
                island_selection_func_params
            )

        # Try incorrect number of pop_sizes
        trainer.pop_sizes = pop_sizes + pop_sizes

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of pop_sizes and
        # try an incorrect number of crossover_funcs
        trainer.pop_sizes = pop_sizes
        trainer.crossover_funcs = crossover_funcs + crossover_funcs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of crossover_funcs and
        # try an incorrect number of mutation_funcs
        trainer.crossover_funcs = crossover_funcs
        trainer.mutation_funcs = mutation_funcs + mutation_funcs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of mutation_funcs and
        # try an incorrect number of selection_funcs
        trainer.mutation_funcs = mutation_funcs
        trainer.selection_funcs = selection_funcs + selection_funcs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of selection_funcs and
        # try an incorrect number of crossover_probs
        trainer.selection_funcs = selection_funcs
        trainer.crossover_probs = crossover_probs + crossover_probs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of crossover_probs and
        # try an incorrect number of mutation_probs
        trainer.crossover_probs = crossover_probs
        trainer.mutation_probs = mutation_probs + mutation_probs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of mutation_probs and
        # try an incorrect number of gene_ind_mutation_probs
        trainer.mutation_probs = mutation_probs
        trainer.gene_ind_mutation_probs = gene_ind_mutation_probs * 2

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of gene_ind_mutation_probs and
        # try an incorrect number of selection_funcs_params
        trainer.gene_ind_mutation_probs = gene_ind_mutation_probs
        trainer.selection_funcs_params = selection_funcs_params * 2

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

    def test_repr(self):
        """Test the repr and str dunder methods."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = MySinglePopEA
        num_subtrainers = 2
        pop_sizes = (13, 15)
        crossover_funcs = (
            solution_cls.crossover1p, solution_cls.crossover2p
        )
        mutation_funcs = (solution_cls.mutate,
                          len)
        selection_funcs = (isinstance, issubclass)
        crossover_probs = (0.33, 0.44)
        mutation_probs = (0.133, 0.144)
        gene_ind_mutation_probs = (0.1133, 0.1144)
        selection_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
        )

        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subtrainer_cls,
            num_subtrainers=num_subtrainers,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
