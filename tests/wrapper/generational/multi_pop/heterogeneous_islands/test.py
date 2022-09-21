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

"""Unit test for :py:class:`wrapper.multi_pop.HeterogeneousIslands`."""

import unittest
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper.single_pop import (
    NSGA,
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_NSGA_SELECTION_FUNC,
    DEFAULT_NSGA_SELECTION_FUNC_PARAMS
)
from culebra.wrapper.multi_pop import HeterogeneousIslands


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyWrapper(HeterogeneousIslands):
    """Dummy implementation of a wrapper method."""

    def _search(self):
        pass


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.HeterogeneousIslands`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.multi_pop.HeterogeneousIslands.__init__`."""
        valid_individual_cls = Individual
        valid_species = Species(dataset.num_feats)
        valid_fitness_func = Fitness(dataset)
        valid_subpop_wrapper_cls = NSGA
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
            valid_individual_cls.crossover1p,
            valid_individual_cls.crossover2p,
            valid_individual_cls.mutate
        )

        invalid_prob_types = (type, {}, len)
        invalid_prob_values = (-1, 2)
        valid_prob = 0.33
        valid_probs = tuple(
            valid_prob + i * 0.1 for i in range(valid_num_subpops)
        )

        invalid_params = (1, 1.5, valid_individual_cls)
        valid_params = {"parameter": 12}
        valid_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
            {"parameter2": 14}
        )

        # Try invalid types for pop_sizes. Should fail
        for pop_size in invalid_pop_size_types:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    pop_sizes=pop_size
                )

        # Try invalid values for pop_size. Should fail
        for pop_size in invalid_pop_size_values:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    pop_sizes=pop_size
                )

        # Try a fixed value for pop_sizes,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            pop_sizes=valid_pop_size
        )

        # Check the length of the sequence
        self.assertEqual(len(wrapper.pop_sizes), wrapper.num_subpops)

        # Check that all the values match
        for island_pop_size in wrapper.pop_sizes:
            self.assertEqual(island_pop_size, valid_pop_size)

        # Try different values of pop_size for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            pop_sizes=valid_pop_sizes
        )
        for pop_size1, pop_size2 in zip(wrapper.pop_sizes, valid_pop_sizes):
            self.assertEqual(pop_size1, pop_size2)

        # Try invalid types for crossover_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    crossover_funcs=func
                )

        # Try a fixed value for all the crossover functions,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(wrapper.crossover_funcs), wrapper.num_subpops)

        # Check that all the values match
        for island_crossover_func in wrapper.crossover_funcs:
            self.assertEqual(island_crossover_func, valid_func)

        # Try different values of crossover_func for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            crossover_funcs=valid_funcs
        )
        for crossover_func1, crossover_func2 in zip(
            wrapper.crossover_funcs, valid_funcs
        ):
            self.assertEqual(crossover_func1, crossover_func2)

        # Try invalid types for mutation_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    mutation_funcs=func
                )

        # Try a fixed value for all the mutation functions,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            mutation_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(wrapper.mutation_funcs), wrapper.num_subpops)

        # Check that all the values match
        for island_mutation_func in wrapper.mutation_funcs:
            self.assertEqual(island_mutation_func, valid_func)

        # Try different values of mutation_func for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            mutation_funcs=valid_funcs
        )
        for mutation_func1, mutation_func2 in zip(
            wrapper.mutation_funcs, valid_funcs
        ):
            self.assertEqual(mutation_func1, mutation_func2)

        # Try invalid types for selection_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    selection_funcs=func
                )

        # Try a fixed value for all the selection functions,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            selection_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(wrapper.selection_funcs), wrapper.num_subpops)

        # Check that all the values match
        for island_selection_func in wrapper.selection_funcs:
            self.assertEqual(island_selection_func, valid_func)

        # Try different values of selection_func for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            selection_funcs=valid_funcs
        )
        for selection_func1, selection_func2 in zip(
            wrapper.selection_funcs, valid_funcs
        ):
            self.assertEqual(selection_func1, selection_func2)

        # Try invalid types for crossover_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    crossover_probs=prob
                )

        # Try invalid values for crossover_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    crossover_probs=prob
                )

        # Try a fixed value for the crossover probability,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            crossover_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(wrapper.crossover_probs), wrapper.num_subpops)

        # Check that all the values match
        for island_crossover_prob in wrapper.crossover_probs:
            self.assertEqual(island_crossover_prob, valid_prob)

        # Try different values of crossover_prob for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            crossover_probs=valid_probs
        )
        for prob1, prob2 in zip(wrapper.crossover_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    mutation_probs=prob
                )

        # Try invalid values for mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(len(wrapper.mutation_probs), wrapper.num_subpops)

        # Check that all the values match
        for island_mutation_prob in wrapper.mutation_probs:
            self.assertEqual(island_mutation_prob, valid_prob)

        # Try different values of mutation_prob for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(wrapper.mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_types:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    gene_ind_mutation_probs=prob
                )

        # Try invalid values for gene_ind_mutation_probs. Should fail
        for prob in invalid_prob_values:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    gene_ind_mutation_probs=prob
                )

        # Try a fixed value for the mutation probability,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            gene_ind_mutation_probs=valid_prob
        )
        # Check the length of the sequence
        self.assertEqual(
            len(wrapper.gene_ind_mutation_probs), wrapper.num_subpops
        )

        # Check that all the values match
        for island_gene_ind_mutation_prob in wrapper.gene_ind_mutation_probs:
            self.assertEqual(island_gene_ind_mutation_prob, valid_prob)

        # Try different values of gene_ind_mutation_prob for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            gene_ind_mutation_probs=valid_probs
        )
        for prob1, prob2 in zip(wrapper.gene_ind_mutation_probs, valid_probs):
            self.assertEqual(prob1, prob2)

        # Try invalid types for selection_funcs_params. Should fail
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops,
                    selection_funcs_params=params
                )

        # Try a fixed value for the selection function parameters,
        # all islands should have the same value
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            selection_funcs_params=valid_params
        )
        # Check the length of the sequence
        self.assertEqual(
            len(wrapper.selection_funcs_params), wrapper.num_subpops
        )

        # Check that all the values match
        for island_selection_func_params in wrapper.selection_funcs_params:
            self.assertEqual(island_selection_func_params, valid_params)

        # Try different values of selection_funcs_params for each island
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops,
            selection_funcs_params=valid_funcs_params
        )
        for selection_func_params1, selection_func_params2 in zip(
            wrapper.selection_funcs_params, valid_funcs_params
        ):
            self.assertEqual(selection_func_params1, selection_func_params2)

        # Test default params
        wrapper = MyWrapper(
            valid_individual_cls,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops
        )

        # Default values for not initialized subpopulations should be None
        for pop_size in wrapper.pop_sizes:
            self.assertEqual(pop_size, None)

        for crossover_func in wrapper.crossover_funcs:
            self.assertEqual(crossover_func, None)

        for mutation_func in wrapper.mutation_funcs:
            self.assertEqual(mutation_func, None)

        for selection_func in wrapper.selection_funcs:
            self.assertEqual(selection_func, None)

        for crossover_prob in wrapper.crossover_probs:
            self.assertEqual(crossover_prob, None)

        for mutation_prob in wrapper.mutation_probs:
            self.assertEqual(mutation_prob, None)

        for gene_ind_mutation_prob in wrapper.gene_ind_mutation_probs:
            self.assertEqual(gene_ind_mutation_prob, None)

        for selection_func_params in wrapper.selection_funcs_params:
            self.assertEqual(selection_func_params, None)

        # Create the islands
        wrapper._generate_subpop_wrappers()

        # Check the default values for all the subpopulations
        for pop_size in wrapper.pop_sizes:
            self.assertEqual(pop_size, DEFAULT_POP_SIZE)

        for crossover_func in wrapper.crossover_funcs:
            self.assertEqual(crossover_func, wrapper.individual_cls.crossover)

        for mutation_func in wrapper.mutation_funcs:
            self.assertEqual(mutation_func, wrapper.individual_cls.mutate)

        for selection_func in wrapper.selection_funcs:
            self.assertEqual(selection_func, DEFAULT_NSGA_SELECTION_FUNC)

        for crossover_prob in wrapper.crossover_probs:
            self.assertEqual(crossover_prob, DEFAULT_CROSSOVER_PROB)

        for mutation_prob in wrapper.mutation_probs:
            self.assertEqual(mutation_prob, DEFAULT_MUTATION_PROB)

        for gene_ind_mutation_prob in wrapper.gene_ind_mutation_probs:
            self.assertEqual(
                gene_ind_mutation_prob, DEFAULT_GENE_IND_MUTATION_PROB
            )

        for selection_func_params in wrapper.selection_funcs_params:
            self.assertEqual(
                selection_func_params, DEFAULT_NSGA_SELECTION_FUNC_PARAMS
            )

    def test_generate_subpop_wrappers(self):
        """Test _generate_subpop_wrappers."""
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        subpop_wrapper_cls = NSGA
        nsga3_reference_points_p = 18
        num_subpops = 2
        pop_sizes = (13, 15)
        crossover_funcs = (
            individual_cls.crossover1p, individual_cls.crossover2p
        )
        mutation_funcs = (individual_cls.mutate,
                          len)
        selection_funcs = (isinstance, issubclass)
        crossover_probs = (0.33, 0.44)
        mutation_probs = (0.133, 0.144)
        gene_ind_mutation_probs = (0.1133, 0.1144)
        selection_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
        )

        wrapper = MyWrapper(
            individual_cls,
            species,
            fitness_func,
            subpop_wrapper_cls,
            num_subpops=num_subpops,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
            nsga3_reference_points_p=nsga3_reference_points_p
        )

        # Islands have not been created yet
        self.assertEqual(wrapper.subpop_wrappers, None)

        # Create the islands
        wrapper._generate_subpop_wrappers()

        # Check the islands
        self.assertIsInstance(wrapper.subpop_wrappers, list)
        self.assertEqual(len(wrapper.subpop_wrappers), num_subpops)

        for index1 in range(wrapper.num_subpops):
            for index2 in range(index1 + 1, wrapper.num_subpops):
                self.assertNotEqual(
                    id(wrapper.subpop_wrappers[index1]),
                    id(wrapper.subpop_wrappers[index2])
                )

        # Check the islands common parameters
        for island_wrapper in wrapper.subpop_wrappers:
            self.assertIsInstance(island_wrapper, subpop_wrapper_cls)

            self.assertEqual(
                island_wrapper.individual_cls, wrapper.individual_cls
            )
            self.assertEqual(island_wrapper.species, wrapper.species)
            self.assertEqual(
                island_wrapper.fitness_function, wrapper.fitness_function
            )
            self.assertEqual(island_wrapper.num_gens, wrapper.num_gens)
            self.assertEqual(
                island_wrapper.checkpoint_enable, wrapper.checkpoint_enable
            )
            self.assertEqual(
                island_wrapper.checkpoint_freq, wrapper.checkpoint_freq
            )
            self.assertEqual(island_wrapper.verbose, wrapper.verbose)
            self.assertEqual(island_wrapper.random_seed, wrapper.random_seed)
            self.assertEqual(island_wrapper.container, wrapper)
            self.assertEqual(
                island_wrapper._preprocess_generation.__name__,
                "receive_representatives"
            )
            self.assertEqual(
                island_wrapper._postprocess_generation.__name__,
                "send_representatives"
            )

            # Check the subpopulation wrapper custom params
            self.assertEqual(
                island_wrapper.nsga3_reference_points_p,
                nsga3_reference_points_p
            )

        # Check the island specific parameters
        for (
            island_wrapper,
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
            wrapper.subpop_wrappers,
            range(wrapper.num_subpops),
            wrapper.subpop_wrapper_checkpoint_filenames,
            wrapper.pop_sizes,
            wrapper.crossover_funcs,
            wrapper.mutation_funcs,
            wrapper.selection_funcs,
            wrapper.crossover_probs,
            wrapper.mutation_probs,
            wrapper.gene_ind_mutation_probs,
            wrapper.selection_funcs_params
        ):
            self.assertEqual(island_wrapper.index, island_index)
            self.assertEqual(
                island_wrapper.checkpoint_filename, island_checkpoint_filename
            )
            self.assertEqual(island_wrapper.pop_size, island_pop_size)
            self.assertEqual(
                island_wrapper.crossover_func, island_crossover_func
            )
            self.assertEqual(
                island_wrapper.mutation_func, island_mutation_func
            )
            self.assertEqual(
                island_wrapper.selection_func, island_selection_func
            )
            self.assertEqual(
                island_wrapper.crossover_prob, island_crossover_prob
            )
            self.assertEqual(
                island_wrapper.mutation_prob, island_mutation_prob
            )
            self.assertEqual(
                island_wrapper.gene_ind_mutation_prob,
                island_gene_ind_mutation_prob
            )
            self.assertEqual(
                island_wrapper.selection_func_params,
                island_selection_func_params
            )

        # Try incorrect number of pop_sizes
        wrapper.pop_sizes = pop_sizes + pop_sizes

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of pop_sizes and
        # try an incorrect number of crossover_funcs
        wrapper.pop_sizes = pop_sizes
        wrapper.crossover_funcs = crossover_funcs + crossover_funcs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of crossover_funcs and
        # try an incorrect number of mutation_funcs
        wrapper.crossover_funcs = crossover_funcs
        wrapper.mutation_funcs = mutation_funcs + mutation_funcs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of mutation_funcs and
        # try an incorrect number of selection_funcs
        wrapper.mutation_funcs = mutation_funcs
        wrapper.selection_funcs = selection_funcs + selection_funcs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of selection_funcs and
        # try an incorrect number of crossover_probs
        wrapper.selection_funcs = selection_funcs
        wrapper.crossover_probs = crossover_probs + crossover_probs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of crossover_probs and
        # try an incorrect number of mutation_probs
        wrapper.crossover_probs = crossover_probs
        wrapper.mutation_probs = mutation_probs + mutation_probs

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of mutation_probs and
        # try an incorrect number of gene_ind_mutation_probs
        wrapper.mutation_probs = mutation_probs
        wrapper.gene_ind_mutation_probs = gene_ind_mutation_probs * 2

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of gene_ind_mutation_probs and
        # try an incorrect number of selection_funcs_params
        wrapper.gene_ind_mutation_probs = gene_ind_mutation_probs
        wrapper.selection_funcs_params = selection_funcs_params * 2

        # Create the islands. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()


if __name__ == '__main__':
    unittest.main()
