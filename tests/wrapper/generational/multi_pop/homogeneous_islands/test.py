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

"""Unit test for :py:class:`wrapper.multi_pop.HomogeneousIslands`."""

import unittest
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper import DEFAULT_NUM_GENS
from culebra.wrapper.single_pop import (
    NSGA,
    DEFAULT_POP_SIZE,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC_PARAMS
)
from culebra.wrapper.multi_pop import HomogeneousIslands


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyWrapper(HomogeneousIslands):
    """Dummy implementation of a wrapper method."""

    def _search(self):
        pass


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.HomogeneousIslands`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.multi_pop.HomogeneousIslands.__init__`."""
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        subpop_wrapper_cls = NSGA

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    pop_size=pop_size
                )

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    pop_size=pop_size
                )

        # Try invalid types for crossover_func. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    crossover_func=func
                )

        # Try invalid types for mutation_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    mutation_func=func
                )

        # Try invalid types for selection_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    selection_func=func
                )

        # Try invalid types for crossover_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    crossover_prob=prob
                )

        # Try invalid types for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    mutation_prob=prob
                )

        # Try invalid types for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    crossover_prob=prob
                )

        # Try invalid values for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    mutation_prob=prob
                )

        # Try invalid values for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid types for selection_func_params. Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    species,
                    fitness_func,
                    subpop_wrapper_cls,
                    selection_func_params=params
                )

        # Test default params
        wrapper = MyWrapper(
            individual_cls,
            species,
            fitness_func,
            subpop_wrapper_cls
        )
        self.assertEqual(wrapper.num_gens, DEFAULT_NUM_GENS)
        self.assertEqual(wrapper.pop_size, DEFAULT_POP_SIZE)
        self.assertEqual(
            wrapper.crossover_func, wrapper.individual_cls.crossover
        )
        self.assertEqual(wrapper.mutation_func, wrapper.individual_cls.mutate)
        self.assertEqual(wrapper.selection_func, DEFAULT_SELECTION_FUNC)
        self.assertEqual(wrapper.crossover_prob, DEFAULT_CROSSOVER_PROB)
        self.assertEqual(wrapper.mutation_prob, DEFAULT_MUTATION_PROB)
        self.assertEqual(
            wrapper.gene_ind_mutation_prob, DEFAULT_GENE_IND_MUTATION_PROB
        )
        self.assertEqual(
            wrapper.selection_func_params, DEFAULT_SELECTION_FUNC_PARAMS
        )

    def test_generate_subpop_wrappers(self):
        """Test _generate_subpop_wrappers."""
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        subpop_wrapper_cls = NSGA
        nsga3_reference_points_p = 18
        num_subpops = 2

        wrapper = MyWrapper(
            individual_cls,
            species,
            fitness_func,
            subpop_wrapper_cls,
            num_subpops=num_subpops,
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

        # Check the islands parameters
        for island_wrapper in wrapper.subpop_wrappers:
            self.assertIsInstance(island_wrapper, subpop_wrapper_cls)

            self.assertEqual(
                island_wrapper.individual_cls,
                wrapper.individual_cls
            )
            self.assertEqual(island_wrapper.species, wrapper.species)
            self.assertEqual(
                island_wrapper.fitness_function,
                wrapper.fitness_function
            )
            self.assertEqual(island_wrapper.num_gens, wrapper.num_gens)
            self.assertEqual(island_wrapper.pop_size, wrapper.pop_size)
            self.assertEqual(
                island_wrapper.crossover_func,
                wrapper.crossover_func
            )
            self.assertEqual(
                island_wrapper.mutation_func,
                wrapper.mutation_func
            )
            self.assertEqual(
                island_wrapper.selection_func,
                wrapper.selection_func
            )
            self.assertEqual(
                island_wrapper.crossover_prob,
                wrapper.crossover_prob
            )
            self.assertEqual(
                island_wrapper.mutation_prob,
                wrapper.mutation_prob
            )
            self.assertEqual(
                island_wrapper.gene_ind_mutation_prob,
                wrapper.gene_ind_mutation_prob
            )
            self.assertEqual(
                island_wrapper.selection_func_params,
                wrapper.selection_func_params
            )
            self.assertEqual(
                island_wrapper.checkpoint_enable, wrapper.checkpoint_enable
            )
            self.assertEqual(
                island_wrapper.checkpoint_freq,
                wrapper.checkpoint_freq
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

        for (
            island_index, (
                island_wrapper,
                island_wrapper_checkpoint_filename
            )
        ) in enumerate(
            zip(
                wrapper.subpop_wrappers,
                wrapper.subpop_wrapper_checkpoint_filenames
            )
        ):
            self.assertEqual(island_wrapper.index, island_index)
            self.assertEqual(
                island_wrapper.checkpoint_filename,
                island_wrapper_checkpoint_filename
            )


if __name__ == '__main__':
    unittest.main()
