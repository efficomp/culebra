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

"""Test for :py:class:`wrapper.multi_pop.HeterogeneousSequentialIslands`."""

import unittest
from culebra.base import (
    Dataset,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME
)
from culebra.fitness_function.feature_selection import KappaNumFeats as Fitness
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import (
    BitVector as Individual
)
from culebra.wrapper import DEFAULT_NUM_GENS
from culebra.wrapper.single_pop import (
    NSGA,
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_NSGA_SELECTION_FUNC,
    DEFAULT_NSGA_SELECTION_FUNC_PARAMS
)
from culebra.wrapper.multi_pop import (
    HeterogeneousSequentialIslands as Wrapper,
    DEFAULT_NUM_SUBPOPS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
)


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.HeterogeneousSequentialIslands`."""

    def test_init(self):
        """Test __init__."""
        individual_cls = Individual
        species = Species(dataset.num_feats)
        fitness_function = Fitness(dataset)
        subpop_wrapper_cls = NSGA
        num_gens = 25
        num_subpops = 3
        pop_sizes = (1234, 2345, 3456)
        crossover_funcs = (abs, all, any)
        mutation_funcs = (bin, bool, bytes)
        selection_funcs = (hash, help, hex)
        crossover_probs = (0.36, 0.46, 0.56)
        mutation_probs = (0.123, 0.133, 0.143)
        gene_ind_mutation_probs = (0.027, 0.033, 0.045)
        selection_funcs_params = (
            {"parameter1": 65},
            {"parameter2": 68},
            {"parameter3": 99})
        representation_size = 3
        representation_freq = 27
        representation_topology_func = max
        representation_topology_func_params = {"parameter2": 45}
        representation_selection_func = min
        representation_selection_func_params = {"parameter3": 15}
        checkpoint_enable = False
        checkpoint_freq = 17
        checkpoint_filename = "my_check_file.gz"
        verbose = False
        random_seed = 149
        nsga3_reference_points_p = 18

        # Test custom params
        wrapper = Wrapper(
            individual_cls,
            species,
            fitness_function,
            subpop_wrapper_cls,
            num_gens=num_gens,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
            num_subpops=num_subpops,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            nsga3_reference_points_p=nsga3_reference_points_p
        )

        self.assertEqual(wrapper.individual_cls, individual_cls)
        self.assertEqual(wrapper.species, species)
        self.assertEqual(wrapper.fitness_function, fitness_function)
        self.assertEqual(wrapper.subpop_wrapper_cls, subpop_wrapper_cls)
        self.assertEqual(wrapper.num_gens, num_gens)
        self.assertEqual(wrapper.num_subpops, num_subpops)
        self.assertEqual(wrapper.representation_size, representation_size)
        self.assertEqual(wrapper.representation_freq, representation_freq)
        self.assertEqual(
            wrapper.representation_topology_func, representation_topology_func
        )
        self.assertEqual(
            wrapper.representation_topology_func_params,
            representation_topology_func_params
        )
        self.assertEqual(
            wrapper.representation_selection_func,
            representation_selection_func
        )
        self.assertEqual(
            wrapper.representation_selection_func_params,
            representation_selection_func_params
        )
        self.assertEqual(wrapper.checkpoint_enable, checkpoint_enable)
        self.assertEqual(wrapper.checkpoint_freq, checkpoint_freq)
        self.assertEqual(wrapper.checkpoint_filename, checkpoint_filename)
        self.assertEqual(wrapper.verbose, verbose)
        self.assertEqual(wrapper.random_seed, random_seed)
        self.assertEqual(
            wrapper.subpop_wrapper_params["nsga3_reference_points_p"],
            nsga3_reference_points_p)

        for index in range(wrapper.num_subpops):
            self.assertEqual(
                wrapper.pop_sizes[index], pop_sizes[index]
            )
            self.assertEqual(
                wrapper.crossover_funcs[index], crossover_funcs[index]
            )
            self.assertEqual(
                wrapper.mutation_funcs[index], mutation_funcs[index]
            )
            self.assertEqual(
                wrapper.selection_funcs[index], selection_funcs[index]
            )
            self.assertEqual(
                wrapper.crossover_probs[index], crossover_probs[index]
            )
            self.assertEqual(
                wrapper.mutation_probs[index], mutation_probs[index]
            )
            self.assertEqual(
                wrapper.gene_ind_mutation_probs[index],
                gene_ind_mutation_probs[index]
            )
            self.assertEqual(
                wrapper.selection_funcs_params[index],
                selection_funcs_params[index]
            )

        # Test default params
        wrapper = Wrapper(
            individual_cls,
            species,
            fitness_function,
            subpop_wrapper_cls,
        )

        self.assertEqual(wrapper.individual_cls, individual_cls)
        self.assertEqual(wrapper.species, species)
        self.assertEqual(wrapper.fitness_function, fitness_function)
        self.assertEqual(wrapper.subpop_wrapper_cls, subpop_wrapper_cls)

        self.assertEqual(wrapper.num_gens, DEFAULT_NUM_GENS)
        self.assertEqual(wrapper.num_subpops, DEFAULT_NUM_SUBPOPS)
        self.assertEqual(
            wrapper.representation_size, DEFAULT_REPRESENTATION_SIZE
        )
        self.assertEqual(
            wrapper.representation_freq, DEFAULT_REPRESENTATION_FREQ
        )
        self.assertEqual(
            wrapper.representation_topology_func,
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
        )
        self.assertEqual(
            wrapper.representation_topology_func_params,
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
        )
        self.assertEqual(
            wrapper.representation_selection_func,
            DEFAULT_REPRESENTATION_SELECTION_FUNC
        )
        self.assertEqual(
            wrapper.representation_selection_func_params,
            DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
        )
        self.assertEqual(wrapper.checkpoint_enable, True)
        self.assertEqual(
            wrapper.checkpoint_freq, DEFAULT_CHECKPOINT_FREQ
        )
        self.assertEqual(
            wrapper.checkpoint_filename, DEFAULT_CHECKPOINT_FILENAME
        )
        self.assertEqual(wrapper.verbose, __debug__)
        self.assertEqual(wrapper.random_seed, None)
        self.assertEqual(wrapper.subpop_wrapper_params, {})

        # Create the islands
        wrapper._generate_subpop_wrappers()

        # Check the default values for all the subpopulations
        for index in range(wrapper.num_subpops):
            self.assertEqual(
                wrapper.pop_sizes[index], DEFAULT_POP_SIZE
            )
            self.assertEqual(
                wrapper.crossover_funcs[index],
                wrapper.individual_cls.crossover
            )
            self.assertEqual(
                wrapper.mutation_funcs[index], wrapper.individual_cls.mutate
            )
            self.assertEqual(
                wrapper.selection_funcs[index], DEFAULT_NSGA_SELECTION_FUNC
            )
            self.assertEqual(
                wrapper.crossover_probs[index], DEFAULT_CROSSOVER_PROB
            )
            self.assertEqual(
                wrapper.mutation_probs[index], DEFAULT_MUTATION_PROB
            )
            self.assertEqual(
                wrapper.gene_ind_mutation_probs[index],
                DEFAULT_GENE_IND_MUTATION_PROB
            )
            self.assertEqual(
                wrapper.selection_funcs_params[index],
                DEFAULT_NSGA_SELECTION_FUNC_PARAMS
            )


if __name__ == '__main__':
    unittest.main()
