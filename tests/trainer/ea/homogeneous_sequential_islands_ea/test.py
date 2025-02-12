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

"""Test for :py:class:`~culebra.trainer.ea.HomogeneousSequentialIslandsEA`."""

import unittest

from culebra import (
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_MAX_NUM_ITERS
)
from culebra.trainer import (
    DEFAULT_NUM_SUBTRAINERS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
)
from culebra.trainer.ea import (
    DEFAULT_POP_SIZE,
    NSGA,
    HomogeneousSequentialIslandsEA as Trainer,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_NSGA_SELECTION_FUNC,
    DEFAULT_NSGA_SELECTION_FUNC_PARAMS
)
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


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.HomogeneousSequentialIslandsEA`."""

    def test_init(self):
        """Test __init__."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_function = Fitness(dataset)
        subtrainer_cls = NSGA
        max_num_iters = 25
        pop_size = 1234
        crossover_func = len
        mutation_func = isinstance
        selection_func = issubclass
        crossover_prob = 0.36
        mutation_prob = 0.123
        gene_ind_mutation_prob = 0.027
        selection_func_params = {"parameter1": 65}
        num_subtrainers = 7
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
        trainer = Trainer(
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            max_num_iters=max_num_iters,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
            num_subtrainers=num_subtrainers,
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

        self.assertEqual(trainer.solution_cls, solution_cls)
        self.assertEqual(trainer.species, species)
        self.assertEqual(trainer.fitness_function, fitness_function)
        self.assertEqual(trainer.subtrainer_cls, subtrainer_cls)
        self.assertEqual(trainer.max_num_iters, max_num_iters)
        self.assertEqual(trainer.pop_size, pop_size)
        self.assertEqual(trainer.crossover_func, crossover_func)
        self.assertEqual(trainer.mutation_func, mutation_func)
        self.assertEqual(trainer.selection_func, selection_func)
        self.assertEqual(trainer.crossover_prob, crossover_prob)
        self.assertEqual(trainer.mutation_prob, mutation_prob)
        self.assertEqual(
            trainer.gene_ind_mutation_prob, gene_ind_mutation_prob
        )
        self.assertEqual(trainer.selection_func_params, selection_func_params)
        self.assertEqual(trainer.num_subtrainers, num_subtrainers)
        self.assertEqual(trainer.representation_size, representation_size)
        self.assertEqual(trainer.representation_freq, representation_freq)
        self.assertEqual(
            trainer.representation_topology_func, representation_topology_func
        )
        self.assertEqual(
            trainer.representation_topology_func_params,
            representation_topology_func_params
        )
        self.assertEqual(
            trainer.representation_selection_func,
            representation_selection_func
        )
        self.assertEqual(
            trainer.representation_selection_func_params,
            representation_selection_func_params
        )
        self.assertEqual(trainer.checkpoint_enable, checkpoint_enable)
        self.assertEqual(trainer.checkpoint_freq, checkpoint_freq)
        self.assertEqual(trainer.checkpoint_filename, checkpoint_filename)
        self.assertEqual(trainer.verbose, verbose)
        self.assertEqual(trainer.random_seed, random_seed)
        self.assertEqual(
            trainer.subtrainer_params["nsga3_reference_points_p"],
            nsga3_reference_points_p)

        # Test default params
        trainer = Trainer(
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls
        )

        self.assertEqual(trainer.solution_cls, solution_cls)
        self.assertEqual(trainer.species, species)
        self.assertEqual(trainer.fitness_function, fitness_function)
        self.assertEqual(trainer.subtrainer_cls, subtrainer_cls)

        # Check the defaults that not depend on subtrainers
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer.num_subtrainers, DEFAULT_NUM_SUBTRAINERS)
        self.assertEqual(
            trainer.representation_size, DEFAULT_REPRESENTATION_SIZE
        )
        self.assertEqual(trainer.representation_freq,
                         DEFAULT_REPRESENTATION_FREQ)
        self.assertEqual(
            trainer.representation_topology_func,
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
        )
        self.assertEqual(
            trainer.representation_topology_func_params,
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
        )
        self.assertEqual(
            trainer.representation_selection_func,
            DEFAULT_REPRESENTATION_SELECTION_FUNC
        )
        self.assertEqual(
            trainer.representation_selection_func_params,
            DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
        )
        self.assertEqual(trainer.checkpoint_enable, True)
        self.assertEqual(
            trainer.checkpoint_freq, DEFAULT_CHECKPOINT_FREQ
        )
        self.assertEqual(
            trainer.checkpoint_filename, DEFAULT_CHECKPOINT_FILENAME
        )
        self.assertEqual(trainer.verbose, __debug__)
        self.assertEqual(trainer.random_seed, None)
        self.assertEqual(trainer.subtrainer_params, {})

        # The default values are only returned once the subtrainers have
        # been generated. Thus they should be None by the moment
        self.assertEqual(trainer.pop_size, None)
        self.assertEqual(trainer.crossover_func, None)
        self.assertEqual(trainer.mutation_func, None)
        self.assertEqual(trainer.selection_func, None)
        self.assertEqual(trainer.crossover_prob, None)
        self.assertEqual(trainer.mutation_prob, None)
        self.assertEqual(trainer.gene_ind_mutation_prob, None)
        self.assertEqual(trainer.selection_func_params, None)

        # Generate the subtrainers
        trainer._generate_subtrainers()

        # Now the trainer should return the default values of the subtrainers
        self.assertEqual(trainer.pop_size, DEFAULT_POP_SIZE)
        self.assertEqual(
            trainer.crossover_func, trainer.solution_cls.crossover
        )
        self.assertEqual(trainer.mutation_func, trainer.solution_cls.mutate)
        self.assertEqual(trainer.selection_func, DEFAULT_NSGA_SELECTION_FUNC)
        self.assertEqual(trainer.crossover_prob, DEFAULT_CROSSOVER_PROB)
        self.assertEqual(trainer.mutation_prob, DEFAULT_MUTATION_PROB)
        self.assertEqual(
            trainer.gene_ind_mutation_prob, DEFAULT_GENE_IND_MUTATION_PROB
        )
        self.assertEqual(
            trainer.selection_func_params, DEFAULT_NSGA_SELECTION_FUNC_PARAMS
        )

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_function = Fitness(dataset)
        subtrainer_cls = NSGA
        max_num_iters = 25
        pop_size = 1234
        crossover_func = len
        mutation_func = isinstance
        selection_func = issubclass
        crossover_prob = 0.36
        mutation_prob = 0.123
        gene_ind_mutation_prob = 0.027
        selection_func_params = {"parameter1": 65}
        num_subtrainers = 7
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
        trainer = Trainer(
            solution_cls,
            species,
            fitness_function,
            subtrainer_cls,
            max_num_iters=max_num_iters,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
            num_subtrainers=num_subtrainers,
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
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
