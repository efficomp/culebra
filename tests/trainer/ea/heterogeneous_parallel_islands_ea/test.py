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

"""Test :class:`~culebra.trainer.ea.HeterogeneousParallelIslandsEA`."""

import unittest
from multiprocess import cpu_count

from culebra import (
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_MAX_NUM_ITERS,
    SERIALIZED_FILE_EXTENSION
)
from culebra.trainer import (
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
    HeterogeneousParallelIslandsEA as Trainer,
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
from culebra.fitness_function import MultiObjectiveFitnessFunction
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats
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
    return MultiObjectiveFitnessFunction(
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


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.trainer.ea.HeterogeneousParallelIslandsEA`."""

    def test_init(self):
        """Test __init__."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_function = KappaNumFeats(dataset)
        subtrainer_cls = NSGA
        max_num_iters = 25
        num_subtrainers = 3
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
        checkpoint_filename = "my_check_file" + SERIALIZED_FILE_EXTENSION
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
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
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

        for index in range(trainer.num_subtrainers):
            self.assertEqual(
                trainer.pop_sizes[index], pop_sizes[index]
            )
            self.assertEqual(
                trainer.crossover_funcs[index], crossover_funcs[index]
            )
            self.assertEqual(
                trainer.mutation_funcs[index], mutation_funcs[index]
            )
            self.assertEqual(
                trainer.selection_funcs[index], selection_funcs[index]
            )
            self.assertEqual(
                trainer.crossover_probs[index], crossover_probs[index]
            )
            self.assertEqual(
                trainer.mutation_probs[index], mutation_probs[index]
            )
            self.assertEqual(
                trainer.gene_ind_mutation_probs[index],
                gene_ind_mutation_probs[index]
            )
            self.assertEqual(
                trainer.selection_funcs_params[index],
                selection_funcs_params[index]
            )

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

        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer.num_subtrainers, cpu_count())
        self.assertEqual(
            trainer.representation_size, DEFAULT_REPRESENTATION_SIZE
        )
        self.assertEqual(
            trainer.representation_freq, DEFAULT_REPRESENTATION_FREQ
        )
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

        # Create the islands
        trainer._generate_subtrainers()

        # Check the default values for all the subpopulations
        for index in range(trainer.num_subtrainers):
            self.assertEqual(
                trainer.pop_sizes[index], DEFAULT_POP_SIZE
            )
            self.assertEqual(
                trainer.crossover_funcs[index],
                trainer.solution_cls.crossover
            )
            self.assertEqual(
                trainer.mutation_funcs[index], trainer.solution_cls.mutate
            )
            self.assertEqual(
                trainer.selection_funcs[index], DEFAULT_NSGA_SELECTION_FUNC
            )
            self.assertEqual(
                trainer.crossover_probs[index], DEFAULT_CROSSOVER_PROB
            )
            self.assertEqual(
                trainer.mutation_probs[index], DEFAULT_MUTATION_PROB
            )
            self.assertEqual(
                trainer.gene_ind_mutation_probs[index],
                DEFAULT_GENE_IND_MUTATION_PROB
            )
            self.assertEqual(
                trainer.selection_funcs_params[index],
                DEFAULT_NSGA_SELECTION_FUNC_PARAMS
            )

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_function = KappaNumFeats(dataset)
        subtrainer_cls = NSGA
        max_num_iters = 25
        num_subtrainers = 3
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
        checkpoint_filename = "my_check_file" + SERIALIZED_FILE_EXTENSION
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
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
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
