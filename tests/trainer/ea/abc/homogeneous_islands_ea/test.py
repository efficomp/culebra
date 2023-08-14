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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.HomogeneousIslandsEA`."""

import unittest

from culebra import DEFAULT_MAX_NUM_ITERS, DEFAULT_POP_SIZE
from culebra.trainer.ea import (
    DEFAULT_SELECTION_FUNC,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC_PARAMS
)
from culebra.trainer.ea.abc import SinglePopEA, HomogeneousIslandsEA
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


class MyIslandsEA(HomogeneousIslandsEA):
    """Dummy implementation of an island-based evolutionary algorithm."""

    def _search(self):
        pass


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.abc.HomogeneousIslandsEA`."""

    def test_init(self):
        """Test the constructor."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        subpop_trainer_cls = MySinglePopEA

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    pop_size=pop_size
                )

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    pop_size=pop_size
                )

        # Try invalid types for crossover_func. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    crossover_func=func
                )

        # Try invalid types for mutation_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    mutation_func=func
                )

        # Try invalid types for selection_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    selection_func=func
                )

        # Try invalid types for crossover_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    crossover_prob=prob
                )

        # Try invalid types for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    mutation_prob=prob
                )

        # Try invalid types for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    crossover_prob=prob
                )

        # Try invalid values for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    mutation_prob=prob
                )

        # Try invalid values for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid types for selection_func_params. Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subpop_trainer_cls,
                    selection_func_params=params
                )

        # Test default params
        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subpop_trainer_cls
        )
        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)
        self.assertEqual(trainer.pop_size, DEFAULT_POP_SIZE)
        self.assertEqual(
            trainer.crossover_func, trainer.solution_cls.crossover
        )
        self.assertEqual(trainer.mutation_func, trainer.solution_cls.mutate)
        self.assertEqual(trainer.selection_func, DEFAULT_SELECTION_FUNC)
        self.assertEqual(trainer.crossover_prob, DEFAULT_CROSSOVER_PROB)
        self.assertEqual(trainer.mutation_prob, DEFAULT_MUTATION_PROB)
        self.assertEqual(
            trainer.gene_ind_mutation_prob, DEFAULT_GENE_IND_MUTATION_PROB
        )
        self.assertEqual(
            trainer.selection_func_params, DEFAULT_SELECTION_FUNC_PARAMS
        )

    def test_generate_subpop_trainers(self):
        """Test _generate_subpop_trainers."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = Fitness(dataset)
        subpop_trainer_cls = MySinglePopEA
        num_subpops = 2

        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subpop_trainer_cls,
            num_subpops=num_subpops
        )

        # Islands have not been created yet
        self.assertEqual(trainer.subpop_trainers, None)

        # Create the islands
        trainer._generate_subpop_trainers()

        # Check the islands
        self.assertIsInstance(trainer.subpop_trainers, list)
        self.assertEqual(len(trainer.subpop_trainers), num_subpops)

        for index1 in range(trainer.num_subpops):
            for index2 in range(index1 + 1, trainer.num_subpops):
                self.assertNotEqual(
                    id(trainer.subpop_trainers[index1]),
                    id(trainer.subpop_trainers[index2])
                )

        # Check the islands parameters
        for island_trainer in trainer.subpop_trainers:
            self.assertIsInstance(island_trainer, subpop_trainer_cls)

            self.assertEqual(
                island_trainer.solution_cls,
                trainer.solution_cls
            )
            self.assertEqual(island_trainer.species, trainer.species)
            self.assertEqual(
                island_trainer.fitness_function,
                trainer.fitness_function
            )
            self.assertEqual(
                island_trainer.max_num_iters,
                trainer.max_num_iters
            )
            self.assertEqual(island_trainer.pop_size, trainer.pop_size)
            self.assertEqual(
                island_trainer.crossover_func,
                trainer.crossover_func
            )
            self.assertEqual(
                island_trainer.mutation_func,
                trainer.mutation_func
            )
            self.assertEqual(
                island_trainer.selection_func,
                trainer.selection_func
            )
            self.assertEqual(
                island_trainer.crossover_prob,
                trainer.crossover_prob
            )
            self.assertEqual(
                island_trainer.mutation_prob,
                trainer.mutation_prob
            )
            self.assertEqual(
                island_trainer.gene_ind_mutation_prob,
                trainer.gene_ind_mutation_prob
            )
            self.assertEqual(
                island_trainer.selection_func_params,
                trainer.selection_func_params
            )
            self.assertEqual(
                island_trainer.checkpoint_enable, trainer.checkpoint_enable
            )
            self.assertEqual(
                island_trainer.checkpoint_freq,
                trainer.checkpoint_freq
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

        for (
            island_index, (
                island_trainer,
                island_trainer_checkpoint_filename
            )
        ) in enumerate(
            zip(
                trainer.subpop_trainers,
                trainer.subpop_trainer_checkpoint_filenames
            )
        ):
            self.assertEqual(island_trainer.index, island_index)
            self.assertEqual(
                island_trainer.checkpoint_filename,
                island_trainer_checkpoint_filename
            )


if __name__ == '__main__':
    unittest.main()
