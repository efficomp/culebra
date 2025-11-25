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

"""Unit test for :class:`~culebra.trainer.ea.abc.HomogeneousIslandsEA`."""

import unittest

from culebra import DEFAULT_MAX_NUM_ITERS
from culebra.trainer.ea import (
    DEFAULT_POP_SIZE,
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
    """Test :class:`~culebra.trainer.ea.abc.HomogeneousIslandsEA`."""

    def test_init(self):
        """Test the constructor."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = MySinglePopEA

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
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
                    subtrainer_cls,
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
                    subtrainer_cls,
                    crossover_func=func
                )

        # Try invalid types for mutation_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
                    mutation_func=func
                )

        # Try invalid types for selection_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
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
                    subtrainer_cls,
                    crossover_prob=prob
                )

        # Try invalid types for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
                    mutation_prob=prob
                )

        # Try invalid types for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
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
                    subtrainer_cls,
                    crossover_prob=prob
                )

        # Try invalid values for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
                    mutation_prob=prob
                )

        # Try invalid values for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyIslandsEA(
                    solution_cls,
                    species,
                    fitness_func,
                    subtrainer_cls,
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
                    subtrainer_cls,
                    selection_func_params=params
                )

        # Test default params
        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subtrainer_cls
        )

        self.assertEqual(trainer.max_num_iters, DEFAULT_MAX_NUM_ITERS)

        # All these parameters should be None if subtrainers are not generated
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

        # Now the default values of subtrainers should be returned
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

    def test_generate_subtrainers(self):
        """Test _generate_subtrainers."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = MySinglePopEA
        num_subtrainers = 2

        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subtrainer_cls,
            num_subtrainers=num_subtrainers
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

        # Check the islands parameters
        for island_trainer in trainer.subtrainers:
            self.assertIsInstance(island_trainer, subtrainer_cls)

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
                trainer.subtrainers,
                trainer.subtrainer_checkpoint_filenames
            )
        ):
            self.assertEqual(island_trainer.index, island_index)
            self.assertEqual(
                island_trainer.checkpoint_filename,
                island_trainer_checkpoint_filename
            )

    def test_repr(self):
        """Test the repr and str dunder methods."""
        solution_cls = Individual
        species = Species(dataset.num_feats)
        fitness_func = KappaNumFeats(dataset)
        subtrainer_cls = MySinglePopEA
        num_subtrainers = 2

        trainer = MyIslandsEA(
            solution_cls,
            species,
            fitness_func,
            subtrainer_cls,
            num_subtrainers=num_subtrainers
        )
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
