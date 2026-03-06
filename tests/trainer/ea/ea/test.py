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

"""Unit test for :class:`culebra.trainer.ea.EA`."""

import unittest

from deap.base import Toolbox

from culebra.trainer.abc import IslandsTrainer
from culebra.trainer.ea import (
    EA,
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC
)

from culebra.solution.feature_selection import (
    Species,
    Solution,
    BitVector as Individual
)
from culebra.fitness_func import MultiObjectiveFitnessFunction
from culebra.fitness_func.feature_selection import (
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
    """Test :class:`culebra.trainer.ea.EA`."""

    def test_init(self):
        """Test __init__."""
        # Test the superclass initialization
        valid_solution_cls = Individual
        valid_species = Species(dataset.num_feats)

        # Try invalid individual classes. Should fail
        invalid_individual_classes = (type, None, 'a', 1, Solution)
        for solution_cls in invalid_individual_classes:
            with self.assertRaises(TypeError):
                EA(KappaNumFeats(dataset), solution_cls, valid_species)

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                EA(KappaNumFeats(dataset), valid_solution_cls, species)

        # Test initialization
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": valid_solution_cls,
            "species": valid_species
        }
        trainer = EA(**params)
        self.assertEqual(trainer._toolbox, None)

    # Try invalid types for max_num_iters. Should fail
        invalid_max_num_iters = (type, 'a', 1.5)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(TypeError):
                EA(max_num_iters=max_num_iters, **params)

        # Try invalid values for max_num_iters. Should fail
        invalid_max_num_iters = (-1, 0)
        for max_num_iters in invalid_max_num_iters:
            with self.assertRaises(ValueError):
                EA(max_num_iters=max_num_iters, **params)

        # Try invalid types for crossover_func. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                EA(crossover_func=func, **params)

        # Try invalid types for mutation_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                EA(mutation_func=func, **params)

        # Try invalid types for selection_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                EA(selection_func=func, **params)

        # Try invalid types for crossover_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                EA(crossover_prob=prob, **params)

        # Try invalid types for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                EA(mutation_prob=prob, **params)

        # Try invalid types for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                EA(gene_ind_mutation_prob=prob, **params)

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                EA(crossover_prob=prob, **params)

        # Try invalid values for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                EA(mutation_prob=prob, **params)

        # Try invalid values for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                EA(gene_ind_mutation_prob=prob, **params)

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                EA(pop_size=pop_size, **params)

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                EA(pop_size=pop_size, **params)

        # Test default params
        trainer = EA(**params)
        self.assertEqual(trainer.fitness_func, params["fitness_func"])
        self.assertEqual(trainer.solution_cls, params["solution_cls"])
        self.assertEqual(trainer.species, params["species"])
        self.assertEqual(
            trainer.crossover_func, trainer.solution_cls.crossover)
        self.assertEqual(trainer.mutation_func, trainer.solution_cls.mutate)
        self.assertEqual(trainer.selection_func, DEFAULT_SELECTION_FUNC)
        self.assertEqual(trainer.crossover_prob, DEFAULT_CROSSOVER_PROB)
        self.assertEqual(trainer.mutation_prob, DEFAULT_MUTATION_PROB)
        self.assertEqual(trainer.gene_ind_mutation_prob,
                         DEFAULT_GENE_IND_MUTATION_PROB)
        self.assertEqual(trainer.pop_size, DEFAULT_POP_SIZE)
        self.assertEqual(trainer.custom_termination_func, None)
        self.assertEqual(trainer.pop, None)
        self.assertEqual(trainer.current_iter, None)

    def test_init_internals(self):
        """Test _init_internals."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats)
        }

        trainer = EA(**params)

        # Init the internals
        trainer._init_internals()
        self.assertIsInstance(trainer._toolbox, Toolbox)
        self.assertEqual(trainer._toolbox.mate.func, trainer.crossover_func)
        self.assertEqual(trainer._toolbox.mutate.func, trainer.mutation_func)
        self.assertEqual(trainer._toolbox.select.func, trainer.selection_func)

    def test_reset_internals(self):
        """Test _reset_internals."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats)
        }
        trainer = EA(**params)

        # Init the internals
        trainer._init_internals()

        # Reset the internals
        trainer._reset_internals()
        self.assertEqual(trainer._toolbox, None)

    def test_new_state(self):
        """Test _new_state."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "verbosity": False
        }
        trainer = EA(**params)

        # Create a new state
        trainer._init_internals()
        trainer._new_state()

        # Check the pop
        self.assertIsInstance(trainer.pop, list)
        self.assertEqual(len(trainer.pop), trainer.pop_size)

        # Check the individuals in the population
        for ind in trainer.pop:
            self.assertIsInstance(ind, trainer.solution_cls)
            self.assertTrue(ind.fitness.is_valid)

        self.assertEqual(trainer.current_iter, 0)
        self.assertEqual(len(trainer.logbook), 1)

    def test_reset_state(self):
        """Test the _reset_state method."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats)
        }
        trainer = EA(**params)

        # Reset the state
        trainer._reset_state()

        # Check the population
        self.assertIsNone(trainer.pop)
        self.assertIsNone(trainer.logbook)

    def test_get_set_state(self):
        """Test _get_state and _set_state."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats)
        }
        trainer1 = EA(**params)

        # Set state attributes to dummy values
        trainer1._pop = [
            Individual(
                params["species"],
                params["fitness_func"].fitness_cls
            ) for _ in range(trainer1.pop_size)
        ]

        trainer1._current_iter = 10

        # Create another trainer
        trainer2 = EA(**params)

        # Check that state attributes in trainer1 are not default values
        self.assertNotEqual(trainer1.pop, None)
        self.assertNotEqual(trainer1.current_iter, None)

        # Check that state attributes in trainer2 are defaults
        self.assertEqual(trainer2.pop, None)
        self.assertEqual(trainer2.current_iter, None)

        # Set the state of trainer2 to that of trainer1
        trainer2._set_state(trainer1._get_state())

        # Check that the state attributes of trainer2 are equal to those of
        # trainer1
        self.assertEqual(trainer1.current_iter, trainer2.current_iter)
        self.assertEqual(len(trainer1.pop), len(trainer2.pop))
        for ind1, ind2 in zip(trainer1.pop, trainer2.pop):
            self.assertEqual(ind1, ind2)

    def test_generate_pop(self):
        """Test the generation of a population."""
        # Check a single population
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }
        trainer = EA(**params)

        self.assertIsNone(trainer.pop)

        trainer._generate_pop()

        self.assertEqual(len(trainer.pop), trainer.pop_size)
        for ind in trainer.pop:
            self.assertIsInstance(ind, Individual)

    def test_evaluate_several(self):
        """Test the evaluation of a population."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }
        trainer = EA(**params)

        # Init the search
        trainer._init_training()

        # Generate some individuals
        sol1 = Individual(
            params["species"],
            params["fitness_func"].fitness_cls,
            (1, 2)
        )

        sol2 = Individual(
            params["species"],
            params["fitness_func"].fitness_cls,
            (0, 2, 3)
        )

        sol3 = Individual(
            params["species"],
            params["fitness_func"].fitness_cls,
            (1, 2, 3)
        )

        # Init the training
        trainer._init_training()

        inds = [sol1, sol2, sol3]

        for ind in inds:
            self.assertFalse(ind.fitness.is_valid)

        # Evaluate the solutions
        trainer._evaluate_several(inds)

        for ind in inds:
            self.assertTrue(ind.fitness.is_valid)

    def test_best_solutions(self):
        """Test the best_solutions method."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }
        trainer = EA(**params)

        # Try before the population has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, tuple)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Generate some individuals
        sol1 = Individual(
            params["species"],
            params["fitness_func"].fitness_cls,
            (1, 2)
        )

        sol2 = Individual(
            params["species"],
            params["fitness_func"].fitness_cls,
            (0, 2, 3)
        )

        sol3 = Individual(
            params["species"],
            params["fitness_func"].fitness_cls,
            (1, 2, 3)
        )

        # Init the training
        trainer._init_training()

        # Evaluate the solutions
        trainer._evaluate_several([sol1, sol2, sol3])

        # Try a population with different number of features
        trainer._pop = [sol1, sol2]

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that both solutions are in the hof
        self.assertEqual(len(best_ones[0]), 2)
        self.assertTrue(sol1 in best_ones[0])
        self.assertTrue(sol2 in best_ones[0])

        # Try a population with the same number of features
        trainer._pop = [sol2, sol3]

        best_ones = trainer.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one solution
        self.assertEqual(len(best_ones[0]), 1)

        # Check that sol2 is in hof
        self.assertTrue(sol2 in best_ones[0])

    def test_do_iteration(self):
        """Test _do_iteration."""
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }
        trainer = EA(**params)

        # Init the training process
        trainer._init_training()

        # Do an iteration
        pop_size_before = len(trainer.pop)
        trainer._do_iteration()
        pop_size_after = len(trainer.pop)
        self.assertEqual(pop_size_before, pop_size_after)

        for ind in trainer.pop:
            self.assertTrue(ind.fitness.is_valid)

    def test_integrate_representatives(self):
        """Test the integrate_representatives method."""
        # Number of representatives
        num_representatives = 5

        # Construct the trainer
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }
        trainer = EA(**params)

        # Set an empty population to trainer
        trainer._pop = []

        # Representatives to be integrated
        representatives = [
            Individual(
                params["species"],
                params["fitness_func"].fitness_cls
            ) for _ in range(num_representatives)
        ]

        # Integrate the representatives
        trainer.integrate_representatives(representatives)

        # Check the integration
        self.assertEqual(len(trainer.pop), num_representatives)
        for ind in representatives:
            self.assertTrue(ind in trainer.pop)

    def test_select_representatives(self):
        """Test the select_representatives method."""
        # Construct the trainer
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }
        trainer = EA(**params)

        # Init the training
        trainer._init_training()

        # Trainers not included in containers return an empty list
        self.assertEqual(trainer.select_representatives(), [])

        # Try with a trainer included within a container
        num_subtrainers = 3
        subtrainers = tuple(
            EA(**params) for _ in range(num_subtrainers)
        )
        container = IslandsTrainer(*subtrainers)
        trainer = subtrainers[0]

        # Init the training
        trainer._init_training()

        # The representatives
        representatives = trainer.select_representatives()

        # Check he representatives
        self.assertEqual(len(representatives), container.num_representatives)
        for ind in representatives:
            self.assertTrue(ind in trainer.pop)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Set custom params
        params = {
            "fitness_func": KappaNumFeats(dataset),
            "solution_cls": Individual,
            "species": Species(dataset.num_feats),
            "checkpoint_activation": False,
            "verbosity": False
        }

        # Construct a parameterized trainer
        trainer = EA(**params)
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

if __name__ == '__main__':
    unittest.main()
