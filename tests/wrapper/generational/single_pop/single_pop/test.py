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

"""Unit test for :py:class:`wrapper.single_pop.SinglePop`."""

import os
import pickle
from copy import copy, deepcopy
from functools import partialmethod
import unittest
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import NumFeats
from culebra.fitness_function.cooperative import KappaNumFeatsC
from culebra.genotype.feature_selection import (
    Species as FeatureSelectionSpecies
)
from culebra.genotype.feature_selection.individual import (
    BitVector as FeatureSelectionIndividual
)
from culebra.genotype.classifier_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.wrapper import DEFAULT_NUM_GENS
from culebra.wrapper.single_pop import (
    SinglePop,
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_SELECTION_FUNC_PARAMS
)


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
    wrapper,
    individual_classes,
    species,
    representation_size
):
    """Init the representatives of other species.

    :param individual_clases: The individual class for each species.
    :type individual_classes: A :py:class:`~collections.abc.Sequence`
        of :py:class:`~base.Individual` subclasses
    :param species: The species to be evolved
    :type species: A :py:class:`~collections.abc.Sequence` of
        :py:class:`~base.Species` instances
    :param representation_size: Number of representative individuals
        from each species
    :type representation_size: :py:class:`int`
    """
    wrapper._representatives = []

    for _ in range(representation_size):
        wrapper._representatives.append(
            [
                ind_cls(
                    spe, wrapper.fitness_function.Fitness
                ) if i != wrapper.index else None
                for i, (ind_cls, spe) in enumerate(
                    zip(
                        individual_classes,
                        species)
                )
            ]

        )


class MyWrapper(SinglePop):
    """Dummy implementation of a wrapper method."""

    def _do_generation(self):
        """Implement a generation of the search process."""
        self._current_gen_evals = 10


class MyCooperativeWrapper(MyWrapper):
    """Dummy implementation of a cooperative wrapper method."""

    _init_representatives = partialmethod(
        init_representatives,
        individual_classes=cooperative_individual_classes,
        species=cooperative_species,
        representation_size=2
    )


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.single_pop.SinglePop`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.single_pop.SinglePop.__init__`."""
        valid_individual_cls = FeatureSelectionIndividual
        valid_species = FeatureSelectionSpecies(dataset.num_feats)
        valid_fitness_func = NumFeats(dataset)

        # Try invalid individual classes. Should fail
        invalid_individual_classes = (type, None, 'a', 1)
        for individual_cls in invalid_individual_classes:
            with self.assertRaises(TypeError):
                MyWrapper(individual_cls, valid_species, valid_fitness_func)

        # Try invalid species. Should fail
        invalid_species = (type, None, 'a', 1)
        for species in invalid_species:
            with self.assertRaises(TypeError):
                MyWrapper(valid_individual_cls, species, valid_fitness_func)

        # Try fitness functions. Should fail
        invalid_fitness_funcs = (type, None, 'a', 1)
        for func in invalid_fitness_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(valid_individual_cls, valid_species, func)

        # Try invalid types for num_gens. Should fail
        invalid_num_gens = (type, 'a', 1.5)
        for num_gens in invalid_num_gens:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    num_gens=num_gens
                )

        # Try invalid values for num_gens. Should fail
        invalid_num_gens = (-1, 0)
        for num_gens in invalid_num_gens:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    num_gens=num_gens
                )

        # Try invalid types for pop_size. Should fail
        invalid_pop_size = (type, 'a', 1.5)
        for pop_size in invalid_pop_size:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    pop_size=pop_size
                )

        # Try invalid values for pop_size. Should fail
        invalid_pop_size = (-1, 0)
        for pop_size in invalid_pop_size:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    pop_size=pop_size
                )

        # Try invalid types for crossover_func. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    crossover_func=func
                )

        # Try invalid types for mutation_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    mutation_func=func
                )

        # Try invalid types for selection_func. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    selection_func=func
                )

        # Try invalid types for crossover_prob. Should fail
        invalid_probs = ('a', type)
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    crossover_prob=prob
                )

        # Try invalid types for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    mutation_prob=prob
                )

        # Try invalid types for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid values for crossover_prob. Should fail
        invalid_probs = (-1, -0.001, 1.001, 4)
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    crossover_prob=prob
                )

        # Try invalid values for mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    mutation_prob=prob
                )

        # Try invalid values for gene_ind_mutation_prob. Should fail
        for prob in invalid_probs:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    gene_ind_mutation_prob=prob
                )

        # Try invalid types for selection_func_params. Should fail
        invalid_params = ('a', type)
        for params in invalid_params:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_cls,
                    valid_species,
                    valid_fitness_func,
                    selection_func_params=params
                )

        # Test default params
        wrapper = MyWrapper(
            valid_individual_cls, valid_species, valid_fitness_func
        )
        self.assertEqual(wrapper.num_gens, DEFAULT_NUM_GENS)
        self.assertEqual(wrapper.pop_size, DEFAULT_POP_SIZE)
        self.assertEqual(
            wrapper.crossover_func, wrapper.individual_cls.crossover)
        self.assertEqual(wrapper.mutation_func, wrapper.individual_cls.mutate)
        self.assertEqual(wrapper.selection_func, DEFAULT_SELECTION_FUNC)
        self.assertEqual(wrapper.crossover_prob, DEFAULT_CROSSOVER_PROB)
        self.assertEqual(wrapper.mutation_prob, DEFAULT_MUTATION_PROB)
        self.assertEqual(wrapper.gene_ind_mutation_prob,
                         DEFAULT_GENE_IND_MUTATION_PROB)
        self.assertEqual(
            wrapper.selection_func_params, DEFAULT_SELECTION_FUNC_PARAMS)
        self.assertEqual(wrapper.pop, None)
        self.assertEqual(wrapper._current_gen, None)

    def test_checkpoining(self):
        """Test checkpointing."""
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset)
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)

        # Set state attributes to dummy values
        wrapper1._pop = [
            FeatureSelectionIndividual(
                params["species"],
                params["fitness_function"].Fitness
            )
        ]

        wrapper1._current_gen = 10

        # Create another wrapper
        wrapper2 = MyWrapper(**params)

        # Check that state attributes in wrapper1 are not default values
        self.assertNotEqual(wrapper1.pop, None)
        self.assertNotEqual(wrapper1._current_gen, None)

        # Check that state attributes in wrapper2 are defaults
        self.assertEqual(wrapper2.pop, None)
        self.assertEqual(wrapper2._current_gen, None)

        # Save the state of wrapper1
        wrapper1._save_state()

        # Load the state of wrapper1 into wrapper2
        wrapper2._load_state()

        # Check that the state attributes of wrapper2 are equal to those of
        # wrapper1
        self.assertEqual(wrapper1._current_gen, wrapper2._current_gen)
        self.assertEqual(len(wrapper1.pop), len(wrapper2.pop))

        # Remove the checkpoint file
        os.remove(wrapper1.checkpoint_filename)

    def test_new_state(self):
        """Test :py:meth:`~wrapper.single_pop.SinglePop._new_state`.

        Also test
        :py:meth:`~wrapper.single_pop.SinglePop._generate_initial_pop`.
        """
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = MyWrapper(**params)

        # Create a new state
        wrapper._init_internals()
        wrapper._new_state()

        # Check the pop size
        self.assertEqual(len(wrapper.pop), wrapper.pop_size)

        # Check the individuals in the population
        for ind in wrapper.pop:
            self.assertIsInstance(ind, wrapper.individual_cls)

    def test_reset_state(self):
        """Test :py:meth:`~wrapper.single_pop.SinglePop._reset_state`."""
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = MyWrapper(**params)

        # Reset the state
        wrapper._reset_state()

        # Check the population
        self.assertEqual(wrapper.pop, None)

    def test_evaluate_pop(self):
        """Test the evaluation of a population."""
        # Check a single population
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = MyWrapper(**params)

        # Init the search
        wrapper._init_search()

        # Check that the individuals in the population are evaluated
        for ind in wrapper.pop:
            self.assertTrue(ind.fitness.valid)

        # Check the number of evaluations performed
        self.assertEqual(wrapper._current_gen_evals, wrapper.pop_size)

        # Check a cooperative problem
        fitness_function = KappaNumFeatsC(dataset)

        # Parameters for the wrapper
        params = {
            "individual_cls": cooperative_individual_classes[0],
            "species": cooperative_species[0],
            "fitness_function": fitness_function,
            "pop_size": 5,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = MyCooperativeWrapper(**params)

        # Init the search
        wrapper._init_search()

        # Check that the individuals in the population are evaluated
        for ind in wrapper.pop:
            self.assertTrue(ind.fitness.valid)

        # Check the number of evaluations performed
        self.assertEqual(
            wrapper._current_gen_evals,
            wrapper.pop_size * len(wrapper._representatives)
        )

    def test_best_solutions(self):
        """Test :py:meth:`~wrapper.single_pop.SinglePop.best_solutions`."""
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "checkpoint_enable": False,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper = MyWrapper(**params)

        # Try before the population has been created
        best_ones = wrapper.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Set state attributes to dummy values
        ind = FeatureSelectionIndividual(
            params["species"],
            params["fitness_function"].Fitness
        )

        # Init the search
        wrapper._init_search()
        wrapper._pop = [ind, ind]
        wrapper._evaluate_pop(wrapper.pop)
        best_ones = wrapper.best_solutions()

        # Check that best_ones contains only one species
        self.assertEqual(len(best_ones), 1)

        # Check that the hof has only one individual
        self.assertEqual(len(best_ones[0]), 1)

        # Check that the individual in hof is ind
        self.assertTrue((best_ones[0][0].features == ind.features).all())

    def test_copy(self):
        """Test the __copy__ method."""
        # Set custom params
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)
        wrapper2 = copy(wrapper1)

        # Copy only copies the first level (wrapper1 != wrapperl2)
        self.assertNotEqual(id(wrapper1), id(wrapper2))

        # The objects attributes are shared
        self.assertEqual(
            id(wrapper1.fitness_function),
            id(wrapper2.fitness_function)
        )
        self.assertEqual(
            id(wrapper1.species),
            id(wrapper2.species)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Set custom params
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)
        wrapper2 = deepcopy(wrapper1)

        # Check the copy
        self._check_deepcopy(wrapper1, wrapper2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Set custom params
        params = {
            "individual_cls": FeatureSelectionIndividual,
            "species": FeatureSelectionSpecies(dataset.num_feats),
            "fitness_function": NumFeats(dataset),
            "pop_size": 2,
            "verbose": False
        }

        # Construct a parameterized wrapper
        wrapper1 = MyWrapper(**params)

        data = pickle.dumps(wrapper1)
        wrapper2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(wrapper1, wrapper2)

    def _check_deepcopy(self, wrapper1, wrapper2):
        """Check if *wrapper1* is a deepcopy of *wrapper2*.

        :param wrapper1: The first wrapper
        :type wrapper1: :py:class:`~wrapper.single_pop.SinglePop`
        :param wrapper2: The second wrapper
        :type wrapper2: :py:class:`~wrapper.single_pop.SinglePop`
        """
        # Copies all the levels
        self.assertNotEqual(id(wrapper1), id(wrapper2))
        self.assertNotEqual(
            id(wrapper1.fitness_function),
            id(wrapper2.fitness_function)
        )
        self.assertNotEqual(
            id(wrapper1.fitness_function.training_data),
            id(wrapper2.fitness_function.training_data)
        )

        self.assertTrue(
            (
                wrapper1.fitness_function.training_data.inputs ==
                wrapper2.fitness_function.training_data.inputs
            ).all()
        )

        self.assertTrue(
            (
                wrapper1.fitness_function.training_data.outputs ==
                wrapper2.fitness_function.training_data.outputs
            ).all()
        )

        self.assertNotEqual(id(wrapper1.species), id(wrapper2.species))
        self.assertEqual(
            id(wrapper1.species.num_feats), id(wrapper2.species.num_feats)
        )


if __name__ == '__main__':
    unittest.main()
