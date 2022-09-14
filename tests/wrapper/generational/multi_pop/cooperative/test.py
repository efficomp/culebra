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

"""Unit test for :py:class:`wrapper.multi_pop.Cooperative`."""

import pickle
from time import sleep
from copy import copy, deepcopy
import unittest
from culebra.base import Dataset
from culebra.fitness_function.cooperative import KappaNumFeatsC as FitnessFunc
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

from culebra.wrapper.single_pop import (
    NSGA, DEFAULT_POP_SIZE
)
from culebra.wrapper.multi_pop import (
    Cooperative,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class MyWrapper(Cooperative):
    """Dummy implementation of a wrapper method."""

    def _new_state(self) -> None:
        """Generate a new wrapper state."""
        super()._new_state()

        # Generate the state of all subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._new_state()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.Cooperative`."""

    def test_init(self):
        """Test :py:meth:`~wrapper.multi_pop.Cooperative.__init__`."""
        valid_individual_classes = [
            ClassifierOptimizationIndividual,
            FeatureSelectionIndividual
        ]
        invalid_individual_class_types = (1, int, len, None)
        invalid_individual_class_values = (
            [valid_individual_classes[0], 2],
            [None, valid_individual_classes[0]]
        )

        valid_species = [
            # Species to optimize a SVM-based classifier
            ClassifierOptimizationSpecies(
                lower_bounds=[0, 0],
                upper_bounds=[100000, 100000],
                names=["C", "gamma"]
            ),
            # Species for the feature selection problem
            FeatureSelectionSpecies(dataset.num_feats),
        ]
        invalid_species_types = (1, int, len, None)
        invalid_species_values = (
            [valid_species[0], 2],
            [None, valid_species[0]]
        )

        valid_fitness_func = FitnessFunc(dataset)
        valid_subpop_wrapper_cls = NSGA
        valid_num_subpops = 2
        invalid_funcs = (1, 1.5, {})
        valid_func = len
        valid_funcs = (
            valid_individual_classes[1].crossover1p,
            valid_individual_classes[1].crossover2p,
            valid_individual_classes[1].mutate
        )

        # Try invalid types for the individual classes. Should fail
        for individual_cls in invalid_individual_class_types:
            with self.assertRaises(TypeError):
                MyWrapper(
                    individual_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops
                )

        # Try invalid values for the individual classes. Should fail
        for individual_classes in invalid_individual_class_values:
            with self.assertRaises(ValueError):
                MyWrapper(
                    individual_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops
                )

        # Try different values of individual_cls for each subpopulation
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops
        )
        for cls1, cls2 in zip(
            wrapper.individual_classes, valid_individual_classes
        ):
            self.assertEqual(cls1, cls2)

        # Try invalid types for the species. Should fail
        for species in invalid_species_types:
            with self.assertRaises(TypeError):
                MyWrapper(
                    valid_individual_classes,
                    species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops
                )

        # Try invalid values for the species. Should fail
        for species in invalid_species_values:
            with self.assertRaises(ValueError):
                MyWrapper(
                    valid_individual_classes,
                    species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    num_subpops=valid_num_subpops
                )

        # Try different values of species for each subpopulation
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            num_subpops=valid_num_subpops
        )
        for species1, species2 in zip(
            wrapper.species, valid_species
        ):
            self.assertEqual(species1, species2)

        # Try invalid types for crossover_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                wrapper = MyWrapper(
                    valid_individual_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    crossover_funcs=func
                )

        # Try invalid values for crossover_funcs. Should fail
        with self.assertRaises(ValueError):
            wrapper = MyWrapper(
                valid_individual_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_wrapper_cls,
                crossover_funcs=invalid_funcs
            )

        # Try a fixed value for all the crossover functions,
        # all subpopulations should have the same value
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(wrapper.crossover_funcs), wrapper.num_subpops)

        # Check that all the values match
        for crossover_func in wrapper.crossover_funcs:
            self.assertEqual(crossover_func, valid_func)

        # Try different values of crossover_func for each subpopulation
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            crossover_funcs=valid_funcs
        )
        for func1, func2 in zip(
            wrapper.crossover_funcs, valid_funcs
        ):
            self.assertEqual(func1, func2)

        # Try invalid types for mutation_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                wrapper = MyWrapper(
                    valid_individual_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    mutation_funcs=func
                )

        # Check the default value for num_subpops
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls
        )
        self.assertEqual(wrapper.num_subpops, len(valid_species))

        # Check a value for num_subpops different from the number os species
        # It should fail
        with self.assertRaises(ValueError):
            MyWrapper(
                valid_individual_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_wrapper_cls,
                num_subpops=18
            )

        # Try invalid representation topology functions. Should fail
        for func in invalid_funcs:
            with self.assertRaises(ValueError):
                wrapper = MyWrapper(
                    valid_individual_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_wrapper_cls,
                    representation_topology_func=func
                )

        # Try the only valid value
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            representation_topology_func=(
                DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
            )
        )

        # Try invalid representation topology function parameters. Should fail
        with self.assertRaises(ValueError):
            wrapper = MyWrapper(
                valid_individual_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_wrapper_cls,
                representation_topology_func_params=valid_func
            )

        # Try the only valid value
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls,
            representation_topology_func_params=(
                DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            )
        )

        # Test default params
        wrapper = MyWrapper(
            valid_individual_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_wrapper_cls
        )

        # Create the subpopulations
        wrapper._generate_subpop_wrappers()

        for pop_size in wrapper.pop_sizes:
            self.assertEqual(pop_size, DEFAULT_POP_SIZE)

        for xover, mut, ind_cls in zip(
            wrapper.crossover_funcs,
            wrapper.mutation_funcs,
            wrapper.individual_classes
        ):
            self.assertEqual(xover, ind_cls.crossover)
            self.assertEqual(mut, ind_cls.mutate)

    def test_generate_subpop_wrappers(self):
        """Test _generate_subpop_wrappers."""
        individual_classes = [
            ClassifierOptimizationIndividual,
            FeatureSelectionIndividual
        ]
        species = [
            # Species to optimize a SVM-based classifier
            ClassifierOptimizationSpecies(
                lower_bounds=[0, 0],
                upper_bounds=[100000, 100000],
                names=["C", "gamma"]
            ),
            # Species for the feature selection problem
            FeatureSelectionSpecies(dataset.num_feats)
        ]

        fitness_func = FitnessFunc(dataset)
        subpop_wrapper_cls = NSGA
        nsga3_reference_points_p = 18
        representation_size = 2
        pop_sizes = (13, 15)
        crossover_funcs = (max, min)
        mutation_funcs = (abs, len)
        selection_funcs = (isinstance, issubclass)
        crossover_probs = (0.33, 0.44)
        mutation_probs = (0.133, 0.144)
        gene_ind_mutation_probs = (0.1133, 0.1144)
        selection_funcs_params = (
            {"parameter0": 12},
            {"parameter1": 13},
        )

        wrapper = MyWrapper(
            individual_classes,
            species,
            fitness_func,
            subpop_wrapper_cls,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
            representation_size=representation_size,
            nsga3_reference_points_p=nsga3_reference_points_p
        )

        # Subpopulations have not been created yet
        self.assertEqual(wrapper.subpop_wrappers, None)

        # Create the subpopulations
        wrapper._generate_subpop_wrappers()

        # Check the subpopulations
        self.assertIsInstance(wrapper.subpop_wrappers, list)
        self.assertEqual(len(wrapper.subpop_wrappers), len(species))

        for index1 in range(wrapper.num_subpops):
            for index2 in range(index1 + 1, wrapper.num_subpops):
                self.assertNotEqual(
                    id(wrapper.subpop_wrappers[index1]),
                    id(wrapper.subpop_wrappers[index2])
                )

        # Check the subpopulations common parameters
        for subpop_wrapper in wrapper.subpop_wrappers:
            self.assertIsInstance(subpop_wrapper, subpop_wrapper_cls)
            self.assertEqual(
                subpop_wrapper.fitness_function,
                wrapper.fitness_function
            )
            self.assertEqual(subpop_wrapper.num_gens, wrapper.num_gens)
            self.assertEqual(
                subpop_wrapper.checkpoint_enable, wrapper.checkpoint_enable
            )
            self.assertEqual(
                subpop_wrapper.checkpoint_freq,
                wrapper.checkpoint_freq
            )
            self.assertEqual(subpop_wrapper.verbose, wrapper.verbose)
            self.assertEqual(subpop_wrapper.random_seed, wrapper.random_seed)
            self.assertEqual(subpop_wrapper.container, wrapper)
            self.assertEqual(
                subpop_wrapper._preprocess_generation.__name__,
                wrapper.receive_representatives.__name__
            )
            self.assertEqual(
                subpop_wrapper._postprocess_generation.__name__,
                wrapper.send_representatives.__name__
            )
            self.assertEqual(
                subpop_wrapper._init_representatives.func.__name__,
                wrapper._init_subpop_wrapper_representatives.__name__
            )

            # Check the subpopulation wrapper custom params
            self.assertEqual(
                subpop_wrapper.nsga3_reference_points_p,
                nsga3_reference_points_p
            )

        # Check the subpopulations specific parameters
        for (
            subpop_wrapper,
            subpop_index,
            subpop_individual_cls,
            subpop_species,
            subpop_checkpoint_filename,
            subpop_pop_size,
            subpop_crossover_func,
            subpop_mutation_func,
            subpop_selection_func,
            subpop_crossover_prob,
            subpop_mutation_prob,
            subpop_gene_ind_mutation_prob,
            subpop_selection_func_params
        ) in zip(
            wrapper.subpop_wrappers,
            range(wrapper.num_subpops),
            wrapper.individual_classes,
            wrapper.species,
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
            self.assertEqual(subpop_wrapper.index, subpop_index)
            self.assertEqual(
                subpop_wrapper.individual_cls,
                subpop_individual_cls
            )
            self.assertEqual(subpop_wrapper.species, subpop_species)
            self.assertEqual(
                subpop_wrapper.checkpoint_filename, subpop_checkpoint_filename
            )
            self.assertEqual(subpop_wrapper.pop_size, subpop_pop_size)
            self.assertEqual(
                subpop_wrapper.crossover_func,
                subpop_crossover_func
            )
            self.assertEqual(
                subpop_wrapper.mutation_func,
                subpop_mutation_func
            )
            self.assertEqual(
                subpop_wrapper.selection_func,
                subpop_selection_func
            )
            self.assertEqual(
                subpop_wrapper.crossover_prob,
                subpop_crossover_prob
            )
            self.assertEqual(
                subpop_wrapper.mutation_prob,
                subpop_mutation_prob
            )
            self.assertEqual(
                subpop_wrapper.gene_ind_mutation_prob,
                subpop_gene_ind_mutation_prob
            )
            self.assertEqual(
                subpop_wrapper.selection_func_params,
                subpop_selection_func_params
            )

        # Try incorrect number of individual_classes
        wrapper.individual_classes = individual_classes + individual_classes

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of individual_classes and
        # try an incorrect number of species
        wrapper.individual_classes = individual_classes
        wrapper.species = species + species

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of species and
        # try an incorrect number of pop_sizes
        wrapper.species = species
        wrapper.pop_sizes = pop_sizes + pop_sizes

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of pop_sizes and
        # try an incorrect number of crossover_funcs
        wrapper.pop_sizes = pop_sizes
        wrapper.crossover_funcs = crossover_funcs + crossover_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of crossover_funcs and
        # try an incorrect number of mutation_funcs
        wrapper.crossover_funcs = crossover_funcs
        wrapper.mutation_funcs = mutation_funcs + mutation_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of mutation_funcs and
        # try an incorrect number of selection_funcs
        wrapper.mutation_funcs = mutation_funcs
        wrapper.selection_funcs = selection_funcs + selection_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of selection_funcs and
        # try an incorrect number of crossover_probs
        wrapper.selection_funcs = selection_funcs
        wrapper.crossover_probs = crossover_probs + crossover_probs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of crossover_probs and
        # try an incorrect number of mutation_probs
        wrapper.crossover_probs = crossover_probs
        wrapper.mutation_probs = mutation_probs + mutation_probs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of mutation_probs and
        # try an incorrect number of gene_ind_mutation_probs
        wrapper.mutation_probs = mutation_probs
        wrapper.gene_ind_mutation_probs = gene_ind_mutation_probs * 2

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

        # Restore the number of gene_ind_mutation_probs and
        # try an incorrect number of selection_funcs_params
        wrapper.gene_ind_mutation_probs = gene_ind_mutation_probs
        wrapper.selection_funcs_params = selection_funcs_params * 2

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            wrapper._generate_subpop_wrappers()

    def test_representatives(self):
        """Test the representatives property."""
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "pop_sizes": 10,
            "representation_size": 2,
            "subpop_wrapper_cls": NSGA,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)
        wrapper._init_search()

        # Get the representatives
        the_representatives = wrapper.representatives

        # Check the representatives
        for (
                subpop_index,
                subpop_wrapper
                ) in enumerate(wrapper.subpop_wrappers):
            for (
                context_index, context
            ) in enumerate(subpop_wrapper.representatives):
                self.assertEqual(
                    the_representatives[context_index][subpop_index - 1],
                    subpop_wrapper.representatives[
                        context_index][subpop_index - 1]
                )

    def test_best_solutions(self):
        """Test best_solutions."""
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Try before the population has been created
        best_ones = wrapper.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), wrapper.num_subpops)
        for best in best_ones:
            self.assertEqual(len(best), 0)

        # Generate the subpopulations
        wrapper._init_search()
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._evaluate_pop(subpop_wrapper.pop)

        # Try again
        best_ones = wrapper.best_solutions()

        # Test that a list with hof per species returned
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), wrapper.num_subpops)
        for hof, ind_cls in zip(best_ones, wrapper.individual_classes):
            for ind in hof:
                self.assertIsInstance(ind, ind_cls)

    def test_best_representatives(self):
        """Test the best_representatives method."""
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "num_gens": 2,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Try before the population has been created
        the_representatives = wrapper.best_representatives()

        # The representatives should be None
        self.assertEqual(the_representatives, None)

        # Train
        wrapper.train()

        # Try after training
        the_representatives = wrapper.best_representatives()

        # Check the representatives
        self.assertIsInstance(the_representatives, list)
        self.assertEqual(
            len(the_representatives), wrapper.representation_size
        )
        for context in the_representatives:
            self.assertIsInstance(context, list)
            self.assertEqual(len(context), wrapper.num_subpops)
            for ind, species in zip(context, wrapper.species):
                self.assertTrue(species.is_member(ind))

    def test_send_representatives(self):
        """Test send_representatives."""
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Generate the subpopulations
        wrapper._init_search()
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._evaluate_pop(subpop_wrapper.pop)

        # Set a generation that should not provoke representatives sending
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._current_gen = wrapper.representation_freq + 1

        # Call to send representatives, assigned to
        # subpop._postprocess_generation at subpopulations generation
        # time
        subpop_wrapper._postprocess_generation()

        # All the queues should be empty
        for index in range(wrapper.num_subpops):
            self.assertTrue(wrapper._communication_queues[index].empty())

        # Set a generation that should provoke representatives sending
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._current_gen = wrapper.representation_freq

            # Call to send representatives, assigned to
            # subpop._postprocess_generation at subpopulations generation
            # time
            subpop_wrapper._postprocess_generation()

            # Wait for the parallel queue processing
            sleep(1)

        # All the queues shouldn't be empty
        for index in range(wrapper.num_subpops):
            self.assertFalse(wrapper._communication_queues[index].empty())
            while not wrapper._communication_queues[index].empty():
                wrapper._communication_queues[index].get()

    def test_receive_representatives(self):
        """Test receive_representatives."""
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = MyWrapper(**params)

        # Generate the subpopulations
        wrapper._init_search()
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._evaluate_pop(subpop_wrapper.pop)

        sender_index = 0
        the_representatives = wrapper.subpop_wrappers[
            sender_index].pop[
            :wrapper.representation_size]

        for index in range(wrapper.num_subpops):
            if index != sender_index:
                wrapper._communication_queues[index].put(
                    (sender_index, the_representatives)
                )

        # Wait for the parallel queue processing
        sleep(1)

        # Call to receive representatives, assigned to
        # subpop._preprocess_generation
        # at subpopulations generation time
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._preprocess_generation()

        # Check the received values
        for recv_index, subpop_wrapper in enumerate(wrapper.subpop_wrappers):
            if recv_index != sender_index:
                for ind_index, ind in enumerate(the_representatives):
                    self.assertEqual(
                        subpop_wrapper.representatives[
                            ind_index][sender_index], ind
                    )

                # Check that all the individuals have been reevaluated
                for ind in subpop_wrapper.pop:
                    self.assertTrue(ind.fitness.valid)

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
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
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper1 = MyWrapper(**params)
        wrapper2 = deepcopy(wrapper1)

        # Check the copy
        self._check_deepcopy(wrapper1, wrapper2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Parameters for the wrapper
        params = {
            "individual_classes": [
                ClassifierOptimizationIndividual,
                FeatureSelectionIndividual
            ],
            "species": [
                # Species to optimize a SVM-based classifier
                ClassifierOptimizationSpecies(
                    lower_bounds=[0, 0],
                    upper_bounds=[100000, 100000],
                    names=["C", "gamma"]
                ),
                # Species for the feature selection problem
                FeatureSelectionSpecies(dataset.num_feats)
            ],
            "fitness_function": FitnessFunc(dataset),
            "subpop_wrapper_cls": NSGA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper1 = MyWrapper(**params)
        data = pickle.dumps(wrapper1)
        wrapper2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(wrapper1, wrapper2)

    def _check_deepcopy(self, wrapper1, wrapper2):
        """Check if *wrapper1* is a deepcopy of *wrapper2*.

        :param wrapper1: The first wrapper
        :type wrapper1: :py:class:`~wrapper.multi_pop.Cooperative`
        :param wrapper2: The second wrapper
        :type wrapper2: :py:class:`~wrapper.multi_pop.Cooperative`
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
        for spe1, spe2 in zip(wrapper1.species, wrapper2.species):
            self.assertNotEqual(id(spe1), id(spe2))


if __name__ == '__main__':
    unittest.main()
