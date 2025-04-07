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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.CooperativeEA`."""

import unittest
from os import remove
from time import sleep
from copy import copy, deepcopy

from culebra.trainer.topology import full_connected_destinations

from culebra.trainer.ea import DEFAULT_POP_SIZE
from culebra.trainer.ea.abc import SinglePopEA, CooperativeEA
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.cooperative import KappaNumFeatsC as FitnessFunc
from culebra.tools import Dataset


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


class MyCooperativeEA(CooperativeEA):
    """Dummy implementation of a cooperative co-evolutionary algorithm."""

    def _new_state(self) -> None:
        """Generate a new trainer state."""
        super()._new_state()

        # Generate the state of all subpopulation trainers
        for subtrainer in self.subtrainers:
            subtrainer._new_state()


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.ea.abc.CooperativeEA`."""

    def test_init(self):
        """Test :py:meth:`~culebra.trainer.ea.abc.CooperativeEA.__init__`."""
        valid_solution_classes = [
            ClassifierOptimizationIndividual,
            FeatureSelectionIndividual
        ]
        invalid_solution_class_types = (1, int, len, None)
        invalid_solution_class_values = (
            [valid_solution_classes[0], 2],
            [None, valid_solution_classes[0]]
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
        valid_subtrainer_cls = MySinglePopEA
        valid_num_subtrainers = 2
        invalid_funcs = (1, 1.5, {})
        valid_func = len
        valid_funcs = (
            valid_solution_classes[1].crossover1p,
            valid_solution_classes[1].crossover2p,
            valid_solution_classes[1].mutate
        )

        # Try invalid types for the individual classes. Should fail
        for solution_cls in invalid_solution_class_types:
            with self.assertRaises(TypeError):
                MyCooperativeEA(
                    solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try invalid values for the individual classes. Should fail
        for solution_classes in invalid_solution_class_values:
            with self.assertRaises(ValueError):
                MyCooperativeEA(
                    solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try different values of solution_cls for each subpopulation
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers
        )
        for cls1, cls2 in zip(
            trainer.solution_classes, valid_solution_classes
        ):
            self.assertEqual(cls1, cls2)

        # Try invalid types for the species. Should fail
        for species in invalid_species_types:
            with self.assertRaises(TypeError):
                MyCooperativeEA(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try invalid values for the species. Should fail
        for species in invalid_species_values:
            with self.assertRaises(ValueError):
                MyCooperativeEA(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try different values of species for each subpopulation
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            num_subtrainers=valid_num_subtrainers
        )
        for species1, species2 in zip(
            trainer.species, valid_species
        ):
            self.assertEqual(species1, species2)

        # Try invalid types for crossover_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                trainer = MyCooperativeEA(
                    valid_solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    crossover_funcs=func
                )

        # Try invalid values for crossover_funcs. Should fail
        with self.assertRaises(ValueError):
            trainer = MyCooperativeEA(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subtrainer_cls,
                crossover_funcs=invalid_funcs
            )

        # Try a fixed value for all the crossover functions,
        # all subpopulations should have the same value
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_funcs), trainer.num_subtrainers)

        # Check that all the values match
        for crossover_func in trainer.crossover_funcs:
            self.assertEqual(crossover_func, valid_func)

        # Try different values of crossover_func for each subpopulation
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls,
            crossover_funcs=valid_funcs
        )
        for func1, func2 in zip(
            trainer.crossover_funcs, valid_funcs
        ):
            self.assertEqual(func1, func2)

        # Try invalid types for mutation_funcs. Should fail
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                trainer = MyCooperativeEA(
                    valid_solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    mutation_funcs=func
                )

        # Check the default value for num_subtrainers
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls
        )
        self.assertEqual(trainer.num_subtrainers, len(valid_species))

        # Check a value for num_subtrainers different from the number of
        # species. It should fail
        with self.assertRaises(ValueError):
            MyCooperativeEA(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subtrainer_cls,
                num_subtrainers=18
            )

        # Test default params
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls
        )

        self.assertEqual(
            trainer.representation_topology_func,
            full_connected_destinations
        )
        self.assertEqual(
            trainer.representation_topology_func_params,
            {}
        )

        # Create the subpopulations
        trainer._generate_subtrainers()

        for pop_size in trainer.pop_sizes:
            self.assertEqual(pop_size, DEFAULT_POP_SIZE)

        for xover, mut, ind_cls in zip(
            trainer.crossover_funcs,
            trainer.mutation_funcs,
            trainer.solution_classes
        ):
            self.assertEqual(xover, ind_cls.crossover)
            self.assertEqual(mut, ind_cls.mutate)

    def test_generate_subtrainers(self):
        """Test _generate_subtrainers."""
        solution_classes = [
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
        subtrainer_cls = MySinglePopEA
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

        trainer = MyCooperativeEA(
            solution_classes,
            species,
            fitness_func,
            subtrainer_cls,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
            representation_size=representation_size
        )

        # Subpopulations have not been created yet
        self.assertEqual(trainer.subtrainers, None)

        # Create the subpopulations
        trainer._generate_subtrainers()

        # Check the subpopulations
        self.assertIsInstance(trainer.subtrainers, list)
        self.assertEqual(len(trainer.subtrainers), len(species))

        for index1 in range(trainer.num_subtrainers):
            for index2 in range(index1 + 1, trainer.num_subtrainers):
                self.assertNotEqual(
                    id(trainer.subtrainers[index1]),
                    id(trainer.subtrainers[index2])
                )

        # Check the subpopulations common parameters
        for subtrainer in trainer.subtrainers:
            self.assertIsInstance(subtrainer, subtrainer_cls)
            self.assertEqual(
                subtrainer.fitness_function,
                trainer.fitness_function
            )
            self.assertEqual(
                subtrainer.max_num_iters, trainer.max_num_iters
            )
            self.assertEqual(
                subtrainer.checkpoint_enable, trainer.checkpoint_enable
            )
            self.assertEqual(
                subtrainer.checkpoint_freq,
                trainer.checkpoint_freq
            )
            self.assertEqual(subtrainer.verbose, trainer.verbose)
            self.assertEqual(subtrainer.random_seed, trainer.random_seed)
            self.assertEqual(subtrainer.container, trainer)
            self.assertEqual(
                subtrainer._preprocess_iteration.__name__,
                trainer.receive_representatives.__name__
            )
            self.assertEqual(
                subtrainer._postprocess_iteration.__name__,
                trainer.send_representatives.__name__
            )
            self.assertEqual(
                subtrainer._init_representatives.func.__name__,
                trainer._init_subtrainer_representatives.__name__
            )

        # Check the subpopulations specific parameters
        for (
            subtrainer,
            subpop_index,
            subpop_solution_cls,
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
            trainer.subtrainers,
            range(trainer.num_subtrainers),
            trainer.solution_classes,
            trainer.species,
            trainer.subtrainer_checkpoint_filenames,
            trainer.pop_sizes,
            trainer.crossover_funcs,
            trainer.mutation_funcs,
            trainer.selection_funcs,
            trainer.crossover_probs,
            trainer.mutation_probs,
            trainer.gene_ind_mutation_probs,
            trainer.selection_funcs_params
        ):
            self.assertEqual(subtrainer.index, subpop_index)
            self.assertEqual(
                subtrainer.solution_cls,
                subpop_solution_cls
            )
            self.assertEqual(subtrainer.species, subpop_species)
            self.assertEqual(
                subtrainer.checkpoint_filename, subpop_checkpoint_filename
            )
            self.assertEqual(subtrainer.pop_size, subpop_pop_size)
            self.assertEqual(
                subtrainer.crossover_func,
                subpop_crossover_func
            )
            self.assertEqual(
                subtrainer.mutation_func,
                subpop_mutation_func
            )
            self.assertEqual(
                subtrainer.selection_func,
                subpop_selection_func
            )
            self.assertEqual(
                subtrainer.crossover_prob,
                subpop_crossover_prob
            )
            self.assertEqual(
                subtrainer.mutation_prob,
                subpop_mutation_prob
            )
            self.assertEqual(
                subtrainer.gene_ind_mutation_prob,
                subpop_gene_ind_mutation_prob
            )
            self.assertEqual(
                subtrainer.selection_func_params,
                subpop_selection_func_params
            )

        # Try incorrect number of solution_classes
        trainer.solution_classes = solution_classes + solution_classes

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of solution_classes and
        # try an incorrect number of species
        trainer.solution_classes = solution_classes
        trainer.species = species + species

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of species and
        # try an incorrect number of pop_sizes
        trainer.species = species
        trainer.pop_sizes = pop_sizes + pop_sizes

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of pop_sizes and
        # try an incorrect number of crossover_funcs
        trainer.pop_sizes = pop_sizes
        trainer.crossover_funcs = crossover_funcs + crossover_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of crossover_funcs and
        # try an incorrect number of mutation_funcs
        trainer.crossover_funcs = crossover_funcs
        trainer.mutation_funcs = mutation_funcs + mutation_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of mutation_funcs and
        # try an incorrect number of selection_funcs
        trainer.mutation_funcs = mutation_funcs
        trainer.selection_funcs = selection_funcs + selection_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of selection_funcs and
        # try an incorrect number of crossover_probs
        trainer.selection_funcs = selection_funcs
        trainer.crossover_probs = crossover_probs + crossover_probs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of crossover_probs and
        # try an incorrect number of mutation_probs
        trainer.crossover_probs = crossover_probs
        trainer.mutation_probs = mutation_probs + mutation_probs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of mutation_probs and
        # try an incorrect number of gene_ind_mutation_probs
        trainer.mutation_probs = mutation_probs
        trainer.gene_ind_mutation_probs = gene_ind_mutation_probs * 2

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

        # Restore the number of gene_ind_mutation_probs and
        # try an incorrect number of selection_funcs_params
        trainer.gene_ind_mutation_probs = gene_ind_mutation_probs
        trainer.selection_funcs_params = selection_funcs_params * 2

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subtrainers()

    def test_representatives(self):
        """Test the representatives property."""
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)
        trainer._init_search()

        # Get the representatives
        the_representatives = trainer.representatives

        # Check the representatives
        for (
                subpop_index,
                subtrainer
                ) in enumerate(trainer.subtrainers):
            for (
                context_index, _
            ) in enumerate(subtrainer.representatives):
                self.assertEqual(
                    the_representatives[context_index][subpop_index - 1],
                    subtrainer.representatives[
                        context_index][subpop_index - 1]
                )

    def test_best_solutions(self):
        """Test best_solutions."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)

        # Try before the population has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), trainer.num_subtrainers)
        for best in best_ones:
            self.assertEqual(len(best), 0)

        # Generate the subpopulations
        trainer._init_search()
        for subtrainer in trainer.subtrainers:
            subtrainer._evaluate_pop(subtrainer.pop)

        # Try again
        best_ones = trainer.best_solutions()

        # Test that a list with hof per species returned
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), trainer.num_subtrainers)
        for hof, ind_cls in zip(best_ones, trainer.solution_classes):
            for ind in hof:
                self.assertIsInstance(ind, ind_cls)

    def test_best_representatives(self):
        """Test the best_representatives method."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "max_num_iters": 2,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)

        # Try before the population has been created
        the_representatives = trainer.best_representatives()

        # The representatives should be None
        self.assertEqual(the_representatives, None)

        # Train
        trainer.train()

        # Try after training
        the_representatives = trainer.best_representatives()

        # Check the representatives
        self.assertIsInstance(the_representatives, list)
        self.assertEqual(
            len(the_representatives), trainer.representation_size
        )
        for context in the_representatives:
            self.assertIsInstance(context, list)
            self.assertEqual(len(context), trainer.num_subtrainers)
            for ind, species in zip(context, trainer.species):
                self.assertTrue(species.is_member(ind))

    def test_send_representatives(self):
        """Test send_representatives."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)

        # Generate the subpopulations
        trainer._init_search()
        for subtrainer in trainer.subtrainers:
            subtrainer._evaluate_pop(subtrainer.pop)

        # Set an iteration that should not provoke representatives sending
        for subtrainer in trainer.subtrainers:
            subtrainer._current_iter = trainer.representation_freq + 1

        # Call to send representatives, assigned to
        # subpop._postprocess_iteration at subpopulations iteration
        # time
        subtrainer._postprocess_iteration()

        # All the queues should be empty
        for index in range(trainer.num_subtrainers):
            self.assertTrue(trainer._communication_queues[index].empty())

        # Set an iteration that should provoke representatives sending
        for subtrainer in trainer.subtrainers:
            subtrainer._current_iter = trainer.representation_freq

            # Call to send representatives, assigned to
            # subpop._postprocess_iteration at subpopulations iteration
            # time
            subtrainer._postprocess_iteration()

            # Wait for the parallel queue processing
            sleep(1)

        # All the queues shouldn't be empty
        for index in range(trainer.num_subtrainers):
            self.assertFalse(trainer._communication_queues[index].empty())
            while not trainer._communication_queues[index].empty():
                trainer._communication_queues[index].get()

    def test_receive_representatives(self):
        """Test receive_representatives."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)

        # Generate the subpopulations
        trainer._init_search()
        for subtrainer in trainer.subtrainers:
            subtrainer._evaluate_pop(subtrainer.pop)

        sender_index = 0
        the_representatives = trainer.subtrainers[
            sender_index].pop[
            :trainer.representation_size]

        for index in range(trainer.num_subtrainers):
            if index != sender_index:
                trainer._communication_queues[index].put(
                    (sender_index, the_representatives)
                )

        # Wait for the parallel queue processing
        sleep(1)

        # Call to receive representatives, assigned to
        # subpop._preprocess_iteration
        # at subpopulations iteration time
        for subtrainer in trainer.subtrainers:
            subtrainer._preprocess_iteration()

        # Check the received values
        for recv_index, subtrainer in enumerate(trainer.subtrainers):
            if recv_index != sender_index:
                for ind_index, ind in enumerate(the_representatives):
                    self.assertEqual(
                        subtrainer.representatives[
                            ind_index][sender_index], ind
                    )

                # Check that all the individuals have been reevaluated
                for ind in subtrainer.pop:
                    self.assertTrue(ind.fitness.valid)

    def test_copy(self):
        """Test the __copy__ method."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyCooperativeEA(**params)
        trainer2 = copy(trainer1)

        # Copy only copies the first level (trainer1 != trainerl2)
        self.assertNotEqual(id(trainer1), id(trainer2))

        # The objects attributes are shared
        self.assertEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertEqual(
            id(trainer1.species),
            id(trainer2.species)
        )

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyCooperativeEA(**params)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyCooperativeEA(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = MyCooperativeEA.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Parameters for the trainer
        params = {
            "solution_classes": [
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
            "subtrainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.ea.abc.CooperativeEA`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.ea.abc.CooperativeEA`
        """
        # Copies all the levels
        self.assertNotEqual(id(trainer1), id(trainer2))
        self.assertNotEqual(
            id(trainer1.fitness_function),
            id(trainer2.fitness_function)
        )
        self.assertNotEqual(
            id(trainer1.fitness_function.training_data),
            id(trainer2.fitness_function.training_data)
        )
        self.assertTrue(
            (
                trainer1.fitness_function.training_data.inputs ==
                trainer2.fitness_function.training_data.inputs
            ).all()
        )
        self.assertTrue(
            (
                trainer1.fitness_function.training_data.outputs ==
                trainer2.fitness_function.training_data.outputs
            ).all()
        )
        self.assertNotEqual(id(trainer1.species), id(trainer2.species))
        for spe1, spe2 in zip(trainer1.species, trainer2.species):
            self.assertNotEqual(id(spe1), id(spe2))


if __name__ == '__main__':
    unittest.main()
