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

"""Unit test for :py:class:`~culebra.trainer.ea.abc.CooperativeEA`."""

import unittest
import pickle
from time import sleep
from copy import copy, deepcopy

from culebra import DEFAULT_POP_SIZE
from culebra.trainer import (
    DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)
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


class MyCooperativeEA(CooperativeEA):
    """Dummy implementation of a cooperative co-evolutionary algorithm."""

    def _new_state(self) -> None:
        """Generate a new trainer state."""
        super()._new_state()

        # Generate the state of all subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._new_state()


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
        valid_subpop_trainer_cls = MySinglePopEA
        valid_num_subpops = 2
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
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try invalid values for the individual classes. Should fail
        for solution_classes in invalid_solution_class_values:
            with self.assertRaises(ValueError):
                MyCooperativeEA(
                    solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try different values of solution_cls for each subpopulation
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops
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
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try invalid values for the species. Should fail
        for species in invalid_species_values:
            with self.assertRaises(ValueError):
                MyCooperativeEA(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try different values of species for each subpopulation
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            num_subpops=valid_num_subpops
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
                    valid_subpop_trainer_cls,
                    crossover_funcs=func
                )

        # Try invalid values for crossover_funcs. Should fail
        with self.assertRaises(ValueError):
            trainer = MyCooperativeEA(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_trainer_cls,
                crossover_funcs=invalid_funcs
            )

        # Try a fixed value for all the crossover functions,
        # all subpopulations should have the same value
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            crossover_funcs=valid_func
        )
        # Check the length of the sequence
        self.assertEqual(len(trainer.crossover_funcs), trainer.num_subpops)

        # Check that all the values match
        for crossover_func in trainer.crossover_funcs:
            self.assertEqual(crossover_func, valid_func)

        # Try different values of crossover_func for each subpopulation
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
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
                    valid_subpop_trainer_cls,
                    mutation_funcs=func
                )

        # Check the default value for num_subpops
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls
        )
        self.assertEqual(trainer.num_subpops, len(valid_species))

        # Check a value for num_subpops different from the number os species
        # It should fail
        with self.assertRaises(ValueError):
            MyCooperativeEA(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_trainer_cls,
                num_subpops=18
            )

        # Try invalid representation topology functions. Should fail
        for func in invalid_funcs:
            with self.assertRaises(ValueError):
                trainer = MyCooperativeEA(
                    valid_solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    representation_topology_func=func
                )

        # Try the only valid value
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            representation_topology_func=(
                DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC
            )
        )

        # Try invalid representation topology function parameters. Should fail
        with self.assertRaises(ValueError):
            trainer = MyCooperativeEA(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_trainer_cls,
                representation_topology_func_params=valid_func
            )

        # Try the only valid value
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            representation_topology_func_params=(
                DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            )
        )

        # Test default params
        trainer = MyCooperativeEA(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls
        )

        # Create the subpopulations
        trainer._generate_subpop_trainers()

        for pop_size in trainer.pop_sizes:
            self.assertEqual(pop_size, DEFAULT_POP_SIZE)

        for xover, mut, ind_cls in zip(
            trainer.crossover_funcs,
            trainer.mutation_funcs,
            trainer.solution_classes
        ):
            self.assertEqual(xover, ind_cls.crossover)
            self.assertEqual(mut, ind_cls.mutate)

    def test_generate_subpop_trainers(self):
        """Test _generate_subpop_trainers."""
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
        subpop_trainer_cls = MySinglePopEA
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
            subpop_trainer_cls,
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
        self.assertEqual(trainer.subpop_trainers, None)

        # Create the subpopulations
        trainer._generate_subpop_trainers()

        # Check the subpopulations
        self.assertIsInstance(trainer.subpop_trainers, list)
        self.assertEqual(len(trainer.subpop_trainers), len(species))

        for index1 in range(trainer.num_subpops):
            for index2 in range(index1 + 1, trainer.num_subpops):
                self.assertNotEqual(
                    id(trainer.subpop_trainers[index1]),
                    id(trainer.subpop_trainers[index2])
                )

        # Check the subpopulations common parameters
        for subpop_trainer in trainer.subpop_trainers:
            self.assertIsInstance(subpop_trainer, subpop_trainer_cls)
            self.assertEqual(
                subpop_trainer.fitness_function,
                trainer.fitness_function
            )
            self.assertEqual(
                subpop_trainer.max_num_iters, trainer.max_num_iters
            )
            self.assertEqual(
                subpop_trainer.checkpoint_enable, trainer.checkpoint_enable
            )
            self.assertEqual(
                subpop_trainer.checkpoint_freq,
                trainer.checkpoint_freq
            )
            self.assertEqual(subpop_trainer.verbose, trainer.verbose)
            self.assertEqual(subpop_trainer.random_seed, trainer.random_seed)
            self.assertEqual(subpop_trainer.container, trainer)
            self.assertEqual(
                subpop_trainer._preprocess_iteration.__name__,
                trainer.receive_representatives.__name__
            )
            self.assertEqual(
                subpop_trainer._postprocess_iteration.__name__,
                trainer.send_representatives.__name__
            )
            self.assertEqual(
                subpop_trainer._init_representatives.func.__name__,
                trainer._init_subpop_trainer_representatives.__name__
            )

        # Check the subpopulations specific parameters
        for (
            subpop_trainer,
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
            trainer.subpop_trainers,
            range(trainer.num_subpops),
            trainer.solution_classes,
            trainer.species,
            trainer.subpop_trainer_checkpoint_filenames,
            trainer.pop_sizes,
            trainer.crossover_funcs,
            trainer.mutation_funcs,
            trainer.selection_funcs,
            trainer.crossover_probs,
            trainer.mutation_probs,
            trainer.gene_ind_mutation_probs,
            trainer.selection_funcs_params
        ):
            self.assertEqual(subpop_trainer.index, subpop_index)
            self.assertEqual(
                subpop_trainer.solution_cls,
                subpop_solution_cls
            )
            self.assertEqual(subpop_trainer.species, subpop_species)
            self.assertEqual(
                subpop_trainer.checkpoint_filename, subpop_checkpoint_filename
            )
            self.assertEqual(subpop_trainer.pop_size, subpop_pop_size)
            self.assertEqual(
                subpop_trainer.crossover_func,
                subpop_crossover_func
            )
            self.assertEqual(
                subpop_trainer.mutation_func,
                subpop_mutation_func
            )
            self.assertEqual(
                subpop_trainer.selection_func,
                subpop_selection_func
            )
            self.assertEqual(
                subpop_trainer.crossover_prob,
                subpop_crossover_prob
            )
            self.assertEqual(
                subpop_trainer.mutation_prob,
                subpop_mutation_prob
            )
            self.assertEqual(
                subpop_trainer.gene_ind_mutation_prob,
                subpop_gene_ind_mutation_prob
            )
            self.assertEqual(
                subpop_trainer.selection_func_params,
                subpop_selection_func_params
            )

        # Try incorrect number of solution_classes
        trainer.solution_classes = solution_classes + solution_classes

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of solution_classes and
        # try an incorrect number of species
        trainer.solution_classes = solution_classes
        trainer.species = species + species

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of species and
        # try an incorrect number of pop_sizes
        trainer.species = species
        trainer.pop_sizes = pop_sizes + pop_sizes

        # Create the subpopulations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of pop_sizes and
        # try an incorrect number of crossover_funcs
        trainer.pop_sizes = pop_sizes
        trainer.crossover_funcs = crossover_funcs + crossover_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of crossover_funcs and
        # try an incorrect number of mutation_funcs
        trainer.crossover_funcs = crossover_funcs
        trainer.mutation_funcs = mutation_funcs + mutation_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of mutation_funcs and
        # try an incorrect number of selection_funcs
        trainer.mutation_funcs = mutation_funcs
        trainer.selection_funcs = selection_funcs + selection_funcs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of selection_funcs and
        # try an incorrect number of crossover_probs
        trainer.selection_funcs = selection_funcs
        trainer.crossover_probs = crossover_probs + crossover_probs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of crossover_probs and
        # try an incorrect number of mutation_probs
        trainer.crossover_probs = crossover_probs
        trainer.mutation_probs = mutation_probs + mutation_probs

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of mutation_probs and
        # try an incorrect number of gene_ind_mutation_probs
        trainer.mutation_probs = mutation_probs
        trainer.gene_ind_mutation_probs = gene_ind_mutation_probs * 2

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

        # Restore the number of gene_ind_mutation_probs and
        # try an incorrect number of selection_funcs_params
        trainer.gene_ind_mutation_probs = gene_ind_mutation_probs
        trainer.selection_funcs_params = selection_funcs_params * 2

        # Create the subpopupations. Should fail
        with self.assertRaises(RuntimeError):
            trainer._generate_subpop_trainers()

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
            "subpop_trainer_cls": MySinglePopEA,
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
                subpop_trainer
                ) in enumerate(trainer.subpop_trainers):
            for (
                context_index, _
            ) in enumerate(subpop_trainer.representatives):
                self.assertEqual(
                    the_representatives[context_index][subpop_index - 1],
                    subpop_trainer.representatives[
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
            "subpop_trainer_cls": MySinglePopEA,
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
        self.assertEqual(len(best_ones), trainer.num_subpops)
        for best in best_ones:
            self.assertEqual(len(best), 0)

        # Generate the subpopulations
        trainer._init_search()
        for subpop_trainer in trainer.subpop_trainers:
            subpop_trainer._evaluate_pop(subpop_trainer.pop)

        # Try again
        best_ones = trainer.best_solutions()

        # Test that a list with hof per species returned
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), trainer.num_subpops)
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
            "subpop_trainer_cls": MySinglePopEA,
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
            self.assertEqual(len(context), trainer.num_subpops)
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
            "subpop_trainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)

        # Generate the subpopulations
        trainer._init_search()
        for subpop_trainer in trainer.subpop_trainers:
            subpop_trainer._evaluate_pop(subpop_trainer.pop)

        # Set an iteration that should not provoke representatives sending
        for subpop_trainer in trainer.subpop_trainers:
            subpop_trainer._current_iter = trainer.representation_freq + 1

        # Call to send representatives, assigned to
        # subpop._postprocess_iteration at subpopulations iteration
        # time
        subpop_trainer._postprocess_iteration()

        # All the queues should be empty
        for index in range(trainer.num_subpops):
            self.assertTrue(trainer._communication_queues[index].empty())

        # Set an iteration that should provoke representatives sending
        for subpop_trainer in trainer.subpop_trainers:
            subpop_trainer._current_iter = trainer.representation_freq

            # Call to send representatives, assigned to
            # subpop._postprocess_iteration at subpopulations iteration
            # time
            subpop_trainer._postprocess_iteration()

            # Wait for the parallel queue processing
            sleep(1)

        # All the queues shouldn't be empty
        for index in range(trainer.num_subpops):
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
            "subpop_trainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyCooperativeEA(**params)

        # Generate the subpopulations
        trainer._init_search()
        for subpop_trainer in trainer.subpop_trainers:
            subpop_trainer._evaluate_pop(subpop_trainer.pop)

        sender_index = 0
        the_representatives = trainer.subpop_trainers[
            sender_index].pop[
            :trainer.representation_size]

        for index in range(trainer.num_subpops):
            if index != sender_index:
                trainer._communication_queues[index].put(
                    (sender_index, the_representatives)
                )

        # Wait for the parallel queue processing
        sleep(1)

        # Call to receive representatives, assigned to
        # subpop._preprocess_iteration
        # at subpopulations iteration time
        for subpop_trainer in trainer.subpop_trainers:
            subpop_trainer._preprocess_iteration()

        # Check the received values
        for recv_index, subpop_trainer in enumerate(trainer.subpop_trainers):
            if recv_index != sender_index:
                for ind_index, ind in enumerate(the_representatives):
                    self.assertEqual(
                        subpop_trainer.representatives[
                            ind_index][sender_index], ind
                    )

                # Check that all the individuals have been reevaluated
                for ind in subpop_trainer.pop:
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
            "subpop_trainer_cls": MySinglePopEA,
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
            "subpop_trainer_cls": MySinglePopEA,
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
            "subpop_trainer_cls": MySinglePopEA,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyCooperativeEA(**params)
        data = pickle.dumps(trainer1)
        trainer2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

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
