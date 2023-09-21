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

"""Unit test for :py:class:`culebra.trainer.abc.CooperativeTrainer`."""

import unittest
import pickle
from time import sleep
from copy import copy, deepcopy
from functools import partialmethod

from culebra import DEFAULT_POP_SIZE
from culebra.trainer import (
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)
from culebra.trainer.abc import SinglePopTrainer, CooperativeTrainer
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Solution as ClassifierOptimizationSolution
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


class MySinglePopTrainerTrainer(SinglePopTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        for _ in range(self.pop_size):
            sol = self.solution_cls(
                self.species, self.fitness_function.Fitness
            )
            self.evaluate(sol)
            self._pop.append(sol)


class MyTrainer(CooperativeTrainer):
    """Dummy implementation of a cooperative co-evolutionary algorithm."""

    _subpop_properties_mapping = {
        "solution_classes": "solution_cls",
        "species": "species",
        "pop_sizes": "pop_size"
    }
    """Map the container names of properties sequences to the different
    subpop property names."""

    def _new_state(self) -> None:
        """Generate a new trainer state."""
        super()._new_state()

        # Generate the state of all subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._new_state()

    def _init_search(self):
        super()._init_search()
        for island_trainer in self.subpop_trainers:
            island_trainer._init_search()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the metrics before each iteration is run.
        """
        super()._start_iteration()
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            # Fix the current iteration
            subpop_trainer._current_iter = self._current_iter
            # Start the iteration
            subpop_trainer._start_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._do_iteration_stats()

    def _generate_subpop_trainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :py:attr:`~culebra.trainer.abc.SinglePopTrainer.index`
        and a :py:attr:`~culebra.trainer.abc.SinglePopTrainer.container` to
        each subpopulation :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        trainer, change the subpopulation trainers'
        :py:attr:`~culebra.trainer.abc.SinglePopTrainer.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._postprocess_iteration`
        methods of the
        :py:attr:`~trainer.ea.MultiPop.subpop_trainer_cls` class
        are dynamically overridden, in order to allow individuals exchange
        between subpopulation trainers, if necessary

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subpopulations.
        """

        def subpop_trainers_properties():
            """Obtain the properties of each subpopulation.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation.
            :rtype: :py:class:`list`
            """
            # Get the common attributes from the container trainer
            cls = self.subpop_trainer_cls
            common_properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            common_properties.update(self.subpop_trainer_params)

            # List with the common properties. Equal for all the subpopulations
            properties = []
            for _ in range(self.num_subpops):
                subpop_properties = {}
                for key, value in common_properties.items():
                    subpop_properties[key] = value
                properties.append(subpop_properties)

            # Particular properties for each subpop
            cls = self.__class__
            for (
                    property_sequence_name,
                    subpop_property_name
            ) in self._subpop_properties_mapping.items():

                # Values of the sequence
                property_sequence_values = getattr(
                    cls, property_sequence_name
                ).fget(self)

                # Check the properties' length
                if len(property_sequence_values) != self.num_subpops:
                    raise RuntimeError(
                        f"The length of {property_sequence_name} does not "
                        "match the number of subpopulations"
                    )
                for (
                        subpop_properties, subpop_property_value
                ) in zip(properties, property_sequence_values):
                    subpop_properties[
                        subpop_property_name] = subpop_property_value

            return properties

        # Get the subpopulations properties
        properties = subpop_trainers_properties()

        # Generate the subpopulations
        self._subpop_trainers = []

        for (
            index, (
                checkpoint_filename,
                subpop_properties
            )
        ) in enumerate(
            zip(self.subpop_trainer_checkpoint_filenames, properties)
        ):
            subpop_trainer = self.subpop_trainer_cls(**subpop_properties)
            subpop_trainer.checkpoint_filename = checkpoint_filename
            subpop_trainer.index = index
            subpop_trainer.container = self
            subpop_trainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subpop_trainer.__class__._postprocess_iteration = (
                self.send_representatives
            )

            subpop_trainer.__class__._init_representatives = partialmethod(
                self._init_subpop_trainer_representatives,
                solution_classes=self.solution_classes,
                species=self.species,
                representation_size=self.representation_size
            )

            self._subpop_trainers.append(subpop_trainer)


class TrainerTester(unittest.TestCase):
    """Test :py:class:`~culebra.trainer.abc.CooperativeTrainer`."""

    def test_init(self):
        """Test :py:meth:`~culebra.trainer.abc.CooperativeTrainer.__init__`."""
        valid_solution_classes = [
            ClassifierOptimizationSolution,
            FeatureSelectionSolution
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
        valid_subpop_trainer_cls = MySinglePopTrainerTrainer
        valid_num_subpops = 2
        invalid_funcs = (1, 1.5, {})
        valid_func = len

        # Try invalid types for the individual classes. Should fail
        for solution_cls in invalid_solution_class_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try invalid values for the individual classes. Should fail
        for solution_classes in invalid_solution_class_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try different values of solution_cls for each subpopulation
        trainer = MyTrainer(
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
                MyTrainer(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try invalid values for the species. Should fail
        for species in invalid_species_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    num_subpops=valid_num_subpops
                )

        # Try different values of species for each subpopulation
        trainer = MyTrainer(
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

        # Check the default value for num_subpops
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls
        )
        self.assertEqual(trainer.num_subpops, len(valid_species))

        # Check a value for num_subpops different from the number os species
        # It should fail
        with self.assertRaises(ValueError):
            MyTrainer(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_trainer_cls,
                num_subpops=18
            )

        # Try invalid representation topology functions. Should fail
        for func in invalid_funcs:
            with self.assertRaises(ValueError):
                trainer = MyTrainer(
                    valid_solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subpop_trainer_cls,
                    representation_topology_func=func
                )

        # Try the only valid value
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            representation_topology_func=(
                DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
            )
        )

        # Try invalid representation topology function parameters. Should fail
        with self.assertRaises(ValueError):
            trainer = MyTrainer(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subpop_trainer_cls,
                representation_topology_func_params=valid_func
            )

        # Try the only valid value
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls,
            representation_topology_func_params=(
                DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            )
        )

        # Test default params
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subpop_trainer_cls
        )

        # Create the subpopulations
        trainer._generate_subpop_trainers()

        for pop_size in trainer.pop_sizes:
            self.assertEqual(pop_size, DEFAULT_POP_SIZE)

    def test_representatives(self):
        """Test the representatives property."""
        params = {
            "solution_classes": [
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Try before the population has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), trainer.num_subpops)
        for best in best_ones:
            self.assertEqual(len(best), 0)

        # Generate the subpopulations
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "max_num_iters": 2,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Generate the subpopulations
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Generate the subpopulations
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyTrainer(**params)
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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyTrainer(**params)
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
                ClassifierOptimizationSolution,
                FeatureSelectionSolution
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
            "subpop_trainer_cls": MySinglePopTrainerTrainer,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyTrainer(**params)
        data = pickle.dumps(trainer1)
        trainer2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: :py:class:`~culebra.trainer.abc.CooperativeTrainer`
        :param trainer2: The second trainer
        :type trainer2: :py:class:`~culebra.trainer.abc.CooperativeTrainer`
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
