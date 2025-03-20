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

"""Unit test for :py:class:`culebra.trainer.abc.CooperativeTrainer`."""

import unittest
from os import remove
from copy import copy, deepcopy
from functools import partialmethod

from deap.tools import ParetoFront

from culebra.trainer.abc import SingleSpeciesTrainer, CooperativeTrainer
from culebra.trainer.topology import full_connected_destinations
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
dataset = Dataset.load_from_uci(name="Wine")

# Remove outliers
dataset.remove_outliers()

# Normalize inputs
dataset.robust_scale()


class MySingleSpeciesTrainer(SingleSpeciesTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self.sol = self.solution_cls(
            self.species, self.fitness_function.Fitness
        )
        self.evaluate(self.sol)

    def best_solutions(self):
        """Get the best solutions found for each species."""
        hof = ParetoFront()
        hof.update([self.sol])
        return [hof]


class MyTrainer(CooperativeTrainer):
    """Dummy implementation of a cooperative co-evolutionary algorithm."""

    _subtrainer_properties_mapping = {
        "solution_classes": "solution_cls",
        "species": "species"
    }
    """Map the container names of properties sequences to the different
    subtrainer property names."""

    def _new_state(self) -> None:
        """Generate a new trainer state."""
        super()._new_state()

        # Generate the state of all subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._new_state()

    def _init_search(self):
        super()._init_search()
        for island_trainer in self.subtrainers:
            island_trainer._init_search()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the metrics before each iteration is run.
        """
        super()._start_iteration()
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            # Fix the current iteration
            subtrainer._current_iter = self._current_iter
            # Start the iteration
            subtrainer._start_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration_stats()

    def _generate_subtrainers(self) -> None:
        """Generate the subtrainers.

        Also assign an
        :py:attr:`~culebra.trainer.abc.SingleSpeciesTrainer.index` and a
        :py:attr:`~culebra.trainer.abc.SingleSpeciesTrainer.container` to each
        subtrainer :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        trainer, change the subtrainers'
        :py:attr:`~culebra.trainer.abc.SingleSpeciesTrainer.checkpoint_filename`
        according to the container checkpointing file name and each
        subtrainer index.

        Finally, the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.abc.DistributedTrainer.subtrainer_cls` class
        are dynamically overridden, in order to allow individuals exchange
        between subtrainers, if necessary

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subtrainers.
        """

        def subtrainers_properties():
            """Obtain the properties of each subtrainer.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subtrainers.

            :return: The properties of each subtrainer.
            :rtype: :py:class:`list`
            """
            # Get the common attributes from the container trainer
            cls = self.subtrainer_cls
            common_properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subtrainer custom atributes
            common_properties.update(self.subtrainer_params)

            # List with the common properties. Equal for all the subtrainers
            properties = []
            for _ in range(self.num_subtrainers):
                subtrainer_properties = {}
                for key, value in common_properties.items():
                    subtrainer_properties[key] = value
                properties.append(subtrainer_properties)

            # Particular properties for each subtrainer
            cls = self.__class__
            for (
                    property_sequence_name,
                    subtrainer_property_name
            ) in self._subtrainer_properties_mapping.items():

                # Values of the sequence
                property_sequence_values = getattr(
                    cls, property_sequence_name
                ).fget(self)

                # Check the properties' length
                if len(property_sequence_values) != self.num_subtrainers:
                    raise RuntimeError(
                        f"The length of {property_sequence_name} does not "
                        "match the number of subtrainers"
                    )
                for (
                        subtrainer_properties, subtrainer_property_value
                ) in zip(properties, property_sequence_values):
                    subtrainer_properties[
                        subtrainer_property_name] = subtrainer_property_value

            return properties

        # Get the subtrainers properties
        properties = subtrainers_properties()

        # Generate the subtrainers
        self._subtrainers = []

        for (
            index, (
                checkpoint_filename,
                subtrainer_properties
            )
        ) in enumerate(
            zip(self.subtrainer_checkpoint_filenames, properties)
        ):
            subtrainer = self.subtrainer_cls(**subtrainer_properties)
            subtrainer.checkpoint_filename = checkpoint_filename
            subtrainer.index = index
            subtrainer.container = self
            subtrainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subtrainer.__class__._postprocess_iteration = (
                self.send_representatives
            )

            subtrainer.__class__._init_representatives = partialmethod(
                self._init_subtrainer_representatives,
                solution_classes=self.solution_classes,
                species=self.species,
                representation_size=self.representation_size
            )

            self._subtrainers.append(subtrainer)


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
        valid_subtrainer_cls = MySingleSpeciesTrainer
        valid_num_subtrainers = 2

        # Try invalid types for the individual classes. Should fail
        for solution_cls in invalid_solution_class_types:
            with self.assertRaises(TypeError):
                MyTrainer(
                    solution_cls,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try invalid values for the individual classes. Should fail
        for solution_classes in invalid_solution_class_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    solution_classes,
                    valid_species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try different values of solution_cls for each subtrainer
        trainer = MyTrainer(
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
                MyTrainer(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try invalid values for the species. Should fail
        for species in invalid_species_values:
            with self.assertRaises(ValueError):
                MyTrainer(
                    valid_solution_classes,
                    species,
                    valid_fitness_func,
                    valid_subtrainer_cls,
                    num_subtrainers=valid_num_subtrainers
                )

        # Try different values of species for each subtrainer
        trainer = MyTrainer(
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

        # Check the default value for num_subtrainers
        trainer = MyTrainer(
            valid_solution_classes,
            valid_species,
            valid_fitness_func,
            valid_subtrainer_cls
        )
        self.assertEqual(trainer.num_subtrainers, len(valid_species))

        # Check a value for num_subtrainers different from the number of
        # species. It should fail
        with self.assertRaises(ValueError):
            MyTrainer(
                valid_solution_classes,
                valid_species,
                valid_fitness_func,
                valid_subtrainer_cls,
                num_subtrainers=18
            )

        # Test default params
        trainer = MyTrainer(
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

        # Create the subtrainers
        trainer._generate_subtrainers()

        for solution_cls, subtrainer in zip(
            trainer.solution_classes, trainer.subtrainers
        ):
            self.assertEqual(solution_cls, subtrainer.solution_cls)

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
            "representation_size": 2,
            "subtrainer_cls": MySingleSpeciesTrainer,
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
                subtrainer_index,
                subtrainer
                ) in enumerate(trainer.subtrainers):
            for (
                context_index, _
            ) in enumerate(subtrainer.representatives):
                self.assertEqual(
                    the_representatives[context_index][subtrainer_index - 1],
                    subtrainer.representatives[
                        context_index][subtrainer_index - 1]
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
            "subtrainer_cls": MySingleSpeciesTrainer,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Try before the subtrainers have been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, list)
        self.assertEqual(len(best_ones), trainer.num_subtrainers)
        for best in best_ones:
            self.assertEqual(len(best), 0)

        # Generate the subtrainers
        trainer._init_search()
        trainer._start_iteration()
        trainer._do_iteration()

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
            "subtrainer_cls": MySingleSpeciesTrainer,
            "max_num_iters": 2,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)

        # Try before the subtrainers have been created
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
            "subtrainer_cls": MySingleSpeciesTrainer,
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
            "subtrainer_cls": MySingleSpeciesTrainer,
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
            "subtrainer_cls": MySingleSpeciesTrainer,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer1 = MyTrainer(**params)

        pickle_filename = "my_pickle.gz"
        trainer1.save_pickle(pickle_filename)
        trainer2 = MyTrainer.load_pickle(pickle_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the pickle file
        remove(pickle_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
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
            "subtrainer_cls": MySingleSpeciesTrainer,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

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
