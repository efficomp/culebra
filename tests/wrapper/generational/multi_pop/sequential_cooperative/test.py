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

"""Unit test for :py:class:`wrapper.multi_pop.SequentialCooperative`."""

import os
import unittest
from time import sleep
from culebra.base import Dataset
from culebra.fitness_function.cooperative import KappaNumFeatsC as Fitness
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
from culebra.wrapper.single_pop import Elitist
from culebra.wrapper.multi_pop import SequentialCooperative


# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()


class WrapperTester(unittest.TestCase):
    """Test :py:class:`wrapper.multi_pop.SequentialCooperative`."""

    def test_init(self):
        """Test the constructor."""
        # Test default params
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
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": Elitist
        }

        # Create the wrapper
        wrapper = SequentialCooperative(**params)

        self.assertEqual(wrapper._current_gen, None)

    def test_new_state(self):
        """Test _new_state."""
        pop_size = 10
        # Create a default wrapper
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
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": Elitist,
            "pop_sizes": pop_size,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = SequentialCooperative(**params)
        wrapper._init_internals()
        wrapper._new_state()

        # Check the current generation
        self.assertEqual(wrapper._current_gen, 0)

        # Test that the logbook is None
        self.assertEqual(wrapper._logbook, None)

        # Test that current_gen is 0 and that all the subpopulations wrappers
        # have been initializated
        for index in range(wrapper.num_subpops):
            self.assertEqual(len(wrapper.subpop_wrappers[index].pop), pop_size)

    def test_checkpoining(self):
        """Test checkpointing."""
        # Create a default wrapper
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
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": Elitist,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False
        }

        # Create the wrapper
        wrapper1 = SequentialCooperative(**params)

        # Create the subpopulations
        wrapper1._init_search()

        # Set state attributes to dummy values
        wrapper1._runtime = 10
        wrapper1._current_gen = 19

        # Save the state of wrapper1
        wrapper1._save_state()

        # Create another wrapper
        wrapper2 = SequentialCooperative(**params)

        # Wrapper2 has no subpopulation wrappers yet
        self.assertEqual(wrapper2.subpop_wrappers, None)

        # Load the state of wrapper1 into wrapper2
        wrapper2._init_search()

        # Check that the state attributes of wrapper2 are equal to those of
        # wrapper1
        self.assertEqual(wrapper1.runtime, wrapper2.runtime)
        self.assertEqual(wrapper1._current_gen, wrapper2._current_gen)
        for (
            subpop_wrapper1,
            subpop_wrapper2
        ) in zip(wrapper1.subpop_wrappers, wrapper2.subpop_wrappers):
            self.assertEqual(
                len(subpop_wrapper1.pop), len(subpop_wrapper2.pop)
            )

        # Remove the checkpoint files
        os.remove(wrapper1.checkpoint_filename)
        for file in wrapper1.subpop_wrapper_checkpoint_filenames:
            os.remove(file)

    def test_representatives_exchange(self):
        """Test _preprocess_generation and _postprocess_generation."""
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
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": Elitist,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = SequentialCooperative(**params)
        wrapper._init_search()

        # Check that all the individuals are evaluated
        for subpop_wrapper in wrapper.subpop_wrappers:
            for ind in subpop_wrapper.pop:
                self.assertTrue(ind.fitness.valid)

        # All the queues should be empty
        for queue in wrapper._communication_queues:
            self.assertTrue(queue.empty())

        # Delete initial representatives
        for subpop_wrapper in wrapper.subpop_wrappers:
            subpop_wrapper._representatives = [
                [None] * wrapper.num_subpops
            ] * wrapper.representation_size

        # Send representatives
        wrapper._postprocess_generation()
        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should not be empty
        for queue in wrapper._communication_queues:
            self.assertFalse(queue.empty())

        # Receive representatives
        wrapper._preprocess_generation()
        # Wait for the parallel queue processing
        sleep(1)

        # All the queues should be empty again
        for queue in wrapper._communication_queues:
            self.assertTrue(queue.empty())

        # Check the received representatives
        for (
            subpop_index,
            subpop_wrapper
        ) in enumerate(wrapper.subpop_wrappers):
            for representatives in subpop_wrapper._representatives:
                for ind_index, ind in enumerate(representatives):
                    if ind_index == subpop_index:
                        self.assertEqual(ind, None)
                    else:
                        self.assertTrue(ind.fitness.valid)

    def test_search(self):
        """Test _search."""
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
            "fitness_function": Fitness(dataset),
            "subpop_wrapper_cls": Elitist,
            "num_gens": 2,
            "pop_sizes": 10,
            "representation_size": 2,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the wrapper
        wrapper = SequentialCooperative(**params)
        wrapper._init_search()

        self.assertEqual(wrapper._current_gen, 0)
        wrapper._search()

        num_gens = params["num_gens"]
        self.assertEqual(wrapper._current_gen, num_gens)
        num_evals = 0
        for subpop_wrapper in wrapper.subpop_wrappers:
            self.assertEqual(subpop_wrapper._current_gen, num_gens)
            num_evals += subpop_wrapper.num_evals

        self.assertEqual(wrapper.num_evals, num_evals)


if __name__ == '__main__':
    unittest.main()
