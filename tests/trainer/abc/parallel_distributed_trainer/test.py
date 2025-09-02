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

"""Unit test for :py:class:`culebra.trainer.abc.ParallelDistributedTrainer`."""

import unittest
from multiprocess.managers import DictProxy

from culebra.trainer.abc import (
    SingleSpeciesTrainer,
    ParallelDistributedTrainer
)
from culebra.trainer.topology import ring_destinations
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
)
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats,
    FSMultiObjectiveDatasetScorer
)
from culebra.tools import Dataset


# Fitness function
def KappaNumFeats(
    training_data,
    test_data=None,
    test_prop=None,
    cv_folds=None,
    classifier=None
):
    """Fitness Function."""
    return FSMultiObjectiveDatasetScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            test_prop=test_prop,
            cv_folds=cv_folds,
            classifier=classifier
        ),
        NumFeats()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)

fitness_func = KappaNumFeats(dataset)


class MySingleSpeciesTrainer(SingleSpeciesTrainer):
    """Dummy implementation of a trainer method."""

    def _do_iteration(self):
        """Implement an iteration of the search process."""
        self.sol = Solution(self.species, self.fitness_function.fitness_cls)
        self.evaluate(self.sol)


class MyDistributedTrainer(ParallelDistributedTrainer):
    """Dummy implementation of a parallel distributed trainer."""

    def _generate_subtrainers(self):
        self._subtrainers = []
        subtrainer_params = {
            "solution_cls": Solution,
            "species": species,
            "fitness_function": self.fitness_function,
            "max_num_iters": self.max_num_iters,
            "checkpoint_enable": self.checkpoint_enable,
            "checkpoint_freq": self.checkpoint_freq,
            "checkpoint_filename": self.checkpoint_filename,
            "verbose": self.verbose,
            "random_seed": self.random_seed
        }

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subtrainer_checkpoint_filenames):
            subtrainer = self.subtrainer_cls(**subtrainer_params)
            subtrainer.checkpoint_filename = checkpoint_filename

            subtrainer.index = index
            subtrainer.container = self
            self._subtrainers.append(subtrainer)

    @property
    def representation_topology_func(self):
        """Get and set the representation topology function."""
        return ring_destinations

    @property
    def representation_topology_func_params(self):
        """Get and set the representation topology function parameters."""
        return {"offset": 3}


class TrainerTester(unittest.TestCase):
    """Test :py:class:`culebra.trainer.abc.ParallelDistributedTrainer`."""

    def test_init(self):
        """Test the constructor."""
        # Test default params
        trainer = MyDistributedTrainer(
            fitness_func,
            MySingleSpeciesTrainer
        )

        self.assertEqual(trainer._manager, None)
        self.assertEqual(trainer._subtrainer_state_proxies, None)

    def test_num_evals(self):
        """Test the num_evals property."""
        # Parameters for the trainer
        num_subtrainers = 2
        max_num_iters = 5
        subtrainer_cls = MySingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "max_num_iters": max_num_iters,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyDistributedTrainer(**params)

        self.assertEqual(trainer._num_evals, None)

        trainer.train()

        # Test the number of evaluations
        global_num_evals = 0
        for subtrainer in trainer.subtrainers:
            global_num_evals += subtrainer.num_evals

        self.assertEqual(global_num_evals, trainer.num_evals)

    def test_runtime(self):
        """Test the runtime property."""
        # Parameters for the trainer
        num_subtrainers = 2
        max_num_iters = 5
        subtrainer_cls = MySingleSpeciesTrainer
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "max_num_iters": max_num_iters,
            "verbose": False,
            "checkpoint_enable": False
        }

        # Create the trainer
        trainer = MyDistributedTrainer(**params)

        self.assertEqual(trainer.runtime, None)

        trainer.train()

        # Test the runtime
        global_runtime = 0
        for subtrainer in trainer.subtrainers:
            if subtrainer.runtime > global_runtime:
                global_runtime = subtrainer.runtime

        self.assertEqual(global_runtime, trainer.runtime)

    def test_new_state(self):
        """Test _new_state."""
        # Create a default trainer
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        params = {
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False
        }

        # Test default params
        trainer = MyDistributedTrainer(**params)
        trainer._init_internals()
        trainer._new_state()

        self.assertEqual(trainer._runtime, None)
        self.assertEqual(trainer._num_evals, None)

    def test_init_internals(self):
        """Test _init_internals."""
        # Create a default trainer
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        params = {
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False
        }

        # Test default params
        trainer = MyDistributedTrainer(**params)
        trainer._init_internals()

        # Test that the communication queues have been created
        self.assertIsInstance(trainer._communication_queues, list)
        self.assertEqual(
            len(trainer._communication_queues), trainer.num_subtrainers
        )

        for index1 in range(trainer.num_subtrainers):
            for index2 in range(index1 + 1, trainer.num_subtrainers):
                self.assertNotEqual(
                    id(trainer._communication_queues[index1]),
                    id(trainer._communication_queues[index2])
                )

        # Test that proxies have been created
        self.assertIsInstance(trainer._subtrainer_state_proxies, list)
        self.assertEqual(
            len(trainer._subtrainer_state_proxies), trainer.num_subtrainers
        )
        for proxy in trainer._subtrainer_state_proxies:
            self.assertIsInstance(proxy, DictProxy)

        for index1 in range(trainer.num_subtrainers):
            for index2 in range(index1 + 1, trainer.num_subtrainers):
                self.assertNotEqual(
                    id(trainer._subtrainer_state_proxies[index1]),
                    id(trainer._subtrainer_state_proxies[index2])
                )

    def test_reset_internals(self):
        """Test _reset_internals."""
        # Create a default trainer
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        params = {
            "species": species,
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "verbose": False
        }

        # Test default params
        trainer = MyDistributedTrainer(**params)
        trainer._init_internals()
        trainer._reset_internals()

        # Check manager
        self.assertEqual(trainer._manager, None)

        # Check the subtrainer_state_proxies
        self.assertEqual(trainer._subtrainer_state_proxies, None)

    def test_search(self):
        """Test _search."""
        # Create a default trainer
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        max_num_iters = 10
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "max_num_iters": max_num_iters,
            "checkpoint_enable": False,
            "verbose": False
        }

        # Test the search method
        trainer = MyDistributedTrainer(**params)
        trainer._init_search()

        trainer._search()

        num_evals = 0
        for subtrainer in trainer.subtrainers:
            self.assertEqual(subtrainer._current_iter, max_num_iters)
            num_evals += subtrainer.num_evals

        self.assertEqual(trainer.num_evals, num_evals)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Create a default trainer
        subtrainer_cls = MySingleSpeciesTrainer
        num_subtrainers = 2
        max_num_iters = 10
        params = {
            "fitness_function": fitness_func,
            "subtrainer_cls": subtrainer_cls,
            "num_subtrainers": num_subtrainers,
            "max_num_iters": max_num_iters,
            "checkpoint_enable": False,
            "verbose": False
        }

        trainer = MyDistributedTrainer(**params)
        trainer._init_search()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
