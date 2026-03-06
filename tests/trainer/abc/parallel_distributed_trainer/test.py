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

"""Unit test for :class:`culebra.trainer.abc.ParallelDistributedTrainer`."""

import unittest
from multiprocess.managers import DictProxy

from culebra.trainer.abc import (
    CentralizedTrainer,
    CommonFitnessFunctionDistributedTrainer,
    ParallelDistributedTrainer
)
from culebra.trainer.topology import ring_destinations
from culebra.solution.feature_selection import (
    Species,
    BinarySolution as Solution
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


class MySubtrainer(CentralizedTrainer):
    """Dummy implementation of a subtrainer."""

    def _do_iteration(self):
        """Implement an iteration of the training process."""
        self.pop = [
            self.solution_cls(
                self.species,
                self.fitness_func.fitness_cls,
                features=[1, 2 ,3]
            ),
            self.solution_cls(
                self.species,
                self.fitness_func.fitness_cls,
                features = [1, 2]
            )
        ]
        for sol in self.pop:
            self._current_iter_evals += self.evaluate(
                sol, self.fitness_func, self.index, self.cooperators
            )

    def _get_objective_stats(self) -> dict:
        """Gather the objective stats."""
        return self._stats.compile(self.pop) if self._stats else {}


class MyTrainer(
    ParallelDistributedTrainer,
    CommonFitnessFunctionDistributedTrainer
):
    """Dummy implementation of a parallel distributed trainer."""

    @property
    def _default_topology_func(self):
        """Default topology function."""
        return ring_destinations

    @staticmethod
    def receive_representatives(subtrainer) -> None:
        pass

    @staticmethod
    def send_representatives(subtrainer) -> None:
        pass


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Default species for all the tests
species = Species(num_feats=dataset.num_feats)

fitness_func = KappaNumFeats(dataset)

# Subtrainers params
subtrainer_params = {
    "fitness_func": fitness_func,
    "solution_cls": Solution,
    "species": species,
    "max_num_iters": 5,
    "checkpoint_activation": False,
    "verbosity": False
}


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.abc.ParallelDistributedTrainer`."""

    def test_runtime(self):
        """Test the runtime property."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)

        self.assertEqual(trainer.runtime, None)

        trainer._init_training()
        for subtrainer in trainer.subtrainers:
            subtrainer.train()
        trainer._finish_training()

        # Test the runtime
        global_runtime = 0
        for subtrainer in trainer.subtrainers:
            global_runtime = max(subtrainer.runtime, global_runtime)

        self.assertEqual(global_runtime, trainer.runtime)

    def test_init_internals(self):
        """Test _init_internals."""
        num_subtrainers = 2
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)

        # Test the internals
        trainer._init_internals()

        # Check manager
        self.assertNotEqual(trainer._manager, None)

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
        for subtr in trainer.subtrainers:
            self.assertIsInstance(subtr.state_proxy, DictProxy)

        for idx1, subtr1 in enumerate(trainer.subtrainers):
            for subtr2 in trainer.subtrainers[idx1+1:]:
                self.assertFalse(subtr1.state_proxy is subtr2.state_proxy)

    def test_reset_internals(self):
        """Test _reset_internals."""
        num_subtrainers = 2
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)
        trainer._init_internals()
        trainer._reset_internals()

        # Check manager
        self.assertEqual(trainer._manager, None)

        # Test that proxies are None
        for subtr in trainer.subtrainers:
            self.assertIsNone(subtr.state_proxy)

    def test_do_training(self):
        """Test _do_training."""
        num_subtrainers = 2
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)
        # Test the training method
        trainer.train()

        num_evals = 0
        for subtrainer in trainer.subtrainers:
            num_evals += subtrainer.num_evals

        self.assertEqual(trainer.num_evals, num_evals)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_subtrainers = 2
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)
        trainer._init_training()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
