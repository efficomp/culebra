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

"""Test for :class:`~culebra.trainer.abc.SequentialDistributedTrainer`."""

import unittest

from culebra.trainer.abc import (
    CentralizedTrainer,
    CommonFitnessFunctionDistributedTrainer,
    SequentialDistributedTrainer
)
from culebra.trainer.topology import full_connected_destinations
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
    SequentialDistributedTrainer,
    CommonFitnessFunctionDistributedTrainer
):
    """Dummy implementation of a sequential distributed trainer."""

    @property
    def _default_topology_func(self):
        """Default topology function."""
        return full_connected_destinations

    @staticmethod
    def receive_representatives(subtrainer):
        pass

    @staticmethod
    def send_representatives(subtrainer):
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
    "max_num_iters": 2,
    "verbosity": False,
    "checkpoint_activation": False
}


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.trainer.abc.SequentialDistributedTrainer`."""

    def test_runtime(self):
        """Test the runtime property."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers,
        )

        self.assertEqual(trainer.runtime, None)

        # Test the runtime
        trainer.train()
        acc_runtime = 0
        for subtr in trainer.subtrainers:
            acc_runtime += subtr.runtime

        self.assertEqual(acc_runtime, trainer.runtime)

        # Test when a subtrainer has done more iterations
        trainer.subtrainers[1]._current_iter = 4
        self.assertEqual(trainer.subtrainers[1].num_iters, trainer.num_iters)

    def test_init_training(self):
        """Test the _init_training method."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        trainer = MyTrainer(*subtrainers)

        # Init the training
        trainer._init_training()

        # Check the internals of trainer
        self.assertIsNotNone(trainer._communication_queues)

        # Check the internal and state ob subtrainers
        for subtr in trainer.subtrainers:
            self.assertIsNotNone(subtr._stats)
            self.assertIsNotNone(subtr.logbook)

    def test_finish_training(self):
        """Test the _finish_training method."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        trainer = MyTrainer(*subtrainers)

        # Init the training
        trainer._init_training()
        trainer._finish_training()

        self.assertTrue(trainer.training_finished)

    def test_do_training(self):
        """Test _do_training and _finish_training."""
        num_subtrainers = 2
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        trainer = MyTrainer(*subtrainers)
        # Force a different termination criterion for subtrainer 0
        trainer.subtrainers[0].max_num_iters = 5
        trainer.train()

        num_evals = 0
        for subtr in trainer.subtrainers:
            self.assertEqual(subtr.num_iters, subtr.max_num_iters)
            num_evals += subtr.num_evals

        self.assertEqual(trainer.num_evals, num_evals)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        # Create a default trainer
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        trainer = MyTrainer(*subtrainers)
        trainer._init_training()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)


if __name__ == '__main__':
    unittest.main()
