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

"""Unit test for :class:`~culebra.trainer.abc.DistributedTrainer`."""

import unittest
import os
from copy import copy, deepcopy

from multiprocess.queues import Queue
from deap.tools import Logbook

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.trainer import (
    DEFAULT_NUM_REPRESENTATIVES,
    DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ
)

from culebra.trainer.abc import CentralizedTrainer, DistributedTrainer
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


class MyTrainer(DistributedTrainer):
    """Dummy implementation of a distributed trainer."""

    @property
    def _default_topology_func(self):
        """Default topology function."""
        return ring_destinations

    @property
    def runtime(self):
        """Dummy implementation."""
        return 0

    @property
    def fitness_func(self):
        """Training fitness function."""
        return self.subtrainers[0].fitness_func

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
    "max_num_iters": 5,
    "checkpoint_activation": False,
    "verbosity": False
}


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.trainer.abc.DistributedTrainer`."""

    def test_init(self):
        """Test :meth:`~culebra.trainer.abc.DistributedTrainer.__init__`."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Try with insufficient subtrainers. Should fail
        insufficient_subtrainers = ((), subtrainers[:1],)
        for subtrainer_seq in insufficient_subtrainers:
            with self.assertRaises(ValueError):
                MyTrainer(*subtrainer_seq)

        # Try invalid subtrainers
        invalid_subtrainers = ((1, subtrainers[1]), (subtrainers[0], 2))
        for subtrainer_seq in invalid_subtrainers:
            with self.assertRaises(ValueError):
                MyTrainer(*subtrainer_seq)

        # Try invalid types for the number of representatives. Should fail
        invalid_num_representatives = (str, 'a', -0.001, 1.001)
        for num_representatives in invalid_num_representatives:
            with self.assertRaises(TypeError):
                MyTrainer(
                    *subtrainers,
                    num_representatives=num_representatives
                )

        # Try invalid values for the number of representatives. Should fail
        invalid_num_representatives = (-1, 0)
        for num_representatives in invalid_num_representatives:
            with self.assertRaises(ValueError):
                MyTrainer(
                    *subtrainers,
                    num_representatives=num_representatives
                )

        # Try invalid representatives selection function. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    *subtrainers,
                    representatives_selection_func=func
                )

        # Try invalid types for the representatives exchange frequency.
        # Should fail
        invalid_repr_exchange_freqs = (str, 'a', 1.5)
        for repr_exchange_freq in invalid_repr_exchange_freqs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    *subtrainers,
                    representatives_exchange_freq=repr_exchange_freq
                )

        # Try invalid values for the representatives exchange frequency.
        # Should fail
        invalid_repr_exchange_freqs = (-1, 0)
        for repr_exchange_freq in invalid_repr_exchange_freqs:
            with self.assertRaises(ValueError):
                MyTrainer(
                    *subtrainers,
                    representatives_exchange_freq=repr_exchange_freq
                )

        # Try invalid topology functions. Should fail
        invalid_funcs = ('a', 1.5)
        for func in invalid_funcs:
            with self.assertRaises(TypeError):
                MyTrainer(
                    *subtrainers, topology_func=func
                )

        # Test default params
        trainer = MyTrainer(*subtrainers)

        self.assertEqual(trainer.subtrainers, subtrainers)
        self.assertEqual(trainer.num_subtrainers, len(subtrainers))
        for idx, subtrainer in enumerate(trainer.subtrainers):
            self.assertEqual(idx, subtrainer.index)
            self.assertEqual(trainer, subtrainer.container)
            self.assertEqual(
                trainer.receive_representatives,
                subtrainer.receive_representatives_func
            )
            self.assertEqual(
                trainer.send_representatives,
                subtrainer.send_representatives_func
            )
            self.assertEqual(trainer.fitness_func, subtrainer.fitness_func)

        self.assertEqual(
            trainer.num_representatives, DEFAULT_NUM_REPRESENTATIVES
        )
        self.assertEqual(
            trainer.representatives_exchange_freq,
            DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ
        )
        self.assertEqual(trainer.topology_func, ring_destinations)

        # Test custom params
        params = {
            "num_representatives": 3,
            "representatives_selection_func": max,
            "representatives_exchange_freq": 2,
            "topology_func": len
        }

        trainer = MyTrainer(*subtrainers, **params)

        self.assertEqual(
            trainer.num_representatives, params["num_representatives"]
        )
        self.assertEqual(
            trainer.representatives_selection_func,
            params["representatives_selection_func"]
        )
        self.assertEqual(
            trainer.representatives_exchange_freq,
            params["representatives_exchange_freq"]
        )
        self.assertEqual(
            trainer.topology_func,
            params["topology_func"]
        )

    def test_iteration_metric_names(self):
        """Test the iteration_metric_names property."""
        # Try several number of subtrainers
        max_num_subtrainers = 10
        for num_subtrainers in range(2, max_num_subtrainers+1):
            subtrainers = tuple(
                MySubtrainer(**subtrainer_params)
                for _ in range(num_subtrainers)
            )

            # Create the trainer
            trainer = MyTrainer(*subtrainers)
            self.assertEqual(
                trainer.iteration_metric_names,
                trainer.subtrainers[1].iteration_metric_names
            )

    def test_iteration_obj_stats(self):
        """Test the iteration_obj_stats property."""
        # Try several number of subtrainers
        max_num_subtrainers = 10
        for num_subtrainers in range(2, max_num_subtrainers+1):
            subtrainers = tuple(
                MySubtrainer(**subtrainer_params)
                for _ in range(num_subtrainers)
            )

            # Create the trainer
            trainer = MyTrainer(*subtrainers)
            self.assertEqual(
                trainer.iteration_obj_stats,
                trainer.subtrainers[1].iteration_obj_stats
            )

    def test_num_subtrainers(self):
        """Test the num_subtrainers property."""
        # Try several number of subtrainers
        max_num_subtrainers = 10
        for num_subtrainers in range(2, max_num_subtrainers+1):
            subtrainers = tuple(
                MySubtrainer(**subtrainer_params)
                for _ in range(num_subtrainers)
            )

            # Create the trainer
            trainer = MyTrainer(*subtrainers)
            self.assertEqual(trainer.num_subtrainers, num_subtrainers)

    def test_training_finished(self):
        """Test the training_finished property."""
        # Try several number of subtrainers
        num_subtrainers = 10
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params)
            for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)

        for subtr in trainer.subtrainers:
            self.assertFalse(trainer.training_finished)
            subtr._training_finished = True

        self.assertTrue(trainer.training_finished)

    def test_logbook(self):
        """Test the logbook property."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)

        self.assertEqual(trainer.logbook, None)

        trainer._init_training()
        for subtrainer in trainer.subtrainers:
            subtrainer.train()

        # Test the logbook property
        global_logbook_len = 0
        for subtrainer in trainer.subtrainers:
            global_logbook_len += len(subtrainer.logbook)

        self.assertIsInstance(trainer.logbook, Logbook)
        self.assertEqual(global_logbook_len, len(trainer.logbook))

    def test_num_evals(self):
        """Test the num_evals property."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)

        self.assertEqual(trainer.num_evals, None)

        trainer._init_training()
        for subtrainer in trainer.subtrainers:
            subtrainer.train()

        # Test the number of evaluations
        global_num_evals = 0
        for subtrainer in trainer.subtrainers:
            global_num_evals += subtrainer.num_evals

        self.assertEqual(global_num_evals, trainer.num_evals)

    def test_num_iters(self):
        """Test the num_iters property."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)

        self.assertEqual(trainer.num_iters, None)

        trainer._init_training()
        for subtrainer in trainer.subtrainers:
            subtrainer.train()

        # Test the number of iterations
        self.assertEqual(trainer.num_iters, subtrainer_params["max_num_iters"])

        # Test when a subtrainer has done more iterations
        trainer.subtrainers[1]._current_iter *= 2
        self.assertEqual(trainer.subtrainers[1].num_iters, trainer.num_iters)

    def test_init_internals(self):
        """Test _init_internals."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)
        trainer._init_internals()

        # Test that the communication queues have been created
        self.assertIsInstance(trainer._communication_queues, list)
        self.assertEqual(
            len(trainer._communication_queues), trainer.num_subtrainers
        )
        for queue in trainer._communication_queues:
            self.assertIsInstance(queue, Queue)

        for index1 in range(trainer.num_subtrainers):
            for index2 in range(index1 + 1, trainer.num_subtrainers):
                self.assertNotEqual(
                    id(trainer._communication_queues[index1]),
                    id(trainer._communication_queues[index2])
                )

    def test_reset_internals(self):
        """Test _reset_internals."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        # Create the trainer
        trainer = MyTrainer(*subtrainers)
        trainer._init_internals()
        trainer._reset_internals()

        # Test the communication queues
        self.assertEqual(trainer._communication_queues, None)

    def test_reset(self):
        """Test reset."""
        self.test_reset_internals()

    def test_init_training(self):
        """Test reset."""
        self.test_init_internals()

    def test_copy(self):
        """Test the __copy__ method."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer1 = MyTrainer(*subtrainers)
        trainer2 = copy(trainer1)

        # Copy only copies the first level
        self.assertFalse(trainer1 is trainer2)

        # The objects attributes are shared
        self.assertTrue(
            trainer1.topology_func is trainer2.topology_func)

        for subtr1, subtr2 in zip(trainer1.subtrainers, trainer2.subtrainers):
            self.assertTrue(subtr1 is subtr2)

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        # Parameters for the trainer
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer1 = MyTrainer(*subtrainers)
        trainer2 = deepcopy(trainer1)

        # Check the copy
        self._check_deepcopy(trainer1, trainer2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer1 = MyTrainer(*subtrainers)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        trainer1.dump(serialized_filename)
        trainer2 = MyTrainer.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(trainer1, trainer2)

        # Remove the serialized file
        os.remove(serialized_filename)

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer = MyTrainer(*subtrainers)
        trainer._init_training()
        self.assertIsInstance(repr(trainer), str)
        self.assertIsInstance(str(trainer), str)

    def _check_deepcopy(self, trainer1, trainer2):
        """Check if *trainer1* is a deepcopy of *trainer2*.

        :param trainer1: The first trainer
        :type trainer1: ~culebra.trainer.abc.DistributedTrainer
        :param trainer2: The second trainer
        :type trainer2: ~culebra.trainer.abc.DistributedTrainer
        """
        self.assertFalse(trainer1 is trainer2)

        for subtr1, subtr2 in zip(trainer1.subtrainers, trainer2.subtrainers):
            self.assertFalse(subtr1 is subtr2)
            self.assertTrue(subtr1.solution_cls is subtr2.solution_cls)
            self.assertFalse(subtr1.species is subtr2.species)
            self.assertTrue(subtr1.container is trainer1)
            self.assertTrue(subtr2.container is trainer2)


if __name__ == '__main__':
    unittest.main()
