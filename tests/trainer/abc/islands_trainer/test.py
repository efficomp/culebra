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

"""Unit test for :class:`culebra.trainer.abc.IslandsTrainer`."""

import unittest
from time import sleep
from queue import Empty

from deap.tools import ParetoFront


from culebra.trainer import (
    DEFAULT_ISLANDS_TOPOLOGY_FUNC
)
from culebra.trainer.abc import (
    CentralizedTrainer,
    SequentialDistributedTrainer,
    IslandsTrainer
)

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

    def __init__(
        self,
        fitness_func,
        solution_cls,
        species,
        custom_termination_func=None,
        max_num_iters=None,
        checkpoint_activation=None,
        checkpoint_freq=None,
        checkpoint_basename=None,
        verbosity=None,
        random_seed=None
    ):
        """Constructor."""
        super().__init__(
            fitness_func=fitness_func,
            solution_cls=solution_cls,
            species=species,
            custom_termination_func=custom_termination_func,
            max_num_iters=max_num_iters,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_basename=checkpoint_basename,
            verbosity=verbosity,
            random_seed=random_seed
        )
        self.pop = None

    def best_solutions(self):
        """Get the best solutions found for each species."""
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return (hof,)

    def select_representatives(self):
        """Select representative solutions."""
        if self.container:
            return self.container.representatives_selection_func(
                self.pop, self.container.num_representatives
                )

        return []

    def integrate_representatives(self, representatives):
        """Integrate representative solutions."""
        self.pop.extend(representatives)

    def _do_iteration(self):
        """Implement an iteration of the training process."""
        self.pop = [
            self.solution_cls(self.species, self.fitness_func.fitness_cls),
            self.solution_cls(self.species, self.fitness_func.fitness_cls),
        ]
        for sol in self.pop:
            self._current_iter_evals += self.evaluate(
                sol, self.fitness_func, self.index, self.cooperators
            )

    def _get_objective_stats(self) -> dict:
        """Gather the objective stats."""
        return self._stats.compile(self.pop) if self._stats else {}


class MyTrainer(SequentialDistributedTrainer, IslandsTrainer):
    """Sequential implementation of an islands-based trainer."""


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
    "checkpoint_activation": False,
    "verbosity": False
}


class TrainerTester(unittest.TestCase):
    """Test :class:`culebra.trainer.abc.IslandsTrainer`."""

    def test_default_topology_func(self):
        """Test _default_topology_func."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer = MyTrainer(*subtrainers)
        self.assertEqual(trainer.topology_func, DEFAULT_ISLANDS_TOPOLOGY_FUNC)

    def test_best_solutions(self):
        """Test best_solutions."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )

        trainer = MyTrainer(*subtrainers)

        # Try before the population has been created
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, tuple)
        self.assertEqual(len(best_ones), 1)
        self.assertEqual(len(best_ones[0]), 0)

        # Train
        trainer.train()

        # Try again
        best_ones = trainer.best_solutions()

        # Test that a list with only one species is returned
        self.assertIsInstance(best_ones, tuple)
        self.assertEqual(len(best_ones), 1)
        for sol in best_ones[0]:
            self.assertIsInstance(sol, Solution)

    def test_receive_representatives(self):
        """Test receive_representatives."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        trainer = MyTrainer(*subtrainers)

        # Train
        trainer.train()

        sol = Solution(species, fitness_func.fitness_cls)
        for queue in trainer._communication_queues:
            queue.put([sol])

            # Wait for the queue processing
            sleep(1)

        # Call to receive representatives, assigned to
        # subtrainer.receive_representatives_func
        for subtr in trainer.subtrainers:
            subtr.receive_representatives_func(subtr)

        # Check that al subtrainers have received the solution
        for subtr in trainer.subtrainers:
            self.assertTrue(sol in subtr.pop)

    def test_send_representatives(self):
        """Test send_representatives."""
        num_subtrainers = 3
        subtrainers = tuple(
            MySubtrainer(**subtrainer_params) for _ in range(num_subtrainers)
        )
        trainer = MyTrainer(*subtrainers)

        # Train
        trainer.train()

        # Wait for the parallel queue processing
        sleep(1)

        # Clear communication queues
        for queue in trainer._communication_queues:
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break

        # Set an iteration that should not provoke representatives sending
        for subtr in trainer.subtrainers:
            subtr._current_iter = trainer.representatives_exchange_freq + 1

            # Call to send representatives, assigned to
            # subtrainer.send_representatives_func
            subtr.send_representatives_func(subtr)

        # All the queues should be empty
        for queue in trainer._communication_queues:
            self.assertTrue(queue.empty())


        # Set an iteration that should provoke representatives sending
        for subtr in trainer.subtrainers:
            subtr._current_iter = trainer.representatives_exchange_freq

            # Call to send representatives, assigned to
            # subtrainer.send_representatives_func
            subtr.send_representatives_func(subtr)

            # Wait for the queue processing
            sleep(1)

        # None of the the queues should be empty
        for queue in trainer._communication_queues:
            self.assertFalse(queue.empty())


if __name__ == '__main__':
    unittest.main()
