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

"""Unit test for :class:`culebra.trainer.abc.CooperativeTrainer`."""

import unittest
from time import sleep
from queue import Empty

from sklearn.svm import SVC

from deap.tools import ParetoFront

from culebra.trainer import DEFAULT_COOPERATIVE_TOPOLOGY_FUNC

from culebra.trainer.abc import (
    CentralizedTrainer,
    SequentialDistributedTrainer,
    CooperativeTrainer
)
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BinarySolution as FeatureSelectionSolution
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Solution as ClassifierOptimizationSolution
)
from culebra.fitness_func.feature_selection import (
    KappaIndex,
    NumFeats
)
from culebra.fitness_func.svc_optimization import C
from culebra.fitness_func.cooperative import FSSVCScorer
from culebra.tools import Dataset


# Fitness function
def KappaNumFeatsC(
    training_data, test_data=None, cv_folds=None
):
    """Fitness Function."""
    return FSSVCScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            classifier=SVC(kernel='rbf'),
            cv_folds=cv_folds
        ),
        NumFeats(),
        C()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Training fitness function
fitness_func = KappaNumFeatsC(dataset, cv_folds=5)


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

    def _new_state(self):
        super()._new_state()
        self._current_iter_evals = 0
        self.pop = [
            self.solution_cls(self.species, self.fitness_func.fitness_cls),
            self.solution_cls(self.species, self.fitness_func.fitness_cls)
        ]
        for sol in self.pop:
            self._current_iter_evals += self.evaluate(
                sol, self.fitness_func, self.index, self.cooperators
            )


    def _do_iteration(self):
        """Implement an iteration of the training process."""
        self.pop = [
            self.solution_cls(self.species, self.fitness_func.fitness_cls),
            self.solution_cls(self.species, self.fitness_func.fitness_cls)
        ]
        for sol in self.pop:
            self._current_iter_evals += self.evaluate(
                sol, self.fitness_func, self.index, self.cooperators
            )

    def _get_objective_stats(self) -> dict:
        """Gather the objective stats."""
        return self._stats.compile(self.pop) if self._stats else {}


class MyTrainer(SequentialDistributedTrainer, CooperativeTrainer):
    """Sequential implementation of a cooperative trainer."""


# Subtrainers parameters
classifier_optimization_species = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[100000, 100000],
    names=["C", "gamma"]
)

feature_selection_species = FeatureSelectionSpecies(dataset.num_feats)

common_subtrainer_params = {
    "fitness_func": fitness_func,
    "max_num_iters": 1,
    "checkpoint_activation": False,
    "verbosity": False
}


class TrainerTester(unittest.TestCase):
    """Test :class:`~culebra.trainer.abc.CooperativeTrainer`."""


    def test_default_topology_func(self):
        """Test _default_topology_func."""
        # Subtrainers
        subtrainers = [
            MySubtrainer(
                solution_cls=ClassifierOptimizationSolution,
                species=classifier_optimization_species,
                **common_subtrainer_params
            ),
            MySubtrainer(
                solution_cls=FeatureSelectionSolution,
                species=feature_selection_species,
                **common_subtrainer_params
            )
        ]

        # Test default params
        trainer = MyTrainer(*subtrainers)

        self.assertEqual(
            trainer.topology_func,
            DEFAULT_COOPERATIVE_TOPOLOGY_FUNC
        )

    def test_best_solutions(self):
        """Test best_solutions."""
        # Subtrainers
        subtrainers = [
            MySubtrainer(
                solution_cls=ClassifierOptimizationSolution,
                species=classifier_optimization_species,
                **common_subtrainer_params
            ),
            MySubtrainer(
                solution_cls=FeatureSelectionSolution,
                species=feature_selection_species,
                **common_subtrainer_params
            )
        ]

        trainer = MyTrainer(*subtrainers)

        # Try before training
        best_ones = trainer.best_solutions()
        self.assertIsInstance(best_ones, tuple)
        self.assertEqual(len(best_ones), trainer.num_subtrainers)
        for best in best_ones:
            self.assertEqual(len(best), 0)

        # Train
        trainer.train()

        # Try again
        best_ones = trainer.best_solutions()

        # Test that a list with hof per species returned
        self.assertIsInstance(best_ones, tuple)
        self.assertEqual(len(best_ones), trainer.num_subtrainers)
        for hof, subtr in zip(best_ones, trainer.subtrainers):
            for sol in hof:
                self.assertIsInstance(sol, subtr.solution_cls)

    def test_best_cooperators(self):
        """Test the best_cooperators method."""
        # Subtrainers
        subtrainers = [
            MySubtrainer(
                solution_cls=ClassifierOptimizationSolution,
                species=classifier_optimization_species,
                **common_subtrainer_params
            ),
            MySubtrainer(
                solution_cls=FeatureSelectionSolution,
                species=feature_selection_species,
                **common_subtrainer_params
            )
        ]

        trainer = MyTrainer(*subtrainers)

        # Try before the subtrainers have been created
        the_cooperators = trainer.best_cooperators()

        # The cooperators should be None
        self.assertIsNone(the_cooperators)

        trainer.train()

        # Try after training
        the_cooperators = trainer.best_cooperators()

        # Check the cooperators
        self.assertIsInstance(the_cooperators, list)
        self.assertEqual(
            len(the_cooperators), trainer.num_representatives
        )
        for context in the_cooperators:
            self.assertIsInstance(context, list)
            self.assertEqual(len(context), trainer.num_subtrainers)
            for sol, subtr in zip(context, trainer.subtrainers):
                self.assertTrue(subtr.species.is_member(sol))

    def test_receive_representatives(self):
        """Test receive_representatives."""
        # Subtrainers
        subtrainers = [
            MySubtrainer(
                solution_cls=ClassifierOptimizationSolution,
                species=classifier_optimization_species,
                **common_subtrainer_params
            ),
            MySubtrainer(
                solution_cls=FeatureSelectionSolution,
                species=feature_selection_species,
                **common_subtrainer_params
            )
        ]

        trainer = MyTrainer(*subtrainers)

        # Train
        trainer.train()

        sender_index = 0
        the_representatives = (
            trainer.subtrainers[sender_index].select_representatives()
        )

        for index in range(trainer.num_subtrainers):
            if index != sender_index:
                trainer._communication_queues[index].put(
                    (sender_index, the_representatives)
                )
            # Wait for the parallel queue processing
            sleep(1)


        # Call to receive representatives, assigned to
        # subtrainer.receive_representatives_func
        for subtr in trainer.subtrainers:
            subtr.receive_representatives_func(subtr)

        # Check the received values
        for recv_index, subtr in enumerate(trainer.subtrainers):
            if recv_index != sender_index:
                for sol_index, sol in enumerate(the_representatives):
                    self.assertEqual(
                        subtr.cooperators[sol_index][sender_index], sol
                    )

                # Check that all the individuals have been reevaluated
                for sol in subtr.pop:
                    self.assertTrue(sol.fitness.is_valid)

    def test_send_representatives(self):
        """Test send_representatives."""
        # Subtrainers
        subtrainers = [
            MySubtrainer(
                solution_cls=ClassifierOptimizationSolution,
                species=classifier_optimization_species,
                **common_subtrainer_params
            ),
            MySubtrainer(
                solution_cls=FeatureSelectionSolution,
                species=feature_selection_species,
                **common_subtrainer_params
            )
        ]

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
            subtr._current_iter = (
                trainer.representatives_exchange_freq + 1
            )

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

            # Wait for the parallel queue processing
            sleep(1)

        # None of the the queues should be empty
        for queue in trainer._communication_queues:
            self.assertFalse(queue.empty())


if __name__ == '__main__':
    unittest.main()
