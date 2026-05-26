#!/usr/bin/env python3
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

"""Unit test for :class:`culebra.tools.abc.DecisionManager`."""

import unittest
from os import remove
from copy import copy, deepcopy

from sklearn.svm import SVC

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_func.feature_selection import (
    KappaIndex,
    NumFeats
)
from culebra.fitness_func.svc_optimization import C
from culebra.fitness_func.cooperative import FSSVCScorer
from culebra.trainer.abc import (
    ParallelDistributedTrainer,
    CooperativeTrainer
)
from culebra.trainer.ea import ElitistEA
from culebra.tools.abc import DecisionManager
from culebra.tools import Dataset


# Fitness function
def KappaNumFeatsC(
    training_data,
    test_data=None,
    cv_num_folds=None,
    cv_fixed_folds=None
):
    """Fitness Function."""
    return FSSVCScorer(
        KappaIndex(
            training_data=training_data,
            test_data=test_data,
            classifier=SVC(kernel='rbf'),
            cv_num_folds=cv_num_folds,
            cv_fixed_folds=cv_fixed_folds
        ),
        NumFeats(),
        C()
    )


# Dataset
dataset = Dataset.load_from_uci(name="Wine")

# Preprocess the dataset
dataset = dataset.drop_missing().scale().remove_outliers(random_seed=0)

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Oversample the training data to make all the clases have the same number
# of samples
training_data = training_data.oversample(random_seed=0)

# Training fitness function
training_fitness_func = KappaNumFeatsC(training_data, cv_num_folds=5)

# Test fitness function
test_fitness_func = KappaNumFeatsC(training_data, test_data)

# Species to optimize a SVM-based classifier
classifierOptimizationSpecies = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[1000, 1000],
    names=["C", "gamma"]
)

# Species for the feature selection problem
featureSelectionSpecies1 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    max_feat=dataset.num_feats//2,
)
featureSelectionSpecies2 = FeatureSelectionSpecies(
    num_feats=dataset.num_feats,
    min_feat=dataset.num_feats//2 + 1,
)

# Subtrainers params
subtrainer_params = {
    "fitness_func": training_fitness_func,
    "crossover_prob": 0.8,
    "mutation_prob": 0.2,
    "pop_size": dataset.num_feats//2,
    "max_num_iters": 10,
    "checkpoint_activation": False,
    "verbosity": False
}

# Parameters for the wrapper
params = {
    "num_representatives": 2
}

# Create the wrapper
subtrainers = (
    ElitistEA(
        solution_cls=ClassifierOptimizationIndividual,
        species=classifierOptimizationSpecies,
        # At least one hyperparameter/feature will be mutated
        gene_ind_mutation_prob=1.0/classifierOptimizationSpecies.num_params,
        **subtrainer_params
    ),
    ElitistEA(
        solution_cls=FeatureSelectionIndividual,
        species=featureSelectionSpecies1,
        gene_ind_mutation_prob=2.0/dataset.num_feats,
        **subtrainer_params
    ),
    ElitistEA(
        solution_cls=FeatureSelectionIndividual,
        species=featureSelectionSpecies2,
        gene_ind_mutation_prob=2.0/dataset.num_feats,
        **subtrainer_params
    )
)

# Trainer
class MyTrainer(ParallelDistributedTrainer, CooperativeTrainer):
    """Parallel implementation of a cooperative trainer."""


class MyDecisionManager(DecisionManager):
    """Decision manager subclass."""

    def _evaluate(self, all_combinations):
        """Evaluate all the combinations of Pareto optimal solutions."""
        fitness_func = self.trainer.fitness_func
        fitness_cls = fitness_func.fitness_cls

        # Evaluate each combination
        all_combinations_fitness = []
        for solution in all_combinations:
            all_combinations_fitness.append(
                fitness_cls(fitness_func.evaluate(solution[0], 0, solution))
            )

        return all_combinations_fitness

    def _choose(self, all_combinations, all_combinations_fitness):
        """Choose one combination of solutions."""
        if len(all_combinations) == 0:
            return None
        # Get the best
        sorted_indices = [
            i[0] for i in sorted(
                enumerate(all_combinations_fitness),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        return tuple(all_combinations[sorted_indices[0]])


class DecisionManagerTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.DecisionManager`."""

    def test_init(self):
        """Test the constructor."""
        trainer = MyTrainer(*subtrainers, **params)

        # Try default params
        dm = MyDecisionManager(trainer)

        self.assertEqual(dm.trainer, trainer)

        # Try an invalid trainer
        with self.assertRaises(TypeError):
            MyDecisionManager('a')

    def test_generate_all_combinations(self):
        """Test the generation of all combinations of Pareto optimal solutions."""
        trainer = MyTrainer(*subtrainers, **params)
        dm = MyDecisionManager(trainer)

        # Try with a not trained trainer
        all_combinations = dm._generate_all_combinations()
        self.assertEqual(len(all_combinations), 0)

        # Generate populations of different individuals with the same fitness
        fitness_values = (1, 1, 1)
        for subtr in trainer.subtrainers:
            repeated_individuals = True
            while repeated_individuals:
                subtr._generate_pop()
                for ind in subtr.pop:
                    ind.fitness.values = fitness_values
                if len(subtr.best_solutions()[0]) == subtr.pop_size:
                    repeated_individuals = False

        all_combinations = dm._generate_all_combinations()

        self.assertEqual(
            len(all_combinations),
            pow(subtrainer_params["pop_size"], trainer.num_subtrainers)
        )

        for sol0 in trainer.subtrainers[0].best_solutions()[0]:
            for sol1 in trainer.subtrainers[1].best_solutions()[0]:
                for sol2 in trainer.subtrainers[2].best_solutions()[0]:
                    self.assertTrue(
                        [sol0, sol1, sol2] in all_combinations
                    )

    def test_select(self):
        """Test the selection method."""
        trainer = MyTrainer(*subtrainers, **params)
        dm = MyDecisionManager(trainer)
        fitness_func = trainer.fitness_func

        # Try with a not trained trainer
        self.assertIsNone(dm.select())

        # Generate populations of different individuals with the same fitness
        fitness_values = (1, 1, 1)
        for subtr in trainer.subtrainers:
            repeated_individuals = True
            while repeated_individuals:
                subtr._generate_pop()
                for ind in subtr.pop:
                    ind.fitness.values = fitness_values
                if len(subtr.best_solutions()[0]) == subtr.pop_size:
                    repeated_individuals = False

        # Let the DM select the best solution
        best_solution = dm.select()

        # Eval the best solution
        for idx, sol in enumerate(best_solution):
            sol.fitness.values = fitness_func.evaluate(
                sol, idx, best_solution
            )

        # Evaluate the individuals of all subtrainers
        for subtr_idx, subtr in enumerate(trainer.subtrainers):
            for ind in subtr.pop:
                ind.fitness.values = fitness_func.evaluate(
                    ind, subtr_idx, best_solution
                )

        # Assess the best solution
        for subtr_idx, subtr in enumerate(trainer.subtrainers):
            for ind in subtr.pop:
                self.assertTrue(best_solution[subtr_idx] >= ind)

    def test_copy(self):
        """Test the :meth:`~culebra.tools.abc.DecisionManager.__copy__` method."""
        trainer = MyTrainer(*subtrainers, **params)

        dm1 = MyDecisionManager(trainer)
        dm2 = copy(dm1)

        # Copy only copies the first level (dm1 != dm2)
        self.assertNotEqual(id(dm1), id(dm2))

        # The trainer is shared
        self.assertEqual(id(dm1._trainer), id(dm2._trainer))

    def test_deepcopy(self):
        """Test :meth:`~culebra.tools.abc.DecisionManager.__deepcopy__`."""
        trainer = MyTrainer(*subtrainers, **params)

        dm1 = MyDecisionManager(trainer)
        dm2 = deepcopy(dm1)

        # Check the copy
        self._check_deepcopy(dm1, dm2)

    def test_serialization(self):
        """Serialization test.

        Test the :meth:`~culebra.tools.abc.DecisionManager.__setstate__` and
        :meth:`~culebra.tools.abc.DecisionManager.__reduce__` methods.
        """
        trainer = MyTrainer(*subtrainers, **params)

        dm1 = MyDecisionManager(trainer)

        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        dm1.dump(serialized_filename)
        dm2 = MyDecisionManager.load(serialized_filename)

        # Check the copy
        self._check_deepcopy(dm1, dm2)

        # Remove the serialized file
        remove(serialized_filename)

    def _check_deepcopy(self, dm1, dm2):
        """Check if *dm1* is a deepcopy of *dm2*.

        :param dm1: The first decision manager
        :type dm1: ~culebra.tools.abc.DecisionManager
        :param dm2: The second decision manager
        :type dm2: ~culebra.tools.abc.DecisionManager
        """
        # Copies all the levels
        self.assertFalse(dm1 is dm2)
        self.assertFalse(dm1.trainer is dm2.trainer)

        for subtr1, subtr2 in zip(
            dm1.trainer.subtrainers, dm2.trainer.subtrainers
        ):
            self.assertFalse(subtr1 is subtr2)
            self.assertTrue(subtr1.solution_cls is subtr2.solution_cls)
            self.assertFalse(subtr1.species is subtr2.species)
            self.assertTrue(subtr1.container is dm1.trainer)
            self.assertTrue(subtr2.container is dm2.trainer)


if __name__ == '__main__':
    unittest.main()
