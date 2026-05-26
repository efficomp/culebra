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

"""Unit test for :class:`culebra.tools.decision_manager.LexicographicDM`."""

import unittest

from sklearn.svm import SVC

from culebra import DEFAULT_SIMILARITY_THRESHOLD
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
from culebra.tools.decision_manager import LexicographicDM
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
    "verbosity": False,
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


class LexicographicDMTester(unittest.TestCase):
    """Test :class:`~culebra.tools.decision_manager.LexicographicDM`."""

    def test_init(self):
        """Test the constructor."""
        trainer = MyTrainer(*subtrainers, **params)

        # Try without arguments
        dm = LexicographicDM(trainer)
        self.assertEqual(dm.trainer, trainer)
        self.assertEqual(
            dm.obj_thresholds,
            (DEFAULT_SIMILARITY_THRESHOLD,) * trainer.fitness_func.num_obj
        )

    def test_obj_thresholds(self):
        """Test the obj_thresholds property."""
        # Try default objective similarity thresholds
        trainer = MyTrainer(*subtrainers, **params)
        num_obj = trainer.fitness_func.num_obj

        # Try without arguments
        dm = LexicographicDM(trainer)

        self.assertEqual(
            dm.obj_thresholds,
            (DEFAULT_SIMILARITY_THRESHOLD,) * num_obj
        )

        invalid_threshold_types = (type, {}, len)
        invalid_threshold_value = -1
        valid_thresholds = [0, 0.33, 0.5, 2]

        # Try invalid types for the thresholds. Should fail
        for threshold in invalid_threshold_types:
            with self.assertRaises(TypeError):
                dm.obj_thresholds = threshold

        # Try invalid values for the threshold. Should fail
        with self.assertRaises(ValueError):
            dm.obj_thresholds = invalid_threshold_value

        # Try a fixed value for all the thresholds
        for threshold in valid_thresholds:
            dm.obj_thresholds = threshold
            # Check the thresholds
            self.assertEqual(
                dm.obj_thresholds, (threshold,) * num_obj
            )

        # Try different values of threshold for each objective
        dm.obj_thresholds = valid_thresholds[:num_obj]
        for th1, th2 in zip(
            valid_thresholds, dm.obj_thresholds
        ):
            self.assertEqual(th1, th2)

        # Try a wrong number of thresholds
        with self.assertRaises(ValueError):
            dm.obj_thresholds = valid_thresholds

    def test_evaluate(self):
        """Test the generation of all combinations of Pareto optimal solutions."""
        trainer = MyTrainer(*subtrainers, **params)
        dm = LexicographicDM(trainer)
        fitness_func = trainer.fitness_func

        # Generate populations of different individuals with the same fitness
        fitness_values = (0.5, 5, 1)
        for subtr in trainer.subtrainers:
            repeated_individuals = True
            while repeated_individuals:
                subtr._generate_pop()
                for ind in subtr.pop:
                    ind.fitness.values = fitness_values
                if len(subtr.best_solutions()[0]) == subtr.pop_size:
                    repeated_individuals = False

        all_combinations = dm._generate_all_combinations()

        # Check the fitnesses before evaluation
        for combination in all_combinations:
            for sol in combination:
                self.assertEqual(sol.fitness.values, fitness_values)

        # Evaluate the combinations
        all_combinations_fitness = dm._evaluate(all_combinations)

        # Check the fitnesses after evaluation
        for comb, comb_fitness in zip(
            all_combinations, all_combinations_fitness
        ):
            self.assertEqual(
                fitness_func.evaluate(comb[0], 0, comb),
                comb_fitness.values
            )

    def test_choose(self):
        """Test the selection method."""
        trainer = MyTrainer(*subtrainers, **params)
        fitness_func = trainer.fitness_func
        fitness_cls = fitness_func.fitness_cls
        dm = LexicographicDM(trainer, obj_thresholds=0.01)

        # Try with a not trained trainer
        self.assertIsNone(dm.select())

        # Generate populations of different individuals with the same fitness
        fitness_values = (0.5, 5, 1)
        for subtr in trainer.subtrainers:
            repeated_individuals = True
            while repeated_individuals:
                subtr._generate_pop()
                for ind in subtr.pop:
                    ind.fitness.values = fitness_values
                if len(subtr.best_solutions()[0]) == subtr.pop_size:
                    repeated_individuals = False

        # Generate all the combinations
        all_combinations = dm._generate_all_combinations()

        # Set the same fitness for all
        all_combinations_fitness = [
            fitness_cls(fitness_values) for _ in range(len(all_combinations))
        ]

        # Improve the fitness of one of them
        best_one_idx = 3
        all_combinations_fitness[best_one_idx] = fitness_cls((0.6, 5, 1))

        # Check that the best ine is at best_one_idx
        best_one = dm._choose(
            all_combinations,
            all_combinations_fitness
        )
        self.assertTrue(best_one == tuple(all_combinations[best_one_idx]))

        # Check the obj_thresholds
        all_combinations_fitness[0] = fitness_cls((0.597, 6, 1))
        all_combinations_fitness[1] = fitness_cls((0.603, 6, 1))
        all_combinations_fitness[2] = fitness_cls((0.4, 2, 1))

        # Check that the best ine is at best_one_idx
        best_one = dm._choose(
            all_combinations,
            all_combinations_fitness
        )
        self.assertTrue(best_one == tuple(all_combinations[best_one_idx]))

        # Check the tie-breaking rule
        all_combinations_fitness[best_one_idx] = fitness_cls((0.604, 6, 1))
        best_one = dm._choose(
            all_combinations,
            all_combinations_fitness
        )
        self.assertTrue(best_one == tuple(all_combinations[best_one_idx]))


if __name__ == '__main__':
    unittest.main()
