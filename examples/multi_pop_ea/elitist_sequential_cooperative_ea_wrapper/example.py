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

"""Usage example of elitist sequential cooperative wrapper."""

from pandas import Series, DataFrame, MultiIndex

from culebra.solution.feature_selection import (
    Species as FeatureSelectionSpecies,
    BitVector as FeatureSelectionIndividual
)
from culebra.solution.parameter_optimization import (
    Species as ClassifierOptimizationSpecies,
    Individual as ClassifierOptimizationIndividual
)
from culebra.fitness_function.cooperative import KappaNumFeatsC
from culebra.trainer.ea import ElitistEA, SequentialCooperativeEA
from culebra.tools import Dataset

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Remove outliers
dataset.remove_outliers()

# Normalize inputs between 0 and 1
dataset.normalize()
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Training fitness function, 50% of samples used for validation
training_fitness_function = KappaNumFeatsC(
    training_data=training_data, test_prop=0.5
)

# Fix the fitness similarity threshold to 0.1 for all the objectives
training_fitness_function.set_fitness_thresholds(0.1)

# Test fitness function
test_fitness_function = KappaNumFeatsC(
    training_data=training_data, test_data=test_data
)

# Species to optimize a SVM-based classifier
classifierOptimizationSpecies = ClassifierOptimizationSpecies(
    lower_bounds=[0, 0],
    upper_bounds=[1000, 1000],
    names=["C", "gamma"]
)

# Species for the feature selection problem
featureSelectionSpecies = FeatureSelectionSpecies(dataset.num_feats)

# Parameters for the wrapper
params = {
    "solution_classes": [
        ClassifierOptimizationIndividual,
        FeatureSelectionIndividual
    ],
    "species": [
        classifierOptimizationSpecies,
        featureSelectionSpecies
    ],
    "fitness_function": training_fitness_function,
    "subtrainer_cls": ElitistEA,
    "representation_size": 2,
    "crossover_probs": 0.8,
    "mutation_probs": 0.2,
    "gene_ind_mutation_probs": (
        # At least one hyperparameter will be mutated
        1.0/classifierOptimizationSpecies.num_params,
        # probability of 0.5 to select/unselect any feature
        0.5
    ),
    "max_num_iters": 100,
    "pop_sizes": 100,
    "verbose": True
}

# Create the wrapper
wrapper = SequentialCooperativeEA(**params)

# Train the wrapper
print("Training ...")
wrapper.train()

# Get the best training solutions
best_ones = wrapper.best_solutions()
best_representatives = wrapper.best_representatives()

# Store the training results in a dataframe
results = DataFrame()
species = Series(dtype=object)
individuals = Series(dtype=object)
training_kappa = Series(dtype=float)
training_nf = Series(dtype=int)
training_C = Series(dtype=float)

for index, pop_best in enumerate(best_ones):
    for ind in pop_best:
        species.loc[len(species)] = index
        individuals.loc[len(individuals)] = ind
        training_kappa.loc[len(training_kappa)] = ind.fitness.getValues()[0]
        training_nf.loc[len(training_nf)] = int(ind.fitness.getValues()[1])
        training_C.loc[len(training_C)] = ind.fitness.getValues()[2]

results['Species'] = species
results['Individual'] = individuals
results['Training Kappa'] = training_kappa
results['Training NF'] = training_nf
results['Training C'] = training_C

# Apply the test data
wrapper.test(
    best_found=best_ones,
    fitness_func=test_fitness_function,
    representatives=best_representatives
)

# Add the test results to the dataframe
test_kappa = Series(dtype=float)
test_nf = Series(dtype=int)
test_C = Series(dtype=float)
for index, pop_best in enumerate(best_ones):
    for ind in pop_best:
        test_kappa.loc[len(test_kappa)] = ind.fitness.getValues()[0]
        test_nf.loc[len(test_nf)] = int(ind.fitness.getValues()[1])
        test_C.loc[len(test_C)] = ind.fitness.getValues()[2]

results['Test Kappa'] = test_kappa
results['Test NF'] = test_nf
results['Test C'] = test_C

# Assign a new index to columns
results.set_index(['Species', 'Individual'], inplace=True)
iterables = [["Training", "Test"], ["Kappa", "NF", "C"]]
index = MultiIndex.from_product(iterables, names=["Phase", "Obj"])
results.columns = index

# Print the results
print("\nResults ...")
print("\nTraining ...")
print(results["Training"])
print("\nTest ...")
print(results["Test"])

print(f"\nNum Evals: {wrapper.num_evals}")
print(f"Runtime  : {wrapper.runtime}")
