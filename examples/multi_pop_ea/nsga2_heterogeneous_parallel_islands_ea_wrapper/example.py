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

"""Usage example of an heterogeneous parallel islands wrapper."""

from os import cpu_count

from pandas import Series, DataFrame, MultiIndex
from deap.tools import selTournament

from sklearn.neighbors import KNeighborsClassifier

from culebra.solution.feature_selection import Species, IntVector
from culebra.fitness_function.feature_selection import (
    KappaIndex,
    NumFeats,
    FSMultiObjectiveDatasetScorer
)
from culebra.trainer.topology import ring_destinations
from culebra.trainer.ea import NSGA, HeterogeneousParallelIslandsEA
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

# Split the dataset
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Oversample the training data to make all the clases have the same number
# of samples
training_data = training_data.oversample(random_seed=0)

n_neighbors = 5
"""Number of neighbors for k-NN."""

knn_classifier = KNeighborsClassifier(n_neighbors)

# Training fitness function
training_fitness_function = KappaNumFeats(
    training_data=training_data, classifier=knn_classifier, cv_folds=5
)

# Test fitness function
test_fitness_function = KappaNumFeats(
    training_data=training_data, test_data=test_data, classifier=knn_classifier
)

# Number of islands
num_subtrainers = cpu_count()

# Parameters for the wrapper
params = {
    "solution_cls": IntVector,
    "species": Species(num_feats=dataset.num_feats, min_size=1),
    "fitness_function": training_fitness_function,
    "subtrainer_cls": NSGA,
    "num_subtrainers": num_subtrainers,
    "representation_topology_func": ring_destinations,
    "representation_topology_func_params": {"offset": 1},
    "representation_selection_func": selTournament,
    "representation_selection_func_params": {"tournsize": 3},
    "crossover_probs": 0.8,
    "mutation_probs": 0.2,
    "gene_ind_mutation_probs": tuple(
        (1.0 + i) / (num_subtrainers + 1) for i in range(num_subtrainers)
    ),
    "max_num_iters": 30,
    "pop_sizes": tuple(
        dataset.num_feats + i*5 for i in range(num_subtrainers)
    ),
    "checkpoint_enable": False
}

# Create the wrapper
wrapper = HeterogeneousParallelIslandsEA(**params)

# Train the wrapper
print("Training ...")
wrapper.train()

# Get the best training solutions
best_ones = wrapper.best_solutions()

# Store the training results in a dataframe
results = DataFrame()
species = Series(dtype=object)
individuals = Series(dtype=object)
training_kappa = Series(dtype=float)
training_nf = Series(dtype=int)

for index, pop_best in enumerate(best_ones):
    for ind in pop_best:
        species.loc[len(species)] = index
        individuals.loc[len(individuals)] = ind
        training_kappa.loc[len(training_kappa)] = ind.fitness.getValues()[0]
        training_nf.loc[len(training_nf)] = int(ind.fitness.getValues()[1])

results['Species'] = species
results['Individual'] = individuals
results['Training Kappa'] = training_kappa
results['Training NF'] = training_nf

# Apply the test data
wrapper.test(best_found=best_ones, fitness_func=test_fitness_function)

# Add the test results to the dataframe
test_kappa = Series(dtype=float)
test_nf = Series(dtype=int)
for index, pop_best in enumerate(best_ones):
    for ind in pop_best:
        test_kappa.loc[len(test_kappa)] = ind.fitness.getValues()[0]
        test_nf.loc[len(test_nf)] = int(ind.fitness.getValues()[1])

results['Test Kappa'] = test_kappa
results['Test NF'] = test_nf

# Assign a new index to columns
results.set_index(['Species', 'Individual'], inplace=True)
iterables = [["Training", "Test"], ["Kappa", "NF"]]
index = MultiIndex.from_product(iterables, names=["Phase", "Obj"])
results.columns = index

# Print the results
print("\nResults ...")
print(results)

print(f"\nNum Evals: {wrapper.num_evals}")
print(f"Runtime  : {wrapper.runtime}")
