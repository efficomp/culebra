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
from functools import partial

from sklearn.neighbors import KNeighborsClassifier

from pandas import Series, DataFrame, MultiIndex
from deap.tools import selTournament

from culebra.solution.feature_selection import Species, IntVector
from culebra.fitness_func import MultiObjectiveFitnessFunction
from culebra.fitness_func.feature_selection import (
    KappaIndex,
    NumFeats
)
from culebra.trainer.abc import (
    ParallelDistributedTrainer,
    IslandsTrainer
)
from culebra.trainer.ea import NSGA
from culebra.trainer.topology import ring_destinations
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


# Wrapper
class Wrapper(ParallelDistributedTrainer, IslandsTrainer):
    """Parallel implementation of an islands-based trainer."""


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
training_fitness_func = KappaNumFeats(
    training_data=training_data, classifier=knn_classifier, cv_folds=5
)

# Test fitness function
test_fitness_func = KappaNumFeats(
    training_data=training_data, test_data=test_data, classifier=knn_classifier
)

# Number of islands
num_subtrainers = cpu_count()

# Subtrainers params
common_subtrainer_params = {
    "fitness_func": training_fitness_func,
    "solution_cls": IntVector,
    "species": Species(num_feats=dataset.num_feats, min_size=1),
    "crossover_prob": 0.8,
    "mutation_prob": 0.2,
    "gene_ind_mutation_prob": 1.0/dataset.num_feats,
    "pop_size": dataset.num_feats,
    "max_num_iters": 30,
    "checkpoint_activation": False
}

# Create the subtrainers
subtrainers = tuple(
    NSGA(**common_subtrainer_params) for _ in range(num_subtrainers)
)

# Introduce heterogeneity
for idx, subtr in enumerate(subtrainers):
    subtr.gene_ind_mutation_prob = (1.0 + idx) / (num_subtrainers + 1)
    subtr.pop_size = dataset.num_feats + idx*5

# Parameters for the wrapper
params = {
    "num_representatives": 3,
    "representatives_selection_func": partial(selTournament, tournsize=3),
    "topology_func": partial(ring_destinations, offset=2)
}

# Create the wrapper
wrapper = Wrapper(*subtrainers, **params)

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

for species_idx, pop_best in enumerate(best_ones):
    for ind in pop_best:
        ind_idx = len(species)
        species.loc[ind_idx] = species_idx
        individuals.loc[ind_idx] = ind
        training_kappa.loc[ind_idx] = ind.fitness.values[0]
        training_nf.loc[ind_idx] = int(ind.fitness.values[1])

results['Species'] = species
results['Individual'] = individuals
results['Training Kappa'] = training_kappa
results['Training NF'] = training_nf

# Apply the test data
wrapper.test(best_found=best_ones, fitness_func=test_fitness_func)

# Add the test results to the dataframe
test_kappa = Series(dtype=float)
test_nf = Series(dtype=int)
for species_idx, pop_best in enumerate(best_ones):
    for ind in pop_best:
        ind_idx = len(test_kappa)
        test_kappa.loc[ind_idx] = ind.fitness.values[0]
        test_nf.loc[ind_idx] = int(ind.fitness.values[1])

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
