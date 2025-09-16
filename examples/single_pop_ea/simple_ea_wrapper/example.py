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

"""Usage example of a simple evolutionary wrapper."""

from pandas import Series, DataFrame, MultiIndex

from sklearn.neighbors import KNeighborsClassifier

from culebra.solution.feature_selection import Species, IntVector
from culebra.fitness_function.feature_selection import KappaIndex
from culebra.trainer.ea import SimpleEA
from culebra.tools import Dataset


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
training_fitness_function = KappaIndex(
    training_data=training_data, classifier=knn_classifier, cv_folds=5
)
training_fitness_function.obj_thresholds = 0.05

# Test fitness function
test_fitness_function = KappaIndex(
    training_data=training_data, test_data=test_data, classifier=knn_classifier
)

# Parameters for the wrapper
params = {
    "solution_cls": IntVector,
    "species": Species(num_feats=dataset.num_feats, min_size=1),
    "fitness_function": training_fitness_function,
    "crossover_prob": 0.8,
    "mutation_prob": 0.2,
    "gene_ind_mutation_prob": 1.0/dataset.num_feats,
    "max_num_iters": 100,
    "pop_size": dataset.num_feats
}

# Create the wrapper
wrapper = SimpleEA(**params)

# Train the wrapper
print("Training ...")
wrapper.train()

# Get the best training solutions
best_ones = wrapper.best_solutions()

# Store the training results in a dataframe
results = DataFrame()
species = Series(dtype=object)
individuals = Series(dtype=object)
training_kappa = Series(dtype=int)

for species_idx, pop_best in enumerate(best_ones):
    for ind in pop_best:
        ind_idx = len(species)
        species.loc[ind_idx] = species_idx
        individuals.loc[ind_idx] = ind
        training_kappa.loc[ind_idx] = ind.fitness.values[0]

results['Species'] = species
results['Individual'] = individuals
results['Training Kappa'] = training_kappa

# Apply the test data
wrapper.test(best_found=best_ones, fitness_func=test_fitness_function)

# Add the test results to the dataframe
test_kappa = Series(dtype=int)
for species_idx, pop_best in enumerate(best_ones):
    for ind in pop_best:
        ind_idx = len(test_kappa)
        test_kappa.loc[ind_idx] = ind.fitness.values[0]

results['Test Kappa'] = test_kappa

# Assign a new index to columns
results.set_index(["Species", "Individual"], inplace=True)
iterables = [["Training", "Test"], ["Kappa"]]
index = MultiIndex.from_product(iterables, names=["Phase", "Obj"])
results.columns = index

# Print the results
print("\nResults ...")
print(results)

print(f"\nNum Evals: {wrapper.num_evals}")
print(f"Runtime  : {wrapper.runtime}")
