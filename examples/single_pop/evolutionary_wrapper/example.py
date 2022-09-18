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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Usage example of a simple evolutionary wrapper."""

from pandas import Series, DataFrame, MultiIndex
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import NumFeats
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import IntVector
from culebra.wrapper.single_pop import Evolutionary

# Dataset
DATASET_PATH = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/australian/australian.dat"
)

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Remove outliers
dataset.remove_outliers()

# Normalize inputs between 0 and 1
dataset.normalize()
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Training fitness function, 50% of samples used for validation
training_fitness_function = NumFeats(
    training_data=training_data, test_prop=0.5
)

# Test fitness function
test_fitness_function = NumFeats(
    training_data=training_data, test_data=test_data
)

# Parameters for the wrapper
params = {
    "individual_cls": IntVector,
    "species": Species(num_feats=dataset.num_feats, min_size=1),
    "fitness_function": training_fitness_function,
    "crossover_prob": 0.8,
    "mutation_prob": 0.2,
    "gene_ind_mutation_prob": 0.5,
    "num_gens": 100,
    "pop_size": 100
}

# Create the wrapper
wrapper = Evolutionary(**params)

# Train the wrapper
print("Training ...")
wrapper.train()

# Get the best training solutions
best_ones = wrapper.best_solutions()

# Store the training results in a dataframe
results = DataFrame()
species = Series(dtype=object)
individuals = Series(dtype=object)
training_nf = Series(dtype=int)

for index, pop_best in enumerate(best_ones):
    for ind in pop_best:
        species.loc[len(species)] = index
        individuals.loc[len(individuals)] = ind
        training_nf.loc[len(training_nf)] = int(ind.fitness.getValues()[0])

results['Species'] = species
results['Individual'] = individuals
results['Training NF'] = training_nf

# Apply the test data
wrapper.test(best_found=best_ones, fitness_func=test_fitness_function)

# Add the test results to the dataframe
test_nf = Series(dtype=int)
for index, pop_best in enumerate(best_ones):
    for ind in pop_best:
        test_nf.loc[len(test_nf)] = int(ind.fitness.getValues()[0])

results['Test NF'] = test_nf

# Assign a new index to columns
results.set_index(["Species", "Individual"], inplace=True)
iterables = [["Training", "Test"], ["NF"]]
index = MultiIndex.from_product(iterables, names=["Phase", "Obj"])
results.columns = index

# Print the results
print("\nResults ...")
print(results)

print(f"\nNum Evals: {wrapper.num_evals}")
print(f"Runtime  : {wrapper.runtime}")
