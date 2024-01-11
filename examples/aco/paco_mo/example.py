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

"""Usage example of the quality-based PACO trainer."""

from pandas import Series, DataFrame

from culebra.solution.tsp import Species, Ant
from culebra.trainer.aco import PACO_MO
from culebra.fitness_function.tsp import DoublePathLength


# Try the KroAB100 problem
fitness_func = DoublePathLength.fromTSPLib(
    "https://raw.githubusercontent.com/mastqe/tsplib/master/kroA100.tsp",
    "https://raw.githubusercontent.com/mastqe/tsplib/master/kroB100.tsp"
)

# Problem graph's number of nodes
num_nodes = fitness_func.num_nodes

# Species for the problem solutions
species = Species(num_nodes)

# Generate and evaluate a greedy solution for the problem
greedy_solution = fitness_func.greedy_solution(species)

initial_pheromone = 1
max_pheromone = 5

# Trainer parameters
params = {
    "solution_cls": Ant,
    "species": species,
    "fitness_function": fitness_func,
    "initial_pheromone": initial_pheromone,
    "max_pheromone": max_pheromone,
    "col_size": 50,
    "pop_size": 10,
    "max_num_iters": 500,
    "checkpoint_enable": False
}

# Create the wrapper
trainer = PACO_MO(**params)

# Train the wrapper
print("Training ...")
trainer.train()

# Get the best training solutions
best_ones = trainer.best_solutions()

# Store the training results in a dataframe
results = DataFrame()
species = Series(dtype=object)
ants = Series(dtype=object)
path_len1 = Series(dtype=float)
path_len2 = Series(dtype=float)

for index, col_best in enumerate(best_ones):
    for ant in col_best:
        species.loc[len(species)] = index
        ants.loc[len(ants)] = ant
        path_len1.loc[len(path_len1)] = int(ant.fitness.getValues()[0])
        path_len2.loc[len(path_len2)] = int(ant.fitness.getValues()[1])

results['Species'] = species
results['Ant'] = ants
results['Length1'] = path_len1
results['Length2'] = path_len2

# Print the results
print("\nResults ...")
print(results[['Length1', 'Length2']])

print(f"\nNum Evals: {trainer.num_evals}")
print(f"Runtime  : {trainer.runtime}")
