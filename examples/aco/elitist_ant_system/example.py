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

"""Usage example of the elitist ant system algorithm."""

from pandas import Series, DataFrame

from culebra.solution.tsp import Species, Ant
from culebra.trainer.aco import (
    ElitistAntSystem,
    DEFAULT_PHEROMONE_EVAPORATION_RATE
)
from culebra.fitness_function.tsp import PathLength


# Load the GR17 distances matrix from TSPLIB
# The minimal tour has length 2085
fitness_func = PathLength.fromTSPLib(
    "https://raw.githubusercontent.com/mastqe/tsplib/master/gr17.tsp"
)

# Problem graph's number of nodes
num_nodes = fitness_func.num_nodes

# Species for the problem solutions
species = Species(num_nodes)

# Generate and evaluate a greedy solution for the problem
greedy_solution = fitness_func.greedy_solution(species)

initial_pheromone = tuple(
    pher / DEFAULT_PHEROMONE_EVAPORATION_RATE
    for pher in greedy_solution.fitness.pheromone_amount
)

# Trainer parameters
params = {
    "solution_cls": Ant,
    "species": species,
    "fitness_function": fitness_func,
    "initial_pheromone": initial_pheromone,
    "max_num_iters": 500,
    "checkpoint_enable": False
}

# Create the wrapper
trainer = ElitistAntSystem(**params)

# Train the wrapper
print("Training ...")
trainer.train()

# Get the best training solutions
best_ones = trainer.best_solutions()

# Store the training results in a dataframe
results = DataFrame()
species = Series(dtype=object)
ants = Series(dtype=object)
path_len = Series(dtype=float)

for index, col_best in enumerate(best_ones):
    for ant in col_best:
        species.loc[len(species)] = index
        ants.loc[len(ants)] = ant
        path_len.loc[len(path_len)] = int(ant.fitness.getValues()[0])

results['Species'] = species
results['Ant'] = ants
results['Length'] = path_len

# Print the results
print("\nResults ...")
print(results)

print(f"\nNum Evals: {trainer.num_evals}")
print(f"Runtime  : {trainer.runtime}")
