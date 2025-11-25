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

"""Usage example of the age-based PACO trainer."""

from pandas import Series, DataFrame

from culebra.solution.tsp import Species, Ant
from culebra.trainer.aco.abc import ACOTSP
from culebra.trainer.aco import AgeBasedPACO
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

# Initial pheromone
initial_pheromone = (1 / num_nodes,) * fitness_func.num_obj
max_pheromone = (5,) * fitness_func.num_obj


class AgeBasedPACOTSP(ACOTSP, AgeBasedPACO):
    """Age Based PACO for TSP."""


# Trainer parameters
params = {
    "solution_cls": Ant,
    "species": species,
    "fitness_function": fitness_func,
    "initial_pheromone": initial_pheromone,
    "exploitation_prob": 0,
    "max_pheromone": max_pheromone,
    "col_size": 15,
    "pop_size": 15,
    "max_num_iters": 500,
    "checkpoint_enable": False
}

# Create the wrapper
trainer = AgeBasedPACOTSP(**params)

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

for ind_col, col_best in enumerate(best_ones):
    for ant in col_best:
        ind_ant = len(species)
        species.loc[ind_ant] = ind_col
        ants.loc[ind_ant] = ant
        path_len.loc[ind_ant] = int(ant.fitness.values[0])

results['Species'] = species
results['Ant'] = ants
results['Length'] = path_len

# Print the results
print("\nResults ...")
print(results)

print(f"\nNum Evals: {trainer.num_evals}")
print(f"Runtime  : {trainer.runtime}")
