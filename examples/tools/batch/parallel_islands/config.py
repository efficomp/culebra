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

"""Usage example of the experiment class."""

from os import cpu_count
from deap.tools import selTournament
from culebra.base import Dataset
from culebra.fitness_function.feature_selection import KappaNumFeats
from culebra.genotype.feature_selection import Species
from culebra.genotype.feature_selection.individual import IntVector
from culebra.wrapper.single_pop import NSGA
from culebra.wrapper.multi_pop import HomogeneousParallelIslands
from culebra.wrapper.multi_pop.topology import ring_destinations

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()
(training_data, test_data) = dataset.split(test_prop=0.3, random_seed=0)

# Training fitness function, 50% of samples used for validation
training_fitness_function = KappaNumFeats(
    training_data=training_data, test_prop=0.5
)

# Test fitness function
test_fitness_function = KappaNumFeats(
    training_data=training_data, test_data=test_data
)

# Number of islands
num_subpops = cpu_count()

# Parameters for the wrapper
params = {
    "individual_cls": IntVector,
    "species": Species(num_feats=dataset.num_feats, min_size=1),
    "fitness_function": training_fitness_function,
    "subpop_wrapper_cls": NSGA,
    "num_gens": 100,
    "pop_size": 100,
    "num_subpops": num_subpops,
    "representation_topology_func": ring_destinations,
    "representation_topology_func_params": {"offset": 1},
    "representation_selection_func": selTournament,
    "representation_selection_func_params": {"tournsize": 10}
}

# Create the wrapper
wrapper = HomogeneousParallelIslands(**params)

# Set the number of experiments
num_experiments = 3
