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

"""Use of the experiment class to evaluate an elitist wrapper."""

from collections import Counter

from culebra.solution.parameter_optimization import Species, Individual
from culebra.fitness_function.svc_optimization import KappaC
from culebra.trainer.ea import ElitistEA
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

# Training fitness function
training_fitness_function = KappaC(training_data=training_data, cv_folds=5)

# Set the training fitness similarity threshold
training_fitness_function.set_fitness_thresholds(0.001)

# Untie fitness function to select the best solution
samples_per_class = Counter(training_data.outputs)
max_folds = samples_per_class[
    min(samples_per_class, key=samples_per_class.get)
]
untie_best_fitness_function = KappaC(
    training_data=training_data,
    cv_folds=max_folds
)

# Test fitness function
test_fitness_function = KappaC(
    training_data=training_data, test_data=test_data
)

# Species to optimize a SVM-based classifier
species = Species(
    lower_bounds=[0, 0],
    upper_bounds=[1000, 1000],
    names=["C", "gamma"]
)

# Parameters for the wrapper
params = {
    "solution_cls": Individual,
    "species": species,
    "fitness_function": training_fitness_function,
    "crossover_prob": 0.8,
    "mutation_prob": 0.2,
    # At least one hyperparameter will be mutated
    "gene_ind_mutation_prob": 1.0/species.num_params,
    "pop_size": dataset.num_feats,
    "max_num_iters": 100,
    "checkpoint_enable": False
}

# Create the wrapper
trainer = ElitistEA(**params)
