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

import pickle
from sklearn.naive_bayes import GaussianNB
from culebra.base.species import Species
from culebra.base.dataset import Dataset
from culebra.fitness.kappa_num_feats_fitness import KappaNumFeatsFitness
from culebra.individual.int_vector import IntVector as Individual
from culebra.wrapper.evolutionary_wrapper import EvolutionaryWrapper

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Proportion of data used to test
TEST_PROP = 0.25

dataset = Dataset.load(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

print("\nSerialization of Datasets")
data = pickle.dumps(dataset)
dataset2 = pickle.loads(data)

print(f"Dataset 1: {dataset}")
print(f"Dataset 2: {dataset2}")

print("\nSerialization of Species")
s = Species.from_proportion(dataset.num_feats)
data = pickle.dumps(s)
s2 = pickle.loads(data)

print(f"Species 1: {s}")
print(f"Species 2: {s2}")

print("\nSerialization of Fitness:")
f = KappaNumFeatsFitness(valid_prop=0.25, classifier=GaussianNB())
f.setValues((1, 4))
data = pickle.dumps(f)
f2 = pickle.loads(data)
print(f"Fitness 1: {f.__repr__()}")
print(f"Fitness 2: {f2.__repr__()}")

print("\nSerialization of IntVectors")

i = Individual(s, f)
i.fitness.setValues(f.eval(i, dataset))
data = pickle.dumps(i)
i2 = pickle.loads(data)

print(f"Individual 1: {i.__repr__()}")
print(f"Individual 2: {i2.__repr__()}")

print("\nSerialization of wrappers")
w = EvolutionaryWrapper(Individual, Species(dataset.num_feats), verbose=False)
data2 = pickle.dumps(w)
w2 = pickle.loads(data2)

print(f"Wrapper 1: {w.__repr__()}")
print(f"Wrapper 2: {w2.__repr__()}")

# Split into training and test data
(tr_data, tst_data) = dataset.split(test_prop=TEST_PROP)

# Train the wrapper
hof, logbook, runtime = w2.train(tr_data, f2)

# Test the wrapper
f2.valid_prop = None
w2.test(hof, tst_data, f2)

# Print the solutions found
print("\nSolutions")
for i in hof:
    print(i)

# Print the training runtime
print(f"\nRuntime: {runtime}")
