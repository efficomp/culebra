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

"""Test the serialization ob culebra objects."""

import pickle
import copy
from sklearn.naive_bayes import GaussianNB
import culebra

b = culebra.base.Base()
data = pickle.dumps(b)
b2 = pickle.loads(data)
b3 = copy.copy(b)
b4 = copy.deepcopy(b)

print(b.__repr__())
print(b2.__repr__())
print(b3.__repr__())
print(b4.__repr__())
print(b4.__str__())

s = culebra.base.Species()
data = pickle.dumps(s)
s2 = pickle.loads(data)
s3 = copy.copy(s)
s4 = copy.deepcopy(s)

print(s.__repr__())
print(s2.__repr__())
print(s3.__repr__())
print(s4.__repr__())
print(s4.__str__())

f = culebra.base.Fitness(thresholds=4)
data = pickle.dumps(f)
f2 = pickle.loads(data)
f3 = copy.copy(f)
f4 = copy.deepcopy(f)

print(f.__repr__())
print(f2.__repr__())
print(f3.__repr__())
print(f4.__repr__())
print(f4.__str__())

i = culebra.base.Individual(s, f)
data = pickle.dumps(i)
i2 = pickle.loads(data)
i3 = copy.copy(i)
i4 = copy.deepcopy(i)

print(i.__repr__())
print(i2.__repr__())
print(i3.__repr__())
print(i4.__repr__())
print(i4.__str__())

d = culebra.base.Dataset()
data = pickle.dumps(d)
d2 = pickle.loads(data)
d3 = copy.copy(d)
d4 = copy.deepcopy(d)

print(d.__repr__())
print(d2.__repr__())
print(d3.__repr__())
print(d4.__repr__())
print(d4.__str__())

w = culebra.base.Wrapper(culebra.base.Individual, s)
data = pickle.dumps(w)
w2 = pickle.loads(data)
w3 = copy.copy(w)
w4 = copy.deepcopy(w)

print(w.__repr__())
print(w2.__repr__())
print(w3.__repr__())
print(w4.__repr__())
print(w4.__str__())

s = culebra.feature_selector.Species(8)
data = pickle.dumps(s)
s2 = pickle.loads(data)
s3 = copy.copy(s)
s4 = copy.deepcopy(s)

print(s.__repr__())
print(s2.__repr__())
print(s3.__repr__())
print(s4.__repr__())
print(s4.__str__())

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Proportion of data used to test
TEST_PROP = 0.25

dataset = culebra.base.Dataset.load(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

print("\nSerialization of Datasets")
data = pickle.dumps(dataset)
dataset2 = pickle.loads(data)

print(f"Dataset 1: {dataset}")
print(f"Dataset 2: {dataset2}")

print("\nSerialization of Species")
s = culebra.feature_selector.Species.from_proportion(dataset.num_feats)
data = pickle.dumps(s)
s2 = pickle.loads(data)

print(f"Species 1: {s}")
print(f"Species 2: {s2}")

print("\nSerialization of Fitness:")
f = culebra.fitness.KappaNumFeatsFitness(
    valid_prop=0.25, classifier=GaussianNB())
f.setValues((1, 4))
data = pickle.dumps(f)
f2 = pickle.loads(data)
print(f"Fitness 1: {f.__repr__()}")
print(f"Fitness 2: {f2.__repr__()}")

print("\nSerialization of IntVectors")

i = culebra.feature_selector.BitVector(s, f)
i.fitness.setValues(f.eval(i, dataset))
data = pickle.dumps(i)
i2 = pickle.loads(data)

print(f"Individual 1: {i.__repr__()}")
print(f"Individual 2: {i2.__repr__()}")

print("\nSerialization of wrappers")
w = culebra.wrapper.EvolutionaryWrapper(
    culebra.feature_selector.IntVector,
    culebra.feature_selector.Species(dataset.num_feats), verbose=False)
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
