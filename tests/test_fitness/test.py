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

from sklearn.naive_bayes import GaussianNB
from culebra.base.species import Species
from culebra.base.dataset import Dataset
from culebra.fitness.kappa_num_feats_fitness import KappaNumFeatsFitness as Fitness
from culebra.individual.bit_vector import BitVector as Individual

# Dataset
DATASET_PATH = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'statlog/australian/australian.dat')

# Load the dataset
dataset = Dataset.load(DATASET_PATH, output_index=-1)

# Normalize inputs between 0 and 1
dataset.normalize()

fitness = Fitness(valid_prop=0.25, classifier=GaussianNB(),
                  thresholds=[0.1, 0.2])

print(f"fitness.classifier: {fitness.classifier}")
print(f"fitness.thresholds: {fitness.thresholds}")
print()

s = Species(dataset.num_feats, min_feat=2)
i1 = Individual(s, fitness, (3, 4))
i2 = Individual(s, fitness, (5, 6))


i1.fitness.setValues(fitness.eval(i1, dataset))
i2.fitness.setValues(fitness.eval(i2, dataset))

print(f"i1: {i1} \t {fitness.names}: {i1.fitness}")
print(f"i2: {i2} \t {fitness.names}: {i2.fitness}")
print()
print(f"i1<i2: {i1<i2}")
print(f"i1 dominates i2: {i1.dominates(i2)}")
