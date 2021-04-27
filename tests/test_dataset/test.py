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

"""Test the Dataset class."""

from culebra.base import Dataset

dataset = Dataset.load("a.dat", "b.dat", output_index=None, sep='\\s+')

print(f"num_feats: {dataset.num_feats}")
print(f"size: {dataset.size}")
print(f"inputs: {dataset.inputs}")
print(f"outputs: {dataset.outputs}")

(training, test) = Dataset.load_train_test(
    "a.dat", "b.dat", "c.dat", "d.dat", test_prop=None, output_index=None,
    sep='\\s+', normalize=True, random_feats=None, random_seed=None)

print(f"\ntraining inputs: {training.inputs}")
print(f"training outputs: {training.outputs}")

print(f"\ntest inputs: {test.inputs}")
print(f"test outputs: {test.outputs}")
