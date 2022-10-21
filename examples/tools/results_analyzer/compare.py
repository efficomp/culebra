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

"""Usage example of the results analyzer. Comparison of results."""

from culebra.tools import Results, ResultsAnalyzer
from os import listdir, path

# Create a results analyzer
analyzer = ResultsAnalyzer()

# Keys of the data to be analyzed
# The dict keys will be used to select a results dataframe
# Each value in its associated list is a column key within the dataframe
data_keys = {
    "execution_metrics": ["Runtime"],
    "test_fitness": ["Kappa", "NF"]
}

# List of batches results in csv format
csv_batches_results = ["elitist"]

# List of serialized batches results
batches_results = ["nsga2", "nsga3"]

# Add the batches results in csv format
for batch in csv_batches_results:
    analyzer[batch] = Results.from_csv_files(
        [
            path.join(batch, file)
            for file in listdir(batch) if file.endswith('.csv')
        ]
    )

# Add the other batches results
for batch in batches_results:
    analyzer[batch] = Results.load(batch + ".gz")

# Compare the results
for dataframe in data_keys.keys():
    for column in data_keys[dataframe]:
        comparison = analyzer.compare(dataframe, column)
        print("\n\n")
        print(comparison)
