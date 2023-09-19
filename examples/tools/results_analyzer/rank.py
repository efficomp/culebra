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

"""Usage example of the results analyzer. Multiple ranking of results."""

from os import listdir, path

from culebra.tools import Results, ResultsAnalyzer


# Create a results analyzer
analyzer = ResultsAnalyzer()

# Keys of the data to be analyzed
dataframes = ("test_fitness", "test_fitness", "execution_metrics")
columns = ("Kappa", "NF", "Runtime")
weights = (1, -1, -1)

# List of batches results in csv format
csv_batches_results = ["elitist_ea_wrapper"]

# List of serialized batches results
batches_results = ["nsga2_ea_wrapper"]

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
    analyzer[batch] = Results.load(
        path.join(batch, Results.default_base_filename + ".gz")
    )

# Rank the results
multiple_rank = analyzer.multiple_rank(dataframes, columns, weights)
print(multiple_rank)
