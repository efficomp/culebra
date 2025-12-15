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

"""Usage example of the results analyzer."""

from os import listdir, path
from sys import argv, exit

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.tools import (
    Results,
    ResultsAnalyzer,
    DEFAULT_RESULTS_BASE_FILENAME
)

CSV_FILE_EXTENSION = ".csv"
"""Extension for csv files."""

valid_commands = ("compare", "rank", "effect-size")
"""Valid commands for this script."""


def init_analyzer(batches_results, csv_batches_results):
    """Load the results of several approaches.

    :param batches_results: Results from a culebra's batch
    :type batches_results: list[~culebra.tools.Batch]
    :param csv_batches_results: Results in csv format
    :type csv_batches_results: list[str]
    :return: A results analyzer for the loaded results
    :rtype: ~culebra.tools.ResultsAnalyzer
    """
    # Create a results analyzer
    analyzer = ResultsAnalyzer()

    # Add the batches results in csv format
    for batch in csv_batches_results:
        analyzer[batch] = Results.from_csv_files(
            [
                path.join(batch, file)
                for file in listdir(batch) if file.endswith(CSV_FILE_EXTENSION)
            ]
        )

    # Add the other batches results
    for batch in batches_results:
        analyzer[batch] = Results.load(
            path.join(
                batch,
                DEFAULT_RESULTS_BASE_FILENAME +
                SERIALIZED_FILE_EXTENSION
            )
        )

    return analyzer


def print_usage():
    """Print the valid script usage."""
    print(f"Usage: {path.basename(__file__)} command")
    print()


def print_valid_commands(valid_commands):
    """Print the valid commands.

    :param valid_commands: List of valid commands.
    :type valid_commands: str
    """
    print("Valid commands:", end=" ")
    first_command = True
    for command in valid_commands:
        if not first_command:
            print(end=f", {command}")
        else:
            print(end=f"{command}")
        first_command = False
    print()
    print_usage()


# Check the arguments
if len(argv) < 2:
    print("\nERROR: Missing command\n")
    print_valid_commands(valid_commands)
    exit(-1)
elif len(argv) > 2:
    print("\nERROR: Too many arguments\n")
    print_usage()
    exit(-1)

# Get the command
command = argv[1].lower()

if command not in valid_commands:
    print(f"\nERROR: Bad command '{argv[1]}'\n")
    print_valid_commands(valid_commands)
    exit(-1)

# List of batches results in csv format
csv_batches_results = ["nsga3_ea_wrapper"]

# List of serialized batches results
batches_results = ["nsga2_ea_wrapper"]

# Init the analyzer with all the results
analyzer = init_analyzer(batches_results, csv_batches_results)

# Multiple ranking of the results
if command == "rank":
    # Keys of the data to be ranked
    multiple_rank_dataframe_keys = (
        "test_best",
        "test_best",
        "execution_metrics"
    )
    multiple_rank_columns = ("Kappa", "NF", "Runtime")
    multiple_rank_weights = (1, -1, -1)

    multiple_rank = analyzer.multiple_rank(
        multiple_rank_dataframe_keys,
        multiple_rank_columns,
        multiple_rank_weights
    )

    print(multiple_rank)
# Comparison or effect size estimation of the results
else:
    # Keys of the data to be compared
    # The dict keys will be used to select a results dataframe
    # Each value in its associated list is a column key within the dataframe
    comparison_data_keys = {
        "execution_metrics": ["Runtime"],
        "test_best": ["Kappa", "NF"]
    }

    # Process the results
    for dataframe in comparison_data_keys.keys():
        for column in comparison_data_keys[dataframe]:
            print(f"{dataframe}.{column}", end=" ")

            if command == "compare":
                print("comparison")
                results = analyzer.compare(dataframe, column)
            else:
                print("effect size")
                results = analyzer.effect_size(dataframe, column)
            print(results)
            print()
