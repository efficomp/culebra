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

"""Provides the :py:class:`~tools.batch.Batch` class."""

import os
import pandas as pd
from shutil import copy, SameFileError
from culebra.tools.experiment import Experiment
from culebra.tools.experiment import Labels as ExpLabels
from culebra.tools.experiment import Files

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Labels:
    """Labels for the results dataframes."""

    max = ExpLabels.max
    """Label for the max column in the dataframes."""

    min = ExpLabels.min
    """Label for the min column in the dataframes."""

    avg = ExpLabels.avg
    """Label for the avg column in the dataframes."""

    std = ExpLabels.std
    """Label for the std column in the dataframes."""

    experiment = "Exp"
    """Label for the experiment column in dataframes."""

    batch = "Batch"
    """Label for the batch column in dataframes."""


DEFAULT_BATCH_STATS = {Labels.avg: pd.DataFrame.mean,
                       Labels.std: pd.DataFrame.std,
                       Labels.min: pd.DataFrame.min,
                       Labels.max: pd.DataFrame.max}
"""Default statistics calculated for the results gathered from all the
experiments."""

DEFAULT_N_EXPERIMENTS = 1
"""Default number of experiments in the batch."""


class Batch:
    """Generate a batch of experiments."""

    results_names = Experiment.results_names + (
        "best_solutions_stats",
        "feature_metrics_stats",
        "execution_metrics_stats")
    """Names of the different results where the stats will be applied."""

    batch_stats = DEFAULT_BATCH_STATS
    """Statistics calculated for the results gathered from all the
    experiments."""

    def __init__(self, config_file, n_experiments=DEFAULT_N_EXPERIMENTS,
                 path=None):
        """Generate a batch of experiments.

        :param config_file: Path to the configuration file
        :type config_file: :py:class:`str`
        :param n_experiments: Number of experiments in the batch, defaults to
            :py:attr:`~tools.batch.DEFAULT_N_EXPERIMENTS`
        :type n_experiments: :py:class:`int`, optional
        :param path: Path for the batch folder. If `None` the current folder is
            assumed. Defaults to `None`
        :type path: :py:class:`str`, optional
        """
        # Absolute path to current folder
        return_path = os.getcwd()

        # Absolute path to the batch folder
        self._path = os.path.abspath(path) if path else return_path

        # Absolute path to the config file
        self._config_file = os.path.abspath(config_file)

        # Number of experiments
        self._n_experiments = n_experiments

        # Create the batch folder
        try:
            os.makedirs(self.path)
        except FileExistsError:
            # The directory already exists
            pass

        # Copy the config file in the batch folder
        try:
            copy(self.config_file, self.path)
        except SameFileError:
            # The file is already there
            pass

        # Change to batch folder
        os.chdir(self.path)

        # For all the experiments to be generated ...
        for exp in self.exp_folders:
            try:
                # Create the expetiment folder
                os.makedirs(exp)
            except FileExistsError:
                # The directory already exists
                pass

            # Change to the experiment folder
            os.chdir(exp)

            # Generate the script to run the experiment
            Experiment.generate_script(os.path.relpath(self.config_file))

            # Return to the batch folder
            os.chdir("..")

        # Generate the script to run the batch
        self.generate_script(os.path.relpath(self.config_file))

        # Return to the current folder
        os.chdir(return_path)

        # Reset results
        self.__reset()

    @property
    def config_file(self):
        """Path to the configuration file.

        :type: :py:class:`str`
        """
        return self._config_file

    @property
    def path(self):
        """Path to the bach folder.

        :type: :py:class:`str`
        """
        return self._path

    @property
    def n_experiments(self):
        """Number of experiments in the batch.

        :type: :py:class:`int`
        """
        return self._n_experiments

    @property
    def exp_folders(self):
        """Experiment folders.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        # Suffix length
        suffix_len = len((self.n_experiments-1).__str__())

        # Generator for the experiment names
        return tuple(
            Labels.experiment.lower() +
            f"{i:0{suffix_len}d}" for i in range(0, self.n_experiments))

    # @property
    # def results_names(self):
    #     """Names of the results provided by this batch of experiments.

    #     :type: :py:class:`tuple` of :py:class:`str`
    #     """
    #     return chain(self._exp_results_names, self._batch_results_names)

    @property
    def results(self):
        """All the results provided by this batch of experiments.

        :type: :py:class:`dict`
        """
        return {name: getattr(self, self.__df_name(name))
                for name in self.results_names}

    def results_to_excel(self, file=None):
        """Save the results to a Excel file.

        :param file: File to store the results. Defaults to
            :py:attr:`~tools.experiment.Files.results`
        :type file: A path to a file, optional
        """
        file = Files.results if file is None else file

        with pd.ExcelWriter(file) as writer:
            for name, data in self.results.items():
                data.to_excel(writer, sheet_name=name)

            writer.save()

    def save(self, file=None):
        """Save this batch.

        :param file: File to store the batch. Defaults to
            :py:attr:`~tools.experiment.Files.backup_file`
        :type file: A path to a file, optional.
        """
        file = Files.backup if file is None else file
        pd.io.pickle.to_pickle(self, file)

    @classmethod
    def load(cls, file=None):
        """Load a batch of experiments.

        :param file: File to load the experiment from. Defaults to
            :py:attr:`~tools.experiment.Files.backup_file`
        :type file: A path to a file, optional.
        """
        file = Files.backup if file is None else file
        return pd.io.pickle.read_pickle(file)

    @classmethod
    def generate_script(cls, config, file=None):
        """Generate a script to run the batch.

        :param config: Path to the configuration file
        :type config: :py:class:`str`
        :param file: File to store the script. Defaults to
            :py:attr:`~tools.experiment.Files.script`
        :type file: A path to a file, optional.
        """
        SCRIPT = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from culebra.tools.config_manager import ConfigManager

# Parameter file
CONFIG = "{config_file}"

# Get the configuration parameters
config_manager = ConfigManager(CONFIG)

# Create the batch
batch = config_manager.build_batch()

# Run the experiments in the batch
batch.run()

# Save the batch
batch.save()

# Save the results to Excel
batch.results_to_excel()

# Print the results
for res, val in batch.results.items():
    print(f"\\n\\n{res}:")
    print(val)
"""
        """Script to run all the experiments in a batch."""

        file = Files.script if file is None else file
        # Create the script
        with open(file, 'w') as script:
            script.write(SCRIPT.format_map({"config_file": config,
                                            "res": "{res}"}))

        # Make the script executable
        os.chmod(file, 0o777)

    def run(self):
        """Run a batch of experiments."""
        # Absolute path to current folder
        return_path = os.getcwd()

        # Change to the batch path
        os.chdir(self.path)

        # For all the experiments ...
        for exp in self.exp_folders:
            # Enter the experiment folder
            os.chdir(exp)

            # Run the experiment script
            with open(Files.script, "rb") as script_file:
                code = compile(script_file.read(), Files.script, "exec")
            exec(code, {})

            # Return th the batch folder
            os.chdir("..")

        # Return to the current folder
        os.chdir(return_path)

        # Gather the results
        self.__gather_results()

    # @property
    # def _exp_results_names(self):
    #     """Names of the different results provided by each experiment.

    #     :type: :py:class:`tuple` of :py:class:`str`
    #     """
    #     return Experiment.results_names

    # @property
    # def _exp_results(self):
    #     """Results provided by each experiment.

    #     :type: :py:class:`dict`
    #     """
    #     return {name: getattr(self, self.__df_name(name))
    #             for name in self.exp_results_names}

    # @property
    # def _batch_results(self):
    #     """Results provided on the experiments results.

    #     :type: :py:class:`dict`
    #     """
    #     return {name: getattr(self, self.__df_name(name))
    #             for name in self.batch_results_names}

    def __df_name(self, name):
        """Return the name of a results dataframe."""
        return "_" + name + "_df"

    def __index_name(self, name):
        """Return the index name for a results dataframe."""
        return "_" + name + "_index"

    def __reset(self):
        """Reset the dataframes used to store the batch results."""
        # Init all the dataframes to None
        for df in (self.__df_name(name) for name in self.results_names):
            setattr(self, df, None)

    def __append_data(self, result_name, exp_name, exp_data):
        """Append data from an experiment to a result dataframe.

        :param result_name: Name of the result
        :type result_name: :py:class:`str`
        :param exp_name: Name of the experiment
        :type exp_name: :py:class:`str`
        :param exp_data: Data of the result
        :type exp_data: :py:class:`~pandas.Series` or
            :py:class:`~pandas.DataFrame`
        """
        # Name of the dataframe
        df_name = self.__df_name(result_name)
        df__index_name = self.__index_name(result_name)

        # Column names of exp_data
        column_names = []

        # Create the dataframe if hasn't been created yet
        if getattr(self, df_name) is None:
            # Create the df
            setattr(self, df_name, pd.DataFrame())

            # Create the df index
            index = [Labels.experiment]

            if isinstance(exp_data, pd.DataFrame):
                if exp_data.index.names[0] is not None:
                    index += exp_data.index.names

            setattr(self, df__index_name, index)

        # Complete the list of columns
        if exp_data.index.names[0] is not None:
            column_names += exp_data.index.names
        column_names += list(exp_data.columns)

        df = pd.DataFrame()
        df[Labels.experiment] = [exp_name]*len(exp_data.index)

        # Append the data
        if isinstance(exp_data, pd.DataFrame):
            exp_data.reset_index(inplace=True)
            df[column_names] = exp_data[column_names]
        elif isinstance(exp_data, pd.Series):
            df[exp_data.name] = exp_data
        else:
            raise TypeError("Only supported pandas Series and DataFrames")

        setattr(self, df_name, getattr(self, df_name).append(df))

        getattr(self, df_name).columns.set_names(
            exp_data.columns.names, inplace=True)

    def __gather_results(self):
        """Gather the results of a batch of experiments."""
        # Absolute path to current folder
        return_path = os.getcwd()

        # Change to the batch path
        os.chdir(self.path)

        # For all the experiments ...
        for exp_folder in self.exp_folders:
            # Enter the experiment folder
            os.chdir(exp_folder)

            # Load the experiment
            exp = Experiment.load()

            # Append the experiment results
            for result, data in exp.results.items():
                self.__append_data(result, exp_folder, data)

            # Return th the batch folder
            os.chdir("..")

        # Sort the results dataframes
        for result in exp.results_names:
            # Name of the dataframe
            df_name = self.__df_name(result)
            df__index_name = self.__index_name(result)

            getattr(self, df_name).set_index(
                    getattr(self, df__index_name), inplace=True)
            getattr(self, df_name).sort_index(inplace=True)

        # Return to the current folder
        os.chdir(return_path)

        # Perform some stats
        self.__do_feature_metrics_stats()
        self.__do_best_solutions_stats()
        self.__do_execution_metrics_stats()

    def __do_execution_metrics_stats(self):
        """Perform some stats on the execution metrics."""
        # Create the execution metrics stats dataframe
        self._execution_metrics_stats_df = pd.DataFrame(
                columns=list(self.batch_stats))

        # For all the metrics
        for metric in self._execution_metrics_df.columns:

            # New row for the dataframe
            stats = {ExpLabels.metric: metric}

            # Apply the stats
            for name, func in self.batch_stats.items():
                stats[name] = func(self._execution_metrics_df[metric])

            # Append the new row
            self._execution_metrics_stats_df = (
                    self._execution_metrics_stats_df.append(
                            stats, ignore_index=True))

        self._execution_metrics_stats_df.set_index(
                ExpLabels.metric, inplace=True)
        self._execution_metrics_stats_df.sort_index(inplace=True)
        self._execution_metrics_stats_df.columns.set_names(
                [ExpLabels.stat], inplace=True)

    def __do_best_solutions_stats(self):
        """Perform stats on the best solutions found for each objective."""
        # Create the batch best solutions stats dataframe
        self._best_solutions_stats_df = pd.DataFrame(
                columns=list(self.batch_stats))

        # Get the name of the different objectives
        fitness_index = self._best_solutions_df.index.levels[1]

        # Get the best value for all the objectives
        best_fitness = self._best_solutions_df[ExpLabels.test]

        # For all the objectives
        for obj in fitness_index:

            # Best fitness values for obj
            best_for_obj = best_fitness[:, obj]

            # New row for the dataframe
            stats = {}

            # Apply the stats
            for name, func in self.batch_stats.items():
                stats[name] = func(best_for_obj)

            # Append the new row
            self._best_solutions_stats_df = (
                    self._best_solutions_stats_df.append(
                            stats, ignore_index=True))

        self._best_solutions_stats_df.set_index(fitness_index, inplace=True)
        self._best_solutions_stats_df.sort_index(inplace=True)
        self._best_solutions_stats_df.columns.set_names([
                ExpLabels.stat], inplace=True)

    def __do_feature_metrics_stats(self):
        """Pertorm stats on the feature metrics of all the experiments."""
        # Create the batch best solutions stats dataframe
        self._feature_metrics_stats_df = pd.DataFrame(
                columns=list(self.batch_stats))

        # Get the features
        features_index = self._feature_metrics_df.index.levels[1]

        metrics = self._feature_metrics_df.columns

        # For all the metrics
        for metric in metrics:
            # Get the values of this metric
            metric_values = self._feature_metrics_df[metric]

            # For all the features
            for feature in features_index:

                # Values for each feature
                best_for_obj = metric_values[:, feature]

                # New row for the dataframe
                stats = {ExpLabels.metric: metric,
                         ExpLabels.feature: feature}

                # Apply the stats
                for name, func in self.batch_stats.items():
                    stats[name] = func(best_for_obj)

                # Append the new row
                self._feature_metrics_stats_df = (
                        self._feature_metrics_stats_df.append(
                                stats, ignore_index=True))

        self._feature_metrics_stats_df[
                ExpLabels.feature] = self._feature_metrics_stats_df[
                        ExpLabels.feature].astype(int)

        self._feature_metrics_stats_df.set_index(
                [ExpLabels.metric, ExpLabels.feature],
                inplace=True)
        self._feature_metrics_stats_df.sort_index(inplace=True)
        self._feature_metrics_stats_df.columns.set_names([
                ExpLabels.stat], inplace=True)
