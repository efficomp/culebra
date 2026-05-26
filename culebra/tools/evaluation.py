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

"""Tools to evaluate the trainers.

Since automated experimentation is a quite valuable characteristic when a
:class:`~culebra.abc.Trainer` method has to be run many times, culebra
provides this features by means of the following classes:

* The :class:`~culebra.tools.evaluation.Batch` class, which allows to run a
  batch of experiments with the same configuration
* The :class:`~culebra.tools.evaluation.Experiment` class, designed to run a
  single experiment with a :class:`~culebra.abc.Trainer`
"""
from __future__ import annotations

from typing import Any
from enum import Enum
from collections.abc import Sequence
from os import makedirs, chdir
from os.path import join

import numpy as np
from pandas import Series, DataFrame, concat
from deap.tools import HallOfFame, ParetoFront

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import (
    Solution,
    FitnessFunction,
    Trainer
)
from culebra.checker import check_int
from culebra.solution.feature_selection import (
    Species as FSSpecies
)
from .constants import DEFAULT_NUM_EXPERIMENTS
from .abc import DecisionManager, Evaluation


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_BATCH_STATS_FUNCS = {
    "Avg": Series.mean,
    "Std": Series.std,
    "Min": Series.min,
    "Max": Series.max
}
"""Default statistics calculated for the results gathered from all the
experiments."""


class Experiment(Evaluation):
    """Run a trainer from the parameters in a config file."""

    class ResultsKeys(str, Enum):
        """Handle the keys for the experiment results."""

        TRAINING_STATS = 'training_stats'
        """Training statistics."""

        TRAINING_FITNESS = 'training_fitness'
        """Training fitness of the best solutions found."""

        TRAIN_BEST = 'train_best'
        """Fitness of the best solution found."""

        TEST_FITNESS = 'test_fitness'
        """Test fitness of the best solutions found."""

        TEST_BEST = 'test_best'
        """Test fitness of the best solution found."""

        TRAINING_FITNESS_STATS = "training_fitness_stats"
        """Training fitness stats."""

        TEST_FITNESS_STATS = "test_fitness_stats"
        """Test fitness stats."""

        EXECUTION_METRICS = 'execution_metrics'
        """Execution metrics."""

        FEATURE_METRICS = 'feature_metrics'
        """Feature metrics."""

    @property
    def best_solutions(self) -> tuple[HallOfFame] | None:
        """Best solutions found by the trainer.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        return self._best_solutions

    @property
    def best_cooperators(self) -> list[list[Solution]] | None:
        """Best cooperators found by the trainer.

        :rtype: list[list[~culebra.abc.Solution]]
        """
        return self._best_cooperators

    def reset(self) -> None:
        """Reset the experiment."""
        super().reset()
        self._best_solutions = None
        self._best_cooperators = None

    def _do_training(self) -> None:
        """Perform the training step.

        Train the trainer and get the best solutions and the training
        stats.
        """
        # Train
        self.trainer.train()

        # Keep the best solutions and best cooperators
        # because the trainer will be reseted
        self._best_solutions = self.trainer.best_solutions()
        self._best_cooperators = self.trainer.best_cooperators()

        # Add the training stats
        self._add_training_stats()

        # Add the training fitness to the best solutions dataframe
        self._add_fitness(self.ResultsKeys.TRAINING_FITNESS.value)

        # Perform the training fitness stats
        self._add_fitness_stats(self.ResultsKeys.TRAINING_FITNESS_STATS.value)

    def _add_training_stats(self) -> None:
        """Add the training stats to the experiment results."""
        # Training_fitness class
        tr_fitness_func = self.trainer.fitness_func

        # Number of training objectives
        num_obj = tr_fitness_func.num_obj

        # Fitness objective names
        obj_names = tr_fitness_func.fitness_cls.names

        # Training logbook
        logbook = self.trainer.logbook

        # Number of entries in the logbook
        n_entries = len(logbook)

        # Results key
        results_key = self.ResultsKeys.TRAINING_STATS.value

        # Create the dataframe
        df = DataFrame()

        # Dataframe index
        index = []

        # Add the hyperparameters (if any)
        if self.hyperparameters is not None:
            logbook_len = len(logbook)
            for hyper_name, hyper_value in self.hyperparameters.items():
                df[hyper_name] = (hyper_value,) * logbook_len * num_obj
                index += [hyper_name]

        # Add the iteration stats
        for stat in self.trainer.iteration_metric_names:
            df[stat.capitalize()] = logbook.select(stat) * num_obj
            index += [stat.capitalize()]

        # Index for each objective stats
        fitness_index = []
        for i in range(num_obj):
            fitness_index.extend([obj_names[i] for _ in range(n_entries)])
        df[self.ResultsLabels.FITNESS.value] = fitness_index

        # For each objective stat
        for stat in self.trainer.iteration_obj_stats.keys():
            # Select the data of this stat for all the objectives
            data = logbook.select(stat)
            stat_data = np.zeros(n_entries * num_obj)
            for i in range(n_entries):
                for j in range(len(data[i])):
                    stat_data[j*n_entries + i] = data[i][j]

            df[stat.capitalize()] = stat_data

        # Set the dataframe index
        df.set_index(index + [self.ResultsLabels.FITNESS.value], inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(self.ResultsLabels.STAT.value, inplace=True)

        # Add the dataframe to the results
        self.results[results_key] = df

    def _add_fitness(self, results_key: str) -> None:
        """Add the fitness values to the solutions found.

        :param results_key: Results key.
        :type results_key: str
        """
        # Objective names
        obj_names = list(self.best_solutions[0][0].fitness.names)

        # Number of species
        num_species = len(self.best_solutions)

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += (
            [
                self.ResultsLabels.SPECIES.value,
                self.ResultsLabels.SOLUTION.value
            ]
            if num_species > 1
            else [self.ResultsLabels.SOLUTION.value]
        )

        # Column names for the dataframe
        column_names = index + obj_names

        # Create the solutions dataframe
        df = DataFrame(columns=column_names)

        # For each species
        for species_index, hof in enumerate(self.best_solutions):
            # For each solution of the species
            for sol in hof:
                # Create a row for the dataframe
                if self.hyperparameters is not None:
                    row_index = tuple(self.hyperparameters.values())
                else:
                    row_index = ()

                row_index += (
                    (species_index, sol)
                    if num_species > 1
                    else (sol, )
                )
                row = Series(
                    row_index + sol.fitness.values,
                    index=column_names
                )

                # Append the row to the dataframe
                df.loc[len(df)] = row

        # Set the dataframe index
        df.set_index(index, inplace=True)
        df.columns.set_names(self.ResultsLabels.FITNESS.value, inplace=True)

        # Add the dataframe to the results
        self.results[results_key] = df.astype(float)

    def _add_fitness_stats(self, results_key: str) -> None:
        """Perform some stats on the best solutions fitness.

        :param results_key: Results key.
        :type results_key: str
        """
        # Objective names
        obj_names = list(self.best_solutions[0][0].fitness.names)

        # Number of objectives
        n_obj = self.best_solutions[0][0].fitness.num_obj

        # Number of species
        num_species = len(self.best_solutions)

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += (
            [
                self.ResultsLabels.SPECIES.value,
                self.ResultsLabels.FITNESS.value
            ]
            if num_species > 1
            else [self.ResultsLabels.FITNESS.value]
        )

        # Column names for the dataframe
        column_names = index + list(self.stats_funcs.keys())

        # Final dataframe (not created yet)
        df = None

        # For each species
        for species_index, hof in enumerate(self.best_solutions):
            # Number of solutions in the hof
            n_sol = len(hof)

            # Array to store all the fitnesses
            fitness = np.zeros([n_obj, n_sol])

            # Get the fitnesses
            for i, sol in enumerate(hof):
                fitness[:, i] = sol.fitness.values

            # Perform the stats
            species_df = DataFrame(columns=column_names)

            if self.hyperparameters is not None:
                for name, value in self.hyperparameters.items():
                    species_df[name] = [value] * n_obj

            if num_species > 1:
                species_df[self.ResultsLabels.SPECIES.value] = (
                    [species_index] * n_obj
                )

            species_df[self.ResultsLabels.FITNESS.value] = obj_names
            for name, func in self.stats_funcs.items():
                species_df[name] = func(fitness, axis=1)

            df = (
                species_df if df is None else concat(
                    [df, species_df], ignore_index=True
                )
            )

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(self.ResultsLabels.STAT.value, inplace=True)

        # Add the dataframe to the results
        self.results[results_key] = df

    def _add_execution_metric(self, metric: str, value: Any) -> None:
        """Add an execution metric to the experiment results.

        :param metric: Name of the metric
        :type metric: str
        :param value: Value of the metric
        :type value: object
        """
        # Results key
        results_key = self.ResultsKeys.EXECUTION_METRICS.value

        # Create the DataFrame if it doesn't exist
        if results_key not in self.results:
            if self.hyperparameters is not None:
                # Index for the dataframe
                index = list(self.hyperparameters.keys())
                self.results[results_key] = DataFrame()
                for hyper_name, hyper_value in self.hyperparameters.items():
                    self.results[results_key][hyper_name] = [hyper_value]

                self.results[results_key].set_index(index, inplace=True)
                self.results[results_key].sort_index(inplace=True)
            else:
                # Index for the dataframe
                index = [self.ResultsLabels.VALUE.value]
                self.results[results_key] = DataFrame(index=index)

            self.results[results_key].columns.set_names(
                self.ResultsLabels.METRIC.value, inplace=True
            )

        # Add a new column to the dataframe
        self.results[results_key][metric] = [value]

    def _add_feature_metrics(self) -> None:
        """Perform stats about features frequency."""
        # Flag to know if there are FS solutions in any hof
        there_are_features = False

        # Results key
        results_key = self.ResultsKeys.FEATURE_METRICS.value

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += [self.ResultsLabels.FEATURE.value]

        # Column names for the dataframe
        column_names = index + list(self.feature_metric_funcs.keys())

        # Create the dataframe
        df = DataFrame(columns=column_names)
        features_hof = ParetoFront()

        # For each species
        for hof in self.best_solutions:
            # If the species codes features
            hof_species = hof[0].species
            if isinstance(hof_species, FSSpecies):
                # Feature selection solutions detected
                there_are_features = True
                features_hof.update(hof)

        # Insert the df only if it is not empty
        if there_are_features:
            # Get the metrics
            is_first_metric = True
            metric = None
            for name, func in self.feature_metric_funcs.items():
                metric = func(features_hof)
                df[name] = metric
                # If there is any metric
                if metric is not None and is_first_metric:
                    is_first_metric = False
                    df[self.ResultsLabels.FEATURE.value] = metric.index
                    num_feats = len(metric.index)
                    if self.hyperparameters is not None:
                        for (
                                hyper_name,
                                hyper_value
                        ) in self.hyperparameters.items():
                            df[hyper_name] = [hyper_value] * num_feats

            # Set the dataframe index
            df.set_index(index, inplace=True)
            df.columns.set_names(self.ResultsLabels.METRIC.value, inplace=True)

            # Add the dataframe to the results
            self.results[results_key] = df

    def _do_test(self) -> None:
        """Perform the test step.

        Test the solutions found by the trainer append their fitness to
        the best solutions dataframe.
        """
        # Test the best solutions found
        self.trainer.test(
            self.best_solutions,
            self.test_fitness_func,
            self.best_cooperators
        )

        # Add the test fitness to the best solutions dataframe
        self._add_fitness(self.ResultsKeys.TEST_FITNESS.value)

        # Perform the test fitness stats
        self._add_fitness_stats(self.ResultsKeys.TEST_FITNESS_STATS.value)

    def _add_best(
        self,
        best: Sequence[Solution],
        fitness_func: FitnessFunction,
        results_key: str
    ) -> None:
        """Add the best solution to the experiment results.

        For cooperative approaches, the solution is evaluated only with the
        species that compose the best solution, without any other cooperator

        :param best: The best solution (one per species)
        :type best: ~collections.abc.Sequence[~culebra.abc.Solution]
        :param fitness_func: Fitness fuction to evaluate the best solution
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param results_key: Results key
        :type results_key: str
        """
        # Number of species
        num_species = len(best)

        # Evaluate the best solution
        best[0].fitness.values = fitness_func.evaluate(best[0], 0, best)
        for sol in best[1:]:
            sol.fitness = best[0].fitness

        # Objective names
        obj_names = list(fitness_func.obj_names)

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += (
            [
                self.ResultsLabels.SPECIES.value,
                self.ResultsLabels.SOLUTION.value
            ]
            if num_species > 1
            else [self.ResultsLabels.SOLUTION.value]
        )

        # Column names for the dataframe
        column_names = index + obj_names

        # Create the solutions dataframe
        df = DataFrame(columns=column_names)

        # For each species
        for species_index, sol in enumerate(best):
            # Create a row for the dataframe
            if self.hyperparameters is not None:
                row_index = tuple(self.hyperparameters.values())
            else:
                row_index = ()

            row_index += (
                (species_index, sol)
                if num_species > 1
                else (sol, )
            )
            row = Series(
                row_index + sol.fitness.values,
                index=column_names
            )

            # Append the row to the dataframe
            df.loc[len(df)] = row

        # Set the dataframe index
        df.set_index(index, inplace=True)
        df.columns.set_names(self.ResultsLabels.FITNESS.value, inplace=True)

        # Add the dataframe to the results
        self.results[results_key] = df.astype(float)

    def _execute(self) -> None:
        """Execute the trainer."""
        # Train the trainer
        self._do_training()

        # Choose one solution
        best = self.decision_manager.select()

        # Add the execution metrics
        self._add_execution_metric(
            self.ResultsLabels.RUNTIME.value, self.trainer.runtime
        )
        self._add_execution_metric(
            self.ResultsLabels.NUM_ITERS.value, self.trainer.num_iters
        )
        self._add_execution_metric(
            self.ResultsLabels.NUM_EVALS.value, self.trainer.num_evals
        )

        # Add the features stats
        self._add_feature_metrics()

        # Test the solutions found
        self._do_test()

        self._add_best(
            best,
            self.trainer.fitness_func,
            self.ResultsKeys.TRAIN_BEST.value
        )

        # Evaluate the best solution with the test data
        self._add_best(
            best,
            self.test_fitness_func,
            self.ResultsKeys.TEST_BEST.value
        )

        # Reset the state of the trainer to allow serialization
        self.trainer.reset()


class Batch(Evaluation):
    """Generate a batch of experiments."""

    class ResultsKeys(str, Enum):
        """Handle the keys for the batch results."""

        TRAINING_STATS = 'training_stats'
        """Training statistics."""

        TRAINING_FITNESS = 'training_fitness'
        """Training fitness of the best solutions found."""

        TRAIN_BEST = 'train_best'
        """Fitness of the best solution found."""

        TEST_FITNESS = 'test_fitness'
        """Test fitness of the best solutions found."""

        TEST_BEST = 'test_best'
        """Test fitness of the best solution found."""

        TRAINING_FITNESS_STATS = "training_fitness_stats"
        """Training fitness stats."""

        TEST_FITNESS_STATS = "test_fitness_stats"
        """Test fitness stats."""

        EXECUTION_METRICS = 'execution_metrics'
        """Execution metrics."""

        FEATURE_METRICS = 'feature_metrics'
        """Feature metrics."""

        BATCH_EXECUTION_METRICS_STATS = 'batch_execution_metrics_stats'
        """Batch execution metrics stats."""

        BATCH_FEATURE_METRICS_STATS = 'batch_feature_metrics_stats'
        """Batch feature metrics stats."""

        BATCH_TRAINING_FITNESS_STATS = "batch_training_fitness_stats"
        """Batch training fitness stats."""

        BATCH_TEST_FITNESS_STATS = "batch_test_fitness_stats"
        """Batch test fitness stats."""

    stats_funcs = DEFAULT_BATCH_STATS_FUNCS
    """Statistics calculated for the results gathered from all the
    experiments."""


    def __init__(
        self,
        trainer: Trainer,
        decision_manager: DecisionManager,
        test_fitness_func: FitnessFunction | None = None,
        results_base_filename: str | None = None,
        hyperparameters: dict | None = None,
        num_experiments: int | None = None
    ) -> None:
        """Generate a batch of experiments.

        :param trainer: The trainer
        :type trainer: ~culebra.abc.Trainer
        :param test_fitness_func: The fitness function used to test. If
            omitted, the training fitness function will be used. Defaults to
            :data:`None`
        :type test_fitness_func: ~culebra.abc.FitnessFunction
        :param results_base_filename: The base filename to save the results
            If omitted,
            :attr:`~culebra.tools.evaluation.Batch._default_results_base_filename` is
            used. Defaults to :data:`None`
        :type results_base_filename: str
        :param hyperparameters: Hyperparameter values used in this evaluation,
            optional
        :type hyperparameters: dict
        :param num_experiments: Number of experiments in the batch. If omitted,
            :attr:`~culebra.tools.evaluation.Batch._default_num_experiments`
            is used. Defaults to :data:`None`
        :type num_experiments: int
        :raises TypeError: If any of the parameters has an incorrect type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        # Init the super class
        super().__init__(
            trainer,
            decision_manager,
            test_fitness_func,
            results_base_filename,
            hyperparameters
        )

        # Number of experiments
        self.num_experiments = num_experiments

    @property
    def _default_num_experiments(self) -> int:
        """Default number of experiments in the batch.

        :return: :attr:`~culebra.tools.DEFAULT_NUM_EXPERIMENTS`
        :rtype: int
        """
        return DEFAULT_NUM_EXPERIMENTS

    @property
    def num_experiments(self) -> int:
        """Number of experiments in the batch.

        :rtype: int
        :setter: Set a new number of experiments
        :param value: New number of experiments. If set to :data:`None`,
            :attr:`~culebra.tools.evaluation.Batch._default_num_experiments` is used
        :type value: int
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not greater than
            zero
        """
        return (
            self._default_num_experiments
            if self._num_experiments is None
            else self._num_experiments
        )

    @num_experiments.setter
    def num_experiments(self, value: int | None) -> None:
        """Set a new number of experiments.

        :param value: New number of experiments. If set to :data:`None`,
            :attr:`~culebra.tools.evaluation.Batch._default_num_experiments` is used
        :type value: int
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not greater than
            zero
        """
        # Check the value
        self._num_experiments = (
            None if value is None else check_int(
                value, "number of experiments", gt=0
            )
        )

        # Reset results
        self.reset()

    @property
    def experiment_basename(self) -> str:
        """Experiments basename.

        :rtype: str
        """
        return self.ResultsLabels.EXPERIMENT.value.lower()

    @property
    def experiment_labels(self) -> tuple[str]:
        """Label to identify each one of the experiments in the batch.

        :rtype: tuple[str]
        """
        # Suffix length
        suffix_len = len(str(self.num_experiments-1))

        # Return the experiment names
        return tuple(
            self.experiment_basename +
            f"{i:0{suffix_len}d}" for i in range(self.num_experiments)
        )

    @classmethod
    def from_config(
        cls,
        config_script_filename: str | None = None
    ) -> Batch:
        """Generate a new evaluation from a configuration file.

        :param config_script_filename: Path to the configuration file. If
            omitted, :attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is
            used. Defaults to :data:`None`
        :type config_script_filename: str
        :raises RuntimeError: If *config_script_filename* is an invalid file
            path or an invalid configuration file
        """
        # Load the config module
        config = cls._load_config(config_script_filename)

        # Generate the Batch from the config module
        return cls(
            trainer=getattr(config, 'trainer', None),
            decision_manager=getattr(
                config, 'decision_manager', None
            ),
            test_fitness_func=getattr(
                config, 'test_fitness_func', None
            ),
            results_base_filename=getattr(
                config, 'results_base_filename', None
            ),
            hyperparameters=getattr(config, 'hyperparameters', None),
            num_experiments=getattr(config, 'num_experiments', None)
        )

    def reset(self) -> None:
        """Reset the results."""
        super().reset()
        self._results_indices = {}

    def _append_data(
        self,
        results_key: str,
        exp_label: str,
        exp_data: Series | DataFrame
    ) -> None:
        """Append data from an experiment to a results dataframe.

        :param results_key: Results key
        :type results_key: str
        :param exp_label: Label of the experiment
        :type exp_label: str
        :param exp_data: Data of the result
        :type exp_data: ~pandas.Series | ~pandas.DataFrame
        """
        # Column names of exp_data
        column_names = []

        # Create the dataframe if hasn't been created yet
        if results_key not in self.results:
            # Create the dataframe
            self.results[results_key] = DataFrame()

            # Create the dataframe index
            if self.hyperparameters is not None:
                index = list(self.hyperparameters.keys())
            else:
                index = []

            index += [self.ResultsLabels.EXPERIMENT.value]

            if isinstance(exp_data, DataFrame):
                if exp_data.index.names[0] is not None:
                    if self.hyperparameters is not None:
                        num_hyperparams = len(self.hyperparameters)
                        index += exp_data.index.names[num_hyperparams:]
                    else:
                        index += exp_data.index.names

            self._results_indices[results_key] = index

        # Reference to the batch results dataframe
        df = self.results[results_key]

        # Complete the list of columns
        if exp_data.index.names[0] is not None:
            column_names += exp_data.index.names
        column_names += list(exp_data.columns)

        # Dataframe with the experiment results
        exp_df = DataFrame()
        exp_df[self.ResultsLabels.EXPERIMENT.value] = (
            [exp_label] * len(exp_data.index)
        )

        # Append the experiment data
        if isinstance(exp_data, DataFrame):
            exp_data.reset_index(inplace=True)
            exp_df[column_names] = exp_data[column_names]
        elif isinstance(exp_data, Series):
            exp_df[exp_data.name] = exp_data
        else:
            raise TypeError("Only supported pandas Series and DataFrames")

        df = concat([df, exp_df], ignore_index=True)

        df.columns.set_names(
            exp_data.columns.names, inplace=True
        )

        # Update the batch results dataframe
        self.results[results_key] = df

    def _add_execution_metrics_stats(self) -> None:
        """Perform some stats on the execution metrics."""
        # Results key
        results_key = self.ResultsKeys.BATCH_EXECUTION_METRICS_STATS.value

        # Input data
        input_data_name = self.ResultsKeys.EXECUTION_METRICS.value
        input_data = self.results[input_data_name]

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += [self.ResultsLabels.METRIC.value]

        # Column names for the dataframe
        column_names = index + list(self.stats_funcs.keys())

        # Create a dataframe
        df = DataFrame(columns=column_names)

        # For all the metrics
        for metric in input_data.columns:
            # New row for the dataframe
            if self.hyperparameters is not None:
                stats = list(self.hyperparameters.values())
            else:
                stats = []

            stats += [metric]

            # Apply the stats
            for func in self.stats_funcs.values():
                stats.append(func(input_data[metric]))

            # Append the row to the dataframe
            df.loc[len(df)] = stats

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(self.ResultsLabels.STAT.value, inplace=True)
        self.results[results_key] = df

    def _add_feature_metrics_stats(self) -> None:
        """Perform stats on the feature metrics of all the experiments."""
        try:
            # Results key
            results_key = self.ResultsKeys.BATCH_FEATURE_METRICS_STATS.value

            # Input data
            input_data_name = self.ResultsKeys.FEATURE_METRICS.value
            input_data = self.results[input_data_name]

            # Index for the dataframe
            if self.hyperparameters is not None:
                index = list(self.hyperparameters.keys())
            else:
                index = []

            index += [
                self.ResultsLabels.METRIC.value,
                self.ResultsLabels.FEATURE.value
            ]

            # Column names for the dataframe
            column_names = index + list(self.stats_funcs.keys())

            # Create a dataframe
            df = DataFrame(columns=column_names)

            # Get the features
            features_index = (
                input_data.index.names.index(self.ResultsLabels.FEATURE.value)
            )
            the_features = input_data.index.levels[features_index]
            feature_indices_slices = (slice(None),)
            if self.hyperparameters is not None:
                num_hypermarams = len(self.hyperparameters)
                feature_indices_slices += (slice(None),) * num_hypermarams

            # For all the metrics
            for metric in input_data.columns:
                # Get the values of this metric
                metric_values = input_data[metric]

                # For all the features
                for feature in the_features:
                    # Indices for the feature
                    feature_indices = feature_indices_slices + (feature,)

                    # Values for the feature
                    feature_metric_values = metric_values[feature_indices]

                    # New row for the dataframe
                    if self.hyperparameters is not None:
                        stats = list(self.hyperparameters.values())
                    else:
                        stats = []

                    stats += [metric, feature]

                    # Apply the stats
                    for func in self.stats_funcs.values():
                        stats.append(func(feature_metric_values))

                    # Append the row to the dataframe
                    df.loc[len(df)] = stats

            # Feature indices should be int
            df[self.ResultsLabels.FEATURE.value] = (
                df[self.ResultsLabels.FEATURE.value].astype(int)
            )

            df.set_index(index, inplace=True)
            df.sort_index(inplace=True)
            df.columns.set_names(self.ResultsLabels.STAT.value, inplace=True)
            self.results[results_key] = df
        except KeyError:
            # The experiments do not have feature metrics
            pass

    def _add_fitness_stats(
        self,
        input_data_key: str,
        results_key: str
    ) -> None:
        """Perform some stats on the best solutions fitness.

        :param input_data_key: Input data key.
        :type input_data_key: str
        :param results_key: Results key.
        :type results_key: str
        """
        # Input data
        input_data = self.results[input_data_key]

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += [self.ResultsLabels.FITNESS.value]

        # Column names for the dataframe
        column_names = index + list(self.stats_funcs.keys())

        # Get the objective names
        obj_names = input_data.columns

        # Create a dataframe
        df = DataFrame(columns=column_names)

        # For all the objectives
        for obj_name in obj_names:
            # New row for the dataframe
            if self.hyperparameters is not None:
                stats = list(self.hyperparameters.values())
            else:
                stats = []

            stats += [obj_name]

            # Apply the stats
            for func in self.stats_funcs.values():
                stats.append(func(input_data[obj_name]))

            # Append the row to the dataframe
            df.loc[len(df)] = stats

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(self.ResultsLabels.STAT.value, inplace=True)
        self.results[results_key] = df

    def setup(self) -> None:
        """Set up the batch.

        Create all the experiments for the batch.
        """
        # Create the experiment
        experiment = Experiment(
            self.trainer,
            self.decision_manager,
            self.test_fitness_func,
            self.results_base_filename,
            self.hyperparameters
        )
        experiment_filename = (
            self.experiment_basename + SERIALIZED_FILE_EXTENSION
        )

        # Save the experiment
        experiment.dump(experiment_filename)

        for exp_folder in self.experiment_labels:
            try:
                # Create the experiment folder
                makedirs(exp_folder)
            except FileExistsError:
                # The directory already exists
                pass

            # Change to the experiment folder
            chdir(exp_folder)

            # Generate the run script for the experiment
            path_to_experiment = join("..", experiment_filename)
            experiment.generate_run_script(path_to_experiment)

            # Return to the batch folder
            chdir("..")

    def run(self) -> None:
        """Execute the batch and save the results."""
        # Setup the batch
        self.setup()

        # Run the batch
        super().run()

    def _execute(self) -> None:
        """Execute a batch of experiments."""
        # Load the experiment
        experiment = Experiment.load(
            self.experiment_basename + SERIALIZED_FILE_EXTENSION
        )

        # For all the experiments to be generated ...
        for exp_label in self.experiment_labels:
            # Reset the experiment
            experiment.reset()

            # Change to the experiment folder
            chdir(exp_label)

            # Run the experiment
            experiment.run()

            # Append the experiment results
            for result, data in experiment.results.items():
                self._append_data(result, exp_label, data)

            # Return to the batch folder
            chdir("..")

        # Sort the results dataframes
        for results_key in self.results:
            self.results[results_key].set_index(
                self._results_indices[results_key], inplace=True
            )
            self.results[results_key].sort_index(inplace=True)

        # Perform some stats
        self._add_execution_metrics_stats()
        self._add_feature_metrics_stats()
        self._add_fitness_stats(
            self.ResultsKeys.TRAINING_FITNESS.value,
            self.ResultsKeys.BATCH_TRAINING_FITNESS_STATS.value
        )
        self._add_fitness_stats(
            self.ResultsKeys.TEST_FITNESS.value,
            self.ResultsKeys.BATCH_TEST_FITNESS_STATS.value
        )


# Exported symbols for this module
__all__ = [
    'Experiment',
    'Batch'
]
