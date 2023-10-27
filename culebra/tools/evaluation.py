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

"""Evaluation of the solutions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple, List, Optional, Dict
from collections.abc import Sequence
from copy import deepcopy
import os
import importlib.util

import numpy as np
from pandas import Series, DataFrame, concat
from deap.tools import HallOfFame, ParetoFront

from culebra.abc import (
    Base,
    Solution,
    FitnessFunction,
    Trainer
)
from culebra.checker import (
    check_int,
    check_instance,
    check_filename
)
from culebra.solution.feature_selection import (
    Species as FSSpecies,
    Metrics
)
from culebra.tools import Results


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class _ResultKeys:
    """Keys of the results obtained.

    This class should be subclassed in every evaluation in order to define
    the keys to access the results it produces.
    """

    @classmethod
    def keys(cls):
        """Return all the keys defined in the class."""
        return list(
            key
            for key in dir(cls)
            if (
                isinstance(getattr(cls, key), str)
                and not key.startswith("_"))
        )


class _Labels:
    """Labels for the results dataframes."""

    species = 'Species'
    """Label for the species column in the dataframes"""

    solution = 'Solution'
    """Label for the solution column in the dataframes"""

    feature = 'Feature'
    """Label for the feature column in the dataframes"""

    value = 'Value'
    """Label for the Value column in the dataframes"""

    fitness = 'Fitness'
    """Label for the fitness column in the dataframes"""

    relevance = 'Relevance'
    """Label for the relevance column in the dataframes"""

    rank = 'Rank'
    """Label for the rank column in the dataframes"""

    max = 'Max'
    """Label for the max column in the dataframes"""

    min = 'Min'
    """Label for the min column in the dataframes"""

    avg = 'Avg'
    """Label for the avg column in the dataframes"""

    std = 'Std'
    """Label for the std column in the dataframes"""

    best = 'Best'
    """Label for the best column in the dataframes"""

    stat = 'Stat'
    """Label for the stat column in the dataframes"""

    metric = 'Metric'
    """Label for the metric column in the dataframes"""

    runtime = "Runtime"
    """Label for the runtime column in dataframes."""

    num_evals = "NEvals"
    """Label for the number of evaluations column in dataframes."""

    experiment = "Exp"
    """Label for the experiment column in dataframes."""

    batch = "Batch"
    """Label for the batch column in dataframes."""


DEFAULT_STATS_FUNCTIONS = {
    _Labels.avg: np.mean,
    _Labels.std: np.std,
    _Labels.min: np.min,
    _Labels.max: np.max
}
"""Default statistics calculated for the results."""

DEFAULT_FEATURE_METRIC_FUNCTIONS = {
    _Labels.relevance: Metrics.relevance,
    _Labels.rank: Metrics.rank
}
"""Default metrics calculated for the features in the set of solutions."""

DEFAULT_BATCH_STATS_FUNCTIONS = {
    _Labels.avg: Series.mean,
    _Labels.std: Series.std,
    _Labels.min: Series.min,
    _Labels.max: Series.max
}
"""Default statistics calculated for the results gathered from all the
experiments."""

DEFAULT_NUM_EXPERIMENTS = 1
"""Default number of experiments in the batch."""


DEFAULT_SCRIPT_FILENAME = "run.py"
"""Default file name for the script to run an evaluation."""

DEFAULT_CONFIG_FILENAME = "config.py"
"""Default name for the configuration file for the evaluation."""


class Evaluation(Base):
    """Base class for results evaluations."""

    class _ResultKeys(_ResultKeys):
        """Result keys for the evaluation.

        It is empty, since :py:class:`~culebra.tools.Evaluation` is an abstract
        class. Subclasses should override this class to fill it with the
        appropriate result keys.
        """

    feature_metric_functions = DEFAULT_FEATURE_METRIC_FUNCTIONS
    """Metrics calculated for the features in the set of solutions."""

    stats_functions = DEFAULT_STATS_FUNCTIONS
    """Statistics calculated for the solutions."""

    _script_code = """#!/usr/bin/env python3

#
# This script relies on the {config_filename} script.
#
# This script is a simple python module defining variables to be passed to
# the {cls_name} constructor. These variables MUST have the same name than
# the constructor parameters.
#

from culebra.tools import {cls_name}

# Create the {var_name}
{var_name} = {cls_name}.from_config({config_filename})

# Run the {var_name}
{var_name}.run()

# Print the results
for res, val in {var_name}.results.items():
    print(f"\\n\\n{res}:")
    print(val)
"""
    """Parameterized script to evaluate the trainer."""

    def __init__(
        self,
        trainer: Trainer,
        test_fitness_function: Optional[FitnessFunction] = None,
        results_base_filename: Optional[str] = None
    ) -> None:
        """Set a trainer evaluation.

        :param trainer: The trainer method
        :type trainer: :py:class:`~culebra.abc.Trainer`
        :param test_fitness_function: The fitness used to test. If
            :py:data:`None`, the training fitness function will be used.
            Defaults to :py:data:`None`.
        :type test_fitness_function: :py:class:`~culebra.abc.FitnessFunction`,
            optional
        :param results_base_filename: The base filename to save the results.
        :type results_base_filename: :py:class:`~str`
        :raises TypeError: If *trainer* is not a valid trainer
        :raises TypeError: If *test_fitness_function* is not a valid
            fitness function
        :raises TypeError: If *results_base_filename* is not a valid file name
        """
        self.trainer = trainer
        self.test_fitness_function = test_fitness_function
        self.results_base_filename = results_base_filename

    @property
    def trainer(self) -> Trainer:
        """Get and set the trainer method.

        :getter: Return the trainer method
        :setter: Set a new trainer method
        :type: :py:class:`~culebra.abc.Trainer`
        :raises TypeError: If set to a value which is not a valid trainer
        """
        return self._trainer

    @trainer.setter
    def trainer(self, value: Trainer) -> None:
        """Set a new trainer method.

        :param value: New trainer
        :type value: :py:class:`~culebra.abc.Trainer`
        :raises TypeError: If *trainer* is not a valid trainer
        """
        # Check the value
        self._trainer = check_instance(value, "trainer", Trainer)

        # Reset results
        self.reset()

    @property
    def test_fitness_function(self) -> FitnessFunction | None:
        """Get and set the test fitness function.

        :getter: Return the test fitness function
        :setter: Set a new test fitness function. If set to :py:data:`None`,
            the training fitness function will also be used for testing.
        :type: :py:class:`~culebra.abc.FitnessFunction`
        :raises TypeError: If set to a value which is not a valid fitness
            funtion
        """
        return self._test_fitness_function

    @test_fitness_function.setter
    def test_fitness_function(self, func: FitnessFunction | None) -> None:
        """Set a new trainer method.

        :param func: New test fitness function. If set to :py:data:`None`,
            the training fitness function will also be used for testing.
        :type func: :py:class:`~culebra.abc.FitnessFunction`
        :raises TypeError: If *func* is not a valid fitness
            function
        """
        # Check the function
        self._test_fitness_function = (
            None if func is None else check_instance(
                func, "test fitness function", FitnessFunction
            )
        )

        # Reset results
        self.reset()

    @property
    def results_base_filename(self) -> str | None:
        """Get and set the results base filename.

        :getter: Return the results base filename
        :setter: Set a new results base filename. If set to :py:data:`None`,
            :py:attr:`culebra.tools.Results.default_base_filename` is used.
            Defaults to :py:data:`None`.
        :raises TypeError: If set to an invalid file name
        """
        return self._results_base_filename

    @results_base_filename.setter
    def results_base_filename(self, filename: str | None) -> None:
        """Set a new results base filename.

        :param filename: New results base filename. If set to :py:data:`None`,
            :py:attr:`culebra.tools.Results.default_base_filename` is used.
            Defaults to :py:data:`None`.
        :type filename: :py:class:`~str`
        :raises TypeError: If *filename* is not a valid file name
        """
        # Check the filename
        self._results_base_filename = (
            None if filename is None else check_filename(
                filename,
                name="base filename to save the results"
            )
        )

        # Reset results
        self.reset()

    @property
    def results(self) -> Dict[str, DataFrame] | None:
        """Get all the results provided.

        :type: :py:class:`~culebra.tools.Results`
        """
        return self._results

    @classmethod
    def from_config(
        cls, config_filename: Optional[str] = None
    ) -> Evaluation:
        """Generate a new evaluation from a configuration file.

        :param config_filename: Path to the configuration file. If set to
            :py:data:`None`, :py:attr:`~culebra.tools.DEFAULT_CONFIG_FILENAME`
            is used. Defaults to :py:data:`None`
        :type config_filename: :py:class:`str`, optional
        :raises TypeError: If *config_filename* is an invalid file path
        """
        config_filename = check_filename(
            (
                DEFAULT_CONFIG_FILENAME
                if config_filename is None
                else config_filename
            ),
            name="configuration file",
            ext=".py"
        )

        # Load the config module
        config = cls._load_config(config_filename)

        # Generate the Evaluation from the config module
        return cls(
            getattr(config, 'trainer', None),
            getattr(config, 'test_fitness_function', None)
        )

    @classmethod
    def generate_script(
        cls,
        config_filename: Optional[str] = None,
        script_filename: Optional[str] = None
    ) -> None:
        """Generate a script to run an evaluation.

        The parameters for the experiment are obtained from a
        configuration file.

        :param config_filename: Path to the configuration file. If set to
            :py:data:`None`, :py:attr:`~culebra.tools.DEFAULT_CONFIG_FILENAME`
            is used. Defaults to :py:data:`None`
        :type config_filename: :py:class:`str`, optional
        :param script_filename: File path to store the script. If set to
            :py:data:`None`, :py:attr:`~culebra.tools.DEFAULT_SCRIPT_FILENAME`
            is used. Defaults to :py:data:`None`
        :type script_filename: :py:class:`str`, optional.
        """
        config_filename = check_filename(
            (
                DEFAULT_CONFIG_FILENAME
                if config_filename is None
                else config_filename
            ),
            name="configuration file",
            ext=".py"
        )

        script_filename = check_filename(
            (
                DEFAULT_SCRIPT_FILENAME
                if script_filename is None
                else script_filename
            ),
            name="script file",
            ext=".py"
        )

        cls_name = cls.__name__
        # Create the script file
        with open(script_filename, 'w', encoding="utf8") as script:
            script.write(
                cls._script_code.format_map(
                    {
                        "config_filename": f"'{config_filename}'",
                        "cls_name": cls_name,
                        "var_name": cls_name.lower(),
                        "res": "{res}"
                    }
                )
            )

        # Make the script file executable
        os.chmod(script_filename, 0o777)

    def reset(self) -> None:
        """Reset the results."""
        self.trainer.reset()
        self._results = None

    def run(self) -> None:
        """Execute the evaluation and save the results."""
        # Forget previous results
        self.reset()

        # Init the results manager
        self._results = Results(self.results_base_filename)

        # Run the evaluation
        self._execute()

        # Save the results
        self.results.save()

        # Save the results to Excel
        self.results.to_excel()

    @abstractmethod
    def _execute(self) -> None:
        """Execute the evaluation.

        This method must be overridden by subclasses to return a correct
        value.
        """
        raise NotImplementedError(
            "The _execute method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @staticmethod
    def _load_config(config_filename: str) -> object:
        """Generate a new evaluation from a configuration file.

        :param config_filename: Path to the config file
        :type config_filename: :py:class:`str`
        :raises TypeError: If *config_filename* is an invalid file path
        """
        # Get the spec
        spec = importlib.util.spec_from_file_location(
            "config",
            check_filename(config_filename, "config filename", ext=".py")
        )

        # Load the module
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        return config

    def __copy__(self) -> Evaluation:
        """Shallow copy the object."""
        cls = self.__class__
        result = cls(self.trainer)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Evaluation:
        """Deepcopy the object.

        :param memo: Object attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the object
        :rtype: The same than the original object
        """
        cls = self.__class__
        result = cls(self.trainer)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the object.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.trainer, ), self.__dict__)


class Experiment(Evaluation):
    """Run a trainer method from the parameters in a config file."""

    class _ResultKeys(_ResultKeys):
        """Handle the keys for the experiment results."""

        training_stats = 'training_stats'
        """Training statistics."""

        training_fitness = 'training_fitness'
        """Training fitness of the best solutions found."""

        test_fitness = 'test_fitness'
        """Test fitness of the best solutions found."""

        training_fitness_stats = "training_fitness_stats"
        """Training fitness stats."""

        test_fitness_stats = "test_fitness_stats"
        """Test fitness stats."""

        execution_metrics = 'execution_metrics'
        """Execution metrics."""

        feature_metrics = 'feature_metrics'
        """Feature metrics."""

    @property
    def best_solutions(self) -> Sequence[HallOfFame] | None:
        """Return the best solutions found by the trainer."""
        return self._best_solutions

    @property
    def best_representatives(self) -> List[List[Solution]] | None:
        """Return the best representatives found by the trainer."""
        return self._best_representatives

    def reset(self) -> None:
        """Reset the results.

        Overridden to reset the best solutions and best representatives.
        """
        super().reset()
        self._best_solutions = None
        self._best_representatives = None

    def _do_training(self) -> None:
        """Perform the training step.

        Train the trainer and get the best solutions and the training
        stats.
        """
        # Search the best solutions
        self.trainer.train()

        # Best solutions found by the trainer
        self._best_solutions = self.trainer.best_solutions()
        self._best_representatives = self.trainer.best_representatives()

        # Add the training stats
        self._add_training_stats()

        # Add the training fitness to the best solutions dataframe
        self._add_fitness(self._ResultKeys.training_fitness)

        # Perform the training fitness stats
        self._add_fitness_stats(self._ResultKeys.training_fitness_stats)

    def _add_training_stats(self) -> None:
        """Add the training stats to the experiment results."""
        # Training_fitness class
        tr_fitness_func = self.trainer.fitness_function

        # Number of training objectives
        num_obj = tr_fitness_func.num_obj

        # Fitness objective names
        obj_names = tr_fitness_func.Fitness.names

        # Training logbook
        logbook = self.trainer.logbook

        # Number of entries in the logbook
        n_entries = len(logbook)

        # Key of the result
        result_key = self._ResultKeys.training_stats

        # Create the dataframe
        df = DataFrame()

        # Add the iteration stats
        index = []
        for stat in self._trainer.stats_names:
            df[stat.capitalize()] = logbook.select(stat) * num_obj
            index += [stat.capitalize()]

        # Index for each objective stats
        fitness_index = []
        for i in range(num_obj):
            fitness_index.extend([obj_names[i] for _ in range(n_entries)])
        df[_Labels.fitness] = fitness_index

        # For each objective stat
        for stat in self.trainer.objective_stats.keys():
            # Select the data of this stat for all the objectives
            data = logbook.select(stat)
            stat_data = np.zeros(n_entries * num_obj)
            for i in range(n_entries):
                for j in range(len(data[i])):
                    stat_data[j*n_entries + i] = data[i][j]

            df[stat.capitalize()] = stat_data

        # Set the dataframe index
        df.set_index(index + [_Labels.fitness], inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(_Labels.stat, inplace=True)

        # Add the dataframe to the results
        self.results[result_key] = df

    def _add_fitness(self, result_key: str) -> None:
        """Add the fitness values to the solutions found.

        :param result_key: Result key.
        :type result_key: :py:class:`str`
        """
        # Objective names
        obj_names = list(self.best_solutions[0][0].fitness.names)

        # Number of species
        num_species = len(self.best_solutions)

        # Index for the dataframe
        index = (
            [_Labels.species, _Labels.solution]
            if num_species > 1
            else [_Labels.solution]
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
                row_index = (
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
        df.columns.set_names(_Labels.fitness, inplace=True)

        # Add the dataframe to the results
        self.results[result_key] = df.astype(float)

    def _add_fitness_stats(self, result_key: str) -> None:
        """Perform some stats on the best solutions fitness.

        :param result_key: Result key.
        :type result_key: :py:class:`str`
        """
        # Objective names
        obj_names = list(self.best_solutions[0][0].fitness.names)

        # Number of objectives
        n_obj = self.best_solutions[0][0].fitness.num_obj

        # Number of species
        num_species = len(self.best_solutions)

        # Index for the dataframe
        index = (
            [_Labels.species, _Labels.fitness]
            if num_species > 1
            else [_Labels.fitness]
        )

        # Column names for the dataframe
        column_names = index + list(self.stats_functions.keys())

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

            if num_species > 1:
                species_df[_Labels.species] = [species_index] * n_obj

            species_df[_Labels.fitness] = obj_names
            for name, func in self.stats_functions.items():
                species_df[name] = func(fitness, axis=1)

            df = (
                species_df if df is None else concat(
                    [df, species_df], ignore_index=True
                )
            )

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(_Labels.stat, inplace=True)

        # Add the dataframe to the results
        self.results[result_key] = df

    def _add_execution_metric(self, metric: str, value: Any) -> None:
        """Add an execution metric to the experiment results.

        :param metric: Name of the metric
        :type metric: :py:class:`str`
        :param value: Value of the metric
        :type value: :py:class:`object`
        """
        # Key of the result
        result_key = self._ResultKeys.execution_metrics

        # Index for the dataframe
        index = [_Labels.value]

        # Create the DataFrame if it doesn't exist
        if result_key not in self.results:
            self.results[result_key] = DataFrame(index=index)
            self.results[result_key].columns.set_names(
                _Labels.metric, inplace=True
            )

        # Add a new column to the dataframe
        self.results[result_key][metric] = [value]

    def _add_feature_metrics(self) -> None:
        """Perform stats about features frequency."""
        # Flag to know if there are FS solutions in any hof
        there_are_features = False

        # Name of the result
        result_key = self._ResultKeys.feature_metrics

        # Index for the dataframe
        index = [_Labels.feature]

        # Column names for the dataframe
        column_names = index + list(self.feature_metric_functions.keys())

        # Create the dataframe
        df = DataFrame(columns=column_names)
        features_hof = ParetoFront()

        # For each species
        for species_index, hof in enumerate(self.best_solutions):
            # If the species codes features
            hof_species = hof[0].species
            if isinstance(hof_species, FSSpecies):
                # Feature selection solutions detected
                there_are_features = True
                features_hof.update(hof)

        # Insert the df only if it is not empty
        if there_are_features:
            # Get the metrics
            metric = None
            for name, func in self.feature_metric_functions.items():
                metric = func(features_hof)
                df[name] = metric
                # If there is any metric
                if metric is not None:
                    df[_Labels.feature] = metric.index

            # Set the dataframe index
            df.set_index(index, inplace=True)
            df.columns.set_names(_Labels.metric, inplace=True)

            # Add the dataframe to the results
            self.results[result_key] = df

    def _do_test(self) -> None:
        """Perform the test step.

        Test the solutions found by the trainer append their fitness to
        the best solutions dataframe.
        """
        # Test the best solutions found
        self._trainer.test(
            self.best_solutions,
            self.test_fitness_function,
            self.best_representatives
        )

        # Add the test fitness to the best solutions dataframe
        self._add_fitness(self._ResultKeys.test_fitness)

        # Perform the test fitness stats
        self._add_fitness_stats(self._ResultKeys.test_fitness_stats)

    def _execute(self) -> None:
        """Execute the trainer method."""
        # Train the trainer
        self._do_training()

        # Add the execution metrics
        self._add_execution_metric(_Labels.runtime, self.trainer.runtime)
        self._add_execution_metric(_Labels.num_evals, self.trainer.num_evals)

        # Add the features stats
        self._add_feature_metrics()

        # Test the best solutions found
        self._do_test()

        # Reset the state of the trainer to allow serialization
        self.trainer.reset()


class Batch(Evaluation):
    """Generate a batch of experiments."""

    class _ResultKeys(_ResultKeys):
        """Handle the keys for the batch results."""

        batch_execution_metrics_stats = 'batch_execution_metrics_stats'
        """Batch execution metrics stats."""

        batch_feature_metrics_stats = 'batch_feature_metrics_stats'
        """Batch feature metrics stats."""

        batch_training_fitness_stats = "batch_training_fitness_stats"
        """Batch training fitness stats."""

        batch_test_fitness_stats = "batch_test_fitness_stats"
        """Batch test fitness stats."""

    stats_functions = DEFAULT_BATCH_STATS_FUNCTIONS
    """Statistics calculated for the results gathered from all the
    experiments."""

    def __init__(
        self,
        trainer: Trainer,
        test_fitness_function: Optional[FitnessFunction] = None,
        results_base_filename: Optional[str] = None,
        num_experiments: Optional[int] = None
    ) -> None:
        """Generate a batch of experiments.

        :param trainer: The trainer method
        :type trainer: :py:class:`~culebra.abc.Trainer`
        :param test_fitness_function: The fitness used to test. If
            :py:data:`None`, the training fitness function will be used.
            Defaults to :py:data:`None`.
        :type test_fitness_function: :py:class:`~culebra.abc.FitnessFunction`,
            optional
        :param results_base_filename: The base filename to save the results.
        :type results_base_filename: :py:class:`~str`
        :param num_experiments: Number of experiments in the batch,
            defaults to :py:attr:`~culebra.tools.DEFAULT_NUM_EXPERIMENTS`
        :type num_experiments: :py:class:`int`, optional
        :raises TypeError: If *trainer* is not a valid trainer
        :raises TypeError: If *test_fitness_function* is not a valid
            fitness function
        :raises TypeError: If *results_base_filename* is not a valid file name
        """
        # Init the super class
        super().__init__(trainer, test_fitness_function, results_base_filename)

        # Number of experiments
        self.num_experiments = num_experiments

    @property
    def num_experiments(self) -> int:
        """Get and set the number of experiments in the batch.

        :getter: Return the number of experiments
        :setter: Set a new number of experiments. If set to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_NUM_EXPERIMENTS` is used
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not greater than
            zero
        """
        return (
            DEFAULT_NUM_EXPERIMENTS
            if self._num_experiments is None
            else self._num_experiments
        )

    @num_experiments.setter
    def num_experiments(self, value: int | None) -> None:
        """Set a new number of experiments.

        :param value: New number of experiments. If set to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_NUM_EXPERIMENTS` is used
        :type value: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not greater than
            zero
        """
        # Check the value
        self._num_experiments = None if value is None else check_int(
            value, "number of experiments", gt=0
        )

        # Reset results
        self.reset()

    @property
    def exp_labels(self) -> Tuple[str]:
        """Get the label to identify each one of the experiments in the batch.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        # Suffix length
        suffix_len = len(str(self.num_experiments-1))

        # Return the experiment names
        return tuple(
            _Labels.experiment.lower() +
            f"{i:0{suffix_len}d}" for i in range(self.num_experiments)
        )

    @classmethod
    def from_config(
        cls, config_filename: Optional[str] = None
    ) -> Evaluation:
        """Generate a new evaluation from a configuration file.

        :param config_filename: Path to the configuration file. If set to
            :py:data:`None`, :py:attr:`~culebra.tools.DEFAULT_CONFIG_FILENAME`
            is used. Defaults to :py:data:`None`
        :type config_filename: :py:class:`str`, optional
        :raises TypeError: If *config_filename* is an invalid file path
        """
        config_filename = check_filename(
            (
                DEFAULT_CONFIG_FILENAME
                if config_filename is None
                else config_filename
            ),
            name="configuration file",
            ext=".py"
        )

        # Load the config module
        config = cls._load_config(config_filename)

        # Generate the Evaluation from the config module
        return cls(
            getattr(config, 'trainer', None),
            getattr(config, 'test_fitness_function', None),
            getattr(config, 'results_base_filename', None),
            getattr(config, 'num_experiments', None)
        )

    def reset(self):
        """Reset the results."""
        super().reset()
        self._results_indices = {}

    def _append_data(
        self,
        result_key: str,
        exp_label: str,
        exp_data: Series | DataFrame
    ) -> None:
        """Append data from an experiment to a results dataframe.

        :param result_key: Key of the result
        :type result_key: :py:class:`str`
        :param exp_label: Label of the experiment
        :type exp_label: :py:class:`str`
        :param exp_data: Data of the result
        :type exp_data: :py:class:`~pandas.Series` or
            :py:class:`~pandas.DataFrame`
        """
        # Column names of exp_data
        column_names = []

        # Create the dataframe if hasn't been created yet
        if result_key not in self.results:
            # Create the dataframe
            self.results[result_key] = DataFrame()

            # Add the result key
            setattr(self._ResultKeys, result_key, result_key)

            # Create the dataframe index
            index = [_Labels.experiment]

            if isinstance(exp_data, DataFrame):
                if exp_data.index.names[0] is not None:
                    index += exp_data.index.names

            self._results_indices[result_key] = index

        # Reference to the batch results dataframe
        df = self.results[result_key]

        # Complete the list of columns
        if exp_data.index.names[0] is not None:
            column_names += exp_data.index.names
        column_names += list(exp_data.columns)

        # Dataframe with the experiment results
        exp_df = DataFrame()
        exp_df[_Labels.experiment] = [exp_label]*len(exp_data.index)

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
        self.results[result_key] = df

    def _add_execution_metrics_stats(self):
        """Perform some stats on the execution metrics."""
        # Name of the result
        result_key = self._ResultKeys.batch_execution_metrics_stats

        # Input data
        input_data_name = self._ResultKeys.execution_metrics
        input_data = self.results[input_data_name]

        # Index for the dataframe
        index = [_Labels.metric]

        # Column names for the dataframe
        column_names = index + list(self.stats_functions.keys())

        # Create a dataframe
        df = DataFrame(columns=column_names)

        # For all the metrics
        for metric in input_data.columns:
            # New row for the dataframe
            stats = [metric]

            # Apply the stats
            for func in self.stats_functions.values():
                stats.append(func(input_data[metric]))

            # Append the row to the dataframe
            df.loc[len(df)] = stats

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(_Labels.stat, inplace=True)
        self.results[result_key] = df

    def _add_feature_metrics_stats(self):
        """Perform stats on the feature metrics of all the experiments."""
        try:
            # Name of the result
            result_key = self._ResultKeys.batch_feature_metrics_stats

            # Input data
            input_data_name = self._ResultKeys.feature_metrics
            input_data = self.results[input_data_name]

            # Index for the dataframe
            index = [_Labels.metric, _Labels.feature]

            # Column names for the dataframe
            column_names = index + list(self.stats_functions.keys())

            # Create a dataframe
            df = DataFrame(columns=column_names)

            # Get the features
            features_index = input_data.index.names.index(_Labels.feature)
            the_features = input_data.index.levels[features_index]

            # For all the metrics
            for metric in input_data.columns:
                # Get the values of this metric
                metric_values = input_data[metric]

                # For all the features
                for feature in the_features:
                    # Values for each feature
                    feature_metric_values = metric_values[:, feature]

                    # New row for the dataframe
                    stats = [metric, feature]

                    # Apply the stats
                    for func in self.stats_functions.values():
                        stats.append(func(feature_metric_values))

                    # Append the row to the dataframe
                    df.loc[len(df)] = stats

            # Feature indices should be int
            df[_Labels.feature] = (df[_Labels.feature].astype(int))

            df.set_index(index, inplace=True)
            df.sort_index(inplace=True)
            df.columns.set_names(_Labels.stat, inplace=True)
            self.results[result_key] = df
        except AttributeError:
            # The experiments do not have feature metrics
            pass

    def _add_fitness_stats(
        self,
        input_data_key: str,
        result_key: str
    ) -> None:
        """Perform some stats on the best solutions fitness.

        :param input_data_key: Input data key.
        :type input_data_key: :py:class:`str`
        :param result_key: Result key.
        :type result_key: :py:class:`str`
        """
        # Input data
        input_data = self.results[input_data_key]

        # Index for the dataframe
        index = [_Labels.fitness]

        # Column names for the dataframe
        column_names = index + list(self.stats_functions.keys())

        # Get the objective names
        obj_names = input_data.columns

        # Create a dataframe
        df = DataFrame(columns=column_names)

        # For all the objectives
        for obj_name in obj_names:
            # New row for the dataframe
            stats = [obj_name]

            # Apply the stats
            for func in self.stats_functions.values():
                stats.append(func(input_data[obj_name]))

            # Append the row to the dataframe
            df.loc[len(df)] = stats

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(_Labels.stat, inplace=True)
        self.results[result_key] = df

    def _execute(self):
        """Execute a batch of experiments."""
        # For all the experiments to be generated ...
        for exp_label in self.exp_labels:
            try:
                # Create the experiment folder
                os.makedirs(exp_label)
            except FileExistsError:
                # The directory already exists
                pass

            # Change to the experiment folder
            os.chdir(exp_label)

            # Create the experiment
            experiment = Experiment(
                self.trainer,
                self.test_fitness_function,
                self.results_base_filename
            )

            # Run the experiment
            experiment.run()

            # Append the experiment results
            for result, data in experiment.results.items():
                self._append_data(result, exp_label, data)

            # Return to the batch folder
            os.chdir("..")

        # Sort the results dataframes
        for result_key in self.results:
            self.results[result_key].set_index(
                self._results_indices[result_key], inplace=True)
            self.results[result_key].sort_index(inplace=True)

        # Perform some stats
        self._add_execution_metrics_stats()
        self._add_feature_metrics_stats()
        self._add_fitness_stats(
            self._ResultKeys.training_fitness,
            self._ResultKeys.batch_training_fitness_stats
        )
        self._add_fitness_stats(
            self._ResultKeys.test_fitness,
            self._ResultKeys.batch_test_fitness_stats
        )


# Exported symbols for this module
__all__ = [
    'Evaluation',
    'Experiment',
    'Batch',
    'DEFAULT_STATS_FUNCTIONS',
    'DEFAULT_FEATURE_METRIC_FUNCTIONS',
    'DEFAULT_BATCH_STATS_FUNCTIONS',
    'DEFAULT_NUM_EXPERIMENTS',
    'DEFAULT_SCRIPT_FILENAME',
    'DEFAULT_CONFIG_FILENAME'
]
