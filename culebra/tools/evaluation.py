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

"""Evaluation of the solutions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple, List, Optional, Dict
from collections.abc import Sequence
from copy import deepcopy
from os import chmod, makedirs, chdir
from os.path import isfile, splitext, join
import importlib.util

import numpy as np
from pandas import Series, DataFrame, concat
from deap.tools import HallOfFame, ParetoFront

from culebra import PICKLE_FILE_EXTENSION
from culebra.abc import (
    Base,
    Solution,
    FitnessFunction,
    Trainer
)
from culebra.checker import (
    check_int,
    check_instance,
    check_filename,
    check_func_params
)
from culebra.solution.feature_selection import (
    Species as FSSpecies,
    Metrics
)
from culebra.tools import Results, EXCEL_FILE_EXTENSION


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

SCRIPT_FILE_EXTENSION = ".py"
"""File extension for python scripts."""

DEFAULT_NUM_EXPERIMENTS = 1
"""Default number of experiments in the batch."""

DEFAULT_RUN_SCRIPT_BASENAME = "run"
"""Default base name for the script to run an evaluation."""

DEFAULT_RUN_SCRIPT_FILENAME = (
    DEFAULT_RUN_SCRIPT_BASENAME + SCRIPT_FILE_EXTENSION
)
"""Default file name for the script to run an evaluation."""

DEFAULT_CONFIG_SCRIPT_BASENAME = "config"
"""Default base name for configuration scripts."""

DEFAULT_CONFIG_SCRIPT_FILENAME = (
    DEFAULT_CONFIG_SCRIPT_BASENAME + SCRIPT_FILE_EXTENSION
)
"""Default file name for configuration scripts."""

DEFAULT_RESULTS_BASENAME = "results"
"""Default base name for results files."""


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

    _run_script_code = """#!/usr/bin/env python3

#
# This script relies on the {config_filename} configuration file.
#
# This script is a simple python module defining variables to be passed to
# the {cls_name} constructor. These variables MUST have the same name than
# the constructor parameters.
#

from culebra.tools import {cls_name}

# Create the {var_name}
{var_name} = {cls_name}.{factory_method}('{config_filename}')

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
        results_base_filename: Optional[str] = None,
        hyperparameters: Optional[dict] = None
    ) -> None:
        """Set a trainer evaluation.

        :param trainer: The trainer method
        :type trainer: :py:class:`~culebra.abc.Trainer`
        :param test_fitness_function: The fitness used to test. If set to
            :py:data:`None`, the training fitness function will be used.
            Defaults to :py:data:`None`.
        :type test_fitness_function: :py:class:`~culebra.abc.FitnessFunction`,
            optional
        :param results_base_filename: The base filename to save the results.
            If set to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_RESULTS_BASENAME` is used.
            Defaults to :py:data:`None`
        :type results_base_filename: :py:class:`~str`, optional
        :param hyperparameters: Hyperparameter values used in this evaluation
        :type hyperparameters: :py:class:`~dict`, optional
        :raises TypeError: If *trainer* is not a valid trainer
        :raises TypeError: If *test_fitness_function* is not a valid
            fitness function
        :raises TypeError: If *results_base_filename* is not a valid file name
        :raises TypeError: If *hyperparameters* is not a dictionary
        :raises ValueError: If the keys in *hyperparameters* are not strings
        :raises ValueError: If any key in *hyperparameters* is reserved
        """
        self.trainer = trainer
        self.test_fitness_function = test_fitness_function
        self.results_base_filename = results_base_filename
        self.hyperparameters = hyperparameters

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
            :py:attr:`~culebra.tools.DEFAULT_RESULTS_BASENAME` is used.
        :raises TypeError: If set to an invalid file name
        """
        return (
            DEFAULT_RESULTS_BASENAME if self._results_base_filename is None
            else self._results_base_filename
        )

    @results_base_filename.setter
    def results_base_filename(self, filename: str | None) -> None:
        """Set a new results base filename.

        :param filename: New results base filename. If set to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_RESULTS_BASENAME` is used.
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
    def results_pickle_filename(self) -> str:
        """Get the filename used to save the pickled results.

        :type: :py:class:`str`
        """
        return self.results_base_filename + PICKLE_FILE_EXTENSION

    @property
    def results_excel_filename(self) -> str:
        """Get the filename used to save the results in Excel format.

        :type: :py:class:`str`
        """
        return self.results_base_filename + EXCEL_FILE_EXTENSION

    def _is_reserved(self, name: str) -> bool:
        """Return :py:data:`True` if the given hyperparameter name is reserved.

        :param name: Hyperparameter name
        :type values: :py:class:`~str`
        """
        reserved_labels = (
            label for label in dir(_Labels) if not label.startswith("_")
        )

        for label in reserved_labels:
            if name == getattr(_Labels, label):
                return True

        return False

    @property
    def hyperparameters(self) -> dict | None:
        """Get and set the hyperparameter values used for the evaluation.

        :getter: Return the hyperparameter values
        :setter: Set a new set of hyperparameter values.
        :raises TypeError: If the hyperparameters are not in a dictionary
        :raises ValueError: If the keys of the dictionary are not strings
        :raises ValueError: If any key in the dictionary is reserved
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, values: dict | None) -> None:
        """Set the hyperparameter values used for the evaluation.

        :param values: Hyperparameter values used in this evaluation
        :type values: :py:class:`~dict`
        :raises TypeError: If *values* is not a dictionary
        :raises ValueError: If the keys in *values* are not strings
        :raises ValueError: If any key in *values* is reserved
        """
        if values is None:
            self._hyperparameters = None
            return

        self._hyperparameters = (
            None if values is None else check_func_params(
                values,
                name="hyperparameters"
            )
        )

        # Check that no parameter name is reserved
        for name in values.keys():
            if self._is_reserved(name):
                raise ValueError(
                    "Attempt to use a reserved label as a hyperparameter "
                    f"name: {name}"
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
        cls, config_script_filename: Optional[str] = None
    ) -> Evaluation:
        """Generate a new evaluation from a configuration file.

        :param config_script_filename: Path to the configuration file. If set
            to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :py:data:`None`
        :type config_script_filename: :py:class:`str`, optional
        :raises RuntimeError: If *config_script_filename* is an invalid file
            path or an invalid configuration file
        """
        # Load the config module
        config = cls._load_config(config_script_filename)

        # Generate the Evaluation from the config module
        return cls(
            trainer=getattr(config, 'trainer', None),
            test_fitness_function=getattr(
                config, 'test_fitness_function', None
            ),
            results_base_filename=getattr(
                config, 'results_base_filename', None
            ),
            hyperparameters=getattr(config, 'hyperparameters', None)
        )

    @classmethod
    def generate_run_script(
        cls,
        config_filename: Optional[str] = None,
        run_script_filename: Optional[str] = None
    ) -> None:
        """Generate a script to run an evaluation.

        The parameters for the evaluation are taken from a configuration file.

        :param config_filename: Path to the configuration file. It can be
            whether a configuration script or a pickled
            :py:attr:`~culebra.tools.Evaluation` instance. If set to
            :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :py:data:`None`
        :type config_filename: :py:class:`str`
        :param run_script_filename: File path to store the run script. If set
            to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_RUN_SCRIPT_FILENAME`
            is used. Defaults to :py:data:`None`
        :type run_script_filename: :py:class:`str`, optional.
        :raises TypeError: If *config_filename* or *run_script_filename*
            are not a valid filename
        :raises ValueError: If the extensions of *config_filename* or
            *run_script_filename* are not valid.
        """
        # Check the configuration filename
        config_filename = check_filename(
            (
                DEFAULT_CONFIG_SCRIPT_FILENAME
                if config_filename is None
                else config_filename
            ),
            name="configuration file"
        )

        # Check the extension of the configuration file
        (_, extension) = splitext(config_filename)
        extension = extension.lower()
        if extension == SCRIPT_FILE_EXTENSION:
            factory_method = 'from_config'
        elif extension == PICKLE_FILE_EXTENSION:
            factory_method = 'load_pickle'
        else:
            raise ValueError(
                "Not valid extension for the configuration file. "
                "Valid extensions are .py for python scripts or .gz for "
                "pickled evaluation objects: {config_filename}"
            )

        # Check the run script filename
        run_script_filename = check_filename(
            (
                DEFAULT_RUN_SCRIPT_FILENAME
                if run_script_filename is None
                else run_script_filename
            ),
            name="run script file",
            ext=SCRIPT_FILE_EXTENSION
        )

        cls_name = cls.__name__
        # Create the script file
        with open(run_script_filename, 'w', encoding="utf8") as run_script:
            run_script.write(
                cls._run_script_code.format_map(
                    {
                        "config_filename": config_filename,
                        "factory_method": factory_method,
                        "cls_name": cls_name,
                        "var_name": cls_name.lower(),
                        "res": "{res}"
                    }
                )
            )

        # Make the run script file executable
        chmod(run_script_filename, 0o777)

    def reset(self) -> None:
        """Reset the results."""
        self.trainer.reset()
        self._results = None

    def run(self) -> None:
        """Execute the evaluation and save the results."""
        # Forget previous results
        self.reset()

        if not isfile(self.results_pickle_filename):
            # Init the results manager
            self._results = Results()

            # Run the evaluation
            self._execute()

            # Save the results
            self.results.save_pickle(self.results_pickle_filename)
        else:
            # Load the results
            self._results = Results.load_pickle(self.results_pickle_filename)

        if not isfile(self.results_excel_filename):
            # Save the results to Excel
            self.results.to_excel(self.results_excel_filename)

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
    def _load_config(config_script_filename: Optional[str] = None) -> object:
        """Generate a new evaluation from a configuration file.

        :param config_script_filename: Path to the configuration file. If set
            to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :py:data:`None`
        :type config_script_filename: :py:class:`str`, optional
        :raises TypeError: If *config_script_filename* is not a valid filename
        :raises ValueError: If the extension of *config_script_filename* is
            not '.py'
        :raises RuntimeError: If *config_script_filename* is an invalid file
            path or an invalid configuration script
        """
        # Check the configuration script filename
        config_script_filename = check_filename(
            (
                DEFAULT_CONFIG_SCRIPT_FILENAME
                if config_script_filename is None
                else config_script_filename
            ),
            name="configuration script file",
            ext=SCRIPT_FILE_EXTENSION
        )

        if not isfile(config_script_filename):
            raise RuntimeError(
                f"Configuration file not found: {config_script_filename}"
            )

        # Try to read the configuration script
        try:
            # Get the spec
            spec = importlib.util.spec_from_file_location(
                "config",
                config_script_filename
            )

            # Load the module
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
        except Exception:
            raise RuntimeError(
                f"Bad configuration script: {config_script_filename}"
            )

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

        # Dataframe index
        index = []

        # Add the hyperparameters (if any)
        if self.hyperparameters is not None:
            logbook_len = len(logbook)
            for hyper_name, hyper_value in self.hyperparameters.items():
                df[hyper_name] = (hyper_value,) * logbook_len * num_obj
                index += [hyper_name]

        # Add the iteration stats
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
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += (
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
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += (
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

            if self.hyperparameters is not None:
                for name, value in self.hyperparameters.items():
                    species_df[name] = [value] * n_obj

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

        # Create the DataFrame if it doesn't exist
        if result_key not in self.results:
            if self.hyperparameters is not None:
                # Index for the dataframe
                index = list(self.hyperparameters.keys())
                self.results[result_key] = DataFrame()
                for hyper_name, hyper_value in self.hyperparameters.items():
                    self.results[result_key][hyper_name] = [hyper_value]

                self.results[result_key].set_index(index, inplace=True)
                self.results[result_key].sort_index(inplace=True)
            else:
                # Index for the dataframe
                index = [_Labels.value]
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
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += [_Labels.feature]

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
            is_first_metric = True
            metric = None
            for name, func in self.feature_metric_functions.items():
                metric = func(features_hof)
                df[name] = metric
                # If there is any metric
                if metric is not None and is_first_metric:
                    is_first_metric = False
                    df[_Labels.feature] = metric.index
                    num_feats = len(metric.index)
                    if self.hyperparameters is not None:
                        for (
                                hyper_name,
                                hyper_value
                        ) in self.hyperparameters.items():
                            df[hyper_name] = [hyper_value] * num_feats

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
        hyperparameters: Optional[dict] = None,
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
            If set to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_RESULTS_BASENAME` is used.
            Defaults to :py:data:`None`
        :type results_base_filename: :py:class:`~str`, optional
        :param hyperparameters: Hyperparameter values used in this evaluation
        :type hyperparameters: :py:class:`~dict`, optional
        :param num_experiments: Number of experiments in the batch,
            defaults to :py:attr:`~culebra.tools.DEFAULT_NUM_EXPERIMENTS`
        :type num_experiments: :py:class:`int`, optional
        :raises TypeError: If *trainer* is not a valid trainer
        :raises TypeError: If *test_fitness_function* is not a valid
            fitness function
        :raises TypeError: If *results_base_filename* is not a valid file name
        :raises TypeError: If *hyperparameters* is not a dictionary
        :raises ValueError: If the keys in *hyperparameters* are not strings
        :raises ValueError: If any key in *hyperparameters* is reserved
        :raises TypeError: If *num_experiments* is not an integer
        :raises ValueError: If *num_experiments* is not greater than zero
        """
        # Init the super class
        super().__init__(
            trainer,
            test_fitness_function,
            results_base_filename,
            hyperparameters
        )

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
    def experiment_basename(self) -> str:
        """Return the experiments basename.

        :type: :py:class:`str`
        """
        return _Labels.experiment.lower()

    @property
    def experiment_labels(self) -> Tuple[str]:
        """Get the label to identify each one of the experiments in the batch.

        :type: :py:class:`tuple` of :py:class:`str`
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
        cls, config_script_filename: str
    ) -> Batch:
        """Generate a new evaluation from a configuration file.

        :param config_script_filename: Path to the configuration file. If set
            to :py:data:`None`,
            :py:attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :py:data:`None`
        :type config_script_filename: :py:class:`str`, optional
        :raises RuntimeError: If *config_script_filename* is an invalid file
            path or an invalid configuration file
        """
        # Load the config module
        config = cls._load_config(config_script_filename)

        # Generate the Batch from the config module
        return cls(
            trainer=getattr(config, 'trainer', None),
            test_fitness_function=getattr(
                config, 'test_fitness_function', None
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
            if self.hyperparameters is not None:
                index = list(self.hyperparameters.keys())
            else:
                index = []

            index += [_Labels.experiment]

            if isinstance(exp_data, DataFrame):
                if exp_data.index.names[0] is not None:
                    if self.hyperparameters is not None:
                        num_hyperparams = len(self.hyperparameters)
                        index += exp_data.index.names[num_hyperparams:]
                    else:
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

    def _add_execution_metrics_stats(self) -> None:
        """Perform some stats on the execution metrics."""
        # Name of the result
        result_key = self._ResultKeys.batch_execution_metrics_stats

        # Input data
        input_data_name = self._ResultKeys.execution_metrics
        input_data = self.results[input_data_name]

        # Index for the dataframe
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += [_Labels.metric]

        # Column names for the dataframe
        column_names = index + list(self.stats_functions.keys())

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
            for func in self.stats_functions.values():
                stats.append(func(input_data[metric]))

            # Append the row to the dataframe
            df.loc[len(df)] = stats

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(_Labels.stat, inplace=True)
        self.results[result_key] = df

    def _add_feature_metrics_stats(self) -> None:
        """Perform stats on the feature metrics of all the experiments."""
        try:
            # Name of the result
            result_key = self._ResultKeys.batch_feature_metrics_stats

            # Input data
            input_data_name = self._ResultKeys.feature_metrics
            input_data = self.results[input_data_name]

            # Index for the dataframe
            if self.hyperparameters is not None:
                index = list(self.hyperparameters.keys())
            else:
                index = []

            index += [_Labels.metric, _Labels.feature]

            # Column names for the dataframe
            column_names = index + list(self.stats_functions.keys())

            # Create a dataframe
            df = DataFrame(columns=column_names)

            # Get the features
            features_index = input_data.index.names.index(_Labels.feature)
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
        if self.hyperparameters is not None:
            index = list(self.hyperparameters.keys())
        else:
            index = []

        index += [_Labels.fitness]

        # Column names for the dataframe
        column_names = index + list(self.stats_functions.keys())

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
            for func in self.stats_functions.values():
                stats.append(func(input_data[obj_name]))

            # Append the row to the dataframe
            df.loc[len(df)] = stats

        df.set_index(index, inplace=True)
        df.sort_index(inplace=True)
        df.columns.set_names(_Labels.stat, inplace=True)
        self.results[result_key] = df

    def setup(self) -> None:
        """Set up the batch.

        Create all the experiments for the batch.
        """
        # Create the experiment
        experiment = Experiment(
            self.trainer,
            self.test_fitness_function,
            self.results_base_filename,
            self.hyperparameters
        )
        experiment_filename = self.experiment_basename + PICKLE_FILE_EXTENSION

        # Save the experiment
        experiment.save_pickle(experiment_filename)

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
        experiment = Experiment.load_pickle(
            self.experiment_basename + PICKLE_FILE_EXTENSION
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
        for result_key in self.results:
            self.results[result_key].set_index(
                self._results_indices[result_key], inplace=True
            )
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
    'DEFAULT_RUN_SCRIPT_FILENAME',
    'DEFAULT_CONFIG_SCRIPT_FILENAME',
    'DEFAULT_RESULTS_BASENAME'
]
