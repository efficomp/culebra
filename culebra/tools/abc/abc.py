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

"""Abstract base classes for tool development.

The :mod:`~culebra.tools.abc` module defines the abstract base classes used to
support experimentation and the selection of optimal solutions. Currently, it
includes the following classes:

* :class:`~culebra.tools.abc.DecisionManager`, which is responsible for
  selecting a solution from among the best candidates identified by the
  trainer.
* :class:`~culebra.tools.abc.Evaluation`, which provides the interface for
  evaluating a trainer.
* :class:`~culebra.tools.abc.GeneratorScript`, which is useful to develop
  scripts that generate files or objects
* :class:`~culebra.tools.abc.Script`, which is the base for command-line
  scripts
"""

from __future__ import annotations

from abc import abstractmethod
import argparse
from enum import Enum
from copy import deepcopy
from os import chmod
from os.path import isfile
import importlib.util

import numpy as np

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import (
    Base,
    Solution,
    Fitness,
    FitnessFunction,
    Trainer
)
from culebra.checker import (
    check_instance,
    check_filename,
    check_params
)
from culebra.solution.feature_selection import Metrics
from ..constants import (
    DEFAULT_SCRIPT_FILE_EXTENSION,
    DEFAULT_RESULTS_BASE_FILENAME,
    DEFAULT_EXCEL_FILE_EXTENSION,
    DEFAULT_RUN_SCRIPT_FILENAME,
    DEFAULT_CONFIG_SCRIPT_FILENAME,
)
from ..results import Results


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_STATS_FUNCS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max
}
"""Default statistics calculated for the results."""

DEFAULT_FEATURE_METRIC_FUNCS = {
    "Relevance": Metrics.relevance,
    "Rank": Metrics.rank
}
"""Default metrics calculated for the features in the set of solutions."""


class DecisionManager(Base):
    """Base class for the decision managers."""

    def __init__(self, trainer: Trainer) -> None:
        """Create a decision manager.

        :param trainer: The trainer
        :type trainer: ~culebra.abc.Trainer
        :raises TypeError: If *trainer* is not a valid trainer
        """
        self.trainer = trainer

    @property
    def trainer(self) -> Trainer:
        """Return the trainer.

        :rtype: ~culebra.abc.Trainer
        :setter: Set a new trainer
        :param value: The new trainer
        :type value: ~culebra.abc.Trainer
        :raises TypeError: If *trainer* is not a valid trainer
        """
        return self._trainer

    @trainer.setter
    def trainer(self, value: Trainer) -> None:
        """Set a new trainer.

        :param value: The new trainer
        :type value: ~culebra.abc.Trainer
        :raises TypeError: If *trainer* is not a valid trainer
        """
        # Check the value
        self._trainer = check_instance(value, "trainer", Trainer)

    def _generate_all_combinations(self) -> list[list[Solution, ...]]:
        """Generate all the combinations of Pareto optimal solutions.

        :return: The combinations
        :rtype: list[list[~culebra.abc.Solution, ...]]
        """
        # Obtain all the combinations of solutions from each species
        all_combinations = [[]]
        for species_best in self.trainer.best_solutions():
            temp = all_combinations
            all_combinations = []
            for comb in temp:
                for item in species_best:
                    all_combinations.append(comb + [item])

        return all_combinations

    @abstractmethod
    def _evaluate(
        self,
        all_combinations: list[list[Solution, ...]]
    ) -> list[Fitness, ...]:
        """Evaluate all the combinations of Pareto optimal solutions.

        :param all_combinations: All the combinations of the Pareto optimal
            solutions from each species
        :type all_combinations: list[list[~culebra.abc.Solution, ...]]
        :return: The fitness for each combination
        :rtype: list[Fitness, ...]

        This method must be overridden by subclasses to return a correct
        value.
        """
        raise NotImplementedError(
            "The _evaluate method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @abstractmethod
    def _choose(
        self,
        all_combinations: list[list[Solution, ...]],
        all_combinations_fitness: list[Fitness, ...]
    ) -> tuple[Solution, ...] | None:
        """Choose one combination of solutions.

        :param all_combinations: All the combinations of the Pareto optimal
            solutions from each species
        :type all_combinations: list[list[~culebra.abc.Solution, ...]]
        :all_combinations_fitness: The fitness for each combination
        :type all_combinations_fitness: list[Fitness, ...]
        :return: The chosen combination
        :rtype: tuple[~culebra.abc.Solution, ...]

        This method must be overridden by subclasses to return a correct
        value.
        """
        raise NotImplementedError(
            "The _choose method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    def select(self) -> tuple[Solution, ...] | None:
        """Select a solution from the trainer's best ones.

        :return: The chosen solution
        :rtype: tuple[~culebra.abc.Solution, ...]
        """
        all_combinations = self._generate_all_combinations()
        all_combinations_fitness = self._evaluate(all_combinations)
        return self._choose(
            all_combinations,
            all_combinations_fitness
        )

    def __copy__(self) -> DecisionManager:
        """Shallow copy the object.

        :return: The copied decision manager
        :rtype: ~culebra.tools.abc.DecisionManager
        """
        cls = self.__class__
        result = cls(self.trainer)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> DecisionManager:
        """Deepcopy the object.

        :param memo: Object attributes
        :type memo: dict
        :return: The copied decision manager
        :rtype: ~culebra.tools.abc.DecisionManager
        """
        cls = self.__class__
        result = cls(
            deepcopy(self.trainer, memo)
        )
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo | {
                    id(self.trainer): result.trainer
                }
            )
        )
        return result

    def __reduce__(self) -> tuple:
        """Reduce the object.

        :return: The reduction
        :rtype: tuple
        """
        return (
            self.__class__,
            (self.trainer, ),
            self.__dict__
        )

    @classmethod
    def __fromstate__(cls, state: dict) -> DecisionManager:
        """Return an evaluation from a state.

        :param state: The state
        :type state: dict
        :return: The decision manager
        :rtype: ~culebra.tools.abc.DecisionManager
        """
        obj = cls(state['_trainer'])
        obj.__setstate__(state)
        return deepcopy(obj)


class Evaluation(Base):
    """Base class for results evaluations."""

    class ResultsLabels(str, Enum):
        """Column labels used in result DataFrames."""

        SPECIES = "Species"
        SOLUTION = "Solution"
        FEATURE = "Feature"
        VALUE = "Value"
        FITNESS = "Fitness"
        RELEVANCE = "Relevance"
        RANK = "Rank"

        MAX = "Max"
        MIN = "Min"
        AVG = "Avg"
        STD = "Std"
        BEST = "Best"

        STAT = "Stat"
        METRIC = "Metric"

        RUNTIME = "Runtime"
        NUM_EVALS = "NEvals"
        NUM_ITERS = "NIters"

        EXPERIMENT = "Exp"
        BATCH = "Batch"

    feature_metric_funcs = DEFAULT_FEATURE_METRIC_FUNCS
    """Metrics calculated for the features in the set of solutions."""

    stats_funcs = DEFAULT_STATS_FUNCS
    """Statistics calculated for the solutions."""

    _run_script_code = """#!/usr/bin/env python3

#
# This script relies on the {config_filename} configuration file.
#
# This script is a simple python module defining variables to be passed to
# the {cls_name} constructor. These variables MUST have the same name than
# the constructor parameters.
#

from culebra.tools.evaluation import {cls_name}

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
        decision_manager: DecisionManager,
        test_fitness_func: FitnessFunction | None = None,
        results_base_filename: str | None = None,
        hyperparameters: dict | None = None
    ) -> None:
        """Set a trainer evaluation.

        :param trainer: The trainer
        :type trainer: ~culebra.abc.Trainer
        :param decision_manager: A decision manager to select the best solution
            from the set of best solutions found by the trainer
        :type decision_manager: ~culebra.tools.abc.DecisionManager
        :param test_fitness_func: The fitness function used to test. If
            omitted, the training fitness function will be used. Defaults to
            :data:`None`
        :type test_fitness_func: ~culebra.abc.FitnessFunction
        :param results_base_filename: The base filename to save the results.
            If omitted,
            :attr:`~culebra.tools.abc.Evaluation._default_results_base_filename` is
            used. Defaults to :data:`None`
        :type results_base_filename: str
        :param hyperparameters: Hyperparameter values used in this evaluation,
            optional
        :type hyperparameters: dict
        :raises TypeError: If any of the parameters has an incorrect type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        self.trainer = trainer
        self.decision_manager = decision_manager
        self.test_fitness_func = test_fitness_func
        self.results_base_filename = results_base_filename
        self.hyperparameters = hyperparameters

    @property
    def trainer(self) -> Trainer:
        """Return the trainer.

        :rtype: ~culebra.abc.Trainer
        :setter: Set a new trainer
        :param value: The new trainer
        :type value: ~culebra.abc.Trainer
        :raises TypeError: If *trainer* is not a valid trainer
        """
        return self._trainer

    @trainer.setter
    def trainer(self, value: Trainer) -> None:
        """Set a new trainer.

        :param value: The new trainer
        :type value: ~culebra.abc.Trainer
        :raises TypeError: If *trainer* is not a valid trainer
        """
        # Check the value
        self._trainer = check_instance(value, "trainer", Trainer)

        # Reset results
        self.reset()

    @property
    def decision_manager(self) -> DecisionManager:
        """Return the decicion manager.

        :rtype: ~culebra.tools.abc.DecisionManager
        :setter: Set a new decision manager
        :param dm: The new decision manager
        :type dm: ~culebra.tools.abc.DecisionManager
        :raises TypeError: If *dm* is not a valid decision manager
        """
        return self._decision_manager

    @decision_manager.setter
    def decision_manager(
        self, dm: DecisionManager
    ) -> None:
        """Set a new decision manager.

        :param dm: The new decision manager
        :type dm: ~culebra.tools.abc.DecisionManager
        :raises TypeError: If *dm* is not a valid decision manager
        """
        # Check the dm
        self._decision_manager = check_instance(
            dm, "decision manager", DecisionManager
        )

        # Reset results
        self.reset()

    @property
    def _default_test_fitness_func(self) -> FitnessFunction:
        """Default test fitness function.

        :return: The trainer's training function
        :rtype: ~culebra.abc.FitnessFunction
        """
        return self.trainer.fitness_func

    @property
    def test_fitness_func(self) -> FitnessFunction:
        """Test fitness function.

        :rtype: ~culebra.abc.FitnessFunction
        :setter: Set a new test fitness function.
        :param func: New test fitness function. If set to :data:`None`,
            the training fitness function will also be used for testing
        :type func: ~culebra.abc.FitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        """
        return (
            self._default_test_fitness_func
            if self._test_fitness_func is None
            else self._test_fitness_func
        )

    @test_fitness_func.setter
    def test_fitness_func(self, func: FitnessFunction | None) -> None:
        """Set a new test fitness function.

        :param func: New test fitness function. If set to :data:`None`,
            the training fitness function will also be used for testing
        :type func: ~culebra.abc.FitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        """
        # Check the function
        self._test_fitness_func = (
            None if func is None else check_instance(
                func, "test fitness function", FitnessFunction
            )
        )

        # Reset results
        self.reset()

    @property
    def _default_results_base_filename(self) -> str:
        """Default base name for results files.

        :return: :attr:`~culebra.tools.DEFAULT_RESULTS_BASE_FILENAME`
        :rtype: str
        """
        return DEFAULT_RESULTS_BASE_FILENAME

    @property
    def results_base_filename(self) -> str | None:
        """Results base filename.

        :rtype: str
        :setter: Set a new results base filename.

        :param filename: New results base filename. If set to :data:`None`,
            :attr:`~culebra.tools.abc.Evaluation._default_results_base_filename` is
            used
        :type filename: str
        :raises TypeError: If *filename* is not a valid file name
        """
        return (
            self._default_results_base_filename
            if self._results_base_filename is None
            else self._results_base_filename
        )

    @results_base_filename.setter
    def results_base_filename(self, filename: str | None) -> None:
        """Set a new results base filename.

        :param filename: New results base filename. If set to :data:`None`,
            :attr:`~culebra.tools.abc.Evaluation._default_results_base_filename` is
            used
        :type filename: str
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
    def serialized_results_filename(self) -> str:
        """Filename used to save the serialized results.

        :rtype: str
        """
        return self.results_base_filename + SERIALIZED_FILE_EXTENSION

    @property
    def excel_results_filename(self) -> str:
        """Filename used to save the results in Excel format.

        :rtype: str
        """
        return self.results_base_filename + DEFAULT_EXCEL_FILE_EXTENSION

    @property
    def hyperparameters(self) -> dict | None:
        """Hyperparameter values used for the evaluation.

        :rtype: dict

        :setter: Set the hyperparameter values used for the evaluation
        :param values: Hyperparameter values used in this evaluation
        :type values: dict
        :raises TypeError: If *values* is not a dictionary
        :raises ValueError: If the keys in *values* are not strings
        :raises ValueError: If any key in *values* is reserved
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, values: dict | None) -> None:
        """Set the hyperparameter values used for the evaluation.

        :param values: Hyperparameter values used in this evaluation
        :type values: dict
        :raises TypeError: If *values* is not a dictionary
        :raises ValueError: If the keys in *values* are not strings
        :raises ValueError: If any key in *values* is reserved
        """
        def is_reserved(label: str) -> bool:
            """Check if a label is reserved

            :param label The label
            :type label: str
            :return: :data:`True` if the given label is reserved
            :rtype: bool
            """
            reserved_labels_upper = (
                rsv.value.strip().upper() for rsv in self.ResultsLabels
            )

            if label.strip().upper() in reserved_labels_upper:
                return True

            return False

        if values is None:
            self._hyperparameters = None
            return

        self._hyperparameters = check_params(
            values,
            name="hyperparameters"
        )

        # Check that no parameter name is reserved
        for name in values.keys():
            if is_reserved(name):
                raise ValueError(
                    "Attempt to use a reserved label as a hyperparameter "
                    f"name: {name}"
                )

        # Reset results
        self.reset()

    @property
    def results(self) -> Results | None:
        """Results obtained.

        :rtype: ~culebra.tools.Results
        """
        return self._results

    @classmethod
    def from_config(
        cls,
        config_script_filename: str | None = None
    ) -> Evaluation:
        """Generate a new evaluation from a configuration file.

        :param config_script_filename: Path to the configuration file. If
            omitted,
            :attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :data:`None`
        :type config_script_filename: str
        :raises RuntimeError: If *config_script_filename* is an invalid file
            path or an invalid configuration file
        """
        # Load the config module
        config = cls._load_config(config_script_filename)

        # Generate the Evaluation from the config module
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
            hyperparameters=getattr(config, 'hyperparameters', None)
        )

    @classmethod
    def generate_run_script(
        cls,
        config_filename: str | None = None,
        run_script_filename: str | None = None
    ) -> None:
        """Generate a script to run an evaluation.

        The parameters for the evaluation are taken from a configuration file.

        :param config_filename: Path to the configuration file. It can be
            whether a configuration script or a serialized
            :attr:`~culebra.tools.abc.Evaluation` instance. If omitted,
            :attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :data:`None`
        :type config_filename: str
        :param run_script_filename: File path to store the run script. If
            omitted, :attr:`~culebra.tools.DEFAULT_RUN_SCRIPT_FILENAME` is
            used. Defaults to :data:`None`
        :type run_script_filename: str
        :raises TypeError: If *config_filename* or *run_script_filename*
            are not a valid filename
        :raises ValueError: If the extensions of *config_filename* or
            *run_script_filename* are not valid
        """
        if config_filename is None:
            config_filename = DEFAULT_CONFIG_SCRIPT_FILENAME

        # Check the configuration filename
        try:
            config_filename = check_filename(
                config_filename,
                name="configuration file",
                ext=DEFAULT_SCRIPT_FILE_EXTENSION
            )
            factory_method = 'from_config'
        except ValueError:
            try:
                config_filename = check_filename(
                    config_filename,
                    name="configuration file",
                    ext=SERIALIZED_FILE_EXTENSION
                )
                factory_method = 'load'
            except ValueError as e:
                raise ValueError(
                    "Not valid extension for the configuration file. "
                    f"Valid extensions are {DEFAULT_SCRIPT_FILE_EXTENSION} "
                    f"for python scripts or {SERIALIZED_FILE_EXTENSION} for "
                    f"serialized evaluation objects: {config_filename}"
                ) from e
            except TypeError as error:
                raise error

        # Check the run script filename
        run_script_filename = check_filename(
            (
                DEFAULT_RUN_SCRIPT_FILENAME
                if run_script_filename is None
                else run_script_filename
            ),
            name="run script file",
            ext=DEFAULT_SCRIPT_FILE_EXTENSION
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
        """Reset the evaluation."""
        self.trainer.reset()
        self._results = None

    def run(self) -> None:
        """Execute the evaluation and save the results."""
        # Forget previous results
        self.reset()

        if not isfile(self.serialized_results_filename):
            # Init the results manager
            self._results = Results()

            # Run the evaluation
            self._execute()

            # Save the results
            self.results.dump(self.serialized_results_filename)
        else:
            # Load the results
            self._results = Results.load(self.serialized_results_filename)

        if not isfile(self.excel_results_filename):
            # Save the results to Excel
            self.results.to_excel(self.excel_results_filename)

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
    def _load_config(config_script_filename: str | None = None) -> object:
        """Generate a new evaluation from a configuration file.

        :param config_script_filename: Path to the configuration file. If
            omitted,
            :attr:`~culebra.tools.DEFAULT_CONFIG_SCRIPT_FILENAME` is used.
            Defaults to :data:`None`
        :type config_script_filename: str
        :return: The configuration
        :rtype: object
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
            ext=DEFAULT_SCRIPT_FILE_EXTENSION
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
        except Exception as e:
            raise RuntimeError(
                f"Bad configuration script: {config_script_filename}"
            ) from e

        return config


    def __copy__(self) -> Evaluation:
        """Shallow copy the object.

        :return: The copied evaluation
        :rtype: ~culebra.tools.abc.Evaluation
        """
        cls = self.__class__
        result = cls(self.trainer, self.decision_manager)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Evaluation:
        """Deepcopy the object.

        :param memo: Object attributes
        :type memo: dict
        :return: The copied evaluation
        :rtype: ~culebra.tools.abc.Evaluation
        """
        cls = self.__class__
        result = cls(
            deepcopy(self.trainer, memo),
            deepcopy(self.decision_manager, memo),
        )
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo | {
                    id(self.trainer): result.trainer,
                    id(self.decision_manager): result.decision_manager,
                }
            )
        )
        return result

    def __reduce__(self) -> tuple:
        """Reduce the object.

        :return: The reduction
        :rtype: tuple
        """
        return (
            self.__class__,
            (self.trainer, self.decision_manager),
            self.__dict__
        )

    @classmethod
    def __fromstate__(cls, state: dict) -> Evaluation:
        """Return an evaluation from a state.

        :param state: The state
        :type state: dict
        :return: The evaluation
        :rtype: ~culebra.tools.abc.Evaluation
        """
        obj = cls(state['_trainer'], state['_decision_manager'])
        obj.__setstate__(state)
        return deepcopy(obj)


class Script(Base):
    """
    Abstract base class for python scripts.

    The expected workflow is:

        script = ConcreteScript()
        script.process()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    """
    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the generator.

        Generate an empty argument parser. Subclasses should add arguments to
        the parser.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        self._parser = argparse.ArgumentParser()
        self._debug_args = args
        self._args = None

    def parse_args(self) -> None:
        """
        Parse command-line arguments.

        Subclasses should validate the arguments.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        self._args = self._parser.parse_args(self._debug_args)

    @abstractmethod
    def process(self) -> None:
        """
        Execute the main processing logic of the script.
        """


class GeneratorScript(Script):
    """
    Abstract base class for python scripts for generators from command-line arguments.

    The expected workflow is:

        generator = ConcreteGenerator()
        generator.generate()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    """
    @abstractmethod
    def generate(self) -> None:
        """
        Generate the output files.
        """


# Exported symbols for this module
__all__ = [
    'DecisionManager',
    'Evaluation',
    'Script',
    'GeneratorScript'
]
