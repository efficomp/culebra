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

"""Absctract base classes for some common scripts.

This module provides the following classes:

* :class:`~culebra.tools.abc.scripts.TrainingTestDataGenerator`, which
  implements the common preprocessing pipeline shared by all datasets
* :class:`~culebra.tools.abc.scripts.TrainingTestFitnessFunctionsGenerator`,
  which generate the training and test fitness functions from the training and
  test datasets
* :class:`~culebra.tools.abc.scripts.TrainerGenerator`, to generate a trainer
  from command-line parameters
* :class:`~culebra.tools.abc.scripts.BatchGenerator`, to generate a batch of
  experiments from command-line parameters
* :class:`~culebra.tools.abc.scripts.BatchResultsAnalyzer`, to analyze the
  results of trainer batches
"""

from abc import abstractmethod
import os
from pathlib import Path

from culebra import SERIALIZED_FILE_EXTENSION, DEFAULT_SIMILARITY_THRESHOLD
from culebra.trainer import DEFAULT_MAX_NUM_ITERS
from culebra.fitness_func.dataset_score import DEFAULT_CV_NUM_FOLDS
from culebra.tools import DEFAULT_NUM_EXPERIMENTS
from ..dataset import Dataset
from .abc import Script, GeneratorScript


class TrainingTestDataGenerator(GeneratorScript):
    """
    Base class for dataset processing workflows.

    This class implements the common preprocessing pipeline shared by all
    datasets. It parses command-line arguments, fetches a dataset through the
    dataset-specific :meth:`get_dataset` method, removes missing values,
    scales features, splits the data into training and test sets, removes
    outliers from the training set, balances the training set by oversampling,
    and stores the resulting datasets to disk.

    Subclasses must implement the :meth:`get_dataset` method to provide the
    logic required to obtain and validate a particular dataset.

    The expected workflow is:

        generator = ConcreteTrainingTestDatasetGenerator()
        generator.generate()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    :ivar _training_dataset: Training dataset
    :vartype _training_dataset: ~culebra.tools.Dataset
    :ivar _test_dataset: Test dataset
    :vartype _test_dataset: ~culebra.tools.Dataset
    """

    default_test_prop = 0.3
    """Default proportion of samples assigned to the test set."""

    default_training_data = "training.data"
    """Default output file for the training dataset."""

    default_test_data = "test.data"
    """Default output file for the test dataset."""

    default_random_seed = 0
    """Default random seed."""

    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the generator.

        Define the command-line arguments.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        super().__init__(args)
        self._training_dataset = None
        self._test_dataset = None

        self._parser.description = "Load and split a dataset."

        self._parser.add_argument(
            "--test-prop",
            type=float,
            default=self.default_test_prop,
            metavar="TEST_PROP",
            help=(
                "Proportion of samples assigned to the test set "
                "(0 < value < 1). Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--training-data",
            default=self.default_training_data,
            metavar="TRAINING_DATA",
            help="Output file for the training dataset. Default: %(default)s."
        )

        self._parser.add_argument(
            "--test-data",
            default=self.default_test_data,
            metavar="TEST_DATA",
            help="Output file for the test dataset. Default: %(default)s."
        )

        self._parser.add_argument(
            "--random-seed",
            type=int,
            default=self.default_random_seed,
            metavar="RANDOM_SEED",
            help=(
                "Random seed used for reproducible splitting. "
                "Default: %(default)s."
            )
        )

    def parse_args(self) -> None:
        """
        Parse and validate command-line arguments.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        super().parse_args()

        # Check the proportion of samples assigned to the test set
        if not 0 < self._args.test_prop < 1:
            self._parser.error(
                "test_prop must be greater than 0 and less than 1: "
                f"{self._args.test_prop}"
            )

        # Check the training and test data filenames
        path = Path(self._args.training_data)
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            self._parser.error(
                "Wrong training data file path: "
                f"{self._args.training_data}"
            )
        path = Path(self._args.test_data)
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            self._parser.error(
                "Wrong test data file path: "
                f"{self._args.test_data}"
            )

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """
        Get and validate the dataset.

        Subclasses must implement this method to get a concrete dataset.

        :return: The dataset.
        :rtype: ~culebra.tools.Dataset

        :raises ValueError: If the dataset does not match the expected
            number of features, samples, or classes.
        """

    def process(self) -> None:
        """
        Process the dataset.

        Get the dataset, remove missing values, scale features, split the
        dataset into training and test sets, remove outliers from the training
        set, balance the training set through oversampling and save both
        datasets.
        """
        # Parse command-line arguments
        self.parse_args()

        # Get and preprocess the dataset
        data = self.get_dataset().drop_missing().scale()

        # Split the data into training and test (stratified)
        (self._training_dataset, self._test_dataset) = data.split(
            test_prop=self._args.test_prop, random_seed=self._args.random_seed
        )

        # Oversample the training data so that all the classes have the same
        # number of samples
        self._training_dataset = self._training_dataset.remove_outliers(
            random_seed=self._args.random_seed
        ).oversample(
            random_seed=self._args.random_seed
        )

    def generate(self) -> None:
        """
        Generate the training and test data files.
        """
        # Process the dataset
        self.process()

        # Save the training and test datasets
        self._training_dataset.to_text(self._args.training_data)
        self._test_dataset.to_text(self._args.test_data)


class TrainingTestFitnessFunctionsGenerator(GeneratorScript):
    """
    Base class for training and test fitness functions generation from
    command-line arguments.

    This class implements the command-line argument parsing and fitness
    functions dumping.

    Subclasses must implement the :meth:`process` method to
    provide the logic required to define particular fitness functions.

    The expected workflow is:

        generator = ConcreteTrainingTestFitnessFunctionsGenerator()
        generator.generate()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    :ivar _training_fitness_func: Training fitness function
    :vartype _training_fitness_func: ~culebra.abc.FitnessFunction
    :ivar _test_fitness_func: Test fitness function
    :vartype _test_fitness_func: ~culebra.abc.FitnessFunction
    """

    default_training_data = TrainingTestDataGenerator.default_training_data
    """Default input file for the training dataset."""

    default_test_data = TrainingTestDataGenerator.default_test_data
    """Default input file for the test dataset."""

    default_training_cv_num_folds = DEFAULT_CV_NUM_FOLDS
    """Default number of training cross-validation folds."""

    default_training_sim_threshold = DEFAULT_SIMILARITY_THRESHOLD
    """Default training similarity threshold"""

    default_training_func_obj = default_training_data.replace(
        ".data", "_func" + SERIALIZED_FILE_EXTENSION
    )
    """Default output file for the serialized training fitness function."""

    default_test_func_obj = default_test_data.replace(
        ".data", "_func" + SERIALIZED_FILE_EXTENSION
    )
    """Default output file for the serialized test fitness function."""

    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the generator.

        Define the command-line arguments.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        super().__init__(args)
        self._training_fitness_func = None
        self._test_fitness_func = None

        self._parser.add_argument(
            "--training-data",
            default=self.default_training_data,
            metavar="TRAINING_DATA",
            help="Input file for the training dataset. Default: %(default)s."
        )

        self._parser.add_argument(
            "--test-data",
            default=self.default_test_data,
            metavar="TEST_DATA",
            help="Input file for the test dataset. Default: %(default)s."
        )

        self._parser.add_argument(
            "--training-cv-num-folds",
            type=int,
            default=self.default_training_cv_num_folds,
            metavar="TRAINING_CV_NUM_FOLDS",
            help=(
                "Number of training cross-validation folds (value >= 2). "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--training-sim-threshold",
            type=float,
            default=self.default_training_sim_threshold,
            metavar="TRAINING_SIM_THRESHOLD",
            help=(
                "Training similarity threshold (value >= 0). "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--training-func-obj",
            default=self.default_training_func_obj,
            metavar="TRAINING_FUNC_OBJ",
            help=(
                "Output file for the serialized training fitness function. "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--test-func-obj",
            default=self.default_test_func_obj,
            metavar="TEST_FUNC_OBJ",
            help=(
                "Output file for the serialized test fitness function. "
                "Default: %(default)s."
            )
        )

    def parse_args(self) -> None:
        """
        Parse and validate command-line arguments.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        super().parse_args()

        # Check the training and test data files
        if not Path(self._args.training_data).is_file():
            self._parser.error(
                "Can't open the training data file: "
                f"{self._args.training_data}"
            )
        if not Path(self._args.test_data).is_file():
            self._parser.error(
                f"Can't open the test data file: {self._args.test_data}"
            )

        # Check the number of training cross-validation folds
        if self._args.training_cv_num_folds < 2:
            self._parser.error(
                "The minimum number of training cv folds is 2: "
                f"{self._args.training_cv_num_folds}"
            )

        # Check the training similarity threshold
        if self._args.training_sim_threshold < 0:
            self._parser.error(
                "The training similarity threshold can't be negative: "
                f"{self._args.training_sim_threshold}"
            )

        # Check the fitness functions filenames
        path = Path(self._args.training_func_obj)
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            self._parser.error(
                "Wrong serialized training fitness function file path: "
                f"{self._args.training_func_obj}"
            )
        if not self._args.training_func_obj.endswith(SERIALIZED_FILE_EXTENSION):
            self._parser.error(
                "Wrong serialized training fitness function file extension: "
                f"{self._args.training_func_obj}"
            )
        path = Path(self._args.test_func_obj)
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            self._parser.error(
                "Wrong serialized test fitness function file path: "
                f"{self._args.test_func_obj}"
            )
        if not self._args.test_func_obj.endswith(SERIALIZED_FILE_EXTENSION):
            self._parser.error(
                "Wrong serialized test fitness function file extension: "
                f"{self._args.test_func_obj}"
            )

    def generate(self) -> None:
        """
        Generate the training and test fitness functions.
        """
        # Create the fitness functions
        self.process()

        # Dump the fitness functions
        self._training_fitness_func.dump(self._args.training_func_obj)
        self._test_fitness_func.dump(self._args.test_func_obj)


class TrainerGenerator(GeneratorScript):
    """
    Base class for trainer generation from command-line arguments.

    This class implements the command-line argument parsing and trainer
    dumping.

    Subclasses must implement the :meth:`process` method to provide the
    logic required to generate a particular trainer.

    The expected workflow is:

        generator = ConcreteTrainerGenerator()
        generator.generate()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    :ivar _trainer: Trainer
    :vartype _trainer: ~culebra.abc.Trainer
    """

    default_training_func_obj = (
        TrainingTestFitnessFunctionsGenerator.default_training_func_obj
    )
    """Default intput file for the serialized training fitness function."""

    default_num_solutions = None
    """The default number of solutions evaluated each iteration will be fixed
    according to each particular problem."""

    default_max_num_iters = DEFAULT_MAX_NUM_ITERS
    """The default number of solutions evaluated each iteration will be fixed
    according to each particular problem.
    E.g., the number of features/nodes."""

    default_trainer_obj = "trainer" + SERIALIZED_FILE_EXTENSION
    """Default output file for the serialized trainer."""

    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the generator.

        Define the command-line arguments.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        super().__init__(args)
        self._trainer = None

        self._parser.add_argument(
            "--training-func-obj",
            default=self.default_training_func_obj,
            metavar="TRAINING_FUNC_OBJ",
            help=(
                "Input file for the serialized training fitness function. "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--num-solutions",
            type=int,
            default=self.default_num_solutions,
            metavar="NUM_SOLUTIONS",
            help=(
                "Number of solutions evaluated each iteration (value > 0)."
                "If omitted, it will be set according to the particular "
                "problem, such as the number of features or nodes.")
        )

        self._parser.add_argument(
            "--max-num-iters",
            type=int,
            default=self.default_max_num_iters,
            metavar="MAX_NUM_ITERS",
            help=(
                "Maximum number of iterations (value > 0). "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--trainer-obj",
            default=self.default_trainer_obj,
            metavar="TRAINER_OBJ",
            help=(
                "Output file for the serialized trainer. Default: %(default)s."
            )
        )

    def parse_args(self) -> None:
        """
        Parse and validate command-line arguments.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        super().parse_args()

        # Check the training fitness function
        if not Path(self._args.training_func_obj).is_file():
            self._parser.error(
                "Can't open the serialized training fitness function file: "
                f"{self._args.training_func_obj}"
            )
        if not self._args.training_func_obj.endswith(
            SERIALIZED_FILE_EXTENSION
        ):
            self._parser.error(
                "Wrong serialized training fitness function file extension: "
                f"{self._args.training_func_obj}"
            )

        # Check the number of solutions
        if (
            self._args.num_solutions is not None and
            self._args.num_solutions <= 0
        ):
            self._parser.error(
                "The number of solutions must be positive: "
                f"{self._args.num_solutions}"
            )

        # Check the maximum number of iterations
        if self._args.max_num_iters <= 0:
            self._parser.error(
                "The maximum number of iterations must be positive: "
                f"{self._args.max_num_iters}"
            )

        # Check the trainer filename
        path = Path(self._args.trainer_obj)
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            self._parser.error(
                "Wrong serialized trainer file path: "
                f"{self._args.trainer_obj}"
            )
        if not self._args.trainer_obj.endswith(SERIALIZED_FILE_EXTENSION):
            self._parser.error(
                "Wrong serialized trainer file extension: "
                f"{self._args.trainer_obj}"
            )

    def generate(self) -> None:
        """
        Generate the trainer.
        """
        # Create the trainer
        self.process()

        # Dump the trainer
        self._trainer.dump(self._args.trainer_obj)


class BatchGenerator(GeneratorScript):
    """
    Base class for batch generation from command-line arguments.

    This class implements the command-line argument parsing and trainer
    dumping.

    Subclasses must implement the :meth:`process` method to provide the
    logic required to generate a particular batch.

    The expected workflow is:

        generator = ConcreteBatchGenerator()
        generator.generate()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    :ivar _batch: Batch
    :vartype _batch: ~culebra.tools.evaluation.Batch
    """

    default_trainer_obj = TrainerGenerator.default_trainer_obj
    """Default input file for the serialized trainer."""

    default_dm_obj_threshold = DEFAULT_SIMILARITY_THRESHOLD
    """Default objective threshold for the decision manager. """

    default_test_func_obj = (
        TrainingTestFitnessFunctionsGenerator.default_test_func_obj
    )
    """Default input file for the serialized test fitness function."""

    default_experiments_per_batch = DEFAULT_NUM_EXPERIMENTS
    """Default number of experiments per batch."""

    default_batch_obj = "batch" + SERIALIZED_FILE_EXTENSION
    """Default output file for the serialized batch."""

    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the generator.

        Define the command-line arguments.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        super().__init__(args)
        self._batch = None

        self._parser.add_argument(
            "--trainer-obj",
            default=self.default_trainer_obj,
            metavar="TRAINER_OBJ",
            help="Input file for the serialized trainer. Default: %(default)s."
        )

        self._parser.add_argument(
            "--dm-obj-threshold",
            type=float,
            default=self.default_dm_obj_threshold,
            metavar="DM_OBJ_THRESHOLD",
            help=(
                "Objective threshold for the decision manager (value >= 0). "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--test-func-obj",
            default=self.default_test_func_obj,
            metavar="TEST_FUNC_OBJ",
            help=(
                "Input file for the serialized test fitness function. "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--experiments-per-batch",
            type=int,
            default=self.default_experiments_per_batch,
            metavar="EXPERIMENTS_PER_BATCH",
            help=(
                "Number of experiments per batch (value > 0). "
                "Default: %(default)s."
            )
        )

        self._parser.add_argument(
            "--batch-obj",
            default=self.default_batch_obj,
            metavar="BATCH_OBJ",
            help="Output file for the serialized batch. Default: %(default)s."
        )

    def parse_args(self) -> None:
        """
        Parse and validate command-line arguments.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        super().parse_args()

        # Check the trainer filename
        if not Path(self._args.trainer_obj).is_file():
            self._parser.error(
                "Can't open the serialized trainer file: "
                f"{self._args.trainer_obj}"
            )
        if not self._args.trainer_obj.endswith(
            SERIALIZED_FILE_EXTENSION
        ):
            self._parser.error(
                "Wrong serialized trainer file extension: "
                f"{self._args.trainer_obj}"
            )

        # Check the objective threshold for the decision manager
        if self._args.dm_obj_threshold < 0:
            self._parser.error(
                "The objective threshold for the decision manager can't be "
                f"negative: {self._args.dm_obj_threshold}"
            )

        # Check the test fitness function filename
        if not Path(self._args.test_func_obj).is_file():
            self._parser.error(
                "Can't open the serialized test fitness function file: "
                f"{self._args.test_func_obj}"
            )
        if not self._args.test_func_obj.endswith(
            SERIALIZED_FILE_EXTENSION
        ):
            self._parser.error(
                "Wrong serialized test fitness function file extension: "
                f"{self._args.test_func_obj}"
            )

        # Check the number of experiments per batch
        if self._args.experiments_per_batch <= 0:
            self._parser.error(
                "The number of experiments per batch must be positive: "
                f"{self._args.experiments_per_batch}"
            )

        # Check the batch filename
        path = Path(self._args.batch_obj)
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            self._parser.error(
                f"Wrong serialized batch file path: {self._args.batch_obj}"
            )
        if not self._args.batch_obj.endswith(SERIALIZED_FILE_EXTENSION):
            self._parser.error(
                "Wrong serialized batch file extension: "
                f"{self._args.batch_obj}"
            )

    def generate(self) -> None:
        """
        Generate the batch.
        """
        # Create the batch
        self.process()

        # Setup the batch
        self._batch.setup()

        # Generate the run script
        self._batch.generate_run_script(self._args.batch_obj)

        # Dump the batch
        self._batch.dump(self._args.batch_obj)


class BatchResultsAnalyzer(Script):
    """
    Base class for batch results analyzer scripts.

    This class implements the command-line argument parsing.

    Subclasses must implement the :meth:`process` method to provide the
    logic required analyze the batch results.

    The expected workflow is:

        analyzer = ConcreteBatchResultsAnalyzer()
        analyzer.process()

    :ivar _parser: Argument parser.
    :vartype _parser: ~argparse.ArgumentParser
    :ivar _debug_args: Sequence of debug arguments.
    :vartype _debug_args: list[str, ...]
    :ivar _args: Parsed command-line arguments.
    :vartype _args: ~argparse.Namespace
    """

    default_method = None
    """Default analysis method. Subclasses should override this attribute."""

    def __init__(self, args: list[str] | None = None) -> None:
        """
        Construct the analyzer.

        Define the command-line arguments.

        :param args: Sequence of command-line arguments to parse. If ``None``
            (default), arguments are read from ``sys.argv``. This parameter
            is mainly intended for testing, allowing command-line arguments
            to be supplied programmatically.
        :type args: list[str] | None
        """
        super().__init__(args)

        self._parser.add_argument(
            "--method",
            default=self.default_method,
            metavar="METHOD",
            help="Analysis method."
        )

        self._parser.add_argument(
            "batches",
            metavar="BATCH",
            nargs="+",
            help="A minimum of two batch folders"
        )

    def parse_args(self) -> None:
        """
        Parse and validate command-line arguments.

        The first argument is expected to be an analysis method, while the
        remaining arguments provide a list of folders with batch results.

        :raises SystemExit: If the command-line arguments are invalid or
            ``--help`` is requested.
        """
        super().parse_args()

        # Check the number of trainers
        if len(self._args.batches) < 2:
            self._parser.error(
                "A minimum of two batches are needed"
            )

        # Check the batches folders
        for batch in self._args.batches:
            if not Path(batch).is_dir():
                self._parser.error(
                    f"Can't open the batch folder: {batch}"
                )


# Exported symbols for this module
__all__ = [
    'TrainingTestDataGenerator',
    'TrainingTestFitnessFunctionsGenerator',
    'TrainerGenerator',
    'BatchGenerator',
    'BatchResultsAnalyzer'
]
