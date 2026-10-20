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

"""Unit test for :class:`culebra.tools.abc.scripts.BatchGenerator`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path
from os import remove
from shutil import rmtree

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import FitnessFunction, Trainer
from culebra.solution.feature_selection import Species, Ant
from culebra.fitness_func.feature_selection import NumFeats
from culebra.trainer.aco import ACOFS1D
from culebra.tools import DEFAULT_RUN_SCRIPT_FILENAME
from culebra.tools.evaluation import Batch
from culebra.tools.decision_manager import LexicographicWithRepeatedCVDM
from culebra.tools.abc.scripts import BatchGenerator


# Script parameters
trainer_obj = "trainer" + SERIALIZED_FILE_EXTENSION
dm_obj_threshold = "0.01"
test_func_obj = "test_func" + SERIALIZED_FILE_EXTENSION
experiments_per_batch = "10"
batch_obj = "batch" + SERIALIZED_FILE_EXTENSION

# Experiment prefix
exp_prefix = "exp"

# Experiment file
exp_file = exp_prefix + SERIALIZED_FILE_EXTENSION


class MyBatchGenerator(BatchGenerator):
    """
    Create and setup a Batch.
    """

    def process(self):
        """Create the Batch."""
        # Parse command-line arguments
        self.parse_args()

        # Load the trainer
        trainer = Trainer.load(
            self._args.trainer_obj
        )

        # Generate the decision manager
        decision_manager = LexicographicWithRepeatedCVDM(
            trainer,
            self._args.dm_obj_threshold
        )

        # Load the test fitness function
        test_fitness_func = FitnessFunction.load(
            self._args.test_func_obj
        )

        self._batch = Batch(
            trainer,
            decision_manager=decision_manager,
            test_fitness_func=test_fitness_func,
            num_experiments=self._args.experiments_per_batch
        )


class BatchGeneratorTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.BatchGenerator`."""

    @classmethod
    def setUpClass(cls):
        """Generate a training fitness function and dump it."""
        # Training fitness function
        training_fitness_func = NumFeats()

        # Species
        species = Species(num_feats=10, min_size=1)

        # Create the trainer
        trainer = ACOFS1D(
            fitness_func=training_fitness_func,
            solution_cls=Ant,
            species=species,
            col_size=species.num_feats,
            max_num_iters=10
        )

        # Dump the trainer
        trainer.dump(trainer_obj)
        trainer.dump(BatchGenerator.default_trainer_obj)

        # Create and dump the test fitness function
        test_fitness_func = NumFeats()
        test_fitness_func.dump(test_func_obj)
        test_fitness_func.dump(BatchGenerator.default_test_func_obj)

    @classmethod
    def tearDownClass(cls):
        """Remove the training fitness function and trainer files."""
        for file in (
             trainer_obj,
             test_func_obj,
             batch_obj,
             BatchGenerator.default_trainer_obj,
             BatchGenerator.default_test_func_obj,
             BatchGenerator.default_batch_obj,
             DEFAULT_RUN_SCRIPT_FILENAME
        ):
            if Path(file).exists():
                remove(file)

        # Remove the experiment-related files and folders
        for item in Path(".").glob(exp_prefix + "*"):
            if item.is_dir():
                rmtree(item)
            else:
                remove(item)

    def test_parse_args(self):
        """Test the parse_args method."""

        stderr = StringIO()
        with redirect_stderr(stderr):
            # Try a wrong trainer file
            wrong_file = "kk/wrong.dat"
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    ["--trainer-obj", wrong_file]
                )
                generator.parse_args()

            # Try a wrong trainer file extension
            wrong_extension = "test.py"
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    ["--trainer-obj", wrong_extension]
                )
                generator.parse_args()

            # Try a wrong objective threshold for the decision manager
            # Should fail ...
            wrong_dm_obj_threshold_values = ["a", "-1.5"]
            for wrong_dm_obj_threshold in wrong_dm_obj_threshold_values:
                with self.assertRaises(SystemExit):
                    generator = MyBatchGenerator(
                        ["--dm-obj-threshold", wrong_dm_obj_threshold]
                    )
                    generator.parse_args()

            # Try a wrong test fitness function file
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    ["--test-func-obj", wrong_file]
                )
                generator.parse_args()

            # Try a wrong test fitness function file extension
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    ["--test-func-obj", wrong_extension]
                )
                generator.parse_args()

            # Try a wrong number of experiments per batch. Should fail ...
            wrong_experiments_per_batch_values = ["a", "1.5", "0"]
            for wrong_experiments_per_batch in wrong_experiments_per_batch_values:
                with self.assertRaises(SystemExit):
                    generator = MyBatchGenerator(
                        [
                            "--experiments-per-batch",
                            wrong_experiments_per_batch
                        ]
                    )
                    generator.parse_args()

            # Try a wrong batch file
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    ["--batch-obj", wrong_file]
                )
                generator.parse_args()

            # Try a wrong batch file extension
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    ["--batch-obj", wrong_extension]
                )
                generator.parse_args()

        # Try the dafault parameters
        generator = MyBatchGenerator()
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(
            generator._args.trainer_obj,
            generator.default_trainer_obj
        )
        self.assertEqual(
            generator._args.dm_obj_threshold,
            generator.default_dm_obj_threshold
        )
        self.assertEqual(
            generator._args.test_func_obj,
            generator.default_test_func_obj
        )
        self.assertEqual(
            generator._args.experiments_per_batch,
            generator.default_experiments_per_batch
        )
        self.assertEqual(
            generator._args.batch_obj,
            generator.default_batch_obj
        )

        # Try valid values
        generator = MyBatchGenerator(
            [
                "--trainer-obj", trainer_obj,
                "--dm-obj-threshold", dm_obj_threshold,
                "--test-func-obj", test_func_obj,
                "--experiments-per-batch", experiments_per_batch,
                "--batch-obj", batch_obj
            ]
        )
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(generator._args.trainer_obj, trainer_obj)
        self.assertEqual(
            generator._args.dm_obj_threshold,
            float(dm_obj_threshold)
        )
        self.assertEqual(generator._args.test_func_obj, test_func_obj)
        self.assertEqual(
            generator._args.experiments_per_batch,
            int(experiments_per_batch)
        )
        self.assertEqual(generator._args.batch_obj, batch_obj)

    def test_generate(self):
        """Test the generate method."""
        def experiment_folders(experiments_per_batch):
            # Suffix length
            num_experiments = int(experiments_per_batch)
            suffix_len = len(str(num_experiments-1))

            # Return the experiment names
            return tuple(
                exp_prefix +
                f"{i:0{suffix_len}d}" for i in range(num_experiments)
            )

        generator = MyBatchGenerator(
            [
                "--trainer-obj", trainer_obj,
                "--dm-obj-threshold", dm_obj_threshold,
                "--test-func-obj", test_func_obj,
                "--experiments-per-batch", experiments_per_batch,
                "--batch-obj", batch_obj
            ]
        )

        # Experiment folder names
        exp_folder_names = experiment_folders(experiments_per_batch)

        # Check that the experiment and batch related files do not exist yet
        self.assertFalse(Path(exp_file).exists())
        self.assertFalse(Path(batch_obj).exists())
        for exp_folder in exp_folder_names:
            self.assertFalse(Path(exp_folder).exists())

        # Generate the batch
        generator.generate()

        # Check that the experiment and batch related files exist now
        self.assertTrue(Path(exp_file).exists())
        self.assertTrue(Path(batch_obj).exists())
        for exp_folder in exp_folder_names:
            self.assertTrue(Path(exp_folder).exists())


if __name__ == '__main__':
    unittest.main()
