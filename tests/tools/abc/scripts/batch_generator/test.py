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
trainer_file = "trainer" + SERIALIZED_FILE_EXTENSION
dm_obj_threshold = "0.01"
test_func_file = "test_func" + SERIALIZED_FILE_EXTENSION
experiments_per_batch = "10"
batch_file = "batch" + SERIALIZED_FILE_EXTENSION

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
            self._args.trainer_file
        )

        # Generate the decision manager
        decision_manager = LexicographicWithRepeatedCVDM(
            trainer,
            self._args.dm_obj_threshold
        )

        # Load the test fitness function
        test_fitness_func = FitnessFunction.load(
            self._args.test_func_file
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
        trainer.dump(trainer_file)

        # Create and dump the test fitness function
        test_fitness_func = NumFeats()
        test_fitness_func.dump(test_func_file)

    @classmethod
    def tearDownClass(cls):
        """Remove the training fitness function and trainer files."""
        # Remove the trainer file
        if Path(trainer_file).exists():
            remove(trainer_file)

        # Remove the test fitness function file
        if Path(test_func_file).exists():
            remove(test_func_file)

        # Remove the experiment-related files and folders
        for item in Path(".").glob(exp_prefix + "*"):
            if item.is_dir():
                rmtree(item)
            else:
                remove(item)

        # Remove the batch file
        if Path(batch_file).exists():
            remove(batch_file)

        # Remove the run script
        if Path(DEFAULT_RUN_SCRIPT_FILENAME).exists():
            remove(DEFAULT_RUN_SCRIPT_FILENAME)

    def test_parse_args(self):
        """Test the parse_args method."""

        stderr = StringIO()
        with redirect_stderr(stderr):
        # Try without parameters. Should fail ...
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator()
                generator.parse_args()

            # Try a wrong trainer file
            wrong_file = "kk/wrong.dat"
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    [
                        wrong_file,
                        dm_obj_threshold,
                        test_func_file,
                        experiments_per_batch,
                        batch_file
                    ]
                )
                generator.parse_args()

            # Try a wrong trainer file extension
            wrong_extension = "test.py"
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    [
                        wrong_extension,
                        dm_obj_threshold,
                        test_func_file,
                        experiments_per_batch,
                        batch_file
                    ]
                )
                generator.parse_args()

            # Try a wrong objective threshold for the decision manager
            # Should fail ...
            wrong_dm_obj_threshold_values = ["a", "-1.5"]
            for wrong_dm_obj_threshold in wrong_dm_obj_threshold_values:
                with self.assertRaises(SystemExit):
                    generator = MyBatchGenerator(
                        [
                            trainer_file,
                            wrong_dm_obj_threshold,
                            test_func_file,
                            experiments_per_batch,
                            batch_file
                        ]
                    )
                    generator.parse_args()

            # Try a wrong test fitness function file
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    [
                        trainer_file,
                        dm_obj_threshold,
                        wrong_file,
                        experiments_per_batch,
                        batch_file
                    ]
                )
                generator.parse_args()

            # Try a wrong test fitness function file extension
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    [
                        trainer_file,
                        dm_obj_threshold,
                        wrong_extension,
                        experiments_per_batch,
                        batch_file
                    ]
                )
                generator.parse_args()

            # Try a wrong number of experiments per batch. Should fail ...
            wrong_experiments_per_batch_values = ["a", "1.5", "0"]
            for wrong_experiments_per_batch in wrong_experiments_per_batch_values:
                with self.assertRaises(SystemExit):
                    generator = MyBatchGenerator(
                        [
                            trainer_file,
                            dm_obj_threshold,
                            test_func_file,
                            wrong_experiments_per_batch,
                            batch_file
                        ]
                    )
                    generator.parse_args()

            # Try a wrong batch file
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    [
                        trainer_file,
                        dm_obj_threshold,
                        test_func_file,
                        experiments_per_batch,
                        wrong_file
                    ]
                )
                generator.parse_args()

            # Try a wrong batch file extension
            with self.assertRaises(SystemExit):
                generator = MyBatchGenerator(
                    [
                        trainer_file,
                        dm_obj_threshold,
                        test_func_file,
                        experiments_per_batch,
                        wrong_extension
                    ]
                )
                generator.parse_args()

        # Try valid values
        generator = MyBatchGenerator(
            [
                trainer_file,
                dm_obj_threshold,
                test_func_file,
                experiments_per_batch,
                batch_file
            ]
        )
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(generator._args.trainer_file, trainer_file)
        self.assertEqual(
            generator._args.dm_obj_threshold,
            float(dm_obj_threshold)
        )
        self.assertEqual(generator._args.test_func_file, test_func_file)
        self.assertEqual(
            generator._args.experiments_per_batch,
            int(experiments_per_batch)
        )
        self.assertEqual(generator._args.batch_file, batch_file)

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
                trainer_file,
                dm_obj_threshold,
                test_func_file,
                experiments_per_batch,
                batch_file
            ]
        )

        # Experiment folder names
        exp_folder_names = experiment_folders(experiments_per_batch)

        # Check that the experiment and batch related files do not exist yet
        self.assertFalse(Path(exp_file).exists())
        self.assertFalse(Path(batch_file).exists())
        for exp_folder in exp_folder_names:
            self.assertFalse(Path(exp_folder).exists())

        # Generate the trainer
        generator.generate()

        # Check that the experiment and batch related files exist now
        self.assertTrue(Path(exp_file).exists())
        self.assertTrue(Path(batch_file).exists())
        for exp_folder in exp_folder_names:
            self.assertTrue(Path(exp_folder).exists())


if __name__ == '__main__':
    unittest.main()
