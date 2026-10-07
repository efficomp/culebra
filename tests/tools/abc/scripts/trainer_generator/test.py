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

"""Unit test for :class:`culebra.tools.abc.scripts.TrainerGenerator`."""

import unittest
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path
from os import remove

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import FitnessFunction
from culebra.solution.feature_selection import Ant, Species
from culebra.fitness_func.feature_selection import NumFeats
from culebra.trainer.aco import ACOFS1D
from culebra.tools.abc.scripts import TrainerGenerator


# Number of neighbors for KNN
KNN_N_NEIGHBORS = 5

# Script parameters
training_func_file = "training_func" + SERIALIZED_FILE_EXTENSION
num_solutions = "50"
max_num_iters = "100"
trainer_file = "trainer" + SERIALIZED_FILE_EXTENSION


class MyTrainerGenerator(TrainerGenerator):
    """
    Create and dump an ACOFS1D trainer.
    """

    def process(self):
        """Create the trainer."""
        # Parse command-line arguments
        self.parse_args()

        # Load the training fitness function
        training_fitness_func = FitnessFunction.load(
            self._args.training_func_file
        )

        # Construct the species
        species = Species(num_feats=10, min_size=1)

        # Create the trainer
        self._trainer = ACOFS1D(
            fitness_func=training_fitness_func,
            solution_cls=Ant,
            species=species,
            col_size=self._args.num_solutions,
            max_num_iters=self._args.max_num_iters
        )


class TrainerGeneratorTester(unittest.TestCase):
    """Test :class:`~culebra.tools.abc.scripts.TrainerGenerator`."""

    @classmethod
    def setUpClass(cls):
        """Generate a training fitness function and dump it."""
        training_fitness_func = NumFeats()
        training_fitness_func.dump(training_func_file)

    @classmethod
    def tearDownClass(cls):
        """Remove the training fitness function and trainer files."""
        if Path(training_func_file).exists():
            remove(training_func_file)
        if Path(trainer_file).exists():
            remove(trainer_file)

    def test_parse_args(self):
        """Test the parse_args method."""
        stderr = StringIO()
        with redirect_stderr(stderr):
        # Try without parameters. Should fail ...
            with self.assertRaises(SystemExit):
                generator = MyTrainerGenerator()
                generator.parse_args()

            # Try a wrong training fitness function file
            wrong_file = "wrong.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainerGenerator(
                    [
                        wrong_file,
                        num_solutions,
                        max_num_iters,
                        trainer_file
                    ]
                )
                generator.parse_args()

            # Try a wrong training fitness function file extension
            wrong_extension = "test.py"
            with self.assertRaises(SystemExit):
                generator = MyTrainerGenerator(
                    [
                        wrong_extension,
                        num_solutions,
                        max_num_iters,
                        trainer_file
                    ]
                )
                generator.parse_args()

            # Try a wrong number of solutions. Should fail ...
            wrong_positive_int_values = ["a", "1.5", "-1", "0"]
            for wrong_num_solutions in wrong_positive_int_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainerGenerator(
                        [
                            training_func_file,
                            wrong_num_solutions,
                            max_num_iters,
                            trainer_file
                        ]
                    )
                    generator.parse_args()

            # Try a wrong maximum number of iterations. Should fail ...
            for wrong_max_num_iters in wrong_positive_int_values:
                with self.assertRaises(SystemExit):
                    generator = MyTrainerGenerator(
                        [
                            training_func_file,
                            num_solutions,
                            wrong_max_num_iters,
                            trainer_file
                        ]
                    )
                    generator.parse_args()

            # Try wrong trainer filepath. Should fail ...
            wrong_filepath = "kk/file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainerGenerator(
                    [
                        training_func_file,
                        num_solutions,
                        max_num_iters,
                        wrong_filepath
                    ]
                )
                generator.parse_args()

            # Try wrong trainer file extension. Should fail ...
            wrong_file_ext = "file.dat"
            with self.assertRaises(SystemExit):
                generator = MyTrainerGenerator(
                    [
                        training_func_file,
                        num_solutions,
                        max_num_iters,
                        wrong_file_ext,
                    ]
                )
                generator.parse_args()

        # Try valid values
        generator = MyTrainerGenerator(
            [
                training_func_file,
                num_solutions,
                max_num_iters,
                trainer_file
            ]
        )
        generator.parse_args()

        # Assess the parameters
        self.assertEqual(
            generator._args.training_func_file,
            training_func_file
        )
        self.assertEqual(generator._args.num_solutions, int(num_solutions))
        self.assertEqual(generator._args.max_num_iters, int(max_num_iters))
        self.assertEqual(generator._args.trainer_file, trainer_file)

    def test_generate(self):
        """Test the generate method."""
        generator = MyTrainerGenerator(
            [
                training_func_file,
                num_solutions,
                max_num_iters,
                trainer_file
            ]
        )

        # Check that the trainer file does not exist yet
        self.assertFalse(Path(trainer_file).exists())

        # Generate the trainer
        generator.generate()

        # Check that the trainer file exists now
        self.assertTrue(Path(trainer_file).exists())


if __name__ == '__main__':
    unittest.main()
