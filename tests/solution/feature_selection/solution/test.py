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

"""Unit test for the feature selection solutions."""

import unittest
from os import remove
import copy
from collections.abc import Sequence
from itertools import repeat
from time import sleep

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import Species as BaseSpecies, Fitness
from culebra.solution.feature_selection import (
    Species,
    Solution,
    BinarySolution,
    IntSolution
)

DEFAULT_NUM_FEATS_VALUES = [10, 100, 1000, 10000]
"""Default list of values for the number of features used to define the
Species."""

DEFAULT_PROP_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
"""Default list of values for the proportion of features used to define the
Species."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an implementation is run."""


class MyFitness(Fitness):
    """Dummy fitness."""

    weights = (1.0, 1.0)
    names = ("obj1", "obj2")
    thresholds = [0.001, 0.001]


class SolutionTester(unittest.TestCase):
    """Tester for the feature selector solutions.

    Test extensively the generation of any subclass
    of :py:class:`~culebra.solution.feature_selection.Solution`
    """

    solution_cls = Solution
    """Feature selector implementation class that will be tested."""

    num_feats_values = DEFAULT_NUM_FEATS_VALUES
    """List of different values for the number of features.

    A :py:class:`~culebra.solution.feature_selection..Species` will be
    generated combining each one of these values for the number of features
    with each one of the different proportions to test the feature selector
    implementation (see
    :py:meth:`~culebra.solution.feature_selection.Species.from_proportion`)."""

    prop_values = DEFAULT_PROP_VALUES
    """List of proportions to generate the different
    :py:class:`~culebra.solution.feature_selection.Species`.

    A :py:class:`~culebra.solution.feature_selection..Species` species will be
    generated combining each one of these proportions with each one of the
    different values for the number of features values to test the featue
    selector implementation (see
    :py:meth:`~culebra.solution.feature_selection.Species.from_proportion`)."""

    times = DEFAULT_TIMES
    """Times each function is executed."""

    def setUp(self):
        """Check that all the parameters are alright.

        :raises TypeError: If any of the parameters is not of the appropriate
            type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        self.__check_solution_cls(self.solution_cls)
        self.__check_number_list(
            self.num_feats_values, int, 'num_feats_values',
            'list of number of features')
        self.__check_number_list(
            self.prop_values, float, 'prop_values', 'list of proportions')
        self.__check_positive_int(self.times)

        self.num_feats_values.sort()
        self.prop_values.sort()

    def test_0_constructor(self):
        """Test the behavior of a feature selector constructor.

        The constructor is executed under different combinations of values for
        the number of features, minimum feature value, maximum feature value,
        minimum size and maximum size.
        """
        print('Testing the',
              self.solution_cls.__name__,
              'constructor ...', end=' ')

        # Check the type of arguments
        with self.assertRaises(TypeError):
            self.solution_cls(BaseSpecies(), MyFitness)
        with self.assertRaises(TypeError):
            self.solution_cls(Species(), Species)

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    # Check that the feature selector meets the species
                    # constraints
                    self.__check_correctness(
                        self.solution_cls(species, MyFitness))

        print('Ok')

    def test_1_features(self):
        """Test the features property."""
        print('Testing the',
              self.solution_cls.__name__,
              'features property ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    sol1 = self.solution_cls(species, MyFitness)
                    sol2 = self.solution_cls(species, MyFitness)
                    sol2.features = sol1.features
                    self.assertTrue((sol1.features == sol2.features).all())
                    self.assertTrue((sol1._features == sol2._features).all())

        print('Ok')

    def test_2_num_feats(self):
        """Test the num_feats property."""
        print('Testing the',
              self.solution_cls.__name__,
              'num_feats property ...', end=' ')

        num_feats = 100
        for size in range(num_feats):
            species = Species(
                num_feats=100, min_size=size, max_size=size)
            for _ in repeat(None, self.times):
                sol = self.solution_cls(species, MyFitness)
                self.assertEqual(sol.num_feats, size)
                self.assertEqual(sol.num_feats, sol.features.shape[0])

        print('Ok')

    def test_3_min_feat(self):
        """Test the min_feat property."""
        print('Testing the',
              self.solution_cls.__name__,
              'min_feat property ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    sol = self.solution_cls(species, MyFitness)
                    if sol.num_feats > 0:
                        self.assertEqual(sol.min_feat, min(sol.features))
                    else:
                        self.assertEqual(sol.min_feat, None)

        print('Ok')

    def test_4_max_feat(self):
        """Test the max_feat property."""
        print('Testing the',
              self.solution_cls.__name__,
              'max_feat property ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    sol = self.solution_cls(species, MyFitness)
                    if sol.num_feats > 0:
                        self.assertEqual(sol.max_feat, max(sol.features))
                    else:
                        self.assertEqual(sol.max_feat, None)

        print('Ok')

    def test_5_serialization(self):
        """Serialization test."""
        print('Testing the',
              self.solution_cls.__name__,
              'serialization ...', end=' ')
        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    sol1 = self.solution_cls(species, MyFitness)

                    sol1.dump(serialized_filename)
                    sol2 = self.solution_cls.load(serialized_filename)

                    self.assertTrue((sol1.features == sol2.features).all())
                    self.assertTrue((sol1._features == sol2._features).all())
                    self.assertEqual(
                        sol1.species.num_feats, sol2.species.num_feats)
                    self.assertEqual(
                        sol1.species.min_feat, sol2.species.min_feat)
                    self.assertEqual(
                        sol1.species.max_feat, sol2.species.max_feat)
                    self.assertEqual(
                        sol1.species.min_size, sol2.species.min_size)
                    self.assertEqual(
                        sol1.species.max_size, sol2.species.max_size)

                    # Remove the serialized file
                    remove(serialized_filename)

        print('Ok')

    def test_6_copy(self):
        """Copy test."""
        print('Testing the',
              self.solution_cls.__name__,
              'copy and deepcopy ...', end=' ')
        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    sol1 = self.solution_cls(species, MyFitness)

                    sol2 = copy.copy(sol1)

                    self.assertTrue((sol1.features == sol2.features).all())
                    self.assertTrue((sol1._features == sol2._features).all())
                    self.assertEqual(
                        sol1.species.num_feats, sol2.species.num_feats)
                    self.assertEqual(
                        sol1.species.min_feat, sol2.species.min_feat)
                    self.assertEqual(
                        sol1.species.max_feat, sol2.species.max_feat)
                    self.assertEqual(
                        sol1.species.min_size, sol2.species.min_size)
                    self.assertEqual(
                        sol1.species.max_size, sol2.species.max_size)

                    sol3 = copy.deepcopy(sol1)

                    self.assertTrue((sol1.features == sol3.features).all())
                    self.assertTrue((sol1._features == sol3._features).all())
                    self.assertEqual(
                        sol1.species.num_feats, sol3.species.num_feats)
                    self.assertEqual(
                        sol1.species.min_feat, sol3.species.min_feat)
                    self.assertEqual(
                        sol1.species.max_feat, sol3.species.max_feat)
                    self.assertEqual(
                        sol1.species.min_size, sol3.species.min_size)
                    self.assertEqual(
                        sol1.species.max_size, sol3.species.max_size)
        print('Ok')

    def test_7_repr(self):
        """Test the repr and str dunder methods."""
        print('Testing the',
              self.solution_cls.__name__,
              '__repr__ and __str__ dunder methods ...', end=' ')
        num_feats = 10
        species = Species(num_feats)
        solution = self.solution_cls(species, MyFitness)
        self.assertIsInstance(repr(solution), str)
        self.assertIsInstance(str(solution), str)
        print('Ok')

    @staticmethod
    def __check_solution_cls(solution_cls):
        """Check if the feature selector class is correct.

        :param solution_cls: Feature selector class to be tested.
        :type solution_cls: Any subclass of
            :py:class:`~culebra.solution.feature_selection.Solution`
        :raises TypeError: If *solution_cls* is not a subclass of
            :py:class:`~culebra.solution.feature_selection.Solution`
        """
        if not (isinstance(solution_cls, type) and
                issubclass(solution_cls, Solution)):
            raise TypeError(f'{solution_cls.__name__} is not a valid '
                            'feature selector class')

    @staticmethod
    def __check_number_list(sequence, number_type, name, desc):
        """Check if all the numbers in a sequence are of a given type.

        :param sequence: Sequence of numbers.
        :type sequence: :py:class:`~collections.abc.Sequence`
        :param number_type: The type of number that the sequence should contain
        :type number_type: :py:class:`int` or :py:class:`float`
        :param name: Name of the sequence
        :type name: :py:class:`str`
        :param desc: Description of the sequence
        :type desc: :py:class:`str`
        :raises TypeError: If the given object is not a
            :py:class:`~collections.abc.Sequence` or if any of its components
            is not of *number_type*
        :raises ValueError: If the sequence is empty
        """
        if not isinstance(sequence, Sequence):
            raise TypeError(f"The {desc} must be a sequence: "
                            f"'{name}: {sequence}'")

        if not any(True for _ in sequence):
            raise ValueError(f"The {desc} can not be empty: "
                             f"'{name}: {sequence}'")

        for val in sequence:
            try:
                _ = number_type(val)
            except Exception as excep:
                raise TypeError(f"The {desc} components must contain "
                                f"{number_type.__name__} values: '{val}'") \
                    from excep

    @staticmethod
    def __check_positive_int(value):
        """Check if the given value is a positive integer.

        :param value: An integer value
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not positive
        """
        try:
            val = int(value)
        except Exception as excep:
            raise TypeError(
                f"times must be an integer value: {value}") from excep
        else:
            if val <= 0:
                raise ValueError(f"times must be greater than 0: {val}")

    def __check_correctness(self, solution):
        """Check the correctness of a feature selector implementation.

        Perform some assertions on the feature selector properties to assure
        that it conforms its species.

        :param solution: Feature selector to be checked
        :type solution: subclass of
            :py:class:`~culebra.solution.feature_selection.Solution`
        """
        # Checks that feature selector's size is lower than or equal to the
        # maximum allowed size fixed in its species
        self.assertLessEqual(
            solution.num_feats, solution.species.max_size,
            f'Solution size: {solution.num_feats} '
            f'Species max size: {solution.species.max_size}')

        # Checks that solution's size is greater than or equal to the
        # minimum allowed size fixed in its species
        self.assertGreaterEqual(
            solution.num_feats, solution.species.min_size,
            f'Solution size: {solution.num_feats} '
            f'Species min size: {solution.species.min_size}')

        # Only if any feature has been selected ...
        if solution.num_feats > 0:
            # Checks that the minimum feature index selected is greater than
            # or equal to the minimum feature index allowed by its species
            self.assertGreaterEqual(
                solution.min_feat, solution.species.min_feat,
                f'Solution min feat: {solution.min_feat} '
                f'Species min feat: {solution.species.min_feat}')

            # Checks that the maximum feature index selected is lower than
            # or equal to the maximum feature index allowed by its species
            self.assertLessEqual(
                solution.max_feat, solution.species.max_feat,
                f'Solution max feat: {solution.max_feat} '
                f'Species max feat: {solution.species.max_feat}')


# Tests the classes in this file
if __name__ == '__main__':

    SolutionTester.times = 5

    # Run the tests for BitVectors
    SolutionTester.solution_cls = BinarySolution
    t = unittest.TestLoader().loadTestsFromTestCase(SolutionTester)
    unittest.TextTestRunner(verbosity=0).run(t)

    # Wait to begin with the following test
    sleep(1)
    print()

    # Run the tests for IntVectors
    SolutionTester.solution_cls = IntSolution
    t = unittest.TestLoader().loadTestsFromTestCase(SolutionTester)
    unittest.TextTestRunner(verbosity=0).run(t)
