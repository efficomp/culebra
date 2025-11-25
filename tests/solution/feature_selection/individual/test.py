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

"""Unit test for the feature selection individuals."""

import unittest
from collections.abc import Sequence
from itertools import repeat
from os import remove
from time import perf_counter, sleep

import numpy as np
from pandas import DataFrame

from culebra import SERIALIZED_FILE_EXTENSION
from culebra.abc import Species as BaseSpecies
from culebra.fitness_function.feature_selection import NumFeats
from culebra.solution.abc import Individual
from culebra.solution.feature_selection import (
    Solution,
    Species,
    BitVector,
    IntVector
)


DEFAULT_NUM_FEATS_VALUES = [10, 100, 1000, 10000]
"""Default list of values for the number of features used to define the
Species."""

DEFAULT_PROP_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
"""Default list of values for the proportion of features used to define the
Species."""

DEFAULT_MUT_PROB_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
"""Default list of values for the independent mutation probability."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an implementation is run."""

fitness_function = NumFeats()


class IndividualTester(unittest.TestCase):
    """Tester for the feature selector individuals.

    Test extensively the generation and breeding operators for any subclass
    of :class:`~culebra.solution.feature_selection.Solution` and
    :class:`~culebra.solution.abc.Individual`
    """

    individual_cls = Solution
    """Feature selector implementation class that will be tested."""

    num_feats_values = DEFAULT_NUM_FEATS_VALUES
    """List of different values for the number of features.

    A :class:`~culebra.solution.feature_selection.Species` will be generated
    combining each one of these values for the number of features with each
    one of the different proportions to test the feature selector
    implementation (see
    :meth:`~culebra.solution.feature_selection.Species.from_proportion`)."""

    prop_values = DEFAULT_PROP_VALUES
    """List of proportions to generate the different
    :class:`~culebra.solution.feature_selection.Species`.

    A :class:`~culebra.solution.feature_selection.Species` species will be
    generated combining each one of these proportions with each one of the
    different values for the number of features values to test the featue
    selector implementation (see
    :meth:`~culebra.solution.feature_selection.Species.from_proportion`)."""

    mut_prob_values = DEFAULT_MUT_PROB_VALUES
    """List of values for the mutation probability.

    Different values for the independent probability for each feature to be
    mutated."""

    times = DEFAULT_TIMES
    """Times each function is executed."""

    def setUp(self):
        """Check that all the parameters are alright.

        :raises TypeError: If any of the parameters is not of the appropriate
            type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        self.__check_individual_cls(self.individual_cls)
        self.__check_number_list(
            self.num_feats_values, int, 'num_feats_values',
            'list of number of features')
        self.__check_number_list(
            self.prop_values, float, 'prop_values', 'list of proportions')
        self.__check_number_list(
            self.mut_prob_values, float, 'mut_prob_values',
            'list of independent mutation probabilities')
        self.__check_positive_int(self.times)

        self.num_feats_values.sort()
        self.prop_values.sort()
        self.mut_prob_values.sort()

    def test_0_constructor(self):
        """Test the behavior of a feature selector constructor.

        The constructor is executed under different combinations of values for
        the number of features, minimum feature value, maximum feature value,
        minimum size and maximum size.
        """
        print('Testing the',
              self.individual_cls.__name__,
              'constructor ...', end=' ')

        # Check the type of arguments
        with self.assertRaises(TypeError):
            self.individual_cls(BaseSpecies(), fitness_function.fitness_cls)
        with self.assertRaises(TypeError):
            self.individual_cls(Species(), Species)

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
                        self.individual_cls(
                            species,
                            fitness_function.fitness_cls
                        )
                    )

        print('Ok')

    def test_1_crossover(self):
        """Test the behavior of the crossover operator.

        The crossover function is executed under different combinations of
        values for the number of features, minimum feature value, maximum
        feature value, minimum size and maximum size.
        """
        print('Testing the',
              self.individual_cls.__name__,
              'crossover ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the crossover function the given number of times
                for _ in repeat(None, self.times):
                    # Generate the two parents
                    parent1 = self.individual_cls(
                        species,
                        fitness_function.fitness_cls
                    )
                    parent2 = self.individual_cls(
                        species,
                        fitness_function.fitness_cls
                    )

                    # Cross the two parents
                    offspring1, offspring2 = parent1.crossover(parent2)

                    # Check that the offspring meet the species constraints
                    self.__check_correctness(offspring1)
                    self.__check_correctness(offspring2)
        print('Ok')

    def test_2_mutation(self):
        """Test the behavior of the mutation operator.

        The mutation function is executed under different combinations of
        values for the number of features, minimum feature value, maximum
        feature value, minimum size and maximum size.
        """
        print('Testing the',
              self.individual_cls.__name__,
              'mutation ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate a feature selector
                individual = self.individual_cls(
                    species, fitness_function.fitness_cls
                )

                # For each possible value of the independent mutation
                # probability
                for mut_prob in self.mut_prob_values:
                    # Execute the mutation function the given number of times
                    for _ in repeat(
                            None, self.times):
                        # Mutate the feature selector
                        mutant, = individual.mutate(mut_prob)
                        # Check that the mutant meets the species constraints
                        self.__check_correctness(mutant)
        print('Ok')

    def test_3_repr(self):
        """Test the repr and str dunder methods."""
        print('Testing the',
              self.individual_cls.__name__,
              '__repr__ and __str__ dunder methods ...', end=' ')
        num_feats = 10
        species = Species(num_feats)
        individual = self.individual_cls(
            species,
            fitness_function.fitness_cls
        )
        self.assertIsInstance(repr(individual), str)
        self.assertIsInstance(str(individual), str)
        print('Ok')

    def test_4_serialization(self):
        """Serialization of individuals."""
        print('Testing the',
              self.individual_cls.__name__,
              'dump and load methods ...', end=' ')
        num_feats = 10
        species = Species(num_feats)
        ind1 = self.individual_cls(species, fitness_function.fitness_cls)
        serialized_filename = "my_file" + SERIALIZED_FILE_EXTENSION
        ind1.dump(serialized_filename)
        ind2 = self.individual_cls.load(serialized_filename)

        # Check the serialization
        self._check_deepcopy(ind1, ind2)

        # Remove the serialized file
        remove(serialized_filename)
        print('Ok')

    def test_5_runtime(self):
        """Runtime of the constructor and breeding operators."""
        dataframe = DataFrame()
        dataframe['constructor'] = self.__constructor_scalability()
        dataframe['crossover'] = self.__crossover_scalability()
        dataframe['mutation'] = self.__mutation_scalability()
        dataframe.index = self.num_feats_values
        dataframe.index.name = 'num_feats'

        print(dataframe)

    def _check_deepcopy(self, ind1, ind2):
        """Check if *ind1* is a deepcopy of *ind2*.

        :param ind1: The first individual
        :type ind1: Any subclass of
            :class:`~culebra.solution.feature_selection.Solution` and
            :class:`~culebra.solution.abc.Individual`
        :param ind2: The second individual
        :type ind2: Any subclass of
            :class:`~culebra.solution.feature_selection.Solution` and
            :class:`~culebra.solution.abc.Individual`
        """
        # Copies all the levels
        self.assertNotEqual(id(ind1), id(ind2))
        self.assertNotEqual(id(ind1._features), id(ind2._features))
        self.assertTrue(np.all(ind1.features == ind2.features))

    @staticmethod
    def __check_individual_cls(individual_cls):
        """Check if the feature selector class is correct.

        :param individual_cls: Feature selector class to be tested.
        :type individual_cls:
            type[~culebra.solution.feature_selection.Solution] &
            type[~culebra.solution.abc.Individual]
        :raises TypeError: If *individual_cls* is not a subclass of
            :class:`~culebra.solution.feature_selection.Solution` and
            :class:`~culebra.solution.abc.Individual`
        """
        if not (
            isinstance(individual_cls, type) and
            issubclass(individual_cls, Solution) and
            issubclass(individual_cls, Individual)
        ):
            raise TypeError(f'{individual_cls.__name__} is not a valid '
                            'feature selector class')

    @staticmethod
    def __check_number_list(sequence, number_type, name, desc):
        """Check if all the numbers in a sequence are of a given type.

        :param sequence: Sequence of numbers.
        :type sequence: ~collections.abc.Sequence
        :param number_type: The type of number that the sequence should contain
        :type number_type: int | float
        :param name: Name of the sequence
        :type name: str
        :param desc: Description of the sequence
        :type desc: str
        :raises TypeError: If the given object is not a
            :class:`~collections.abc.Sequence` or if any of its components
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
        :type value: int
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

    def __check_correctness(self, individual):
        """Check the correctness of a feature selector implementation.

        Perform some assertions on the feature selector properties to assure
        that it conforms its species.

        :param individual: Feature selector to be checked
        :type individual: subclass of
            :class:`~culebra.solution.feature_selection.Solution` and
            :class:`~culebra.solution.abc.Individual`
        """
        # Checks that feature selector's size is lower than or equal to the
        # maximum allowed size fixed in its species
        self.assertLessEqual(
            individual.num_feats, individual.species.max_size,
            f'Individual size: {individual.num_feats} '
            f'Species max size: {individual.species.max_size}')

        # Checks that individual's size is greater than or equal to the
        # minimum allowed size fixed in its species
        self.assertGreaterEqual(
            individual.num_feats, individual.species.min_size,
            f'Individual size: {individual.num_feats} '
            f'Species min size: {individual.species.min_size}')

        # Only if any feature has been selected ...
        if individual.num_feats > 0:
            # Checks that the minimum feature index selected is greater than
            # or equal to the minimum feature index allowed by its species
            self.assertGreaterEqual(
                individual.min_feat, individual.species.min_feat,
                f'Individual min feat: {individual.min_feat} '
                f'Species min feat: {individual.species.min_feat}')

            # Checks that the maximum feature index selected is lower than
            # or equal to the maximum feature index allowed by its species
            self.assertLessEqual(
                individual.max_feat, individual.species.max_feat,
                f'Individual max feat: {individual.max_feat} '
                f'Species max feat: {individual.species.max_feat}')

    @staticmethod
    def __runtime(func, times=DEFAULT_TIMES, *args, **kwargs):
        """Estimate the __runtime of a function.

        :param func: The function whose __runtime will be estimated
        :type func: ~collections.abc.Callable
        :param times: Number of execution times for func, defaults to
            :attr:`DEFAULT_TIMES`
        :type times: int
        :param args: Unnamed arguments for *func*
        :type args: tuple
        :param kwargs: Named arguments for func
        :type kwargs: dict
        :return: Total __runtime of all the executions of the function
        :rtype: float
        """
        # Measure time just before executing the function
        start_time = perf_counter()

        # Execute the function the given number of times
        for _ in repeat(None, times):
            func(*args, **kwargs)

        # Measure time just after finishing the execution
        end_time = perf_counter()

        # Return the total __runtime
        return end_time - start_time

    def __constructor_scalability(self):
        """Scalability for the feature selector constructor.

        Measure the execution time for the different values of the number of
        features.

        :return: Average execution time of the feature selector constructor for
            each value of number of features
        :rtype: list
        """
        print('Calculating the scalability of the',
              self.individual_cls.__name__,
              'constructor ...', end=' ')

        # Will store the average runtime of the constructor for each different
        # number of features
        runtimes = []

        # For each different value for the number of features ...
        for num_feats in self.num_feats_values:
            # Initialize the runtime accumulator
            runtime = 0
            # Accumulate __runtimes for the different proportions ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Obtain the average execution time
                runtime += self.__runtime(
                    self.individual_cls,
                    self.times,
                    species=species,
                    fitness_cls=fitness_function.fitness_cls
                )
            # Store it
            runtimes.append(runtime / (self.times * len(self.prop_values)))

        print('Ok')
        return runtimes

    def __crossover_scalability(self):
        """Scalability for the feature selector crossover operator.

        Measure the execution time for the different values of the number of
        features.

        :return: Average execution time of the feature selector crossover
            method for each value of number of features.
        :rtype: list
        """
        print('Calculating the scalability of the',
              self.individual_cls.__name__,
              'crossover method ...', end=' ')

        # Will store the average runtime of the crossover method for each
        # different number of features
        runtimes = []

        # For each different value for the number of features ...
        for num_feats in self.num_feats_values:
            # Initialize the runtime accumulator
            runtime = 0
            # Accumulate runtimes for the different proportions ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate the two parents
                parent1 = self.individual_cls(
                    species,
                    fitness_function.fitness_cls
                )
                parent2 = self.individual_cls(
                    species,
                    fitness_function.fitness_cls
                )
                # Obtain the average execution time
                runtime += self.__runtime(
                    self.individual_cls.crossover,
                    self.times, parent1, parent2)
            # Store it
            runtimes.append(runtime / (self.times * len(self.prop_values)))

        print('Ok')
        return runtimes

    def __mutation_scalability(self):
        """Scalability for the feature selector mutation operator.

        Measure the execution time for the different values of the number of
        features.

        :return: Average execution time of the feature selector mutation method
            for each value of number of features.
        :rtype: list
        """
        print('Calculating the scalability of the',
              self.individual_cls.__name__,
              'mutation method ...', end=' ')

        # Will store the average runtime of the mutation method for each
        # different number of features
        runtimes = []

        # For each different value for the number of features ...
        for num_feats in self.num_feats_values:
            runtime = 0
            # Accumulate runtimes for the different proportions ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate a feature selector
                individual = self.individual_cls(
                    species, fitness_function.fitness_cls
                )
                # For each possible value of the mutation probability
                for mut_prob in self.mut_prob_values:
                    # Obtain the average execution time
                    runtime += self.__runtime(
                        self.individual_cls.mutate,
                        self.times,
                        individual, mut_prob)
            # Store it
            runtimes.append(runtime / (self.times * len(self.prop_values) *
                                       len(self.mut_prob_values)))

        print('Ok')
        return runtimes


# Tests the classes in this file
if __name__ == '__main__':

    # Number of times each function is executed
    IndividualTester.times = 5

    # Run the tests for BitVectors
    BitVector.crossover = BitVector.crossover1p
    IndividualTester.individual_cls = BitVector
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)

    # Wait to begin with the following test
    sleep(1)
    print()

    BitVector.crossover = BitVector.crossover2p
    IndividualTester.individual_cls = BitVector
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)

    # Wait to begin with the following test
    sleep(1)
    print()

    # Run the tests for IntVectors
    IndividualTester.individual_cls = IntVector
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)
