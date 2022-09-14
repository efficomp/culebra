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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Unit test for the individuals defined in :py:mod:`feature_selector`."""

import unittest
import pickle
import copy

from collections.abc import Sequence
from itertools import repeat
from time import perf_counter, sleep
from pandas import DataFrame
from culebra.base import Species as BaseSpecies
from culebra.fitness_function.feature_selection import NumFeats
from culebra.genotype.feature_selection import Individual, Species
from culebra.genotype.feature_selection.individual import BitVector, IntVector


Fitness = NumFeats.Fitness
"""Default fitness class."""

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


class IndividualTester(unittest.TestCase):
    """Tester for the feature selector generator and breeding functions.

    Test extensively the generation and breeding operators of any subclass
    of :py:class:`~feature_selector.BaseIndividual`
    """

    feature_selector_cls = Individual
    """Feature selector implementation class that will be tested."""

    num_feats_values = DEFAULT_NUM_FEATS_VALUES
    """List of different values for the number of features.

    A :py:class:`~base.Species` will be generated combining each one
    of these values for the number of features with each one of the different
    proportions to test the feature selector implementation (see
    :py:meth:`~feature_selector.Species.from_proportion`)."""

    prop_values = DEFAULT_PROP_VALUES
    """List of proportions to generate the different
    :py:class:`~base.Species`.

    A :py:class:`~base.Species` species will be generated combining
    each one of these proportions with each one of the different values for the
    number of features values to test the featue selector implementation (see
    :py:meth:`~feature_selector.Species.from_proportion`)."""

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
        self.__check_feature_selector_cls(self.feature_selector_cls)
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
              self.feature_selector_cls.__name__,
              'constructor ...', end=' ')

        # Check the type of arguments
        with self.assertRaises(TypeError):
            self.feature_selector_cls(BaseSpecies(), Fitness)
        with self.assertRaises(TypeError):
            self.feature_selector_cls(Species(), Species)

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
                        self.feature_selector_cls(species, Fitness))

        print('Ok')

    def test_1_features(self):
        """Test the features property."""
        print('Testing the',
              self.feature_selector_cls.__name__,
              'features property ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    ind1 = self.feature_selector_cls(species, Fitness)
                    ind2 = self.feature_selector_cls(species, Fitness)
                    ind2.features = ind1.features
                    self.assertTrue((ind1.features == ind2.features).all())
                    self.assertTrue((ind1._features == ind2._features).all())

        print('Ok')

    def test_2_num_feats(self):
        """Test the num_feats property."""
        print('Testing the',
              self.feature_selector_cls.__name__,
              'num_feats property ...', end=' ')

        num_feats = 100
        for size in range(num_feats):
            species = Species(
                num_feats=100, min_size=size, max_size=size)
            for _ in repeat(None, self.times):
                ind = self.feature_selector_cls(species, Fitness)
                self.assertEqual(ind.num_feats, size)
                self.assertEqual(ind.num_feats, ind.features.shape[0])

        print('Ok')

    def test_3_min_feat(self):
        """Test the min_feat property."""
        print('Testing the',
              self.feature_selector_cls.__name__,
              'min_feat property ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    ind = self.feature_selector_cls(species, Fitness)
                    if ind.num_feats > 0:
                        self.assertEqual(ind.min_feat, min(ind.features))
                    else:
                        self.assertEqual(ind.min_feat, None)

        print('Ok')

    def test_4_max_feat(self):
        """Test the max_feat property."""
        print('Testing the',
              self.feature_selector_cls.__name__,
              'max_feat property ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    ind = self.feature_selector_cls(species, Fitness)
                    if ind.num_feats > 0:
                        self.assertEqual(ind.max_feat, max(ind.features))
                    else:
                        self.assertEqual(ind.max_feat, None)

        print('Ok')

    def test5_serialization(self):
        """Serialization test."""
        print('Testing serialization ...', end=' ')
        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    ind1 = self.feature_selector_cls(species, Fitness)

                    data = pickle.dumps(ind1)
                    ind2 = pickle.loads(data)

                    self.assertTrue((ind1.features == ind2.features).all())
                    self.assertTrue((ind1._features == ind2._features).all())
                    self.assertEqual(
                        ind1.species.num_feats, ind2.species.num_feats)
                    self.assertEqual(
                        ind1.species.min_feat, ind2.species.min_feat)
                    self.assertEqual(
                        ind1.species.max_feat, ind2.species.max_feat)
                    self.assertEqual(
                        ind1.species.min_size, ind2.species.min_size)
                    self.assertEqual(
                        ind1.species.max_size, ind2.species.max_size)
        print('Ok')

    def test6_copy(self):
        """Copy test."""
        print('Testing copy and deepcopy ...', end=' ')
        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in repeat(None, self.times):
                    ind1 = self.feature_selector_cls(species, Fitness)

                    ind2 = copy.copy(ind1)

                    self.assertTrue((ind1.features == ind2.features).all())
                    self.assertTrue((ind1._features == ind2._features).all())
                    self.assertEqual(
                        ind1.species.num_feats, ind2.species.num_feats)
                    self.assertEqual(
                        ind1.species.min_feat, ind2.species.min_feat)
                    self.assertEqual(
                        ind1.species.max_feat, ind2.species.max_feat)
                    self.assertEqual(
                        ind1.species.min_size, ind2.species.min_size)
                    self.assertEqual(
                        ind1.species.max_size, ind2.species.max_size)

                    ind3 = copy.deepcopy(ind1)

                    self.assertTrue((ind1.features == ind3.features).all())
                    self.assertTrue((ind1._features == ind3._features).all())
                    self.assertEqual(
                        ind1.species.num_feats, ind3.species.num_feats)
                    self.assertEqual(
                        ind1.species.min_feat, ind3.species.min_feat)
                    self.assertEqual(
                        ind1.species.max_feat, ind3.species.max_feat)
                    self.assertEqual(
                        ind1.species.min_size, ind3.species.min_size)
                    self.assertEqual(
                        ind1.species.max_size, ind3.species.max_size)
        print('Ok')

    def test_7_crossover(self):
        """Test the behavior of the crossover operator.

        The crossover function is executed under different combinations of
        values for the number of features, minimum feature value, maximum
        feature value, minimum size and maximum size.
        """
        print('Testing the',
              self.feature_selector_cls.__name__,
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
                    parent1 = self.feature_selector_cls(species, Fitness)
                    parent2 = self.feature_selector_cls(species, Fitness)
                    # Cross the two parents
                    offspring1, offspring2 = parent1.crossover(parent2)

                    # Check that the offspring meet the species constraints
                    self.__check_correctness(offspring1)
                    self.__check_correctness(offspring2)
        print('Ok')

    def test_8_mutation(self):
        """Test the behavior of the mutation operator.

        The mutation function is executed under different combinations of
        values for the number of features, minimum feature value, maximum
        feature value, minimum size and maximum size.
        """
        print('Testing the',
              self.feature_selector_cls.__name__,
              'mutation ...', end=' ')

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate a feature selector
                feature_selector = self.feature_selector_cls(
                    species, Fitness
                )
                # For each possible value of the independent mutation
                # probability
                for mut_prob in self.mut_prob_values:
                    # Execute the mutation function the given number of times
                    for _ in repeat(
                            None, self.times):
                        # Mutate the feature selector
                        mutant, = feature_selector.mutate(mut_prob)
                        # Check that the mutant meets the species constraints
                        self.__check_correctness(mutant)
        print('Ok')

    def test_9_runtime(self):
        """Runtime of the constructor and breeding operators."""
        dataframe = DataFrame()
        dataframe['constructor'] = self.__constructor_scalability()
        dataframe['crossover'] = self.__crossover_scalability()
        dataframe['mutation'] = self.__mutation_scalability()
        dataframe.index = self.num_feats_values
        dataframe.index.name = 'num_feats'

        print(dataframe)

    @staticmethod
    def __check_feature_selector_cls(feature_selector_cls):
        """Check if the feature selector class is correct.

        :param feature_selector_cls: Feature selector class to be tested.
        :type feature_selector_cls: Any subclass of
            :py:class:`~feature_selector.BaseIndividual`
        :raises TypeError: If *feature_selector_cls* is not a subclass of
            :py:class:`~feature_selector.BaseIndividual`
        """
        if not (isinstance(feature_selector_cls, type) and
                issubclass(feature_selector_cls, Individual)):
            raise TypeError(f'{feature_selector_cls.__name__} is not a valid '
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

    def __check_correctness(self, feature_selector):
        """Check the correctness of a feature selector implementation.

        Perform some assertions on the feature selector properties to assure
        that it conforms its species.

        :param feature_selector: Feature selector to be checked
        :type feature_selector: subclass of
            :py:class:`~feature_selector.BaseIndividual`
        """
        # Checks that feature selector's size is lower than or equal to the
        # maximum allowed size fixed in its species
        self.assertLessEqual(
            feature_selector.num_feats, feature_selector.species.max_size,
            f'Individual size: {feature_selector.num_feats} '
            f'Species max size: {feature_selector.species.max_size}')

        # Checks that individual's size is greater than or equal to the
        # minimum allowed size fixed in its species
        self.assertGreaterEqual(
            feature_selector.num_feats, feature_selector.species.min_size,
            f'Individual size: {feature_selector.num_feats} '
            f'Species min size: {feature_selector.species.min_size}')

        # Only if any feature has been selected ...
        if feature_selector.num_feats > 0:
            # Checks that the minimum feature index selected is greater than
            # or equal to the minimum feature index allowed by its species
            self.assertGreaterEqual(
                feature_selector.min_feat, feature_selector.species.min_feat,
                f'Individual min feat: {feature_selector.min_feat} '
                f'Species min feat: {feature_selector.species.min_feat}')

            # Checks that the maximum feature index selected is lower than
            # or equal to the maximum feature index allowed by its species
            self.assertLessEqual(
                feature_selector.max_feat, feature_selector.species.max_feat,
                f'Individual max feat: {feature_selector.max_feat} '
                f'Species max feat: {feature_selector.species.max_feat}')

    @staticmethod
    def __runtime(func, times=DEFAULT_TIMES, *args, **kwargs):
        """Estimate the __runtime of a function.

        :param func: The function whose __runtime will be estimated
        :type func: :py:obj:`function`
        :param times: Number of execution times for func, defaults to
            :py:attr:`~feature_selector.DEFAULT_TIMES`
        :type times: :py:class:`int`, optional
        :param args: Unnamed arguments for *func*
        :type args: :py:class:`tuple`
        :param kwargs: Named arguments for func
        :type kwargs: :py:class:`dict`
        :return: Total __runtime of all the executions of the function
        :rtype: :py:class:`float`
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
        :rtype: :py:class:`list`
        """
        print('Calculating the scalability of the',
              self.feature_selector_cls.__name__,
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
                    self.feature_selector_cls,
                    self.times, species=species, fitness_cls=Fitness)
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
        :rtype: :py:class:`list`
        """
        print('Calculating the scalability of the',
              self.feature_selector_cls.__name__,
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
                parent1 = self.feature_selector_cls(species, Fitness)
                parent2 = self.feature_selector_cls(species, Fitness)
                # Obtain the average execution time
                runtime += self.__runtime(
                    self.feature_selector_cls.crossover,
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
        :rtype: :py:class:`list`
        """
        print('Calculating the scalability of the',
              self.feature_selector_cls.__name__,
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
                feature_selector = self.feature_selector_cls(
                    species, Fitness
                )
                # For each possible value of the mutation probability
                for mut_prob in self.mut_prob_values:
                    # Obtain the average execution time
                    runtime += self.__runtime(
                        self.feature_selector_cls.mutate,
                        self.times,
                        feature_selector, mut_prob)
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
    IndividualTester.feature_selector_cls = BitVector
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)

    # Wait to begin with the following test
    sleep(1)
    print()

    BitVector.crossover = BitVector.crossover2p
    IndividualTester.feature_selector_cls = BitVector
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)

    # Wait to begin with the following test
    sleep(1)
    print()

    # Run the tests for IntVectors
    IndividualTester.feature_selector_cls = IntVector
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)