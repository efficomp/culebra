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

"""Tester for individuals correctness and runtime.

Since individuals are generated randomly, it is necessary to perform tests
extensively, with different species parameterizations, in order to check that
both the constructor and the breeding operators generate valid individuals,
that is, individuals meeting its species constraints.

This tester generate different :py:class:`~base.species.Species` from several
values for the number of features and for the proportion of features used to
configure the rest of parameteres of the species (see
:py:meth:`base.species.Species.from_proportion`) and executes the constructor
and the breeding operators many times to assert if all the generated
individuals meet their species parameters.

The mutation operator is also tested with several mutation probabilities.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import unittest
import itertools
import pandas as pd
import time
from culebra.base.species import Species
from culebra.base.individual import Individual
from culebra.fitness.num_feats_fitness import NumFeatsFitness as Fitness

DEFAULT_INDIVIDUAL = Individual
"""Default individual class."""

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
    """Tester for the individuals generator and breeding functions.

    Test extensively the generation and breeding operators of any subclass
    of :py:class:`~base.individual.Individual`
    """

    ind_cls = Individual
    """Individual implementation class that will be tested.  Defaults to
        :py:attr:`~tools.individual_tester.DEFAULT_INDIVIDUAL`
        """

    num_feats_values = DEFAULT_NUM_FEATS_VALUES
    """List of different values for the number of features.

    A :py:class:`~base.species.Species` will be generated combining each one
    of these values for the number of features with each one of the different
    proportions to test the individual implementation (see
    :py:meth:`base.species.Species.from_proportion`). Defaults to
    :py:attr:`~tools.individual_tester.DEFAULT_NUM_FEATS_VALUES`
    """

    prop_values = DEFAULT_PROP_VALUES
    """List of proportions to generate the different
    :py:class:`~base.species.Species`.

    A :py:class:`~base.species.Species` species will be generated combining
    each one of these proportions with each one of the different values for the
    number of features values to test the individual implementation (see
    :py:meth:`base.species.Species.from_proportion`). Defaults to
    :py:attr:`~tools.individual_tester.DEFAULT_PROP_VALUES`
    """

    mut_prob_values = DEFAULT_MUT_PROB_VALUES
    """List of values for the mutation probability.

    Different values for the independent probability for each feature to be
    mutated. Defaults to
    :py:attr:`~tools.individual_tester.DEFAULT_MUT_PROB_VALUES`
    """

    times = DEFAULT_TIMES
    """Times each function is executed. Defaults to
    :py:attr:`~tools.individual_tester.DEFAULT_TIMES`
    """

    def setUp(self):
        """Check that all the parameters are alright.

        :raises TypeError: If any of the parameters is not of the appropriate
            type
        :raises ValueError: If any of the parameters has an incorrect value
        """
        self.__check_ind_cls(self.ind_cls)
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
        """Test the behavior of an individual constructor.

        The constructor is executed under different combinations of values for
        the number of features, minimum feature value, maximum feature value,
        minimum size and maximum size.
        """
        print('Testing the', IndividualTester.ind_cls.__name__,
              'constructor ...', end=' ')

        # Default fitness
        fitness = Fitness()

        # For each value for the number of features ...
        for nf in IndividualTester.num_feats_values:
            # And for each proportion ...
            for p in IndividualTester.prop_values:
                # Create the species from nf and p
                s = Species.from_proportion(nf, prop=p)
                # Execute the generator function the given number of times
                for _ in itertools.repeat(None, IndividualTester.times):
                    # Check that individual meet the species constraints
                    self.__check_correctness(
                        IndividualTester.ind_cls(s, fitness))

        print('Ok')

    def test_1_crossover(self):
        """Test the behavior of the crossover operator.

        The crossover function is executed under different combinations of
        values for the number of features, minimum feature value, maximum
        feature value, minimum size and maximum size.
        """
        print('Testing the', IndividualTester.ind_cls.__name__,
              'crossover ...', end=' ')

        # Default fitness
        fitness = Fitness()

        # For each value for the number of features ...
        for nf in IndividualTester.num_feats_values:
            # And for each proportion ...
            for p in IndividualTester.prop_values:
                # Create the species from nf and p
                s = Species.from_proportion(nf, prop=p)
                # Execute the crossover function the given number of times
                for _ in itertools.repeat(None, IndividualTester.times):
                    # Generate the two parents
                    parent1 = IndividualTester.ind_cls(s, fitness)
                    parent2 = IndividualTester.ind_cls(s, fitness)
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
        print('Testing the', IndividualTester.ind_cls.__name__,
              'mutation ...', end=' ')

        # Default fitness
        fitness = Fitness()

        # For each value for the number of features ...
        for nf in IndividualTester.num_feats_values:
            # And for each proportion ...
            for p in IndividualTester.prop_values:
                # Create the species from nf and p
                s = Species.from_proportion(nf, prop=p)
                # Generate an individual
                ind = IndividualTester.ind_cls(s, fitness)
                # For each possible value of the ind. mutation probability
                for pb in IndividualTester.mut_prob_values:
                    # Execute the mutation function the given number of times
                    for _ in itertools.repeat(None, IndividualTester.times):
                        # Mutate the individual
                        mutant, = ind.mutate(pb)
                        # Check that the mutant meets the species constraints
                        self.__check_correctness(mutant)
        print('Ok')

    def test_3_runtime(self):
        """__runtime of the individual constructor and breeding operators."""
        df = pd.DataFrame()
        df['constructor'] = self.__constructor_scalability()
        df['crossover'] = self.__crossover_scalability()
        df['mutation'] = self.__mutation_scalability()
        df.index = IndividualTester.num_feats_values
        df.index.name = 'num_feats'

        print(df)

    @staticmethod
    def __check_ind_cls(ind_cls):
        """Check if the individual class is correct.

        :param ind_cls: Individual class to be tested.
        :type ind_cls: Any subclass of :py:class:`~base.individual.Individual`
        :raises TypeError: If *ind_cls* is not a subclass of
            :py:class:`~base.individual.Individual`
        """
        if not (isinstance(ind_cls, type) and
                issubclass(ind_cls, Individual)):
            raise TypeError(f'{ind_cls.__name__} is not a valid '
                            'individual class')

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
            except Exception as e:
                raise TypeError(f"The {desc} components must contain "
                                f"{number_type.__name__} values: '{val}'") \
                    from e

    @staticmethod
    def __check_positive_int(value):
        """Check if the given value is a positive integer.

        :param value: An integer value
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not positive
        """
        try:
            v = int(value)
        except Exception as e:
            raise TypeError(f"times must be an integer value: {value}") from e
        else:
            if v <= 0:
                raise ValueError(f"times must be greater than 0: {v}")

    def __check_correctness(self, ind):
        """Check the correctness of an individual implementation.

        Perform some assertions on the individual properties to assure that it
        conforms its species.

        :param ind: Individual to be checked
        :type ind: subclass of :py:class:`~base.individual.Individual`
        """
        # Checks that individual's size is lower than or equal to the maximum
        # allowed size fixed in its species
        self.assertLessEqual(ind.num_feats, ind.species.max_size,
                             f'Individual size: {ind.num_feats} '
                             f'Species max size: {ind.species.max_size}')

        # Checks that individual's size is greater than or equal to the
        # minimum allowed size fixed in its species
        self.assertGreaterEqual(ind.num_feats, ind.species.min_size,
                                f'Individual size: {ind.num_feats} '
                                f'Species min size: {ind.species.min_size}')

        # Only if any feature has been selected ...
        if ind.num_feats > 0:
            # Checks that the minimum feature index selected is greater than
            # or equal to the minimum feature index allowed by its species
            self.assertGreaterEqual(ind.min_feat, ind.species.min_feat,
                                    f'Individual min feat: {ind.min_feat} '
                                    'Species min feat: '
                                    f'{ind.species.min_feat}')

            # Checks that the maximum feature index selected is lower than
            # or equal to the maximum feature index allowed by its species
            self.assertLessEqual(ind.max_feat, ind.species.max_feat,
                                 f'Individual max feat: {ind.max_feat} '
                                 f'Species max feat: {ind.species.max_feat}')

    @staticmethod
    def __runtime(func, times=DEFAULT_TIMES, *args, **kwargs):
        """Estimate the __runtime of a function.

        :param func: The function whose __runtime will be estimated
        :type func: :py:obj:`function`
        :param times: Number of execution times for func, defaults to
            :py:attr:`~tools.individual_tester.DEFAULT_TIMES`
        :type times: :py:class:`int`, optional
        :param args: Unnamed arguments for *func*
        :type args: :py:class:`tuple`
        :param kwargs: Named arguments for func
        :type kwargs: :py:class:`dict`
        :return: Total __runtime of all the executions of the function
        :rtype: :py:class:`float`
        """
        # Measure time just before executing the function
        start_time = time.perf_counter()

        # Execute the function the given number of times
        for _ in itertools.repeat(None, times):
            func(*args, **kwargs)

        # Measure time just after finishing the execution
        end_time = time.perf_counter()

        # Return the total __runtime
        return end_time - start_time

    def __constructor_scalability(self):
        """Scalability for the individual constructor.

        Measure the execution time for the different values of the number of
        features.

        :return: Average execution time of the individual constructor for each
            value of number of features
        :rtype: :py:class:`list`
        """
        print('Calculating the scalability of the',
              IndividualTester.ind_cls.__name__, 'constructor ...', end=' ')

        # Will store the average runtime of the constructor for each different
        # number of features
        runtimes = []

        # Default fitness
        fitness = Fitness()

        # For each different value for the number of features ...
        for nf in IndividualTester.num_feats_values:
            # Initialize the runtime accumulator
            t = 0
            # Accumulate __runtimes for the different proportions ...
            for p in IndividualTester.prop_values:
                # Create the species from nf and p
                s = Species.from_proportion(nf, prop=p)
                # Obtain the average execution time
                t += IndividualTester.__runtime(
                    IndividualTester.ind_cls, IndividualTester.times, s,
                    fitness)
            # Store it
            runtimes.append(t / (IndividualTester.times *
                                 len(IndividualTester.prop_values)))

        print('Ok')
        return runtimes

    def __crossover_scalability(self):
        """Scalability for the individual crossover operator.

        Measure the execution time for the different values of the number of
        features.

        :return: Average execution time of the individual crossover method for
            each value of number of features.
        :rtype: :py:class:`list`
        """
        print('Calculating the scalability of the',
              IndividualTester.ind_cls.__name__,
              'crossover method ...', end=' ')

        # Will store the average runtime of the crossover method for each
        # different number of features
        runtimes = []

        # Default fitness
        fitness = Fitness()

        # For each different value for the number of features ...
        for nf in IndividualTester.num_feats_values:
            # Initialize the runtime accumulator
            t = 0
            # Accumulate runtimes for the different proportions ...
            for p in IndividualTester.prop_values:
                # Create the species from nf and p
                s = Species.from_proportion(nf, prop=p)
                # Generate the two parents
                parent1 = IndividualTester.ind_cls(s, fitness)
                parent2 = IndividualTester.ind_cls(s, fitness)
                # Obtain the average execution time
                t += IndividualTester.__runtime(
                    IndividualTester.ind_cls.crossover, IndividualTester.times,
                    parent1, parent2)
            # Store it
            runtimes.append(t / (IndividualTester.times *
                                 len(IndividualTester.prop_values)))

        print('Ok')
        return runtimes

    def __mutation_scalability(self):
        """Scalability for the individual mutation operator.

        Measure the execution time for the different values of the number of
        features.

        :return: Average execution time of the individual mutation method for
            each value of number of features.
        :rtype: :py:class:`list`
        """
        print('Calculating the scalability of the',
              IndividualTester.ind_cls.__name__,
              'mutation method ...', end=' ')

        # Will store the average runtime of the mutation method for each
        # different number of features
        runtimes = []

        # Default fitness
        fitness = Fitness()

        # For each different value for the number of features ...
        for nf in IndividualTester.num_feats_values:
            t = 0
            # Accumulate runtimes for the different proportions ...
            for p in IndividualTester.prop_values:
                # Create the species from nf and p
                s = Species.from_proportion(nf, prop=p)
                # Generate an individual
                ind = IndividualTester.ind_cls(s, fitness)
                # For each possible value of the mutation probability
                for pb in IndividualTester.mut_prob_values:
                    # Obtain the average execution time
                    t += IndividualTester.__runtime(
                        IndividualTester.ind_cls.mutate,
                        IndividualTester.times,
                        ind, pb)
            # Store it
            runtimes.append(t / (IndividualTester.times *
                                 len(IndividualTester.prop_values) *
                                 len(IndividualTester.mut_prob_values)))

        print('Ok')
        return runtimes
