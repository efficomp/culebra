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

"""Implementation of feature selector individuals.

This module provides two :py:class:`numpy.ndarray`-based implementations of
feature selector individuals:

    * :py:class:`~feature_selector.BitVector`:
        Individuals are boolean vectors of a fixed length, the number of
        selectable features. One boolean value is associated to each selectable
        feature. If the the boolean value is true then the feature is selected,
        otherwhise the feature is ignored.
    * :py:class:`~feature_selector.IntVector`:
        Individuals are vector of feature indices. Only the features whose
        indices are present in the individual are selected. Thus, individuals
        may have different lengths and can not contain repeated indices.

Both implementations inherit from :py:class:`~feature_selector.Individual`,
which defines the interface for feature selectors and must meet the constraints
imposed by the :py:class:`~feature_selector.Species` class.

Since individuals are generated and modified randomly, it is necessary a
:py:class:`~feature_selector.Tester` class to test extensively the random
generation, crossover and mutation of each individual implementation.
"""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import copy
import numbers
import random
import unittest
import itertools
import time
import numpy as np
import pandas as pd
from culebra.base import Species as BaseSpecies
from culebra.base import Individual as CulebraIndividual
from culebra.fitness import NumFeatsFitness


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_PROP = 0.15
"""Default proportion for the generation of a species."""

MAX_PROP = 0.25
"""Maximum proportion for the generation of a species."""


class Species(BaseSpecies):
    """Species for the feature selector individuals."""

    def __init__(self, num_feats, min_feat=0, max_feat=-1, min_size=0,
                 max_size=-1):
        """Create a new species.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param min_feat: Smallest feature index considered in this species.
            Must be in the interval [0, *num_feats*). Defaults to 0
        :type min_feat: :py:class:`int`, optional
        :param max_feat: Largest feature index considered in this species.
            Must be in the interval [*min_feat*, *num_feats*). Negative values
            are interpreted as the maximum possible feature index
            (*num_feats* - 1). Defaults to -1
        :type max_feat: :py:class:`int`, optional
        :param min_size: Minimum size of individuals (minimum number of
            features selected by individuals in the species). Must be in the
            interval [0, *max_feat - min_feat + 1*]. Defaults to 0
        :type min_size: :py:class:`int`, optional
        :param max_size: Maximum size of individuals. Must be in the interval
            [*min_size*, *max_feat - min_feat + 1*]. Negative values are
            interpreted as the maximum possible size. Defaults to -1
        :type max_size: :py:class:`int`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        self.__num_feats = self.__check_num_feats(num_feats)
        self.__min_feat = self.__check_min_feat(min_feat)
        self.__max_feat = self.__check_max_feat(max_feat)
        self.__min_size = self.__check_min_size(min_size)
        self.__max_size = self.__check_max_size(max_size)

    @classmethod
    def from_proportion(cls, num_feats, prop=DEFAULT_PROP):
        """Create a parametrized species for testing purposes.

        Fix *min_feat*, *max_feat*, *min_size* and *max_size* proportionally
        to the number of features, according to *prop*, in this way:

            - *min_feat* = *num_feats* * *prop*
            - *max_feat* = *num_feats* - *min_feat* - 1
            - *min_size* = *min_feat*
            - *max_size* = *max_feat* - (2 * *min_feat*) + 1

        Here are some examples for *num_feats* = 1000

        ======  ==========  ==========  ==========  ==========
        *prop*  *min_feat*  *max_feat*  *min_size*  *max_size*
        ======  ==========  ==========  ==========  ==========
          0.00           0         999           0        1000
          0.05          50         949          50         850
          0.10         100         899         100         700
          0.15         150         849         150         550
          0.20         200         799         200         400
          0.25         250         749         250         250
        ======  ==========  ==========  ==========  ==========

        The maximum value for *prop* is
        :py:attr:`~feature_selector.MAX_PROP`.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param prop: Proportion of the number of features used to fix the
            species parameters. Defaults to
            :py:attr:`~feature_selector.DEFAULT_PROP`. The maximum
            allowed value is :py:attr:`~feature_selector.MAX_PROP`.
        :type prop: :py:class:`float`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        :return: A Species object
        :rtype: :py:class:`~base.Species`
        """
        numf = cls.__check_num_feats(num_feats)

        # Checks prop
        prop = cls.__check_prop(prop)

        # Parametrize the species from num_feats and prop
        minf = int(numf * prop)
        maxf = numf - minf - 1
        mins = minf
        maxs = maxf - (2 * minf) + 1

        return cls(numf, min_feat=minf, max_feat=maxf, min_size=mins,
                   max_size=maxs)

    @property
    def num_feats(self):
        """Get the number of features for this species.

        :type: :py:class:`int`
        """
        return self.__num_feats

    @property
    def min_feat(self):
        """Get the minimum feature index for this species..

        :type: :py:class:`int`
        """
        return self.__min_feat

    @property
    def max_feat(self):
        """Get the maximum feature index for this species.

        :type: :py:class:`int`
        """
        return self.__max_feat

    @property
    def min_size(self):
        """Get the minimum subset size for this species.

        :type: :py:class:`int`
        """
        return self.__min_size

    @property
    def max_size(self):
        """Get the maximum subset size for this species.

        :type: :py:class:`int`
        """
        return self.__max_size

    def check(self, ind):
        """Check if an individual meets the constraints imposed by the species.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.Individual`
        :return: `True` if the individual belong to the species. `False`
            otherwise
        :rtype: :py:class:`tuple`
        """
        # Check the number of features
        num_feats = ind.num_feats
        if num_feats < self.min_size or num_feats > self.max_size:
            return False

        # Check the minimum and maximum index
        if ind.min_feat < self.min_feat or ind.max_feat > self.max_feat:
            return False

        return True

    @property
    def __size_limit(self):
        """Get the Limit for the individuals size for this species.

        :type: :py:class:`int`
        """
        return self.__max_feat - self.__min_feat + 1

    @staticmethod
    def __check_num_feats(num_feats):
        """Check if the type or value of num_feats is valid.

        :param num_feats: Proposed value for num_feats
        :type num_feats: :py:class:`int`
        :raise TypeError: If num_feats is not an integer
        :raise ValueError: If num_feats is lower than or equal to 0
        :return: A valid value for the number of features
        :rtype: :py:class:`int`
        """
        if not isinstance(num_feats, numbers.Integral):
            raise TypeError("The number of features should be an integer "
                            f"number: 'num_feats = {num_feats}'")
        if num_feats <= 0:
            raise ValueError("The number of features should be an integer "
                             f"greater than 0: 'num_feats = {num_feats}'")
        return num_feats

    def __check_min_feat(self, min_feat):
        """Check if the type or value of min_feat is valid.

        :param min_feat: Proposed value for min_feat
        :type min_feat: :py:class:`int`
        :raise TypeError: If min_feat is not an integer
        :raise ValueError: If min_feat is lower than 0 or if it is greater than
            or equal to the number of features
        :return: A valid value for the minimum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(min_feat, numbers.Integral):
            raise TypeError("The minimum feature index should be an integer "
                            f"number: 'min_feat = {min_feat}'")
        if min_feat < 0:
            raise ValueError("The minimum feature index should be greater "
                             f"than or equal to 0: 'min_feat = {min_feat}'")
        if min_feat >= self.__num_feats:
            raise ValueError("The minimum feature index should be lower than "
                             "the number of features: 'num_feats = "
                             f"{self.__num_feats}, min_feat = {min_feat}'")
        return min_feat

    def __check_max_feat(self, max_feat):
        """Check if the type or value of max_feat is valid.

        :param max_feat: Proposed value for max_feat
        :type max_feat: :py:class:`int`
        :raise TypeError: If max_feat is not an integer
        :raise ValueError: If max_feat is lower than checked_min_feat or if it
            is greater than or equal to the number of features.
        :return: A valid value for the maximum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(max_feat, numbers.Integral):
            raise TypeError("The maximum feature index should be an integer "
                            f"number: 'max_feat = {max_feat}'")

        if max_feat < 0:
            max_feat = self.__num_feats - 1

        if max_feat < self.__min_feat:
            raise ValueError("The maximum feature index should be greater "
                             "than or equal to min_feat: 'min_feat = "
                             f"{self.__min_feat}, max_feat = {max_feat}'")
        if max_feat >= self.__num_feats:
            raise ValueError("The maximum feature index should be lower than "
                             "the number of features: 'num_feats = "
                             f"{self.__num_feats}, max_feat = {max_feat}'")
        return max_feat

    def __check_min_size(self, min_size):
        """Check if the type or value of min_size is valid.

        :param min_size: Proposed value for min_size
        :type min_size: :py:class:`int`
        :raise TypeError: If min_size is not an integer
        :raise ValueError: If min_size is lower than 0 or if it is greater than
            checked_max_feat - checked_min_feat + 1.
        :return: A valid value for the minimum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(min_size, numbers.Integral):
            raise TypeError("The minimum subset size should be an integer "
                            f"number: 'min_size = {min_size}'")

        if min_size < 0:
            raise ValueError("The minimum subset size should be greater than "
                             f"or equal to 0: 'min_size = {min_size}'")

        if min_size > self.__size_limit:
            raise ValueError("The minimum subset size should be lower than or "
                             "equal to (max_feat - min_feat + 1): 'min_feat = "
                             f"{self.__min_feat}, max_feat = "
                             f"{self.__max_feat}, min_size = {min_size}'")

        return min_size

    def __check_max_size(self, max_size):
        """Check if the type or value of max_size is valid.

        :param max_size: Proposed value for max_size
        :type max_size: :py:class:`int`
        :raise TypeError: If max_size is not an integer
        :raise ValueError: If max_size is lower than checked_min_size or if it
            is greater than checked_max_feat - checked_min_feat + 1.
        :return: A valid value for the maximum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(max_size, numbers.Integral):
            raise TypeError("The maximum subset size should be an integer "
                            f"number: 'max_size = {max_size}'")

        if max_size < 0:
            max_size = self.__size_limit

        if max_size < self.__min_size:
            raise ValueError("The maximum subset size should be greater than "
                             "or equal to the minimum size: 'min_size = "
                             f"{self.__min_size}, max_size = {max_size}'")

        if max_size > self.__size_limit:
            raise ValueError("The maximum subset size should be lower than or "
                             "equal to (max_feat - min_feat + 1): 'min_feat = "
                             f"{self.__min_feat}, max_feat = "
                             f"{self.__max_feat}, max_size = {max_size}'")

        return max_size

    @staticmethod
    def __check_prop(prop):
        """Check if the type or value of prop is valid.

        :param prop: Proportion of the number of features used to fix the
            attributes of a species
        :type prop: :py:class:`float`
        :raise TypeError: If prop is not a valid real number
        :raise ValueError: If prop is not in the interval
            [0, *MAX_PROP*]
        :return: A vaild value for the proportion of the number of features
        :rtype: :py:class:`float`
        """
        if not isinstance(prop, numbers.Real):
            raise TypeError("The proportion must be a real value: 'prop = "
                            f"{prop}'")

        if not 0 <= prop <= MAX_PROP:
            raise ValueError("The proportion must be in the interval "
                             f"[0, {MAX_PROP}]: "
                             f"'prop = {prop}'")

        return prop

    def __reduce__(self):
        """Reduce the species.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.num_feats, ), self.__dict__)

    def __copy__(self):
        """Shallow copy the species."""
        cls = self.__class__
        result = cls(self.num_feats)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the species
        :rtype: :py:class:`~feature_selector.Species`
        """
        cls = self.__class__
        result = cls(self.num_feats)
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result


class Individual(CulebraIndividual):
    """Base class for all the feature selector individuals."""

    @property
    def features(self):
        """Get and set the indices of the features selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :getter: Return the indices of the selected features
        :setter: Set the new feature indices. An array-like object of
            feature indices is expected

        :raises NotImplementedError: if has not been overriden
        :type: Array-like object
        """
        raise NotImplementedError("The features property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @features.setter
    def features(self, values):
        """Set the indices of the new features selected by the individual.

        This property setter must be overriden by subclasses.

        :param values: The new feature indices
        :type values: Array-like object
        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError("The features property seter has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    def num_feats(self):
        """Get the nmber of features selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The num_feats property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    def min_feat(self):
        """Minimum feature index selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The min_feat property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    def max_feat(self):
        """Maximum feature index selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The max_feat property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def __str__(self):
        """Return the individual as a string."""
        return self.features.__str__()

    def __repr__(self):
        """Return the individual representation."""
        cls_name = self.__class__.__name__
        species_info = self.species.__str__()
        fitness_info = self.fitness.values

        return (f"{cls_name}(species={species_info}, fitness={fitness_info}, "
                f"features={self.__str__()})")


class BitVector(Individual):
    """Feature selector implementation based on boolean arrays."""

    def __init__(self, species, fitness, features=None):
        """Create a feature selector individual.

        :param species: The species the individual will belong to
        :type species: :py:class:`~base.Species`
        :param fitness: The fitness object for the individual
        :type fitness: Instance of any subclass of
            :py:class:`~base.Fitness`
        :param features: Features to init the individual. If `None` the
            individual is initialized randomly.
        :type features: Any subclass of :py:class:`~collections.abc.Sequence`
            containing valid feature indices
        :raises TypeError: If any parameter type is wrong
        """
        # Initialize the individual
        super().__init__(species, fitness)

        # If initial features are provided, use them
        if features is not None:
            self.features = features
        # Else the features are randomly generated
        else:

            # Select the features randomly
            num_feats = species.max_feat - species.min_feat + 1
            self._features = np.random.random(num_feats) < 0.5

            # Number of selected features
            num_feats = np.count_nonzero(self._features)

            # Repair too small BitVectors
            if num_feats < species.min_size:
                # Indices of the unselected features
                unselected = np.where(~self._features)[0]

                # Random choice of the minimal set of features to be selected
                np.random.shuffle(unselected)
                self._features[
                    unselected[:species.min_size - num_feats]] = True

                # The number of features is now the minimim allowed size
                num_feats = species.min_size

            # Repair too large BitVectors
            if num_feats > species.max_size:
                # Indices of the selected features
                selected = np.where(self._features)[0]

                # Random choice of the minimal set of features to be deselected
                np.random.shuffle(selected)
                self._features[selected[:num_feats - species.max_size]] = False

    @property
    def features(self):
        """Get and set the indices of the features selected by the individual.

        :getter: Return the indices of the selected features
        :setter: Set the new feature indices. An array-like object of
            feature indices is expected
        :type: :py:class:`numpy.ndarray`
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        return self._features.nonzero()[0] + self.species.min_feat

    @features.setter
    def features(self, values):
        """Set the indices of the new features selected by the individual.

        :param values: The new feature indices
        :type values: Array-like object
        :raises ValueError: If the values do not meet the species constraints.
        """
        # Get the set of indices
        indices = np.unique(
                    np.asarray(values, dtype=int)
                            ) - self.species.min_feat

        num_feats = self.species.max_feat - self.species.min_feat + 1
        self._features = np.zeros(num_feats, dtype=bool)
        self._features[indices] = True

        if not self.species.check(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

    @property
    def num_feats(self):
        """Get the number of features selected by the individual.

        :type: :py:class:`int`
        """
        return np.count_nonzero(self._features)

    @property
    def min_feat(self):
        """Minimum feature index selected by the individual.

        :type: :py:class:`int`
        """
        return self._features.nonzero()[0][0] + self.species.min_feat

    @property
    def max_feat(self):
        """Maximum feature index selected by the individual.

        :type: :py:class:`int`
        """
        return self._features.nonzero()[0][-1] + self.species.min_feat

    def crossover1p(self, other):
        """Cross this individual with another one.

        This method implements the single-point crossover.

        :param other: The other individual
        :type other: :py:class:`~feature_selector.BitVector`
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        while True:
            # Cross point. Use randint to include also max_feat
            min_feat = self.species.min_feat
            max_feat = self.species.max_feat
            cross_point = random.randint(0, max_feat - min_feat)

            # Try a crossing of the the individuals
            (self._features[:cross_point],
             other._features[:cross_point]) = \
                (other._features[:cross_point].copy(),
                 self._features[:cross_point].copy())

            # Check if the number of features meets the species constraints
            if (self.species.min_size <= self.num_feats <=
                    self.species.max_size) and \
               (self.species.min_size <= other.num_feats <=
                    self.species.max_size):
                break

        # Return the offspring
        return self, other

    def crossover2p(self, other):
        """Cross this individual with another one.

        This method implements the two-points crossover.

        :param other: The other individual
        :type other: :py:class:`~feature_selector.BitVector`
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        while True:
            # Cross points. Use randint to include also max_feat
            min_feat = self.species.min_feat
            max_feat = self.species.max_feat
            cross_point_0 = random.randint(0, max_feat - min_feat)
            cross_point_1 = random.randint(0, max_feat - min_feat)
            if cross_point_0 > cross_point_1:
                cross_point_0, cross_point_1 = cross_point_1, cross_point_0

            # Cross the individuals
            (self._features[cross_point_0:cross_point_1],
             other._features[cross_point_0:cross_point_1]) = \
                (other._features[cross_point_0:cross_point_1].copy(),
                 self._features[cross_point_0:cross_point_1].copy())

            # Check if the number of features meets the species constraints
            if (self.species.min_size <= self.num_feats <=
                    self.species.max_size) and \
               (self.species.min_size <= other.num_feats <=
                    self.species.max_size):
                break

        # Return the offspring
        return self, other

    def mutate(self, indpb):
        """Mutate the individual.

        Each feature is independently mutated according to the given
        probability.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: :py:class:`float`
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        while True:
            # Mask to select the features that will be flipped
            mask = np.random.random(self._features.size) < indpb

            self._features[mask] = ~self._features[mask]
            # Check if the number of features meets the species constraints
            if (self.species.min_size <= self.num_feats <=
                    self.species.max_size):
                break

        # Return the mutant
        return (self,)

    crossover = crossover1p
    """The default crossover operator is the single-point crossover."""


class IntVector(Individual):
    """Feature selector implementation based on arrays of indices."""

    def __init__(self, species, fitness, features=None):
        """Create a feature selector individual.

        :param species: The species the individual will belong to
        :type species: :py:class:`~base.Species`
        :param fitness: The fitness object for the individual
        :type fitness: Instance of any subclass of
            :py:class:`~base.Fitness`
        :param features: Features to init the individual. If `None` the
            individual is initialized randomly.
        :type features: Any subclass of :py:class:`~collections.abc.Sequence`
            containing valid feature indices
        :raises TypeError: If any parameter type is wrong
        """
        # Initialize the individual
        super().__init__(species, fitness)

        # If initial features are provided, use them
        if features is not None:
            self.features = features
        # Else the features are randomly generated
        else:
            # All possible indices for the species
            indices = np.arange(self.species.min_feat,
                                self.species.max_feat + 1)

            # Random size
            size = random.randint(self.species.min_size,
                                  self.species.max_size)

            # Select the features of the new individual
            np.random.shuffle(indices)
            self._features = indices[:size]

    @property
    def features(self):
        """Get ans set the indices of the features selected by the individual.

        :getter: Return the indices of the selected features
        :setter: Set the new feature indices. An array-like object of
            feature indices is expected
        :type: :py:class:`numpy.ndarray`
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        # Sort the array
        self._features.sort()
        return self._features

    @features.setter
    def features(self, values):
        """Set the indices of the new features selected by the individual.

        :param values: The new feature indices
        :type values: Array-like object
        :raises ValueError: If the values do not meet the species constraints.
        """
        # Get the set of indices
        self._features = np.unique(
                    np.asarray(values, dtype=int)
                            ) - self.species.min_feat

        if not self.species.check(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

    @property
    def num_feats(self):
        """Get the number of features selected by the individual.

        :type: :py:class:`int`
        """
        return self._features.size

    @property
    def min_feat(self):
        """Minimum feature index selected by the individual.

        :type: :py:class:`int`
        """
        return min(self._features)

    @property
    def max_feat(self):
        """Maximum feature index selected by the individual.

        :type: :py:class:`int`
        """
        return max(self._features)

    def crossover(self, other):
        """Cross this individual with another one.

        All the common features will remain common in the new offspring. The
        remaining features will be randomly distributed to generate two new
        individuals.

        :param other: The other individual
        :type other: :py:class:`feature_selector.IntVector`
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        # Common features to both individuals
        common = np.intersect1d(self._features, other._features,
                                assume_unique=True)

        # Uncommon features
        uncommon = np.setxor1d(self._features, other._features,
                               assume_unique=True)

        # Create the two offspring
        np.random.shuffle(uncommon)

        lower_limit = max(0, self.species.min_size - common.size,
                          common.size + uncommon.size - self.species.max_size)

        upper_limit = uncommon.size - lower_limit

        cross_point = random.randint(lower_limit, upper_limit)
        self._features = np.concatenate((common, uncommon[:cross_point]))
        other._features = np.concatenate((common, uncommon[cross_point:]))

        # Return the new offspring
        return self, other

    def mutate(self, indpb):
        """Mutate the individual.

        Each feature is independently mutated according to the given
        probability.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: :py:class:`float`
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        # All possible features for the species
        all_feats = np.arange(self.species.min_feat, self.species.max_feat + 1)

        # Features to be kept
        to_be_kept = self._features[np.random.random(self.num_feats) >
                                    indpb]

        # Features not selected by the original individual
        not_selected = np.setdiff1d(all_feats, self._features)

        # Features to be added to the mutated individual
        to_be_added = not_selected[np.random.random(not_selected.size) <=
                                   indpb]

        # Generate the new mutated individual
        self._features = np.concatenate((to_be_kept, to_be_added))

        # Repair too small individuals
        if self._features.size < self.species.min_size:
            # Features not considered by the individual
            not_considered = np.setdiff1d(all_feats, self._features)

            # Number of needed features to achieve the minimum size for the
            # species
            needed = self.species.min_size - self._features.size

            # Obtain the patch for the individual
            patch = np.random.choice(not_considered, needed, replace=False)

            # Apply the patch
            self._features = np.concatenate((self._features, patch))

        # Repair too large individuals
        if self._features.size > self.species.max_size:
            # Select only some of the features to maintain the maximum size
            # for this species
            self._features = np.random.choice(self._features,
                                              self.species.max_size,
                                              replace=False)

        # Return the new individual
        return (self,)


DEFAULT_FEATURE_SELECTOR = Individual
"""Default feature selector class for the
:py:class:`~feature_selector.Tester`."""

DEFAULT_FITNESS = NumFeatsFitness
"""Default fitness for the :py:class:`~feature_selector.Tester`."""

DEFAULT_NUM_FEATS_VALUES = [10, 100, 1000, 10000]
"""Default list of values for the number of features used to define the
Species for the :py:class:`~feature_selector.Tester`."""

DEFAULT_PROP_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
"""Default list of values for the proportion of features used to define the
Species for the :py:class:`~feature_selector.Tester`."""

DEFAULT_MUT_PROB_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
"""Default list of values for the independent mutation probability for the
:py:class:`~feature_selector.Tester`."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an implementation is run in the
:py:class:`~feature_selector.Tester`."""


class Tester(unittest.TestCase):
    """Tester for the feature selector generator and breeding functions.

    Test extensively the generation and breeding operators of any subclass
    of :py:class:`~feature_selector.BaseIndividual`
    """

    feature_selector_cls = DEFAULT_FEATURE_SELECTOR
    """Feature selector implementation class that will be tested. Defaults to
        :py:attr:`~feature_selector.DEFAULT_FEATURE_SELECTOR`
        """

    num_feats_values = DEFAULT_NUM_FEATS_VALUES
    """List of different values for the number of features.

    A :py:class:`~base.Species` will be generated combining each one
    of these values for the number of features with each one of the different
    proportions to test the feature selector implementation (see
    :py:meth:`~feature_selector.Species.from_proportion`). Defaults to
    :py:attr:`~feature_selector.DEFAULT_NUM_FEATS_VALUES`
    """

    prop_values = DEFAULT_PROP_VALUES
    """List of proportions to generate the different
    :py:class:`~base.Species`.

    A :py:class:`~base.Species` species will be generated combining
    each one of these proportions with each one of the different values for the
    number of features values to test the featue selector implementation (see
    :py:meth:`~feature_selector.Species.from_proportion`). Defaults to
    :py:attr:`~feature_selector.DEFAULT_PROP_VALUES`
    """

    mut_prob_values = DEFAULT_MUT_PROB_VALUES
    """List of values for the mutation probability.

    Different values for the independent probability for each feature to be
    mutated. Defaults to
    :py:attr:`~feature_selector.DEFAULT_MUT_PROB_VALUES`
    """

    times = DEFAULT_TIMES
    """Times each function is executed. Defaults to
    :py:attr:`~feature_selector.DEFAULT_TIMES`
    """

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

        # Default fitness
        fitness = DEFAULT_FITNESS()

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the generator function the given number of times
                for _ in itertools.repeat(None, self.times):
                    # Check that the feature selector meets the species
                    # constraints
                    self.__check_correctness(
                        self.feature_selector_cls(
                            species, fitness))

        print('Ok')

    def test_1_crossover(self):
        """Test the behavior of the crossover operator.

        The crossover function is executed under different combinations of
        values for the number of features, minimum feature value, maximum
        feature value, minimum size and maximum size.
        """
        print('Testing the',
              self.feature_selector_cls.__name__,
              'crossover ...', end=' ')

        # Default fitness
        fitness = DEFAULT_FITNESS()

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Execute the crossover function the given number of times
                for _ in itertools.repeat(None, self.times):
                    # Generate the two parents
                    parent1 = self.feature_selector_cls(
                        species, fitness)
                    parent2 = self.feature_selector_cls(
                        species, fitness)
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
              self.feature_selector_cls.__name__,
              'mutation ...', end=' ')

        # Default fitness
        fitness = DEFAULT_FITNESS()

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # And for each proportion ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate a feature selector
                feature_selector = self.feature_selector_cls(
                    species, fitness)
                # For each possible value of the independent mutation
                # probability
                for mut_prob in self.mut_prob_values:
                    # Execute the mutation function the given number of times
                    for _ in itertools.repeat(
                            None, self.times):
                        # Mutate the feature selector
                        mutant, = feature_selector.mutate(mut_prob)
                        # Check that the mutant meets the species constraints
                        self.__check_correctness(mutant)
        print('Ok')

    def test_3_runtime(self):
        """Runtime of the constructor and breeding operators."""
        dataframe = pd.DataFrame()
        dataframe['constructor'] = self.__constructor_scalability()
        dataframe['crossover'] = self.__crossover_scalability()
        dataframe['mutation'] = self.__mutation_scalability()
        dataframe.index = self.num_feats_values
        dataframe.index.name = 'num_feats'

        print(dataframe)

    @staticmethod
    def __check_feature_selector_cls(feature_selector_cls):
        """Check if the festure selector class is correct.

        :param feature_selector_cls: Feature selector class to be tested.
        :type feature_selector_cls: Any subclass of
            :py:class:`~feature_selector.BaseIndividual`
        :raises TypeError: If *feature_selector_cls* is not a subclass of
            :py:class:`~feature_selector.BaseIndividual`
        """
        if not (isinstance(feature_selector_cls, type) and
                issubclass(feature_selector_cls, DEFAULT_FEATURE_SELECTOR)):
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
        start_time = time.perf_counter()

        # Execute the function the given number of times
        for _ in itertools.repeat(None, times):
            func(*args, **kwargs)

        # Measure time just after finishing the execution
        end_time = time.perf_counter()

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

        # Default fitness
        fitness = DEFAULT_FITNESS()

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
                    self.times, species, fitness)
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

        # Default fitness
        fitness = DEFAULT_FITNESS()

        # For each different value for the number of features ...
        for num_feats in self.num_feats_values:
            # Initialize the runtime accumulator
            runtime = 0
            # Accumulate runtimes for the different proportions ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate the two parents
                parent1 = self.feature_selector_cls(
                    species, fitness)
                parent2 = self.feature_selector_cls(
                    species, fitness)
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

        # Default fitness
        fitness = DEFAULT_FITNESS()

        # For each different value for the number of features ...
        for num_feats in self.num_feats_values:
            runtime = 0
            # Accumulate runtimes for the different proportions ...
            for prop in self.prop_values:
                # Create the species from num_feats and prop
                species = Species.from_proportion(num_feats, prop)
                # Generate a feature selector
                feature_selector = self.feature_selector_cls(
                    species, fitness)
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

    Tester.times = 50

    # Run the tests for BitVectors
    Tester.feature_selector_cls = BitVector
    t = unittest.TestLoader().loadTestsFromTestCase(Tester)
    unittest.TextTestRunner(verbosity=0).run(t)

    time.sleep(1)
    print()

    # Run the tests for IntVectors
    Tester.feature_selector_cls = IntVector
    t = unittest.TestLoader().loadTestsFromTestCase(Tester)
    unittest.TextTestRunner(verbosity=0).run(t)
