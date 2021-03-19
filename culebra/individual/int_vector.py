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

"""Implementation of :py:class:`~base.individual.Individual` as a
:py:class:`numpy.ndarray` of :py:class:`int` feature indices.

This module implements the creation of individuals, along with the
crossover and mutation operators based on a :py:class:`numpy.ndarray` of
:py:class:`int` values, where each element represents the index of a selected
feature.
"""

import random
import unittest
import numpy as np
from culebra.base.individual import Individual

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class IntVector(Individual):
    """Individuals implementation based on arrays of indices."""

    def __init__(self, species, fitness, features=None):
        """Create an individual.

        :param species: The species the individual will belong to
        :type species: :py:class:`~base.species.Species`
        :param fitness: The fitness object for the individual
        :type fitness: Instance of any subclass of
            :py:class:`~base.fitness.Fitness`
        :param features: Features to init the individual. If `None` the
            individual is initialized randomly.
        :type features: Any subclass of :py:class:`~collections.abc.Sequence`
            containing valid feature indices
        :raises TypeError: If any parameter type is wrong
        """
        # Initialize the individual
        super(IntVector, self).__init__(species, fitness, features)

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
        """Indices of the features selected by the individual.

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
        # Get the set of indices (ordered)
        self._features = self._check_features(values)

    @property
    def num_feats(self):
        """Number of features selected by the individual.

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
        :type other: :py:class:`~individual.int_vector.IntVector`
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

        r = random.randint(lower_limit, upper_limit)
        self._features = np.concatenate((common, uncommon[:r]))
        other._features = np.concatenate((common, uncommon[r:]))

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


# Tests the functions in this file
if __name__ == '__main__':
    from culebra.tools.individual_tester import IndividualTester

    # Configure the tests parameters
    IndividualTester.ind_cls = IntVector

    # Run the tests
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)
