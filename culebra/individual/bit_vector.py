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

"""Implementation of :py:class:`~base.individual.Individual` as a boolean
:py:class:`numpy.ndarray`.

This module implements the creation of individuals, along with the
crossover and mutation operators based on a :py:class:`numpy.ndarray` of
:py:class:`bool` values, where each element represents whether the
corresponding feature has been selected or not.
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


class BitVector(Individual):
    """Individuals implementation based on boolean arrays."""

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
        super(BitVector, self).__init__(species, fitness, features)

        # If initial features are provided, use them
        if features is not None:
            self.features = features
        # Else the features are randomly generated
        else:

            # Select the features randomly
            nf = species.max_feat - species.min_feat + 1
            self._features = np.random.random(nf) < 0.5

            # Number of selected features
            nf = np.count_nonzero(self._features)

            # Repair too small BitVectors
            if nf < species.min_size:
                # Indices of the unselected features
                unselected = np.where(~self._features)[0]

                # Random choice of the minimal set of features to be selected
                np.random.shuffle(unselected)
                self._features[unselected[:species.min_size - nf]] = True

                # The number of features is now the minimim allowed size
                nf = species.min_size

            # Repair too large BitVectors
            if nf > species.max_size:
                # Indices of the selected features
                selected = np.where(self._features)[0]

                # Random choice of the minimal set of features to be deselected
                np.random.shuffle(selected)
                self._features[selected[:nf - species.max_size]] = False

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
        return self._features.nonzero()[0] + self.species.min_feat

    @features.setter
    def features(self, values):
        """Set the indices of the new features selected by the individual.

        :param values: The new feature indices
        :type values: Array-like object
        :raises ValueError: If the values do not meet the species constraints.
        """
        # Get the set of indices (ordered)
        indices = self._check_features(values) - self.species.min_feat
        nf = self.species.max_feat - self.species.min_feat + 1
        self._features = np.zeros(nf, dtype=bool)
        self._features[indices] = True

    @property
    def num_feats(self):
        """Number of features selected by the individual.

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
        :type other: :py:class:`~individual.bit_vector.BitVector`
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        while True:
            # Cross point. Use randint to include also max_feat
            min_feat = self.species.min_feat
            max_feat = self.species.max_feat
            p = random.randint(0, max_feat - min_feat)

            # Try a crossing the individuals
            self._features[:p], other._features[:p] = \
                other._features[:p].copy(), self._features[:p].copy()

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
        :type other: :py:class:`~individual.bit_vector.BitVector`
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        while True:
            # Cross points. Use randint to include also max_feat
            min_feat = self.species.min_feat
            max_feat = self.species.max_feat
            p0 = random.randint(0, max_feat - min_feat)
            p1 = random.randint(0, max_feat - min_feat)
            if p0 > p1:
                p0, p1 = p1, p0
            # Cross the individuals
            self._features[p0:p1], other._features[p0:p1] = \
                other._features[p0:p1].copy(), self._features[p0:p1].copy()
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


# Tests the functions in this file
if __name__ == '__main__':
    from culebra.tools.individual_tester import IndividualTester

    # Configure the tests parameters
    IndividualTester.ind_cls = BitVector

    # Run the tests
    t = unittest.TestLoader().loadTestsFromTestCase(IndividualTester)
    unittest.TextTestRunner(verbosity=0).run(t)
