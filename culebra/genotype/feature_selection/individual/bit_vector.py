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

"""Individual base class."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Tuple
import random
import numpy as np
from culebra.genotype.feature_selection import Individual


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class BitVector(Individual):
    """Feature selector implementation based on boolean arrays."""

    def _random_init(self) -> None:
        """Init the features of this individual randomly."""
        # Select the features randomly
        max_num_feats = self.species.max_feat - self.species.min_feat + 1
        self._features = np.random.random(max_num_feats) < 0.5

        # Number of selected features
        selected_num_feats = np.count_nonzero(self._features)

        # Repair too small BitVectors
        if selected_num_feats < self.species.min_size:
            # Indices of the unselected features
            unselected = np.where(~self._features)[0]

            # Random choice of the minimal set of features to be selected
            np.random.shuffle(unselected)
            self._features[
                unselected[:self.species.min_size - selected_num_feats]] = True

            # The number of selected features is now the minimim allowed size
            selected_num_feats = self.species.min_size

        # Repair too large BitVectors
        if selected_num_feats > self.species.max_size:
            # Indices of the selected features
            selected = np.where(self._features)[0]

            # Random choice of the minimal set of features to be deselected
            np.random.shuffle(selected)
            self._features[
                selected[:selected_num_feats - self.species.max_size]
            ] = False

    @property
    def features(self) -> Sequence[int]:
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
    def features(self, values: Sequence[int]) -> None:
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

        if not self.species.is_member(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

    @property
    def num_feats(self) -> int:
        """Get the number of features selected by the individual.

        :type: :py:class:`int`
        """
        return np.count_nonzero(self._features)

    @property
    def min_feat(self) -> int | None:
        """Minimum feature index selected by the individual.

        :type: :py:class:`int` or :py:data:`None` if no feature has been
            selected
        """
        index = None
        if self.num_feats > 0:
            index = self._features.nonzero()[0][0] + self.species.min_feat
        return index

    @property
    def max_feat(self) -> int | None:
        """Maximum feature index selected by the individual.

        :type: :py:class:`int` or :py:data:`None` if no feature has been
            selected
        """
        index = None
        if self.num_feats > 0:
            index = self._features.nonzero()[0][-1] + self.species.min_feat
        return index

    def crossover1p(self, other: BitVector) -> Tuple[BitVector, BitVector]:
        """Cross this individual with another one.

        This method implements the single-point crossover.

        :param other: The other individual
        :type other:
            :py:class:`~genotype.feature_selection.individual.BitVector`
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

    def crossover2p(self, other: BitVector) -> Tuple[BitVector, BitVector]:
        """Cross this individual with another one.

        This method implements the two-points crossover.

        :param other: The other individual
        :type other:
            :py:class:`~genotype.feature_selection.individual.BitVector`
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

    def mutate(self, indpb: float) -> Tuple[BitVector]:
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
    """Default crossover operator.

    Implemented as the single-point crossover.

    :param other: The other individual
    :type other:
        :py:class:`~genotype.feature_selection.individual.BitVector`
    :return: The two offspring
    :rtype: :py:class:`tuple`
    """


# Exported symbols for this module
__all__ = ['BitVector']
