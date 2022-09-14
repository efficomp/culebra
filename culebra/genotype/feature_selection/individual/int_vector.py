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


class IntVector(Individual):
    """Feature selector implementation based on arrays of indices."""

    def _random_init(self) -> None:
        """Init the features of this individual randomly."""
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
    def features(self) -> Sequence[int]:
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
    def features(self, values: Sequence[int]) -> None:
        """Set the indices of the new features selected by the individual.

        :param values: The new feature indices
        :type values: Array-like object
        :raises ValueError: If the values do not meet the species constraints.
        """
        # Get the set of indices
        self._features = np.unique(np.asarray(values, dtype=int))

        if not self.species.is_member(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

    @property
    def num_feats(self) -> int:
        """Get the number of features selected by the individual.

        :type: :py:class:`int`
        """
        return self._features.size

    @property
    def min_feat(self) -> int | None:
        """Minimum feature index selected by the individual.

        :type: :py:class:`int` or :py:data:`None` if no feature has been
            selected
        """
        return min(self._features) if self.num_feats > 0 else None

    @property
    def max_feat(self) -> int | None:
        """Maximum feature index selected by the individual.

        :type: :py:class:`int` or :py:data:`None` if no feature has been
            selected
        """
        return max(self._features) if self.num_feats > 0 else None

    def crossover(self, other: IntVector) -> Tuple[IntVector, IntVector]:
        """Cross this individual with another one.

        All the common features will remain common in the new offspring. The
        remaining features will be randomly distributed to generate two new
        individuals.

        :param other: The other individual
        :type other:
            :py:class:`~genotype.feature_selection.individual.IntVector`
        :return: The two offspring
        :rtype: :py:class:`tuple`
        """
        # Common features to both individuals
        common = np.intersect1d(
            self._features, other._features, assume_unique=True
        )

        # Uncommon features
        uncommon = np.setxor1d(
            self._features, other._features, assume_unique=True
        )

        # Create the two offspring
        np.random.shuffle(uncommon)

        lower_limit = max(
            0,
            self.species.min_size - common.size,
            common.size + uncommon.size - self.species.max_size
        )

        upper_limit = uncommon.size - lower_limit

        cross_point = random.randint(lower_limit, upper_limit)

        # Return the new offspring
        self._features = np.concatenate((common, uncommon[:cross_point]))
        other._features = np.concatenate((common, uncommon[cross_point:]))
        return self, other

    def mutate(self, indpb: float) -> Tuple[IntVector]:
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
        to_be_kept = self._features[np.random.random(self.num_feats) >=
                                    indpb]

        # Features not selected by the original individual
        not_selected = np.setdiff1d(all_feats, self._features)

        # Features to be added to the mutated individual
        to_be_added = not_selected[np.random.random(not_selected.size) <
                                   indpb]

        # Generate the new mutated individual
        new_features = np.concatenate((to_be_kept, to_be_added))

        # Repair too small individuals
        if new_features.size < self.species.min_size:
            # Features not considered by the individual
            not_considered = np.setdiff1d(all_feats, new_features)

            # Number of needed features to achieve the minimum size for the
            # species
            needed = self.species.min_size - new_features.size

            # Obtain the patch for the individual
            patch = np.random.choice(not_considered, needed, replace=False)

            # Apply the patch
            new_features = np.concatenate((new_features, patch))

        # Repair too large individuals
        if new_features.size > self.species.max_size:
            # Select only some of the features to maintain the maximum size
            # for this species
            new_features = np.random.choice(
                new_features, self.species.max_size, replace=False
            )

        self._features = new_features

        # Return the new individual
        return (self,)


# Exported symbols for this module
__all__ = ['IntVector']
