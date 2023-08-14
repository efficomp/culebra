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

"""Unit test for the feature selection species."""

import unittest
import pickle
from copy import copy, deepcopy
from culebra.solution.feature_selection import Species, MAX_PROP


class SpeciesTester(unittest.TestCase):
    """Test :py:class:`~culebra.solution.feature_selection.Species`."""

    def test_init(self):
        """Test the constructor."""
        # Default parameters
        valid_num_feats = (10, 50, 1000)
        for num_feats in valid_num_feats:
            species = Species(num_feats)

            # Check the default values for the attributes
            self.assertEqual(species.num_feats, num_feats)
            self.assertEqual(species.min_feat, 0)
            self.assertEqual(species.max_feat, num_feats - 1)
            self.assertEqual(species.min_size, 0)
            self.assertEqual(species.max_size, num_feats)

        # Invalid num_feats types
        invalid_num_feats = (len, 'a')
        for num_feats in invalid_num_feats:
            with self.assertRaises(TypeError):
                Species(num_feats=num_feats)

        # Invalid num_feats values
        invalid_num_feats = (-1, 0)
        for num_feats in invalid_num_feats:
            with self.assertRaises(ValueError):
                Species(num_feats=num_feats)

        # Invalid min_feat types
        num_feats = 10
        invalid_min_feats = (len, 'a')
        for min_feat in invalid_min_feats:
            with self.assertRaises(TypeError):
                Species(num_feats=num_feats, min_feat=min_feat)

        # Invalid min_feat values
        invalid_min_feats = (-1, 10)
        for min_feat in invalid_min_feats:
            with self.assertRaises(ValueError):
                Species(num_feats=num_feats, min_feat=min_feat)

        # Valid min_feat values
        valid_min_feats = (0, 5, 9)
        for min_feat in valid_min_feats:
            Species(num_feats=num_feats, min_feat=min_feat)

        # Invalid max_feat types
        min_feat = 5
        invalid_max_feats = (len, 'a')
        for max_feat in invalid_max_feats:
            with self.assertRaises(TypeError):
                Species(
                    num_feats=num_feats, min_feat=min_feat, max_feat=max_feat
                )

        # Invalid max_feat values
        invalid_max_feats = (4, 10)
        for max_feat in invalid_max_feats:
            with self.assertRaises(ValueError):
                Species(
                    num_feats=num_feats, min_feat=min_feat, max_feat=max_feat
                )

        # Valid max_feat values
        valid_max_feats = (5, 7, 9)
        for max_feat in valid_max_feats:
            Species(
                num_feats=num_feats, min_feat=min_feat, max_feat=max_feat
            )

        # Invalid min_size types
        max_feat = 8
        invalid_min_sizes = (len, 'a')
        for min_size in invalid_min_sizes:
            with self.assertRaises(TypeError):
                Species(
                    num_feats=num_feats,
                    min_feat=min_feat,
                    max_feat=max_feat,
                    min_size=min_size
                )

        # Invalid min_size values
        invalid_min_sizes = (-1, 5)
        for min_size in invalid_min_sizes:
            with self.assertRaises(ValueError):
                Species(
                    num_feats=num_feats,
                    min_feat=min_feat,
                    max_feat=max_feat,
                    min_size=min_size
                )

        # Valid min_size values
        valid_min_sizes = (0, 2, 4)
        for min_size in valid_min_sizes:
            Species(
                num_feats=num_feats,
                min_feat=min_feat,
                max_feat=max_feat,
                min_size=min_size
            )

        # Invalid max_size types
        min_size = 1
        invalid_max_sizes = (len, 'a')
        for max_size in invalid_max_sizes:
            with self.assertRaises(TypeError):
                Species(
                    num_feats=num_feats,
                    min_feat=min_feat,
                    max_feat=max_feat,
                    min_size=min_size,
                    max_size=max_size
                )

        # Invalid max_size values
        invalid_max_sizes = (0, 5)
        for max_size in invalid_max_sizes:
            with self.assertRaises(ValueError):
                Species(
                    num_feats=num_feats,
                    min_feat=min_feat,
                    max_feat=max_feat,
                    min_size=min_size,
                    max_size=max_size
                )

        # Valid max_size values
        valid_max_sizes = (1, 2, 4)
        for max_size in valid_max_sizes:
            Species(
                num_feats=num_feats,
                min_feat=min_feat,
                max_feat=max_feat,
                min_size=min_size,
                max_size=max_size
            )

    def test_from_proportion(self):
        """Test the from_proportion factory method."""
        with self.assertRaises(TypeError):
            Species.from_proportion(1000, 'a')
        with self.assertRaises(ValueError):
            Species.from_proportion(1000, -0.1)
        with self.assertRaises(ValueError):
            Species.from_proportion(1000, MAX_PROP+0.2)

        species = Species.from_proportion(1000, 0)
        self.assertEqual(species.num_feats, 1000)
        self.assertEqual(species.min_feat, 0)
        self.assertEqual(species.max_feat, 999)
        self.assertEqual(species.min_size, 0)
        self.assertEqual(species.max_size, 1000)

        species = Species.from_proportion(1000, MAX_PROP)
        self.assertEqual(species.num_feats, 1000)
        self.assertEqual(species.min_feat, 250)
        self.assertEqual(species.max_feat, 749)
        self.assertEqual(species.min_size, 250)
        self.assertEqual(species.max_size, 250)

    def test_copy(self):
        """Test the __copy__ method."""
        species1 = Species(15)
        species2 = copy(species1)

        # Copy only copies the first level (species1 != species2)
        self.assertNotEqual(id(species1), id(species2))

        # The species attributes are shared
        self.assertEqual(id(species1._num_feats), id(species2._num_feats))
        self.assertEqual(id(species1._max_size), id(species2._max_size))

    def test_deepcopy(self):
        """Test the __deepcopy__ method."""
        species1 = Species(18)
        species2 = deepcopy(species1)

        # Check the copy
        self._check_deepcopy(species1, species2)

    def test_serialization(self):
        """Serialization test.

        Test the __setstate__ and __reduce__ methods.
        """
        species1 = Species(18)

        data = pickle.dumps(species1)
        species2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(species1, species2)

    def _check_deepcopy(self, species1, species2):
        """Check if *species1* is a deepcopy of *species2*.

        :param species1: The first species
        :type species1: :py:class:`~culebra.solution.feature_selection.Species`
        :param species2: The second species
        :type species2: :py:class:`~culebra.solution.feature_selection.Species`
        """
        # Copies all the levels
        self.assertNotEqual(id(species1), id(species2))
        self.assertEqual(species1._num_feats, species2._num_feats)


if __name__ == '__main__':
    unittest.main()
