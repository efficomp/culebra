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

"""Unit test for the feature selection ants."""

import unittest
from os import remove
import random
from copy import copy, deepcopy
from itertools import repeat

import numpy as np

from culebra.abc import Species as BaseSpecies
from culebra.fitness_function.feature_selection import NumFeats
from culebra.solution.feature_selection import Species, Ant


Fitness = NumFeats.Fitness
"""Default fitness class."""

DEFAULT_NUM_FEATS_VALUES = [10, 100, 1000, 10000]
"""Default list of values for the number of features used to define the
Species."""

DEFAULT_TIMES = 1000
"""Default value for the number of times an function is tested."""


class AntTester(unittest.TestCase):
    """Test :py:class:`culebra.solution.feature_selection.Ant`."""

    num_feats_values = DEFAULT_NUM_FEATS_VALUES
    """List of different values for the number of features.

    A :py:class:`~culebra.solution.feature_selection.Species` will be generated
    combining each one of these values for the number of features with each one
    of the different proportions to test the feature selector implementation
    (see
    :py:meth:`~culebra.solution.feature_selection.Species.from_proportion`)."""

    times = DEFAULT_TIMES
    """Times each function is executed."""

    def test_init(self):
        """Test the __init__ method."""
        # Check the type of arguments
        with self.assertRaises(TypeError):
            Ant(BaseSpecies(), Fitness)
        with self.assertRaises(TypeError):
            Ant(Species(), Species)

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # Create the species from num_feats
            species = Species(num_feats)
            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                # Create an ant with an empty path
                empty_ant = Ant(species, Fitness)
                # Check that the default path is empty
                self.assertEqual(
                    empty_ant.num_feats, 0,
                    f'Ant size: {empty_ant.num_feats}'
                )
                # Check that there are not any discasred feature
                self.assertEqual(
                    len(empty_ant.discarded), 0,
                    f'Discarded: {empty_ant.discarded}'
                )

                # Create an ant with an full path
                full_ant = Ant(
                    species,
                    Fitness,
                    np.random.permutation(num_feats)
                )
                # Check that the default path is empty
                self.assertEqual(
                    full_ant.num_feats, num_feats,
                    f'Ant size: {full_ant.num_feats}'
                )
                full_ant.__repr__()
                # Check that there are not any discasred feature
                self.assertEqual(
                    len(full_ant.discarded), 0,
                    f'Discarded: {full_ant.discarded}'
                )

    def test_setup(self):
        """Test the _setup method."""
        # Check the type of arguments
        with self.assertRaises(TypeError):
            Ant(BaseSpecies(), Fitness)
        with self.assertRaises(TypeError):
            Ant(Species(), Species)

        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            # Create the species from num_feats and prop
            species = Species(num_feats)
            # Execute the generator function the given number of times
            for _ in repeat(None, self.times):
                ant = Ant(species, Fitness)
                # Check that the default path is empty
                self.assertEqual(
                    ant.num_feats, 0,
                    f'Ant size: {ant.num_feats}'
                )

    def test_features(self):
        """Test the features property."""
        num_feats = 10
        species = Species(num_feats)

        # All possible indices for the species
        indices = np.arange(0, num_feats)

        # Construct an ant with a complete path
        ant = Ant(species, Fitness, indices)

        # Test repeated features, should fail
        for index in indices:
            with self.assertRaises(ValueError):
                ant.features = np.append(ant.features, (index))

        # Test invalid values, should fail
        with self.assertRaises(ValueError):
            ant.features = np.append(ant.features, (num_feats))
        with self.assertRaises(ValueError):
            ant.features = np.append(ant.features, (-1))

        # Try correct indices
        for _ in repeat(None, self.times):
            size = random.randint(1, num_feats)
            feats = np.random.choice(indices, size=(size), replace=False)
            ant = Ant(species, Fitness, feats)

            # The features property returns an ordered path
            feats.sort()
            self.assertTrue((ant.features == feats).all())

    def test_path(self):
        """Test the path property."""
        num_feats = 10
        species = Species(num_feats)

        # All possible indices for the species
        indices = np.arange(0, num_feats)

        for _ in repeat(None, self.times):
            size = random.randint(1, num_feats)
            feats = np.random.choice(indices, size=(size), replace=False)
            ant = Ant(species, Fitness, feats)

            self.assertTrue((ant.path == feats).all())

    def test_discard(self):
        """Test the discard method and the discarded property."""
        num_feats = 10
        species = Species(num_feats)
        ant = Ant(species, Fitness)

        # Test that discarded is empty
        self.assertEqual(len(ant.discarded), 0)

        # All possible indices for the species
        indices = np.arange(0, num_feats)

        # Discard a feature
        for index in indices:
            ant.discard(index)
            self.assertEqual(len(ant.discarded), index + 1)
            self.assertTrue(index in ant.discarded)

            # Try to discard the feature again. Should fail
            with self.assertRaises(ValueError):
                ant.discard(index)

            # Try to append the discarded feature. Should fail
            with self.assertRaises(ValueError):
                ant.append(index)

        # Try to discard an invalid feature
        invalid_features = ('a', self, None)
        for feature in invalid_features:
            with self.assertRaises(TypeError):
                ant.discard(feature)

    def test_append_current(self):
        """Test the append method."""
        num_feats = 10
        species = Species(num_feats)

        # Test repeated features, should fail
        ant = Ant(species, Fitness)

        # Test invalid feature types
        invalid_features = ('a', self, None)
        for feature in invalid_features:
            with self.assertRaises(TypeError):
                ant.append(feature)

        # All possible indices for the species
        indices = np.arange(0, num_feats)

        for index in indices:
            ant.append(index)
            with self.assertRaises(ValueError):
                ant.append(index)

        # Test discarded features, should fail
        ant = Ant(species, Fitness)
        for index in indices:
            ant.discard(index)
            with self.assertRaises(ValueError):
                ant.append(index)

        # Try correct indices
        ant = Ant(species, Fitness)
        for index, feature in enumerate(indices):
            ant.append(feature)
            self.assertEqual(index + 1, ant.num_feats)
            self.assertEqual(feature, ant.current)

    def test_serialization(self):
        """Serialization test."""
        # For each value for the number of features ...
        pickle_filename = "my_pickle.gz"

        for num_feats in self.num_feats_values:
            species = Species(num_feats)
            # Build an ant the given number of times
            for _ in repeat(None, self.times):
                size = random.randint(1, num_feats)
                indices = np.arange(0, num_feats)
                feats = np.random.choice(indices, size=(size), replace=False)
                ant1 = Ant(species, Fitness, feats)

                ant1.save_pickle(pickle_filename)
                ant2 = Ant.load_pickle(pickle_filename)

                self.assertTrue((ant1.path == ant2.path).all())

                # Remove the pickle file
                remove(pickle_filename)

    def test_copy(self):
        """Copy test."""
        # For each value for the number of features ...
        for num_feats in self.num_feats_values:
            species = Species(num_feats)
            # Build an ant the given number of times
            for _ in repeat(None, self.times):
                size = random.randint(1, num_feats)
                indices = np.arange(0, num_feats)
                feats = np.random.choice(indices, size=(size), replace=False)
                ant1 = Ant(species, Fitness, feats)

                ant2 = copy(ant1)
                ant3 = deepcopy(ant1)

                self.assertTrue((ant1.path == ant2.path).all())
                self.assertTrue((ant1.path == ant3.path).all())

    def test_repr(self):
        """Test the repr and str dunder methods."""
        num_feats = 10
        species = Species(num_feats)
        ant = Ant(species, Fitness)
        self.assertIsInstance(repr(ant), str)
        self.assertIsInstance(str(ant), str)


if __name__ == '__main__':
    unittest.main()
