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

"""Test the abstract base feature selection fitness functions."""

import unittest

from culebra.fitness_function.svc_optimization.abc import SVCScorer
from culebra.solution.parameter_optimization import (
    Species as ParamOptSpecies,
    Individual as ParamOptIndividual
)
from culebra.solution.feature_selection import (
    Species as FSSpecies,
    BitVector as FSIndividual
)


class MySVCScorer(SVCScorer):
    """Dummy implementation of a fitness function."""

    @property
    def obj_weights(self):
        """Objective weights."""
        return (1, )

    def evaluate(self, sol, index, representatives):
        """Evaluate a solution."""
        return (0,)


class SVCScorerTester(unittest.TestCase):
    """Test SVCScorer."""

    def test_is_evaluable(self):
        """Test the is_evaluable method."""
        func = MySVCScorer()

        # Try an invalid solution class. Should fail...
        fs_species = FSSpecies(num_feats=3)
        fs_ind = FSIndividual(fs_species, func.fitness_cls)

        self.assertFalse(func.is_evaluable(fs_ind))

        # Try a valid solution class, but with invalis parameter names
        invalid_species = ParamOptSpecies(
            lower_bounds=[0, 0],
            upper_bounds=[100000, 100000],
            names=["P1", "P2"]
        )
        invalid_ind = ParamOptIndividual(invalid_species, func.fitness_cls)
        self.assertFalse(func.is_evaluable(invalid_ind))

        # Try a valid solution
        valid_species = ParamOptSpecies(
            lower_bounds=[0, 0],
            upper_bounds=[100000, 100000],
            names=["C", "gamma"]
        )
        valid_ind = ParamOptIndividual(valid_species, func.fitness_cls)
        self.assertTrue(func.is_evaluable(valid_ind))


if __name__ == '__main__':
    unittest.main()
