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

"""Fitness functions related to feature selection.

This module is composed by:
    
* The :mod:`~culebra.fitness_function.feature_selection.abc` sub-module,
  where some abstract base classes are defined to support fitness functions
  developed in this module
* Some single-objective scoring fitness functions:
      
  * :class:`~culebra.fitness_function.feature_selection.Accuracy`:
    Single-objective function that maximizes the Accuracy for classification
    problems.
  * :class:`~culebra.fitness_function.feature_selection.FeatsProportion`:
    Dummy single-objective function that minimizes the number of selected
    features from a :class:`~culebra.tools.Dataset`. The difference with
    :class:`~culebra.fitness_function.feature_selection.NumFeats` is just that
    :class:`~culebra.fitness_function.feature_selection.FeatsProportion`
    returns a normalized number in [0, 1].
  * :class:`~culebra.fitness_function.feature_selection.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for
    classification problems.
  * :class:`~culebra.fitness_function.feature_selection.NumFeats`: Dummy
    single-objective function that minimizes the number of selected features
    from a :class:`~culebra.tools.Dataset`.
"""

from . import abc

from .functions import (
    NumFeats,
    FeatsProportion,
    KappaIndex,
    Accuracy
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


# Exported symbols for this module
__all__ = [
    'abc',
    'NumFeats',
    'FeatsProportion',
    'KappaIndex',
    'Accuracy'
]
