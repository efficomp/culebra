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

"""Fitness functions related to dataset scoring.

This module is composed by:

  * The :py:mod:`~culebra.fitness_function.dataset_score.abc` sub-module,
    where some abstract base classes are defined to support fitness functions
    developed in this module

  * Some popular scoring fitness functions:

    * :py:class:`~culebra.fitness_function.dataset_score.KappaIndex`:
      Single-objective function that maximizes the Kohen's Kappa index.

    * :py:class:`~culebra.fitness_function.dataset_score.Accuracy`:
      Single-objective function that maximizes the Accuracy.
"""

from .constants import (
    DEFAULT_CLASSIFIER,
    DEFAULT_CV_FOLDS
)

from . import abc

from .functions import (
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
    'KappaIndex',
    'Accuracy',
    'DEFAULT_CLASSIFIER',
    'DEFAULT_CV_FOLDS'
]
