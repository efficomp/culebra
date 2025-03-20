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

"""Fitness functions.

This module provides several fitness functions, all related to the feature
selection problem. Currently:

  * The :py:mod:`~culebra.fitness_function.abc` sub-module provides abstract
    classes to define the remaining fitness functions.

  * The :py:mod:`~culebra.fitness_function.feature_selection` sub-module is
    centered in datasets dimensionality reduction.

  * The :py:mod:`~culebra.fitness_function.svc_optimization` sub-module
    provides several fitness functions intended to optimize the Support Vector
    Classifier (SVC) hyperparameters for a given dataset.

  * The :py:mod:`~culebra.fitness_function.cooperative` sub-module provides
    fitness functions designed to the cooperative solving of a feature
    selection problem while the classifier hyperparamters are also being
    optimized.

  * The :py:mod:`~culebra.fitness_function.tsp` sub-module offers fitness
    functions for the traveling salesman problem.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


from .constants import DEFAULT_CLASSIFIER, DEFAULT_THRESHOLD, DEFAULT_CV_FOLDS
from . import abc, feature_selection, svc_optimization, cooperative, tsp


__all__ = [
    'abc',
    'feature_selection',
    'svc_optimization',
    'cooperative',
    'tsp',
    'DEFAULT_CLASSIFIER',
    'DEFAULT_THRESHOLD',
    'DEFAULT_CV_FOLDS'
]
