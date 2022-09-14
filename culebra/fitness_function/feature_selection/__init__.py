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

"""Fitness functions for feature selection.

This module provides the following classes:

  * :py:class:`~fitness_function.feature_selection.NumFeats`: Dummy
    single-objective function that minimizes the number of selected features
    from a :py:class:`~base.Dataset`.
  * :py:class:`~fitness_function.feature_selection.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for
    classification problems.
  * :py:class:`~fitness_function.feature_selection.KappaNumFeats`:
    Bi-objective function composed by the two former functions. It tries to
    both maximize the Kohen's Kappa index and minimize the number of features
    that an individual has selected.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .feature_selection import *
