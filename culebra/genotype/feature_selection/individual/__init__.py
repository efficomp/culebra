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

"""Individual implementations for feature selection problems.

This module provides two :py:class:`numpy.ndarray`-based implementations of
feature selector individuals:

    * :py:class:`~genotype.feature_selection.individual.BitVector`:
        Individuals are boolean vectors of a fixed length, the number of
        selectable features. One boolean value is associated to each selectable
        feature. If the the boolean value is :py:data:`True`, then the feature
        is selected, otherwhise the feature is ignored.
    * :py:class:`~genotype.feature_selection.individual.IntVector`:
        Individuals are vector of feature indices. Only the features whose
        indices are present in the individual are selected. Thus, individuals
        may have different lengths and can not contain repeated indices.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .bit_vector import *
from .int_vector import *
