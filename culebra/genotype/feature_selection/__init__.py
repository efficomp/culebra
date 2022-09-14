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

"""Implementation of feature selector genotypes.

This module provides the :py:class:`~genotype.feature_selection.Individual`
abstract class, which defines the interface for feature selectors, and the
:py:class:`~genotype.feature_selection.Species` class, to impose constraints
on the individuals.

Some implementations of the :py:class:`~genotype.feature_selection.Individual`
class are provided in the :py:class:`~genotype.feature_selection.individual`
sub-module. Besides, this module also provides the
:py:class:`~genotype.feature_selection.Metrics` class, which supplies several
metrics about the selected features obtained by a :py:class:`~base.Wrapper`.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .abc import *
from .metrics import *
from . import individual

__all__ = ['individual']