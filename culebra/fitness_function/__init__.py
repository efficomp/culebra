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

"""Implementation of different fitness functions.

This module provides fitness functions for several problems. They are grouped
within the following sub-modules:

  * :py:mod:`~fitness_function.feature_selection` sub-module: Fitness functions
    to select a minimal subset of features from a :py:class:`~base.Dataset`.
  * :py:mod:`~fitness_function.classifier_optimization` sub-module: Fitness
    functions to find the optimal hyper-parameters of the classifier used
    within a :py:class:`~base.Wrapper` procedure.
  * :py:mod:`~fitness_function.cooperative` sub-module: Fitness functions to
    solve the two former problems in a cooperative way.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


from . import feature_selection
from . import classifier_optimization
from . import cooperative

__all__ = [
    'feature_selection',
    'classifier_optimization',
    'cooperative'
]
