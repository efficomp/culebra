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

"""Solution module.

This module provides several solution and species implementations for several
problems. Currently, the following sub-modules are available:

  * :py:mod:`~culebra.solution.abc`: Abstract base classes needed for each type
    of metaheuristic developed as a culebra's :py:class:`~culebra.abc.Trainer`

  * :py:mod:`~culebra.solution.feature_selection`: Solutions and species
    defined for feature-selection problems.

  * :py:mod:`~culebra.solution.parameter_optimization`: Solutions and species
    targeted for parameter optimization problems.

  * :py:mod:`~culebra.solution.tsp`: Solutions and species for the traveling
    salesman problem.
"""

from . import abc, feature_selection, parameter_optimization, tsp


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


__all__ = [
    'abc',
    'feature_selection',
    'parameter_optimization',
    'tsp'
]
