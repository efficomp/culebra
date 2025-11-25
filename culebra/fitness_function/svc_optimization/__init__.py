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

"""Fitness functions for parameter optimization problems.

This module is composed by:

* The :mod:`~culebra.fitness_function.svc_optimization.abc` sub-module,
  where some abstract base classes are defined to support fitness functions
  developed in this module
* Some scoring fitness functions:

  * :class:`~culebra.fitness_function.svc_optimization.Accuracy`:
    Single-objective function that maximizes the Accuracy for a
    SVM-based classifier with RBF kernels.
  * :class:`~culebra.fitness_function.svc_optimization.C`: Dummy
    single-objective function that minimizes the regularization parameter *C*
    of a SVM-based classifier with RBF kernels.
  * :class:`~culebra.fitness_function.svc_optimization.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for a
    SVM-based classifier with RBF kernels.
"""

from . import abc

from .functions import (
    C,
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
    'C',
    'KappaIndex',
    'Accuracy'
]
