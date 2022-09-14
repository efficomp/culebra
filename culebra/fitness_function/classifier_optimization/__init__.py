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

"""Fitness functions for classifier optimization.

This module provides the following classes:

  * :py:class:`~fitness_function.classifier_optimization.RBFSVCFitnessFunction`:
    Abstract base class for fitness functions to optimize the hyperparameters
    of SVM-based classifier with RBF kernels.
  * :py:class:`~fitness_function.classifier_optimization.C`: Dummy
    single-objective function that minimizes the regularization parameter *C*
    of a SVM-based classifier with RBF kernels.
  * :py:class:`~fitness_function.classifier_optimization.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for
    a SVM-based classifier with RBF kernels.
  * :py:class:`~fitness_function.classifier_optimization.KappaC`:
    Bi-objective function that tries to both maximize the Kohen's Kappa index
    and minimize the regularization parameter *C*
    of a SVM-based classifier with RBF kernels.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .rbf_svc_optimization import *