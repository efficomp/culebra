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

"""The classifier optimization genotype allows the co-evolution of the
hyper-parameters defining the classifier used within the
:py:class:`~base.Wrapper` procedure while the best subset of features is also
being found. The genotype is defined by the following classes:

  * An :py:class:`~genotype.classifier_optimization.Individual` class
    containing all the hyper-parameters needed to define a classifier. Since
    the hyper-parameters will be real numbers, the crossover and mutation
    operators are implemented using the well-known SBX crossover and polynomial
    mutation, respectively
  * A :py:class:`~genotype.classifier_optimization.Species` class to define the
    characteristics of the individuals
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .classifier_optimization import *
