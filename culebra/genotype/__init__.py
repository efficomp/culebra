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

"""Genotypes definition.

This module defines genotypes (genetic codings) for the different kinds of
individuals involved in feature selection problems. The genotype of one
individual is basically defined by the :py:class:`~base.Individual` subclass
it belongs to, and also by the :py:class:`~base.Species` constraining all the
individuals of that class.

By the moment, two genotypes are defined:

  * The :py:mod:`~genotype.feature_selection` sub-module defines individuals
    and species designed to select the most relevant features of a dataset
  * The :py:mod:`~genotype.classifier_optimization` sub-module defines
    individuals and species to support the co-evolution of the classifier used
    within the :py:class:`~base.Wrapper` while the better features are also
    being selected
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

__all__ = ['feature_selection', 'classifier_optimization']
