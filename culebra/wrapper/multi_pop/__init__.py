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

"""Implementation of distributed evolutionary algorithms.

The base abstract class for all distributed generational wrappers is
:py:class:`~wrapper.multi_pop.MultiPop`, which provides the basic mechanisms
to support multiple subpopulations and :py:class:`~base.Individual` exchanges
between them. Each subpopulation will be handled by a
:py:class:`~wrapper.single_pop.SinglePop` wrapper.

However, any distributed wrapper could be executed both sequentially or in
parallel. Thus, the :py:class:`~wrapper.multi_pop.MultiPop` class has two
abstract subclasses to allow users to choose the way they want to execute a
distributed wrapper:

    * :py:class:`~wrapper.multi_pop.SequentialMultiPop`: Abstract sequential
      multi-population model for generational evolutionary wrappers
    * :py:class:`~wrapper.multi_pop.ParallelMultiPop`: Abstract parallel
      multi-population model for generational evolutionary wrappers

Regarding the evolutionary models, by the moment the following ones have been
implemented:

  * :py:class:`~wrapper.multi_pop.Islands`: Abstract island-based model for
    evolutionary wrappers. Each island evolves a subpopulation and all the
    islands evolve the same kind (:py:class:`~base.Species`) of
    :py:class:`~base.Individual`
  * :py:class:`~wrapper.multi_pop.HomogeneousIslands`: Abstract island-based
    model for evolutionary wrappers where all the islands must use the same
    hyperparameters (population size, breeding operators, probabilities,
    etc.)
  * :py:class:`~wrapper.multi_pop.HeterogeneousIslands`: Abstract island-based
    model for evolutionary wrappers where each island can have different
    values for its hyper-parameters (population size, breeding operators,
    probabilities, etc.)
  * :py:class:`~wrapper.multi_pop.Cooperative`: Abstract island-based
    model for cooperative co-evolutionary wrappers, where each island evolves
    a different :py:class:`~base.Species` of :py:class:`~base.Individual` that
    cooperate to solve a problem. The hyperparameters of each island can be
    different too.


The above models have been used to develop the following distributed wrapper
implementations:

  * Island model implementations:

      * :py:class:`~wrapper.multi_pop.HomogeneousSequentialIslands`: Sequential
        implementation of the homogeneous island model
      * :py:class:`~wrapper.multi_pop.HomogeneousParallelIslands`: Parallel
        implementation of the homogeneous island model
      * :py:class:`~wrapper.multi_pop.HeterogeneousSequentialIslands`:
        Sequential implementation of the heterogeneous island model
      * :py:class:`~wrapper.multi_pop.HeterogeneousParallelIslands`: Parallel
        implementation of the heterogeneous island model

  * Cooperative model implementations:
      * :py:class:`~wrapper.multi_pop.SequentialCooperative`: Sequential
        implementation of the cooperative co-evolutionary model
      * :py:class:`~wrapper.multi_pop.ParallelCooperative`: Parallel
        implementation of the cooperative co-evolutionary model

Independently of the chosen model, :py:class:`~wrapper.multi_pop.MultiPop`
subclasses may send individuals from any subpopulation wrapper to others. The
:py:mod:`~wrapper.multi_pop.topology` sub-module provides several commonly used
communication topologies.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .topology import *
from .abc import *
from .islands import *
from .cooperative import *
