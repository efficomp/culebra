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

"""Implementation of some evolutionary trainers.

This module is composed by:

* The :mod:`~culebra.trainer.ea.abc` sub-module, where some abstract base
  classes are defined to support the evolutionary trainers developed in this
  module

* Some popular single-population evolutionary algorithms:

  * The :class:`~culebra.trainer.ea.ElitistEA` class, which provides an
    elitist EA
  * The :class:`~culebra.trainer.ea.NSGA` class, which implements a
    multi-objective EA, based on Non-dominated sorting, able to run both
    the NSGA-II and the NSGA-III algorithms
  * The :class:`~culebra.trainer.ea.SimpleEA` class, which implements
    the simplest EA

* Some variants of the multi-population island-based EA:

    * The :class:`~culebra.trainer.ea.HeterogeneousParallelIslandsEA`
      class, a parallel implementation of the heterogeneous islands model
    * The :class:`~culebra.trainer.ea.HeterogeneousSequentialIslandsEA`
      class, providing a sequential implementation of the heterogeneous
      islands model
    * The :class:`~culebra.trainer.ea.HomogeneousParallelIslandsEA`
      class, which implements a parallel implementation of the homogeneous
      islands model
    * The :class:`~culebra.trainer.ea.HomogeneousSequentialIslandsEA`
      class, which provides a sequential implementation of the island model
      with homogeneous hyperparameters for all the islands

* A couple of cooperative co-evolutionary implementations:

    * The :class:`~culebra.trainer.ea.ParallelCooperativeEA` class,
      which provides a parallel implementation of the cooperative
      co-evolutionary model         
    * The :class:`~culebra.trainer.ea.SequentialCooperativeEA` class,
      which implements a sequential implementation of the cooperative
      co-evolutionary model
"""

from .constants import (
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_SELECTION_FUNC_PARAMS
)

from . import abc

from .single_pop_ea import (
    SimpleEA,
    ElitistEA,
    NSGA,
    DEFAULT_ELITE_SIZE,
    DEFAULT_NSGA_SELECTION_FUNC,
    DEFAULT_NSGA_SELECTION_FUNC_PARAMS,
    DEFAULT_NSGA3_REFERENCE_POINTS_P
)

from .islands_ea import (
    HomogeneousSequentialIslandsEA,
    HomogeneousParallelIslandsEA,
    HeterogeneousSequentialIslandsEA,
    HeterogeneousParallelIslandsEA
)

from .cooperative_ea import (
    SequentialCooperativeEA,
    ParallelCooperativeEA
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

# Exported symbols for this module
__all__ = [
    'abc',
    'SimpleEA',
    'ElitistEA',
    'NSGA',
    'HomogeneousSequentialIslandsEA',
    'HomogeneousParallelIslandsEA',
    'HeterogeneousSequentialIslandsEA',
    'HeterogeneousParallelIslandsEA',
    'SequentialCooperativeEA',
    'ParallelCooperativeEA',
    'DEFAULT_POP_SIZE',
    'DEFAULT_CROSSOVER_PROB',
    'DEFAULT_MUTATION_PROB',
    'DEFAULT_GENE_IND_MUTATION_PROB',
    'DEFAULT_SELECTION_FUNC',
    'DEFAULT_SELECTION_FUNC_PARAMS',
    'DEFAULT_ELITE_SIZE',
    'DEFAULT_NSGA_SELECTION_FUNC',
    'DEFAULT_NSGA_SELECTION_FUNC_PARAMS',
    'DEFAULT_NSGA3_REFERENCE_POINTS_P'
]
