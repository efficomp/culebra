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

"""Base classes of culebra.

Culebra is based in some base classes to define the fundamental pieces that are
necessary to solve a feature selection problem. The :py:mod:`~base`
module defines:

  * A :py:class:`~base.Base` class from which all the classes of culebra
    inherit
  * A :py:class:`~base.Dataset` class containing samples, where the most
    relevant input features need to be selected
  * An :py:class:`~base.Individual` class, which will be used within the
    :py:class:`~base.Wrapper` class to search the best subset of features
  * A :py:class:`~base.Species` class to define the characteristics of the
    individuals
  * A :py:class:`~base.FitnessFunction` class to evaluate the individuals
    during the search process
  * A :py:class:`~base.Fitness` class to store the fitness values for each
    :py:class:`~base.Individual`
  * A :py:class:`~base.Wrapper` class to find the most representative features
    of the :py:class:`~base.Dataset`
  * A :py:mod:`~base.checker` module with functions to verify different kinds
    of objects.
"""

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

from .checker import (
    check_bool,
    check_str,
    check_limits,
    check_int,
    check_float,
    check_instance,
    check_subclass,
    check_func,
    check_func_params,
    check_sequence,
    check_filename
)
from .base import Base
from .dataset import Dataset, DEFAULT_SEP
from .abc import (
    Fitness,
    FitnessFunction,
    Species,
    Individual,
    Wrapper,
    DEFAULT_STATS_NAMES,
    DEFAULT_OBJECTIVE_STATS,
    DEFAULT_CHECKPOINT_ENABLE,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_VERBOSITY,
    DEFAULT_INDEX,
    DEFAULT_CLASSIFIER
)

__all__ = [
    'check_bool',
    'check_str',
    'check_limits',
    'check_int',
    'check_float',
    'check_instance',
    'check_subclass',
    'check_func',
    'check_func_params',
    'check_sequence',
    'check_filename',
    'Base',
    'Dataset',
    'Fitness',
    'FitnessFunction',
    'Species',
    'Individual',
    'Wrapper',
    'DEFAULT_SEP',
    'DEFAULT_STATS_NAMES',
    'DEFAULT_OBJECTIVE_STATS',
    'DEFAULT_CHECKPOINT_ENABLE',
    'DEFAULT_CHECKPOINT_FREQ',
    'DEFAULT_CHECKPOINT_FILENAME',
    'DEFAULT_VERBOSITY',
    'DEFAULT_INDEX',
    'DEFAULT_CLASSIFIER'
]
