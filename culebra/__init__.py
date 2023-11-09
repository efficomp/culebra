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

"""Culebra was born as a `DEAP <https://deap.readthedocs.io/en/master/>`_-based
evolutionary computation library designed to solve feature selection problems.
However, it has been redesigned to support different kind of problems and also
different metaheuristics.

Experiments and experiment batchs are automatized by means of the
:py:class:`~culebra.tools.Experiment` and :py:class:`~culebra.tools.Batch`
classes, both in the :py:mod:`~culebra.tools` module. Statistical analysis of
the :py:class:`~culebra.tools.Results` is also provided by the
:py:class:`~culebra.tools.ResultsAnalyzer` class.

Culebra is structured in the following modules:

  * The :py:mod:`~culebra.abc` module, which defines the abstract base classes
    that support culebra

  * The :py:mod:`~culebra.checker` module, which provides several checker
    functions used within culebra to prevent wrong arguments to functions
    and methods

  * The :py:mod:`~culebra.solution` module, which define solutions and solution
    species for several problems

  * The :py:mod:`~culebra.fitness_function` module, which provides fitness
    functions for several problems

  * The :py:mod:`~culebra.trainer` module, which implement several training
    algorithms

  * The :py:mod:`~culebra.tools` module, which implements several tools to
    handle data and make easier the experimentation and the analysis of the
    obtained results
"""

from .constants import (
    DEFAULT_MAX_NUM_ITERS,
    DEFAULT_CHECKPOINT_ENABLE,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_VERBOSITY,
    DEFAULT_INDEX
)
from . import (
    checker,
    abc,
    solution,
    fitness_function,
    trainer,
    tools
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


__all__ = [
    'checker',
    'abc',
    'solution',
    'fitness_function',
    'trainer',
    'tools',
    'DEFAULT_MAX_NUM_ITERS',
    'DEFAULT_CHECKPOINT_ENABLE',
    'DEFAULT_CHECKPOINT_FREQ',
    'DEFAULT_CHECKPOINT_FILENAME',
    'DEFAULT_VERBOSITY',
    'DEFAULT_INDEX'
]
