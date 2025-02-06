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

"""Constants of the module."""

from deap.tools import selTournament


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_POP_SIZE = 100
"""Default population size."""

DEFAULT_CROSSOVER_PROB = 0.8
"""Default crossover probability."""

DEFAULT_MUTATION_PROB = 0.2
"""Default mutation probability."""

DEFAULT_GENE_IND_MUTATION_PROB = 0.1
"""Default gene independent mutation probability."""

DEFAULT_SELECTION_FUNC = selTournament
"""Default selection function."""

DEFAULT_SELECTION_FUNC_PARAMS = {'tournsize': 2}
"""Default selection function parameters."""


# Exported symbols for this module
__all__ = [
    'DEFAULT_POP_SIZE',
    'DEFAULT_CROSSOVER_PROB',
    'DEFAULT_MUTATION_PROB',
    'DEFAULT_GENE_IND_MUTATION_PROB',
    'DEFAULT_SELECTION_FUNC',
    'DEFAULT_SELECTION_FUNC_PARAMS'
]
