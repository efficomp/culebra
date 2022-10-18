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

"""Implementation of the simplest evolutionary algorithm."""

from __future__ import annotations
from deap.base import Toolbox
from deap.algorithms import varAnd

from . import SinglePop


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class Evolutionary(SinglePop):
    """This class implements the simplest evolutionary algorithm."""

    def _init_internals(self) -> None:
        """Set up the wrapper internal data structures to start searching.

        Overriden to create and initialize the Deap's
        :py:class:`~deap.base.Toolbox`.
        """
        super()._init_internals()

        # Create the toolbox
        self._toolbox = Toolbox()

        # Register the crossover function
        self._toolbox.register("mate", self.crossover_func)

        # Register the mutation function
        self._toolbox.register(
            "mutate", self.mutation_func, indpb=self.gene_ind_mutation_prob)

        # Register the selection function
        self._toolbox.register(
            "select", self.selection_func, **self.selection_func_params)

    def _reset_internals(self) -> None:
        """Reset the internal structures of the wrapper.

        Overriden to reset the Deap's :py:class:`~deap.base.Toolbox`.
        """
        super()._reset_internals()
        self._toolbox = None

    def _do_generation(self) -> None:
        """Implement a generation of the search process.

        In this case, the most simple evolutionary algorithm, as
        presented in chapter 7 of [Back2000]_, is implemented.

        .. [Back2000] T. Back, D. Fogel and Z. Michalewicz, eds. *Evolutionary
           Computation 1: Basic Algorithms and Operators*, CRC Press, 2000.

        """
        # Select the next generation individuals
        self.pop[:] = self._toolbox.select(self.pop, self.pop_size)

        # Vary the pool of individuals
        self.pop[:] = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob)

        # Evaluate the individuals with an invalid fitness and append the
        # current generation statistics to the logbook
        self._evaluate_pop(self.pop)
        self._do_generation_stats(self.pop)


# Exported symbols for this module
__all__ = ['Evolutionary']
