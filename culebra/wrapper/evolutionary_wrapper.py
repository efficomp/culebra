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

"""Provides the :py:class:`~wrapper.evolutionary_wrapper.EvolutionaryWrapper`
class.

This class implements the simplest evolutionary algorithm.
"""

import numbers
from deap.tools import selTournament
from deap.algorithms import varAnd
from culebra.wrapper.generational_wrapper import GenerationalWrapper

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_XOVER_PB = 0.5
"""Default xover probability."""

DEFAULT_MUT_PB = 0.1
"""Default mutation probability."""

DEFAULT_MUT_IND_PB = 0.05
"""Default independent gene mutation probability."""

DEFAULT_SEL_FUNC = selTournament
"""Default selection function."""

DEFAULT_SEL_FUNC_PARAMS = {'tournsize': 2}
"""Default selection function parameters."""


class EvolutionaryWrapper(GenerationalWrapper):
    """Base class for all the evolutionary wrapper methods."""

    def __init__(self, individual_cls, species, **params):
        """Initialize the wrapper method.

        :param individual_cls: Individual representation.
        :type individual_cls: Any subclass of
            :py:class:`~base.individual.Individual`
        :param species: The species the individual will belong to
        :type species: :py:class:`~base.species.Species`
        :param pop_size: Population size, defaults to
            :py:attr:`~wrapper.generational_wrapper.DEFAULT_POP_SIZE`
        :type pop_size: :py:class:`int`, optional
        :param n_gens: Number of generations, defaults to
            :py:attr:`~wrapper.generational_wrapper.DEFAULT_N_GENS`
        :type n_gens: :py:class:`int`, optional
        :param xover_func: Crossover function, defaults to the
            :py:meth:`~base.individual.Individual.crossover` method of
            *individual_cls*
        :type xover_func: Any callable object, optional
        :param xover_pb: Crossover rate, defaults to
            :py:attr:`~wrapper.evolutionary_wrapper.DEFAULT_XOVER_PB`
        :type xover_pb: :py:class:`float`, optional
        :param mut_func: Mutation function, defaults to the
            :py:meth:`~base.individual.Individual.mutate` method of
            *individual_cls*
        :type mut_func: Any callable object, optional
        :param mut_pb: Mutation rate, defaults to
            :py:attr:`~wrapper.evolutionary_wrapper.DEFAULT_MUT_PB`
        :type mut_pb: :py:class:`float`, optional
        :param mut_ind_pb: Independent gene mutation probability, defaults to
            :py:attr:`~wrapper.evolutionary_wrapper.DEFAULT_MUT_IND_PB`
        :type mut_ind_pb: :py:class:`float`, optional
        :param sel_func: Selection function, defaults to
            :py:attr:`~wrapper.evolutionary_wrapper.DEFAULT_SEL_FUNC`
        :type sel_func: Any callable object, optional
        :param sel_func_params: Selection function parameters, defaults to
            :py:attr:`~wrapper.evolutionary_wrapper.DEFAULT_SEL_FUNC_PARAMS`
        :type sel_func_params: :py:class:`dict`, optional
        :param checkpoint_freq: Frequency for checkpointing, defaults to
            :py:attr:`~base.wrapper.DEFAULT_CHECKPOINT_FREQ`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_file: File path for checkpointing, defaults to
            :py:attr:`~base.wrapper.DEFAULT_CHECKPOINT_FILE`
        :type checkpoint_file: :py:class:`str`, optional
        :param random_seed: Random seed for the random generator, defaults to
            `None`
        :type random_seed: :py:class:`int`, optional
        :param verbose: Whether or not to log the statistics, defaults to
            :py:data:`__debug__`
        :type verbose: :py:class:`bool`
        :raises TypeError: If any parameter has a wrong type
        """
        # Initialize the wrapper process
        super().__init__(individual_cls, species, **params)

        # Register the crossover function
        self.xover_func = params.pop('xover_func',
                                     self._individual_cls.crossover)
        self._toolbox.register("mate", self.xover_func)

        # Get the xover probability
        self.xover_pb = params.pop('xover_pb', DEFAULT_XOVER_PB)

        # Register the mutation function
        self.mut_func = params.pop('mut_func', self._individual_cls.mutate)
        self.mut_ind_pb = params.pop('mut_ind_pb', DEFAULT_MUT_IND_PB)
        self._toolbox.register("mutate", self.mut_func, indpb=self.mut_ind_pb)

        # Get the mutation probability
        self.mut_pb = params.pop('mut_pb', DEFAULT_MUT_PB)

        # Register the selection function
        self.sel_func = params.pop('sel_func', DEFAULT_SEL_FUNC)
        self.sel_func_params = params.pop('sel_func_params',
                                          DEFAULT_SEL_FUNC_PARAMS)
        self._toolbox.register("select", self.sel_func, **self.sel_func_params)

    @property
    def xover_func(self):
        """Crossover function.

        :getter: Return the current crossover function
        :setter: Set a new crossover function
        :type: Any callable object
        :raises TypeError: If set to a non callable object
        """
        return self._xover_func

    @xover_func.setter
    def xover_func(self, func):
        """Set the crossover function.

        :param func: The new crossover function
        :type func: Any callable object
        :raises TypeError: If *func* is not callable
        """
        # Check the func
        if not callable(func):
            raise TypeError("The crossover function is not callable")

        # Set the value
        self._xover_func = func

    @property
    def mut_func(self):
        """Mutation function.

        :getter: Return the current mutation function
        :setter: Set a new mutation function
        :type: Any callable object
        :raises TypeError: If set to a non callable object
        """
        return self._mut_func

    @mut_func.setter
    def mut_func(self, func):
        """Set the mutation function.

        :param func: The new mutation function
        :type func: Any callable object
        :raises TypeError: If *func* is not callable
        """
        # Check the func
        if not callable(func):
            raise TypeError("The mutation function is not callable")

        # Set the value
        self._mut_func = func

    @property
    def xover_pb(self):
        """Crossover probability.

        :getter: Return the current crossover probability
        :setter: Set a new crossover probability
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return self._xover_pb

    @xover_pb.setter
    def xover_pb(self, value):
        """Set the crossover probability.

        :param value: The new crossover probability
        :type value: A real value
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        # Check the value
        if not isinstance(value, numbers.Real):
            raise TypeError("The crossover probability must be an real "
                            "number")
        if not 0 < value < 1:
            raise ValueError("The crossover probability must be in (0, 1)")

        # Set the value
        self._xover_pb = value

    @property
    def mut_pb(self):
        """Mutation probability.

        :getter: Return the current mutation probability
        :setter: Set a new mutation probability
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return self._mut_pb

    @mut_pb.setter
    def mut_pb(self, value):
        """Set the mutation probability.

        :param value: The new mutation probability
        :type value: A real value
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        # Check the value
        if not isinstance(value, numbers.Real):
            raise TypeError("The mutation probability must be an real number")
        if not 0 < value < 1:
            raise ValueError("The mutation probability must be in (0, 1)")

        # Set the value
        self._mut_pb = value

    @property
    def mut_ind_pb(self):
        """Gene independent mutation probability.

        :getter: Return the current gene independent mutation probability
        :setter: Set a new gene independent mutation probability
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return self._mut_ind_pb

    @mut_ind_pb.setter
    def mut_ind_pb(self, value):
        """Set the gene independent mutation probability.

        :param value: The new gene independent mutation probability
        :type value: A real value
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        # Check the value
        if not isinstance(value, numbers.Real):
            raise TypeError("The gene independent mutation probability must "
                            "be an real number")
        if not 0 < value < 1:
            raise ValueError("The gene independent mutation probability must "
                             "be in (0, 1)")

        # Set the value
        self._mut_ind_pb = value

    @property
    def sel_func(self):
        """Selection function.

        :getter: Return the current selection function
        :setter: Set a new selection function
        :type: Any callable object
        :raises TypeError: If set to a non callable object
        """
        return self._sel_func

    @sel_func.setter
    def sel_func(self, func):
        """Set the selection function.

        :param func: The new selection function
        :type func: Any callable object
        :raises TypeError: If *func* is not callable
        """
        # Check the func
        if not callable(func):
            raise TypeError("The selection function is not callable")

        # Set the value
        self._sel_func = func

    @property
    def sel_func_params(self):
        """Selection function paramters.

        :getter: Return the current selection function parameters
        :setter: Set new selection function parameters
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a dictionary
        """
        return self._sel_func_params

    @sel_func_params.setter
    def sel_func_params(self, params):
        """Set the selection function parameters.

        :param params: The parameters
        :type params: :py:class:`dict`
        :raises TypeError: If *params* is not a dictionary
        """
        # Check the params
        if not isinstance(params, dict):
            raise TypeError("The selection function parameters must be "
                            "provided within a dictionary")

        # Set the value
        self._sel_func_params = params

    def _do_generation(self, pop, gen_num, logbook):
        """Implement a generation of the search process.

        :param pop: The population
        :type pop: Any iterable type
        :param gen_num: The generation number
        :type gen_num: :py:class:`int`
        :param logbook: Logbook for the stats
        :type logbook: :py:class:`~deap.tools.Logbook`
        """
        # Select the next generation individuals
        offspring = self._toolbox.select(pop, self.pop_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, self._toolbox, self.xover_pb,
                           self.mut_pb)

        # Evaluate the individuals with an invalid fitness
        num_evals = self._eval(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        self._do_stats(pop, gen_num, num_evals, logbook)
