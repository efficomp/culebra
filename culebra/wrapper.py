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

"""Implementation of some wrapper algorithms.

This module provides the following wrapper method implementations:
  * :py:class:`~wrapper.GenerationalWrapper`: Base class for all the
    generational-based wrapper algorithms
  * :py:class:`~wrapper.EvolutionaryWrapper`: Simple evolutionary algorithm
  * :py:class:`~wrapper.NSGAWrapper`: Multi-objective evolutionary algorithm,
    based on Non-dominated sorting, able to run the NSGA-II or the NSGA-III
    algorithms
"""

import time
import numbers
import random
import numpy as np
import pandas.io
from deap.algorithms import varAnd
from deap.tools import initRepeat, HallOfFame, Logbook, selTournament, \
    ParetoFront, selNSGA2, selNSGA3, uniform_reference_points
from culebra.base import Wrapper


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_STATS_NAMES = ("Gen", "NEvals")
"""Default statistics calculated for each generation."""

DEFAULT_POP_SIZE = 100
"""Default population size."""

DEFAULT_N_GENS = 100
"""Default number of generations."""


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

DEFAULT_NSGA_SEL_FUNC = selNSGA2
"""Default selection function for NSGA-based algorithms."""

DEFAULT_NSGA_SEL_FUNC_PARAMS = {}
"""Default selection function parameters for NSGA-based algorithms."""


class GenerationalWrapper(Wrapper):
    """Base class for all the generational wrapper methods.

    It is an abstract class. The
    :py:meth:`~wrapper.GenerationalWrapper._do_generation` method must be
    implemented in its subclasses to obtain a complete generational algorithm.
    """

    stats_names = DEFAULT_STATS_NAMES
    """Statistics calculated for each generation."""

    def __init__(self, individual_cls, species, **params):
        """Initialize the wrapper method.

        :param individual_cls: Individual representation.
        :type individual_cls: Any subclass of :py:class:`~base.Individual`
        :param species: The species the individual will belong to
        :type species: Any sublass of :py:class:`~base.Species`
        :param pop_size: Population size, defaults to
            :py:attr:`~wrapper.DEFAULT_POP_SIZE`
        :type pop_size: :py:class:`int`, optional
        :param n_gens: Number of generations, defaults to
            :py:attr:`~wrapper.DEFAULT_N_GENS`
        :type n_gens: :py:class:`int`, optional
        :param checkpoint_freq: Frequency for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FREQ`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_file: File path for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FILE`
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

        # Get the population size
        self.pop_size = params.pop('pop_size', DEFAULT_POP_SIZE)

        # Get the number of generations
        self.n_gens = params.pop('n_gens', DEFAULT_N_GENS)

    @property
    def pop_size(self):
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return self._pop_size

    @pop_size.setter
    def pop_size(self, value):
        """Set the population size.

        :param value: The new population size
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        if not isinstance(value, numbers.Integral):
            raise TypeError("The population size must be an integer number")
        if value <= 0:
            raise ValueError("The population size must be a positive number")

        # Set the value
        self._pop_size = value

    @property
    def n_gens(self):
        """Get and set the number of generations.

        :getter: Return the current number of generations
        :setter: Set a new value for the number of generations
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return self._n_gens

    @n_gens.setter
    def n_gens(self, value):
        """Set the number of generations.

        :param value: The new number of generations
        :type value: An integer value
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        if not isinstance(value, numbers.Integral):
            raise TypeError("The number of generations must be an integer "
                            "number")
        if value <= 0:
            raise ValueError("The number of generations must be a positive "
                             "number")

        # Set the value
        self._n_gens = value

    def _eval(self, pop):
        """Evaluate the individuals of a population with an invalid fitness.

        :param pop: The population.
        :type pop: Any iterable type.
        :return: The number of individuals evaluated.
        :rtype: :py:class:`int`
        """
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        return len(invalid_ind)

    def _do_stats(self, pop, gen_num, num_evals, logbook):
        """Compile the stats of a generation.

        :param pop: The population
        :type pop: Any iterable type
        :param gen_num: The generation number
        :type gen_num: :py:class:`int`
        :param num_evals: Number of evaluations performed
        :type num_evals: :py:class:`int`
        :param logbook: Logbook for the stats
        :type logbook: :py:class:`~deap.tools.Logbook`
        """
        record = self._stats.compile(pop) if self._stats else {}
        logbook.record(Gen=gen_num, NEvals=num_evals, **record)
        if self.verbose:
            print(logbook.stream)

    def _do_generation(self, pop, gen_num, logbook):
        """Implement a generation of the search process.

        This method must be overriden by subclasses to implement the different
        generational algorithms.

        :param pop: The population
        :type pop: Any iterable type
        :param gen_num: The generation number
        :type gen_num: :py:class:`int`
        :param logbook: Logbook for the stats
        :type logbook: :py:class:`~deap.tools.Logbook`
        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError("The _do_generation method has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def _select_best(self, pop):
        """Select the best individuals of the population.

        :param pop: The population
        :type pop: Any iterable type
        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        """
        hof = HallOfFame(self.pop_size)
        hof.update(pop)
        return hof

    def __load_checkpoint(self):
        """Load the state of the last checkpoint.

        :raises Exception: If the checkpoint file can't be loaded
        :return: The last saved population
        :rtype: :py:class:`list`
        :return: The last executed generation
        :rtype: :py:class:`int`
        :return: The last saved logbook
        :rtype: :py:class:`~deap.tools.Logbook`
        :return: The runtime until checkpoint was reached
        :rtype: :py:class:`float`
        """
        state = pandas.io.pickle.read_pickle(self.checkpoint_file)
        pop = state["population"]
        start_gen = state["generation"]
        logbook = state["logbook"]
        runtime = state["runtime"]
        random.setstate(state["rnd_state"])
        np.random.set_state(state["np_rnd_state"])

        return pop, start_gen, logbook, runtime

    def __save_checkpoint(self, pop, gen, logbook, runtime):
        """Save the state at a new checkpoint.

        :param pop: The last population
        :type pop: :py:class:`list`
        :param gen: The last executed generation
        :type gen: :py:class:`int`
        :param logbook: The last logbook
        :type logbook: :py:class:`~deap.tools.Logbook`
        :param runtime: The runtime until checkpoint was reached
        :type runtime: :py:class:`float`
        """
        # Fill in the dictionary with the wrapper state
        state = dict(population=pop,
                     generation=gen,
                     logbook=logbook,
                     runtime=runtime,
                     rnd_state=random.getstate(),
                     np_rnd_state=np.random.get_state())

        # Save the state
        pandas.io.pickle.to_pickle(state, self.checkpoint_file)

    def _search(self):
        """Apply the search algorithm.

        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        :return: A logbook with the statistics of the evolution
        :rtype: :py:class:`~deap.tools.Logbook`
        :return: The runtime of the algorithm
        :rtype: :py:class:`float`
        """
        # The population will be a list of individuals
        self._toolbox.register("population", initRepeat, list,
                               self._toolbox.individual)

        # Try to load the state of the last checkpoint
        try:
            pop, start_gen, logbook, runtime = self.__load_checkpoint()
        # If a checkpoint can't be loaded, start a new execution
        except Exception:
            # Create the initial population
            pop = self._toolbox.population(n=self.pop_size)

            # First generation
            start_gen = 0

            # Computing runtime
            runtime = 0

            # Create the logbook
            logbook = Logbook()
            logbook.header = list(self.stats_names) + \
                (self._stats.fields if self._stats else [])

            # Evaluate the individuals with an invalid fitness
            num_evals = self._eval(pop)

            # Compile statistics about the population
            self._do_stats(pop, start_gen, num_evals, logbook)

        # Run all the generations
        for gen in range(start_gen + 1, self.n_gens + 1):
            start_time = time.perf_counter()
            self._do_generation(pop, gen, logbook)
            end_time = time.perf_counter()
            runtime += end_time - start_time

            # Save the wrapper state at each checkpoint
            if gen % self.checkpoint_freq == 0:
                self.__save_checkpoint(pop, gen, logbook, runtime)

        # Save the last state
        self.__save_checkpoint(pop, self.n_gens, logbook, runtime)

        return self._select_best(pop), logbook, runtime


class EvolutionaryWrapper(GenerationalWrapper):
    """Base class for all the evolutionary wrapper methods.

    This class implements the simplest evolutionary algorithm.
    """

    def __init__(self, individual_cls, species, **params):
        """Initialize the wrapper method.

        :param individual_cls: Individual representation.
        :type individual_cls: Any subclass of
            :py:class:`~base.Individual`
        :param species: The species the individual will belong to
        :type species: Any sublass of :py:class:`~base.Species`
        :param pop_size: Population size, defaults to
            :py:attr:`~wrapper.DEFAULT_POP_SIZE`
        :type pop_size: :py:class:`int`, optional
        :param n_gens: Number of generations, defaults to
            :py:attr:`~wrapper.DEFAULT_N_GENS`
        :type n_gens: :py:class:`int`, optional
        :param xover_func: Crossover function, defaults to the
            :py:meth:`~base.Individual.crossover` method of
            *individual_cls*
        :type xover_func: Any callable object, optional
        :param xover_pb: Crossover rate, defaults to
            :py:attr:`~wrapper.DEFAULT_XOVER_PB`
        :type xover_pb: :py:class:`float`, optional
        :param mut_func: Mutation function, defaults to the
            :py:meth:`~base.Individual.mutate` method of
            *individual_cls*
        :type mut_func: Any callable object, optional
        :param mut_pb: Mutation rate, defaults to
            :py:attr:`~wrapper.DEFAULT_MUT_PB`
        :type mut_pb: :py:class:`float`, optional
        :param mut_ind_pb: Independent gene mutation probability, defaults to
            :py:attr:`~wrapper.DEFAULT_MUT_IND_PB`
        :type mut_ind_pb: :py:class:`float`, optional
        :param sel_func: Selection function, defaults to
            :py:attr:`~wrapper.DEFAULT_SEL_FUNC`
        :type sel_func: Any callable object, optional
        :param sel_func_params: Selection function parameters, defaults to
            :py:attr:`~wrapper.DEFAULT_SEL_FUNC_PARAMS`
        :type sel_func_params: :py:class:`dict`, optional
        :param checkpoint_freq: Frequency for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FREQ`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_file: File path for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FILE`
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
        """Get the selection function.

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
        """Get the selection function parameters.

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


class NSGAWrapper(EvolutionaryWrapper):
    """NSGA-based wrapper method.

    This class allows to run the NSGA2 or NSGA3 algorithm as the search method
    within the wrapper method.
    """

    def __init__(self, individual_cls, species, **params):
        """Initialize the wrapper method.

        :param individual_cls: Individual representation.
        :type individual_cls: Any subclass of
            :py:class:`~base.Individual`
        :param species: The species the individual will belong to
        :type species: Any sublass of :py:class:`~base.Species`
        :param pop_size: Population size, defaults to
            :py:attr:`~wrapper.DEFAULT_POP_SIZE`
        :type pop_size: :py:class:`int`, optional
        :param n_gens: Number of generations, defaults to
            :py:attr:`~wrapper.DEFAULT_N_GENS`
        :type n_gens: :py:class:`int`, optional
        :param xover_func: Crossover function, defaults to the
            :py:meth:`~base.Individual.crossover` method of
            *individual_cls*
        :type xover_func: Any callable object, optional
        :param xover_pb: Crossover rate, defaults to
            :py:attr:`~wrapper.DEFAULT_XOVER_PB`
        :type xover_pb: :py:class:`float`, optional
        :param mut_func: Mutation function, defaults to the
            :py:meth:`~base.Individual.mutate` method of
            *individual_cls*
        :type mut_func: Any callable object, optional
        :param mut_pb: Mutation rate, defaults to
            :py:attr:`~wrapper.DEFAULT_MUT_PB`
        :type mut_pb: :py:class:`float`, optional
        :param mut_ind_pb: Independent gene mutation probability, defaults to
            :py:attr:`~wrapper.DEFAULT_MUT_IND_PB`
        :type mut_ind_pb: :py:class:`float`, optional
        :param sel_func: Selection function (:py:func:`~deap.tools.selNSGA2` or
            :py:func:`~deap.tools.selNSGA3`), defaults to
            :py:attr:`~wrapper.DEFAULT_NSGA_SEL_FUNC`
        :type sel_func: Any callable object, optional
        :param sel_func_params: Selection function parameters. If NSGA-III is
            used, this attribute must include a key named *'ref_points'*
            containing a :py:class:`dict` with the parameters needed to
            generate the reference points (the arguments of
            :py:func:`~deap.tools.uniform_reference_points`). Since *sel_func*
            defaults to NSGA-II, the default value for *sel_func_params*
            is :py:attr:`~wrapper.DEFAULT_NSGA_SEL_FUNC_PARAMS`
        :type sel_func_params: :py:class:`dict`, optional
        :param checkpoint_freq: Frequency for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FREQ`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_file: File path for checkpointing, defaults to
            :py:attr:`~base.DEFAULT_WRAPPER_CHECKPOINT_FILE`
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

        # Register the selection operator
        self.sel_func = params.pop('sel_func', DEFAULT_NSGA_SEL_FUNC)
        self.sel_func_params = params.pop('sel_func_params',
                                          DEFAULT_NSGA_SEL_FUNC_PARAMS)
        ref_points = self.sel_func_params.pop('ref_points', None)

        # If NSGA3 is selected, the reference points are mandatory
        if self.sel_func is selNSGA3:
            # If sel_func_params doesn't define the ref points for NSGA-III
            if ref_points is None:
                raise ValueError("The reference points parameters are missing")

            ref_points = uniform_reference_points(**ref_points)
            self.sel_func_params['ref_points'] = ref_points

        self._toolbox.register("select", self.sel_func, **self.sel_func_params)

    def _do_generation(self, pop, gen_num, logbook):
        """Implement a generation of the search process.

        :param pop: The population
        :type pop: Any iterable type
        :param gen_num: The generation number
        :type gen_num: :py:class:`int`
        :param logbook: Logbook for the stats
        :type logbook: :py:class:`~deap.tools.Logbook`
        """
        offspring = varAnd(pop, self._toolbox, self.xover_pb, self.mut_pb)

        # Evaluate the individuals with an invalid fitness
        num_evals = self._eval(offspring)

        # Select the next generation population from parents and offspring
        pop[:] = self._toolbox.select(pop + offspring, self.pop_size)

        # Append the current generation statistics to the logbook
        self._do_stats(pop, gen_num, num_evals, logbook)

    def _select_best(self, pop):
        """Select the best indiividuals of the population.

        :param pop: The population
        :type pop: Any iterable type
        :return: The best individuals found
        :rtype: :py:class:`~deap.tools.HallOfFame` of individuals
        """
        hof = ParetoFront()
        hof.update(pop)
        return hof
