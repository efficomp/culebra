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

"""Provides the :py:class:`~wrapper.generational_wrapper.GenerationalWrapper`
class.

This class is the base class for all the generational wrapper procedures
developed within the Wrapper subpackage. It is an abstract class. The
:py:meth:`~wrapper.generational_wrapper.GenerationalWrapper._do_generation`
method must be implemented in its subclasses to obtain a complete generational
algorithm.
"""

import time
import numbers
import random
import numpy as np
import pandas.io
from deap import base
from deap.tools import initRepeat, HallOfFame, Logbook
from culebra.base.wrapper import Wrapper

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


class GenerationalWrapper(Wrapper):
    """Base class for all the generational wrapper methods."""

    stats_names = DEFAULT_STATS_NAMES
    """Statistics calculated for each generation."""

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

        # Get the population size
        self.pop_size = params.pop('pop_size', DEFAULT_POP_SIZE)

        # Get the number of generations
        self.n_gens = params.pop('n_gens', DEFAULT_N_GENS)

        # Initialize the toolbox of DEAP
        self._toolbox = base.Toolbox()

    @property
    def pop_size(self):
        """Population size.

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
        """Number of generations.

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
        """Application of the search algorithm.

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
