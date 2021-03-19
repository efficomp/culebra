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

"""Provides the :py:class:`~wrapper.nsga_wrapper.NSGAWrapper`
class.

This class allows to run the NSGA2 or NSGA3 algorithm as the search method
within the wrapper method.
"""
from deap.tools import selNSGA2, selNSGA3, ParetoFront
from deap.tools import uniform_reference_points
from deap.algorithms import varAnd
from culebra.wrapper.evolutionary_wrapper import EvolutionaryWrapper

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_SEL_FUNC = selNSGA2
"""Default selection function."""

DEFAULT_SEL_FUNC_PARAMS = {}
"""Default selection function parameters."""


class NSGAWrapper(EvolutionaryWrapper):
    """NSGA-based wrapper method."""

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
        :param sel_func: Selection function (:py:func:`~deap.tools.selNSGA2` or
            :py:func:`~deap.tools.selNSGA3`), defaults to
            :py:attr:`~wrapper.nsga_wrapper.DEFAULT_SEL_FUNC`
        :type sel_func: Any callable object, optional
        :param sel_func_params: Selection function parameters. If NSGA-III is
            used, this attribute must include a key named *'ref_points'*
            containing a :py:class:`dict` with the parameters needed to
            generate the reference points (the arguments of
            :py:func:`~deap.tools.uniform_reference_points`). Since *sel_func*
            defaults to NSGA-II, the default value for *sel_func_params*
            is :py:attr:`~wrapper.nsga_wrapper.DEFAULT_SEL_FUNC_PARAMS`
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

        # Register the selection operator
        self.sel_func = params.pop('sel_func', DEFAULT_SEL_FUNC)
        self.sel_func_params = params.pop('sel_func_params',
                                          DEFAULT_SEL_FUNC_PARAMS)
        ref_points = self.sel_func_params.pop('ref_points', None)

        # If NSGA3 is selected, the reference points are mandatory
        if self.sel_func is selNSGA3:
            # If sel_func_params doesn't define the ref points for NSGA-III
            if ref_points is None:
                raise ValueError("The reference points parameters are missing")
            else:
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
