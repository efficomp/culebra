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

"""NSGA-based wrapper method."""

from __future__ import annotations
from typing import Any, Type, Callable, Tuple, List, Dict, Optional
from collections.abc import Sequence
from copy import copy
from deap.algorithms import varAnd
from deap.tools import (
    selNSGA2,
    selNSGA3,
    uniform_reference_points,
    HallOfFame,
    ParetoFront
)
from culebra.base import (
    Individual,
    Species,
    FitnessFunction,
    check_int,
    check_float
)
from . import Evolutionary

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_NSGA_SELECTION_FUNC = selNSGA2
"""Default selection function for NSGA-based algorithms."""

DEFAULT_NSGA_SELECTION_FUNC_PARAMS = {}
"""Default selection function parameters for NSGA-based algorithms."""

DEFAULT_NSGA3_REFERENCE_POINTS_P = 4
"""Default number of divisions along each objective for the reference points
of NSGA-III."""


class NSGA(Evolutionary):
    """NSGA-based wrapper method.

    This class allows to run the NSGA2 or NSGA3 algorithm as the search method
    within the wrapper method.
    """

    def __init__(
        self,
        individual_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        num_gens: Optional[int] = None,
        pop_size: Optional[int] = None,
        crossover_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual, Individual]]
        ] = None,
        mutation_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual]]
        ] = None,
        selection_func: Optional[
            Callable[[List[Individual], int, Any], List[Individual]]
        ] = None,
        crossover_prob: Optional[float] = None,
        mutation_prob: Optional[float] = None,
        gene_ind_mutation_prob: Optional[float] = None,
        selection_func_params: Optional[Dict[str, Any]] = None,
        nsga3_reference_points_p: Optional[int] = None,
        nsga3_reference_points_scaling: Optional[float] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new wrapper.

        :param individual_cls: The individual class
        :type individual_cls: An :py:class:`~base.Individual` subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~base.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param pop_size: The population size. If set to
            :py:data:`None`, :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE`
            is used for nsga2 or the number of reference points is chosen for
            nsga3.
            Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
        :param crossover_func: The crossover function. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.crossover` method will be used.
            Defaults to :py:data:`None`
        :type crossover_func: :py:class:`~collections.abc.Callable`, optional
        :param mutation_func: The mutation function. If set to
            :py:data:`None`, the *individual_cls*
            :py:meth:`~base.Individual.mutate` method will be used. Defaults
            to :py:data:`None`
        :type mutation_func: :py:class:`~collections.abc.Callable`, optional
        :param selection_func: The selection function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_func: :py:class:`~collections.abc.Callable`, optional
        :param crossover_prob: The crossover probability. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_prob: :py:class:`float` in (0, 1), optional
        :param mutation_prob: The mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` will be used.
            Defaults to :py:data:`None`
        :type mutation_prob: :py:class:`float` in (0, 1), optional
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_prob: :py:class:`float` in (0, 1), optional
        :param selection_func_params: The parameters for the selection
            function. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_func_params: :py:class:`dict`, optional
        :param nsga3_reference_points_p: Number of divisions along each
            objective to obtain the reference points of NSGA3. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA3_REFERENCE_POINTS_P`
            will be used. Defaults to :py:data:`None`
        :type nsga3_reference_points_p: :py:class:`int`, optional
        :param nsga3_reference_points_scaling: Scaling factor for the reference
            points of NSGA3. Defaults to :py:data:`None`
        :type nsga3_reference_points_scaling: :py:class:`float`, optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FILENAME` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            individual_cls,
            species=species,
            fitness_function=fitness_function,
            num_gens=num_gens,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        self.nsga3_reference_points_p = nsga3_reference_points_p
        self.nsga3_reference_points_scaling = nsga3_reference_points_scaling

    @Evolutionary.pop_size.getter
    def pop_size(self) -> int:
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size. If set to
            :py:data:`None`, :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE`
            is used for nsga2 or the number of reference points is chosen for
            nsga3
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        if self._pop_size is None and self.selection_func is selNSGA3:
            the_pop_size = len(self.nsga3_reference_points)
        else:
            the_pop_size = super().pop_size

        return the_pop_size

    @Evolutionary.selection_func.getter
    def selection_func(
        self
    ) -> Callable[[List[Individual], int, Any], List[Individual]]:
        """Get and set the selection function.

        :getter: Return the current selection function
        :setter: Set the new selection function. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA_SELECTION_FUNC` is
            chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            DEFAULT_NSGA_SELECTION_FUNC
            if self._selection_func is None
            else self._selection_func
        )

    @Evolutionary.selection_func_params.getter
    def selection_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the selection function.

        :getter: Return the current parameters for the selection function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA_SELECTION_FUNC_PARAMS`
            is chosen
        :type: A :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        func_params = (
            DEFAULT_NSGA_SELECTION_FUNC_PARAMS
            if self._selection_func_params is None
            else self._selection_func_params)

        if self.selection_func is selNSGA3:
            func_params = copy(func_params)
            # Assign the ref_points to the selNSGA3 parameters
            func_params['ref_points'] = self.nsga3_reference_points

        return func_params

    @property
    def nsga3_reference_points(self) -> Sequence:
        """Get the set of reference points for NSGA3.

        :type: :py:class:`~collections.abc.Sequence`
        """
        if self._nsga3_ref_points is None:
            # Obtain the reference points
            self._nsga3_ref_points = uniform_reference_points(
                nobj=self.fitness_function.num_obj,
                p=self.nsga3_reference_points_p,
                scaling=self.nsga3_reference_points_scaling
            )

        return self._nsga3_ref_points

    @property
    def nsga3_reference_points_p(self) -> int:
        """Get and set the *p* parameter for NSGA3.

        The *p* parameter indicates the number of divisions to be made along
        each objective to obtain the reference points of NSGA3.

        :getter: Return the number of divisions
        :setter: Set a new number of divisions. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA3_REFERENCE_POINTS_P` is
            chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value lower than 1
        """
        return (
            DEFAULT_NSGA3_REFERENCE_POINTS_P
            if self._nsga3_reference_points_p is None
            else self._nsga3_reference_points_p
        )

    @nsga3_reference_points_p.setter
    def nsga3_reference_points_p(self, value: int | None) -> None:
        """Set a new value for the *p* parameter for NSGA3.

        The *p* parameter indicates the number of divisions to be made along
        each objective to obtain the reference points of NSGA3.

        :param value: New number of divisions. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NSGA3_REFERENCE_POINTS_P` is
            chosen
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* < 1
        """
        # Check the value
        self._nsga3_reference_points_p = (
            None if value is None else check_int(
                value, "NSGA3 p parameter", ge=1
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def nsga3_reference_points_scaling(self) -> float | None:
        """Get and set the scaling factor for the reference points of NSGA3.

        :getter: Return the scaling factor
        :setter: Set a new scaling factor
        :type: :py:class:`float` or :py:data:`None`
        :raises TypeError: If set to a value which is not a float number
        """
        return self._nsga3_reference_points_scaling

    @nsga3_reference_points_scaling.setter
    def nsga3_reference_points_scaling(self, value: float | None) -> None:
        """Set a new scaling factor for the reference points of NSGA3.

        :param value: New scaling factor
        :type value: :py:class:`float` or :py:data:`None`
        :raises TypeError: If *value* is not a float number
        """
        # Check the value
        self._nsga3_reference_points_scaling = (
            None if value is None else check_float(
                value, "NSGA3 reference points scaling factor"
            )
        )

        # Reset the algorithm
        self.reset()

    def _init_internals(self) -> None:
        """Set up the wrapper internal data structures to start searching.

        Overriden to support the NSGA3 reference points.
        """
        super()._init_internals()

        # Generated when property nsga3_reference_points is invoked
        self._nsga3_ref_points = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the wrapper.

        Overriden to reset the NSGA3 reference points.
        """
        super()._reset_internals()
        self._nsga3_ref_points = None

    def best_solutions(self) -> List[HallOfFame]:
        """Get the best individuals found for each species.

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return [hof]

    def _do_generation(self) -> None:
        """Implement a generation of the search process.

        In this case, a generation of the NSGA is implemented.
        """
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob
        )

        # Evaluate the individuals with an invalid fitness
        self._evaluate_pop(offspring)
        self._do_generation_stats(offspring)

        # Select the next generation population from parents and offspring
        self.pop[:] = self._toolbox.select(
            self.pop[:] + offspring, self.pop_size
        )


# Exported symbols for this module
__all__ = [
    'NSGA',
    'DEFAULT_NSGA_SELECTION_FUNC',
    'DEFAULT_NSGA_SELECTION_FUNC_PARAMS',
    'DEFAULT_NSGA3_REFERENCE_POINTS_P'
]
