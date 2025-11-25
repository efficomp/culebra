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

"""Implementation of single population evolutionary algorithms."""

from __future__ import annotations

from typing import Any, Type, Callable, Tuple, List, Dict, Optional
from collections.abc import Sequence
from copy import copy

from deap.base import Toolbox
from deap.algorithms import varAnd
from deap.tools import (
    HallOfFame,
    selNSGA2,
    selNSGA3,
    uniform_reference_points
)

from culebra.abc import Species, FitnessFunction
from culebra.solution.abc import Individual
from culebra.trainer.ea.abc import SinglePopEA
from culebra.checker import check_int, check_float


DEFAULT_ELITE_SIZE = 5
"""Default number of elite individuals."""

DEFAULT_NSGA_SELECTION_FUNC = selNSGA2
"""Default selection function for NSGA-based algorithms."""

DEFAULT_NSGA_SELECTION_FUNC_PARAMS = {}
"""Default selection function parameters for NSGA-based algorithms."""

DEFAULT_NSGA3_REFERENCE_POINTS_P = 4
"""Default number of divisions along each objective for the reference points
of NSGA-III."""


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class SimpleEA(SinglePopEA):
    """Implement the simplest evolutionary algorithm."""

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Overridden to create and initialize the Deap's
        :class:`~deap.base.Toolbox`.
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
        """Reset the internal structures of the trainer.

        Overridden to reset the Deap's :class:`~deap.base.Toolbox`.
        """
        super()._reset_internals()
        self._toolbox = None

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        In this case, the simplest evolutionary algorithm, as
        presented in chapter 7 of [Back2000]_, is implemented.

        .. [Back2000] T. Back, D. Fogel and Z. Michalewicz, eds. *Evolutionary
           Computation 1: Basic Algorithms and Operators*, CRC Press, 2000.

        """
        # Select the next iteration individuals
        self.pop[:] = self._toolbox.select(self.pop, self.pop_size)

        # Vary the pool of individuals
        self.pop[:] = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob
        )

        # Evaluate the individuals with an invalid fitness and append the
        # current iteration statistics to the logbook
        self._evaluate_pop(self.pop)


class ElitistEA(SimpleEA):
    """Elitist evolutionary algorithm."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[Callable[[ElitistEA], bool]] = None,
        pop_size: Optional[int] = None,
        crossover_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual, Individual]]
        ] = None,
        mutation_func: Optional[
            Callable[[Individual, float], Tuple[Individual]]
        ] = None,
        selection_func: Optional[
            Callable[[List[Individual], int, Any], List[Individual]]
        ] = None,
        crossover_prob: Optional[float] = None,
        mutation_prob: Optional[float] = None,
        gene_ind_mutation_prob: Optional[float] = None,
        selection_func_params: Optional[Dict[str, Any]] = None,
        elite_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If set to
            :data:`None`, the *solution_cls*
            :meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If set to
            :data:`None`, the *solution_cls*
            :meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param elite_size: Number of individuals that will be preserved
            as the elite. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_ELITE_SIZE` will be
            used. Defaults to :data:`None`
        :type elite_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            will be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`,
            :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` will be used.
            Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbose: The verbosity. If set to
            :data:`None`, :data:`__debug__` will be used. Defaults to
            :data:`None`
        :type verbose: bool
        :param random_seed: The seed, defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            solution_cls,
            species=species,
            fitness_function=fitness_function,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
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

        self.elite_size = elite_size

    @property
    def elite_size(self) -> int:
        """Elite size.

        :rtype: int
        :setter: Set a new value for the elite size
        :param value: The new size. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_ELITE_SIZE` is chosen
        :type value: int
        :raises TypeError: If *value* is not an int
        :raises ValueError: If *value* is lower than or equal to 0
        """
        return (
            DEFAULT_ELITE_SIZE if self._elite_size is None
            else self._elite_size
        )

    @elite_size.setter
    def elite_size(self, value: int | None) -> None:
        """Set a new value for the elite size.

        :param value: The new size. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_ELITE_SIZE` is chosen
        :type value: int
        :raises TypeError: If *value* is not an int
        :raises ValueError: If *value* is lower than or equal to 0
        """
        # Check the value
        self._elite_size = (
            None if value is None else check_int(value, "elite size", gt=0)
        )

        # Reset the algorithm
        self.reset()

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current elite to the trainer's state.

        :rtype: dict
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["elite"] = self._elite

        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current elite to the trainer's state.

        :param state: The last loaded state
        :type state: dict
        """
        # Set the state of the superclass
        super()._set_state(state)

        # Set the state of this class
        self._elite = state["elite"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to initialize the elite.
        """
        super()._new_state()

        # Create the elite
        self._elite = HallOfFame(maxsize=min(self.pop_size, self.elite_size))

        # Update the elite
        self._elite.update(self.pop)

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the elite.
        """
        super()._reset_state()
        self._elite = None

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        In this case, the best
        :attr:`~culebra.trainer.ea.ElitistEA.elite_size` individuals of
        each iteration (the elite) are preserved for the next iteration. The
        breeding and selection are implemented as in the
        :class:`~culebra.trainer.ea.SimpleEA` trainer.
        """
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob)

        # Evaluate the individuals with an invalid fitness
        self._evaluate_pop(offspring)

        # Update the elite
        self._elite.update(offspring)

        # Select the next iteration population from offspring and elite
        self.pop[:] = self._toolbox.select(
            offspring + [ind for ind in self._elite],
            self.pop_size
        )


class NSGA(SimpleEA):
    """NSGA-based evolutionary algorithm.

    This class allows to run the NSGA-II or NSGA-III .
    """

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[Callable[[NSGA], bool]] = None,
        pop_size: Optional[int] = None,
        crossover_func: Optional[
            Callable[[Individual, Individual], Tuple[Individual, Individual]]
        ] = None,
        mutation_func: Optional[
            Callable[[Individual, float], Tuple[Individual]]
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
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is used for
            NSGA-II or the number of reference points is chosen for NSGA-III.
            Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If set to
            :data:`None`, the *solution_cls*
            :meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If set to
            :data:`None`, the *solution_cls*
            :meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC`
            will be used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param nsga3_reference_points_p: Number of divisions along each
            objective to obtain the reference points of NSGA-III. If set to
            :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA3_REFERENCE_POINTS_P`
            will be used. Defaults to :data:`None`
        :type nsga3_reference_points_p: int
        :param nsga3_reference_points_scaling: Scaling factor for the reference
            points of NSGA-III. Defaults to :data:`None`
        :type nsga3_reference_points_scaling: float
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            will be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`,
            :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` will be used.
            Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbose: The verbosity. If set to
            :data:`None`, :data:`__debug__` will be used. Defaults to
            :data:`None`
        :type verbose: bool
        :param random_seed: The seed, defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            solution_cls,
            species=species,
            fitness_function=fitness_function,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
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

    @SimpleEA.pop_size.getter
    def pop_size(self) -> int:
        """Population size.

        :rtype: int
        :setter: Set a new population size
        :param size: The new population size. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is used for
            NSGA-II or the number of reference points is chosen for NSGA-III
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        if self._pop_size is None and self.selection_func is selNSGA3:
            the_pop_size = len(self.nsga3_reference_points)
        else:
            the_pop_size = super().pop_size

        return the_pop_size

    @SimpleEA.selection_func.getter
    def selection_func(
        self
    ) -> Callable[[List[Individual], int, Any], List[Individual]]:
        """Selection function.

        :rtype: ~collections.abc.Callable
        :setter: Set a new selection function
        :param func: The new selection function. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC` is
            chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return (
            DEFAULT_NSGA_SELECTION_FUNC
            if self._selection_func is None
            else self._selection_func
        )

    @SimpleEA.selection_func_params.getter
    def selection_func_params(self) -> Dict[str, Any]:
        """Parameters of the selection function.

        :rtype: dict
        :setter: Set new parameters for the selection function
        :param params: The new parameters. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC_PARAMS`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
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
        """Reference points for NSGA-III.

        :rtype: ~collections.abc.Sequence
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
        """NSGA-III's *p* parameter.

        The *p* parameter indicates the number of divisions to be made along
        each objective to obtain the reference points of NSGA-III.

        :rtype: int
        :setter: Set a new value for the *p* parameter for NSGA-III
        :param value: The new number of divisions. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA3_REFERENCE_POINTS_P`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* < 1
        """
        return (
            DEFAULT_NSGA3_REFERENCE_POINTS_P
            if self._nsga3_reference_points_p is None
            else self._nsga3_reference_points_p
        )

    @nsga3_reference_points_p.setter
    def nsga3_reference_points_p(self, value: int | None) -> None:
        """Set a new value for the *p* parameter for NSGA-III.

        The *p* parameter indicates the number of divisions to be made along
        each objective to obtain the reference points of NSGA-III.

        :param value: The new number of divisions. If set to :data:`None`,
            :attr:`~culebra.trainer.ea.DEFAULT_NSGA3_REFERENCE_POINTS_P`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* < 1
        """
        # Check the value
        self._nsga3_reference_points_p = (
            None if value is None else check_int(
                value, "NSGA-III p parameter", ge=1
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def nsga3_reference_points_scaling(self) -> float | None:
        """Scaling factor for the reference points of NSGA-III.

        :return: The scaling factor or :data:`None` if it is not defined
        :rtype: float
        :setter: Set a new scaling factor for the reference points of NSGA-III
        :param value: New scaling factor
        :type value: float
        :raises TypeError: If *value* is not a float number
        """
        return self._nsga3_reference_points_scaling

    @nsga3_reference_points_scaling.setter
    def nsga3_reference_points_scaling(self, value: float | None) -> None:
        """Set a new scaling factor for the reference points of NSGA-III.

        :param value: New scaling factor
        :type value: float
        :raises TypeError: If *value* is not a float number
        """
        # Check the value
        self._nsga3_reference_points_scaling = (
            None if value is None else check_float(
                value, "NSGA-III reference points scaling factor"
            )
        )

        # Reset the algorithm
        self.reset()

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Overridden to support the NSGA-III reference points.
        """
        super()._init_internals()

        # Generated when property nsga3_reference_points is invoked
        self._nsga3_ref_points = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the NSGA-III reference points.
        """
        super()._reset_internals()
        self._nsga3_ref_points = None

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        In this case, a generation of NSGA is implemented.
        """
        # Generate the offspring
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob
        )

        # Evaluate the individuals with an invalid fitness
        self._evaluate_pop(offspring)

        # Select the next generation population from parents and offspring
        self.pop[:] = self._toolbox.select(
            self.pop[:] + offspring, self.pop_size
        )


# Exported symbols for this module
__all__ = [
    'SimpleEA',
    'ElitistEA',
    'NSGA',
    'DEFAULT_ELITE_SIZE',
    'DEFAULT_NSGA_SELECTION_FUNC',
    'DEFAULT_NSGA_SELECTION_FUNC_PARAMS',
    'DEFAULT_NSGA3_REFERENCE_POINTS_P'
]
