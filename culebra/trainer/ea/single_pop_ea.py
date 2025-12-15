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

from typing import Any
from collections.abc import Callable

from numpy import ndarray

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
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[ElitistEA], bool] | None = None,
        pop_size: int | None = None,
        crossover_func:
            Callable[[Individual, Individual], tuple[Individual, Individual]] |
            None = None,
        mutation_func:
            Callable[[Individual, float], tuple[Individual]] | None = None,
        selection_func:
            Callable[[list[Individual], int, Any], list[Individual]] |
            None = None,
        crossover_prob: float | None = None,
        mutation_prob: float | None = None,
        gene_ind_mutation_prob: float | None = None,
        selection_func_params: dict[str, Any] | None = None,
        elite_size: int | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.ElitistEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_crossover_func`
            is used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_mutation_func`
            is used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_selection_func`
            is used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_crossover_prob`
            is used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_mutation_prob`
            is used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_gene_ind_mutation_prob`
            is used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_selection_func_params`
            is used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param elite_size: Number of individuals that will be preserved
            as the elite. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_elite_size`
            will be used. Defaults to :data:`None`
        :type elite_size: int
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

        self.elite_size = elite_size

    @property
    def _default_elite_size(self) -> int:
        """Default elite size.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_ELITE_SIZE`
        :rtype: int
        """
        return DEFAULT_ELITE_SIZE

    @property
    def elite_size(self) -> int:
        """Elite size.

        :rtype: int
        :setter: Set a new value for the elite size
        :param value: The new size. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_elite_size` is chosen
        :type value: int
        :raises TypeError: If *value* is not an int
        :raises ValueError: If *value* is lower than or equal to 0
        """
        return self._elite_size

    @elite_size.setter
    def elite_size(self, value: int | None) -> None:
        """Set a new value for the elite size.

        :param value: The new size. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_elite_size` is chosen
        :type value: int
        :raises TypeError: If *value* is not an int
        :raises ValueError: If *value* is lower than or equal to 0
        """
        # Check the value
        self._elite_size = (
            self._default_elite_size
            if value is None else check_int(value, "elite size", gt=0)
        )

        # Reset the algorithm
        self.reset()

    def _get_state(self) -> dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current elite to the trainer's state.

        :rtype: dict
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["elite"] = self._elite

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
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
            offspring + list(self._elite), self.pop_size
        )


class NSGA(SimpleEA):
    """NSGA-based evolutionary algorithm.

    This class allows to run the NSGA-II or NSGA-III .
    """

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[NSGA], bool] | None = None,
        pop_size: int | None = None,
        crossover_func:
            Callable[[Individual, Individual], tuple[Individual, Individual]] |
            None = None,
        mutation_func:
            Callable[[Individual, float], tuple[Individual]] | None = None,
        selection_func:
            Callable[[list[Individual], int, Any], list[Individual]] |
            None = None,
        crossover_prob: float | None = None,
        mutation_prob: float | None = None,
        gene_ind_mutation_prob: float | None = None,
        selection_func_params: dict[str, Any] | None = None,
        nsga3_reference_points_p: int | None = None,
        nsga3_reference_points_scaling: float | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted, :meth:`~culebra.trainer.ea.NSGA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_crossover_func`
            is used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_mutation_func`
            is used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_selection_func`
            is used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_crossover_prob`
            is used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_mutation_prob`
            is used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_gene_ind_mutation_prob`
            is used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_selection_func_params`
            is used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param nsga3_reference_points_p: Number of divisions along each
            objective to obtain the reference points of NSGA-III. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_nsga3_reference_points_p`
            will be used. Defaults to :data:`None`
        :type nsga3_reference_points_p: int
        :param nsga3_reference_points_scaling: Scaling factor for the reference
            points of NSGA-III. Defaults to :data:`None`
        :type nsga3_reference_points_scaling: float
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # The selection function and all the NSGA related stuff
        # must be set before calling the superclass constructor because the
        # _default_pop_size property depends on it
        self.selection_func = selection_func
        self.nsga3_reference_points_p = nsga3_reference_points_p
        self.nsga3_reference_points_scaling = nsga3_reference_points_scaling

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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

    @property
    def _default_pop_size(self) -> int:
        """Default population size.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` for
            NSGA-II or the number of reference points for NSGA-III
        :rtype: int
        """
        if self.selection_func is selNSGA3:
            return len(self.nsga3_reference_points)
        return SimpleEA._default_pop_size.fget(self)

    @property
    def _default_selection_func(
        self
    ) -> Callable[[list[Individual], int, Any], list[Individual]]:
        """Default selection function.
        self.nsga3_reference_points_p = nsga3_reference_points_p
        self.nsga3_reference_points_scaling = nsga3_reference_points_scaling

        :return: :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC`

        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_NSGA_SELECTION_FUNC

    @property
    def _default_selection_func_params(self) -> dict[str, Any]:
        """Parameters of the default selection function.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC_PARAMS`
        :rtype: float
        """
        return DEFAULT_NSGA_SELECTION_FUNC_PARAMS

    @SimpleEA.selection_func_params.setter
    def selection_func_params(self, params: dict[str, Any] | None) -> None:
        """Set new parameters for the selection function.

        The reference points are also appended in case NSGA3 is used.

        :param params: The new parameters. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_selection_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        SimpleEA.selection_func_params.fset(self, params)

        if self.selection_func is selNSGA3:
            self._selection_func_params['ref_points'] = (
                self.nsga3_reference_points
            )

    @property
    def nsga3_reference_points(self) -> ndarray:
        """Reference points for NSGA-III.

        :rtype: ~numpy.ndarray
        """
        if self._nsga3_reference_points is None:
            self._nsga3_reference_points = uniform_reference_points(
                nobj=self.fitness_function.num_obj,
                p=self.nsga3_reference_points_p,
                scaling=self.nsga3_reference_points_scaling
            )
        return self._nsga3_reference_points

    @property
    def _default_nsga3_reference_points_p(self) -> int:
        """Default NSGA-III's *p* parameter.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_NSGA3_REFERENCE_POINTS_P`
        :rtype: int
        """
        return DEFAULT_NSGA3_REFERENCE_POINTS_P

    @property
    def nsga3_reference_points_p(self) -> int:
        """NSGA-III's *p* parameter.

        The *p* parameter indicates the number of divisions to be made along
        each objective to obtain the reference points of NSGA-III.

        :rtype: int
        :setter: Set a new value for the *p* parameter for NSGA-III
        :param value: The new number of divisions. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_nsga3_reference_points_p`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* < 1
        """
        return self._nsga3_reference_points_p

    @nsga3_reference_points_p.setter
    def nsga3_reference_points_p(self, value: int | None) -> None:
        """Set a new value for the *p* parameter for NSGA-III.

        :param value: The new number of divisions. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_nsga3_reference_points_p`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* < 1
        """
        # Check the value
        self._nsga3_reference_points_p = (
            self._default_nsga3_reference_points_p
            if value is None else check_int(
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
        self._nsga3_reference_points = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the NSGA-III reference points.
        """
        super()._reset_internals()
        self._nsga3_reference_points = None

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        In this case, a generation of NSGA is implemented.
        """
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob
        )

        # Evaluate the individuals with an invalid fitness
        self._evaluate_pop(offspring)

        # Select the next generation population from parents and offspring
        self.pop[:] = self._toolbox.select(
            self.pop + offspring, self.pop_size
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
