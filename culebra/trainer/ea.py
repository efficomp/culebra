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

"""Implementation of some evolutionary trainers.

This module provides some popular evolutionary algorithms:

  * The :class:`~culebra.trainer.ea.EA` class, which implements the simplest
    EA
  * The :class:`~culebra.trainer.ea.ElitistEA` class, which provides an
    elitist EA
  * The :class:`~culebra.trainer.ea.NSGA` class, which implements a
    multi-objective EA, based on Non-dominated sorting, able to run both
    the NSGA-II and the NSGA-III algorithms
"""

from __future__ import annotations

from functools import partial
from typing import Any
from collections.abc import Sequence, Callable
from itertools import repeat

from numpy import ndarray

from deap.base import Toolbox
from deap.algorithms import varAnd
from deap.tools import (
    HallOfFame,
    ParetoFront,
    selTournament,
    selNSGA2,
    selNSGA3,
    uniform_reference_points
)

from culebra.abc import Species, FitnessFunction
from culebra.solution.abc import Individual
from culebra.trainer.abc import CentralizedTrainer
from culebra.checker import (
    check_int,
    check_float,
    check_subclass,
    check_func
)

DEFAULT_POP_SIZE = 100
"""Default population size."""

DEFAULT_CROSSOVER_PROB = 0.8
"""Default crossover probability."""

DEFAULT_MUTATION_PROB = 0.2
"""Default mutation probability."""

DEFAULT_GENE_IND_MUTATION_PROB = 0.1
"""Default gene independent mutation probability."""

DEFAULT_SELECTION_FUNC_PARAMS = {'tournsize': 2}
"""Default selection function parameters."""

DEFAULT_SELECTION_FUNC = partial(
    selTournament, **DEFAULT_SELECTION_FUNC_PARAMS
)
"""Default selection function."""

DEFAULT_ELITE_SIZE = 5
"""Default number of elite individuals."""

DEFAULT_NSGA_SELECTION_FUNC = selNSGA2
"""Default selection function for NSGA-based algorithms."""

DEFAULT_NSGA3_REFERENCE_POINTS_P = 4
"""Default number of divisions along each objective for the reference points
of NSGA-III."""


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class EA(CentralizedTrainer):
    """Base class for all the evolutionary algorithms."""

    def __init__(
        self,
        fitness_func: FitnessFunction,
        solution_cls: type[Individual],
        species: Species,
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
        pop_size: int | None = None,
        custom_termination_func: Callable[[EA], bool] | None = None,
        max_num_iters: int | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_basename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param fitness_func: The training fitness function
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for the solutions
        :type species: ~culebra.abc.Species
        :param crossover_func: The crossover function. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_crossover_func` will be
            used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_mutation_func` will be
            used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_selection_func` will be
            used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. Must be in (0, 1).
            If omitted,
            :attr:`~culebra.trainer.ea.EA._default_crossover_prob` will be
            used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. Must be in (0, 1).
            If omitted,
            :attr:`~culebra.trainer.ea.EA._default_mutation_prob` will be
            used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. Must be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.EA._default_gene_ind_mutation_prob`
            will be used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param pop_size: The population size. Must be greater then zero. If
            omitted, :attr:`~culebra.trainer.ea.EA._default_pop_size` will
            be used. Defaults to :data:`None`
        :type pop_size: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.EA._default_termination_func` is
            used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_max_num_iters` will be
            used. Defaults to :data:`None`
        :type max_num_iters: int
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_checkpoint_freq` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_basename: The checkpoint base file path. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_checkpoint_basename`
            will be used. Defaults to :data:`None`
        :type checkpoint_basename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.EA._default_verbosity` will be used.
            Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__(
            fitness_func=fitness_func,
            solution_cls=solution_cls,
            species=species,
            custom_termination_func=custom_termination_func,
            max_num_iters=max_num_iters,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_basename=checkpoint_basename,
            verbosity=verbosity,
            random_seed=random_seed
        )

        # Get the parameters
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.selection_func = selection_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_ind_mutation_prob = gene_ind_mutation_prob
        self.pop_size = pop_size

    @CentralizedTrainer.solution_cls.setter
    def solution_cls(self, value: type[Individual]) -> None:
        """Set a new solution class.

        :param value: The new class
        :type value: type[~culebra.solution.abc.Individual]
        :raises TypeError: If *value* is not an
            :class:`~culebra.solution.abc.Individual` subclass
        """
        check_subclass(value, "solution class", Individual)
        CentralizedTrainer.solution_cls.fset(self, value)

    @property
    def _default_crossover_func(self) -> Callable[
        [Individual, Individual], tuple[Individual, Individual]
    ]:
        """Default crossover function.

        :return: The :meth:`~culebra.solution.abc.Individual.crossover` method
            of :attr:`~culebra.trainer.ea.EA.solution_cls`
        :rtype: ~collections.abc.Callable
        """
        return self.solution_cls.crossover

    @property
    def crossover_func(self) -> Callable[
        [Individual, Individual], tuple[Individual, Individual]
    ]:
        """Crossover function.

        :rtype: ~collections.abc.Callable
        :setter: Set the crossover function
        :param value: The new crossover function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_crossover_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        """
        return (
            self._default_crossover_func
            if self._crossover_func is None
            else self._crossover_func
        )

    @crossover_func.setter
    def crossover_func(
        self,
        value:
            Callable[
                [Individual, Individual], tuple[Individual, Individual]
            ] | None
    ) -> None:
        """Set a new crossover function.

        :param value: The new crossover function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_crossover_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        """
        self._crossover_func = (
            None if value is None else check_func(value, "crossover function")
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_mutation_func(self) -> Callable[
        [Individual, float], tuple[Individual]
    ]:
        """Default mutation function.

        :return: The :meth:`~culebra.solution.abc.Individual.mutate` method
            of :attr:`~culebra.trainer.ea.EA.solution_cls`
        :rtype: ~collections.abc.Callable
        """
        return self.solution_cls.mutate

    @property
    def mutation_func(self) -> Callable[
        [Individual, float], tuple[Individual]
    ]:
        """Mutation function.

        :rtype: ~collections.abc.Callable
        :setter: Set the mutation function
        :param value: The new mutation function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_mutation_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        """
        return (
            self._default_mutation_func
            if self._mutation_func is None
            else self._mutation_func
        )

    @mutation_func.setter
    def mutation_func(
        self,
        value:
            Callable[[Individual, float], tuple[Individual]] | None
    ) -> None:
        """Set a new mutation function.

        :param value: The new mutation function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_mutation_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        """
        self._mutation_func = (
            None if value is None else check_func(value, "mutation function")
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_selection_func(self) -> Callable[
        [list[Individual], int, Any], list[Individual]
    ]:
        """Default selection function.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_SELECTION_FUNC

    @property
    def selection_func(self) -> Callable[
        [list[Individual], int, Any], list[Individual]
    ]:
        """Selection function.

        :rtype: ~collections.abc.Callable
        :setter: Set the selection function
        :param value: The new selection function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_selection_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        """
        return (
            self._default_selection_func
            if self._selection_func is None
            else self._selection_func
        )

    @selection_func.setter
    def selection_func(
        self,
        value: Callable[[list[Individual], int, Any], list[Individual]] | None
    ) -> None:
        """Set a new selection function.

        :param value: The new selection function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_selection_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        """
        self._selection_func = (
            None if value is None else check_func(value, "selection function")
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_crossover_prob(self) -> float:
        """Default crossover probability.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB`
        :rtype: float
        """
        return DEFAULT_CROSSOVER_PROB

    @property
    def crossover_prob(self) -> float:
        """Crossover probability.

        :rtype: float
        :setter: Set the crossover probability
        :param value: The new crossover probability. Must be in (0, 1). If set
            to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_crossover_prob` is
            chosen
        :type value: float
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        return (
            self._default_crossover_prob
            if self._crossover_prob is None
            else self._crossover_prob
        )

    @crossover_prob.setter
    def crossover_prob(self, value: float | None) -> None:
        """Set a new crossover probability.

        :param value: The new crossover probability. Must be in (0, 1). If set
            to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_crossover_prob` is
            chosen
        :type value: float
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        self._crossover_prob = (
            None if value is None else check_float(
                value, "crossover probability", gt=0, lt=1
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_mutation_prob(self) -> float:
        """Default mutation probability.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB`
        :rtype: float
        """
        return DEFAULT_MUTATION_PROB

    @property
    def mutation_prob(self) -> float:
        """Mutation probability.

        :rtype: float
        :setter: Set the mutation probability
        :param value: The new mutation probability. Must be in (0, 1). If set
            to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_mutation_prob` is
            chosen
        :type value: float
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        return (
            self._default_mutation_prob
            if self._mutation_prob is None
            else self._mutation_prob
        )

    @mutation_prob.setter
    def mutation_prob(self, value: float | None) -> None:
        """Set a new mutation probability.

        :param value: The new mutation probability. Must be in (0, 1). If set
            to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_mutation_prob` is
            chosen
        :type value: float
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        self._mutation_prob = (
            None if value is None else check_float(
                value, "mutation probability", gt=0, lt=1
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_gene_ind_mutation_prob(self) -> float:
        """Default gene independent mutation probability.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
        :rtype: float
        """
        return DEFAULT_GENE_IND_MUTATION_PROB

    @property
    def gene_ind_mutation_prob(self) -> float:
        """Gene independent mutation probability.

        :rtype: float
        :setter: Set the gene independent mutation probability
        :param value: The new gene independent mutation probability. Must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_gene_ind_mutation_prob`
            is chosen
        :type value: float
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        return (
            self._default_gene_ind_mutation_prob
            if self._gene_ind_mutation_prob is None
            else self._gene_ind_mutation_prob
        )

    @gene_ind_mutation_prob.setter
    def gene_ind_mutation_prob(
        self,
        value: float | Sequence[float] | None
    ) -> None:
        """Set a new gene independent mutation probability.

        :param value: The new gene independent mutation probability. Must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_gene_ind_mutation_prob`
            is chosen
        :type value: float
        :raises TypeError: If *value* is not a real number
        :raises ValueError: If *value* is not in (0, 1)
        """
        self._gene_ind_mutation_prob = (
            None if value is None else check_float(
                value, "gene independent mutation probability", gt=0, lt=1
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_pop_size(self) -> int:
        """Default population size.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE`
        :rtype: int
        """
        return DEFAULT_POP_SIZE

    @property
    def pop_size(self) -> int:
        """Population size.

        :return: The population size
        :rtype: int
        :setter: Set the population size
        :param value: The new population size. Must be greater then zero. If
            set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_pop_size` is chosen
        :type value: int
        :raises TypeError: If *value* is not an :class:`int`
        :raises ValueError: If *value* is not greater than zero
        """
        return (
            self._default_pop_size
            if self._pop_size is None
            else self._pop_size
        )

    @pop_size.setter
    def pop_size(self, value: int | None) -> None:
        """Set the population size.

        :param value: The new population size. Must be greater then zero. If
            set to :data`None`,
            :attr:`~culebra.trainer.ea.EA._default_pop_size` is chosen
        :type value: int
        :raises TypeError: If *value* is not an :class:`int`
        :raises ValueError: If *value* is not greater than zero
        """
        self._pop_size = (
            None if value is None else check_int(
                value, "population size", gt=0
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def pop(self) -> list[Individual] | None:
        """Population.

        :return: The population or :data:`None` if it has not been generated
            yet
        :rtype: list[~culebra.solution.abc.Individual]
        """
        return self._pop

    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return (hof,)

    def select_representatives(self) -> list[Individual]:
        """Select representative solutions.

        This method is intended to be called within distributed trainers to
        make the implementation of migrations easier.

        :return: A list of solutions
        :rtype: list[~culebra.solution.abc.Individual]
        """
        if self.container:
            return self.container.representatives_selection_func(
                self.pop, self.container.num_representatives
                )

        return []

    def integrate_representatives(
        self, representatives: list[Individual]
    ) -> None:
        """Integrate representative solutions.

        This method is intended to be called within distributed trainers to
        make the implementation of migrations easier.

        :param representatives: A list of solutions
        :type representatives: list[~culebra.solution.abc.Individual]
        """
        self.pop.extend(representatives)

    def _evaluate_several(self, inds: Sequence[Individual]) -> None:
        """Evaluate the individuals that have an invalid fitness.

        :param inds: A sequence of individuals
        :type inds: ~collections.abc.Sequence[~culebra.solution.abc.Individual]
        """
        invalid_inds = [ind for ind in inds if not ind.fitness.is_valid]

        for ind in invalid_inds:
            self._current_iter_evals += self.evaluate(
                ind, self.fitness_func, self.index, self.cooperators
            )

    def _generate_pop(self) -> None:
        """Generate the initial population.

        The population is filled with random generated individuals.
        """
        self._pop = []
        for _ in repeat(None, self.pop_size):
            self._pop.append(
                self.solution_cls(
                    species=self.species,
                    fitness_cls=self.fitness_func.fitness_cls)
            )

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start training.

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
        self._toolbox.register("select", self.selection_func)

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the Deap's :class:`~deap.base.Toolbox`.
        """
        super()._reset_internals()
        self._toolbox = None

    def _get_state(self) -> dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :rtype: dict
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["pop"] = self.pop

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :param state: The last loaded state
        :type state: dict
        """
        # Set the state of the superclass
        super()._set_state(state)

        # Set the state of this class
        self._pop = state["pop"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to fill the population with evaluated random individuals.
        """
        # Call superclass to get an initial empty population
        super()._new_state()

        # Generate the initial population
        self._generate_pop()

        # Not really in the state,
        # but needed to evaluate the initial population
        self._current_iter_evals = 0

        # Evaluate the initial population and append its
        # statistics to the logbook
        # Since the evaluation of the initial population is performed
        # before the first iteration, fix self.current_iter = -1
        self._current_iter = -1
        self._evaluate_several(self.pop)
        self._update_logbook()
        self._num_evals += self._current_iter_evals
        self._current_iter += 1

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the initial population.
        """
        super()._reset_state()
        self._pop = None

    def _do_iteration(self) -> None:
        """Implement an iteration of the training process.

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
        self._evaluate_several(self.pop)

    def _get_objective_stats(self) -> dict:
        """Gather the objective stats.

        :return: The stats
        :rtype: dict
        """
        return self._stats.compile(self.pop) if self._stats else {}


class ElitistEA(EA):
    """Elitist evolutionary algorithm."""

    def __init__(
        self,
        fitness_func: FitnessFunction,
        solution_cls: type[Individual],
        species: Species,
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
        pop_size: int | None = None,
        elite_size: int | None = None,
        custom_termination_func: Callable[[ElitistEA], bool] | None = None,
        max_num_iters: int | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_basename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param fitness_func: The training fitness function
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
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
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param elite_size: Number of individuals that will be preserved
            as the elite. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_elite_size`
            will be used. Defaults to :data:`None`
        :type elite_size: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.ElitistEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_basename: The checkpoint base file path. If omitted,
            :attr:`~culebra.trainer.ea.ElitistEA._default_checkpoint_basename`
            will be used. Defaults to :data:`None`
        :type checkpoint_basename: str
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
            fitness_func=fitness_func,
            solution_cls=solution_cls,
            species=species,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            pop_size=pop_size,
            custom_termination_func=custom_termination_func,
            max_num_iters=max_num_iters,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_basename=checkpoint_basename,
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
        return (
            self._default_elite_size
            if self._elite_size is None
            else self._elite_size
        )

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
            None if value is None else check_int(value, "elite size", gt=0)
        )

        # Reset the trainer
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
        """Implement an iteration of the training process.

        In this case, the best
        :attr:`~culebra.trainer.ea.ElitistEA.elite_size` individuals of
        each iteration (the elite) are preserved for the next iteration. The
        breeding and selection are implemented as in the
        :class:`~culebra.trainer.ea.EA` trainer.
        """
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob)

        # Evaluate the individuals with an invalid fitness
        self._evaluate_several(offspring)

        # Update the elite
        self._elite.update(offspring)

        # Select the next iteration population from offspring and elite
        self.pop[:] = self._toolbox.select(
            offspring + list(self._elite), self.pop_size
        )


class NSGA(EA):
    """NSGA-based evolutionary algorithm.

    This class allows to run the NSGA-II or NSGA-III .
    """

    def __init__(
        self,
        fitness_func: FitnessFunction,
        solution_cls: type[Individual],
        species: Species,
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
        pop_size: int | None = None,
        nsga3_reference_points_p: int | None = None,
        nsga3_reference_points_scaling: float | None = None,
        custom_termination_func: Callable[[NSGA], bool] | None = None,
        max_num_iters: int | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_basename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param fitness_func: The training fitness function
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
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
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param nsga3_reference_points_p: Number of divisions along each
            objective to obtain the reference points of NSGA-III. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_nsga3_reference_points_p`
            will be used. Defaults to :data:`None`
        :type nsga3_reference_points_p: int
        :param nsga3_reference_points_scaling: Scaling factor for the reference
            points of NSGA-III. Defaults to :data:`None`
        :type nsga3_reference_points_scaling: float
        :param custom_termination_func: Custom termination criterion. If
            omitted, :meth:`~culebra.trainer.ea.NSGA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_basename: The checkpoint base file path. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_checkpoint_basename`
            will be used. Defaults to :data:`None`
        :type checkpoint_basename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.NSGA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            fitness_func=fitness_func,
            solution_cls=solution_cls,
            species=species,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            pop_size=pop_size,
            custom_termination_func=custom_termination_func,
            max_num_iters=max_num_iters,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_basename=checkpoint_basename,
            verbosity=verbosity,
            random_seed=random_seed
        )
        self.nsga3_reference_points_p = nsga3_reference_points_p
        self.nsga3_reference_points_scaling = nsga3_reference_points_scaling

    @property
    def _default_pop_size(self) -> int:
        """Default population size.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` for
            NSGA-II or the number of reference points for NSGA-III
        :rtype: int
        """
        if self.selection_func is selNSGA3:
            return len(self.nsga3_reference_points)
        return EA._default_pop_size.fget(self)

    @property
    def _default_selection_func(
        self
    ) -> Callable[[list[Individual], int, Any], list[Individual]]:
        """Default selection function.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_NSGA_SELECTION_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_NSGA_SELECTION_FUNC

    @EA.selection_func.setter
    def selection_func(
        self,
        value: Callable[[list[Individual], int, Any], list[Individual]] | None
    ) -> None:
        """Set a new selection function.

        :param value: The new selection function. If set to :data`None`,
            :attr:`~culebra.trainer.ea.NSGA._default_selection_func` is
            chosen
        :type value: ~collections.abc.Callable
        :raises TypeError: If *value* is not :class:`~collections.abc.Callable`
        :raises ValueError: If *value* is not :func:`~deap.tools.selNSGA2` or
            :func:`~deap.tools.selNSGA3`
        """
        EA.selection_func.fset(self, value)
        if self.selection_func not in [selNSGA2, selNSGA3]:
            raise ValueError(
                "Invalid selection function."
            )

    @property
    def nsga3_reference_points(self) -> ndarray:
        """Reference points for NSGA-III.

        :rtype: ~numpy.ndarray
        """
        return uniform_reference_points(
            nobj=self.fitness_func.num_obj,
            p=self.nsga3_reference_points_p,
            scaling=self.nsga3_reference_points_scaling
        )

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
        return (
            self._default_nsga3_reference_points_p
            if self._nsga3_reference_points_p is None
            else self._nsga3_reference_points_p
        )

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
            None if value is None else check_int(
                value, "NSGA-III p parameter", ge=1
            )
        )

        # Reset the trainer
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

        # Reset the trainer
        self.reset()

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start training.

        Overridden to support the NSGA-III reference points.
        """
        super()._init_internals()

        if self.selection_func is selNSGA3:
            # Unregister the selection function
            self._toolbox.unregister("select")

            # Register the selection function with the current reference points
            self._toolbox.register(
                "select",
                self.selection_func,
                ref_points=self.nsga3_reference_points
            )

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the NSGA-III reference points.
        """
        super()._reset_internals()
        self._nsga3_reference_points = None

    def _do_iteration(self) -> None:
        """Implement an iteration of the training process.

        In this case, a generation of NSGA is implemented.
        """
        # Generate the offspring
        offspring = varAnd(
            self.pop, self._toolbox, self.crossover_prob, self.mutation_prob
        )

        # Evaluate the individuals with an invalid fitness
        self._evaluate_several(offspring)

        # Select the next generation population from parents and offspring
        self.pop[:] = self._toolbox.select(
            self.pop + offspring, self.pop_size
        )


# Exported symbols for this module
__all__ = [
    'EA',
    'ElitistEA',
    'NSGA',
    'DEFAULT_POP_SIZE',
    'DEFAULT_CROSSOVER_PROB',
    'DEFAULT_MUTATION_PROB',
    'DEFAULT_GENE_IND_MUTATION_PROB',
    'DEFAULT_SELECTION_FUNC',
    'DEFAULT_ELITE_SIZE',
    'DEFAULT_NSGA_SELECTION_FUNC',
    'DEFAULT_NSGA3_REFERENCE_POINTS_P'
]
