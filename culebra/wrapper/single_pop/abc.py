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

"""Base class for single-population evolutionary algorithms."""

from __future__ import annotations
from typing import Any, Type, Callable, Tuple, List, Dict, Optional
from copy import deepcopy
from itertools import repeat
from deap.tools import selTournament, HallOfFame
from culebra.base import (
    Individual,
    Species,
    FitnessFunction,
    check_subclass,
    check_instance,
    check_int,
    check_float,
    check_func,
    check_func_params
)
from culebra.wrapper import Generational

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_POP_SIZE = 100
"""Default population size."""

DEFAULT_CROSSOVER_PROB = 0.9
"""Default crossover probability."""

DEFAULT_MUTATION_PROB = 0.1
"""Default mutation probability."""

DEFAULT_GENE_IND_MUTATION_PROB = 0.1
"""Default gene independent mutation probability."""

DEFAULT_SELECTION_FUNC = selTournament
"""Default selection function."""

DEFAULT_SELECTION_FUNC_PARAMS = {'tournsize': 2}
"""Default selection function parameters."""


class SinglePop(Generational):
    """Base class for all the single population wrapper methods."""

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
        :param num_gens: The number of generations. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_NUM_GENS` will be used.
            Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param pop_size: The populaion size. If set to
            :py:data:`None`, :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE`
            will be used. Defaults to :py:data:`None`
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
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` will be
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
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` will
            be used. Defaults to :py:data:`None`
        :type selection_func_params: :py:class:`dict`, optional
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
        # Init the superclass
        super().__init__(
            fitness_function=fitness_function,
            num_gens=num_gens,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.individual_cls = individual_cls
        self.species = species
        self.pop_size = pop_size
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.selection_func = selection_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_ind_mutation_prob = gene_ind_mutation_prob
        self.selection_func_params = selection_func_params

    @property
    def individual_cls(self) -> Type[Individual]:
        """Get and set the individual class.

        :getter: Return the individual class
        :setter: Set a new individual class
        :type: An :py:class:`~base.Individual` subclass
        :raises TypeError: If set to a value which is not an
            :py:class:`~base.Individual` subclass
        """
        return self._individual_cls

    @individual_cls.setter
    def individual_cls(self, cls: Type[Individual]) -> None:
        """Set a new individual class.

        :param cls: The new class
        :type cls: An :py:class:`~base.Individual` subclass
        :raises TypeError: If *cls* is not an :py:class:`~base.Individual`
        """
        # Check cls
        self._individual_cls = check_subclass(
            cls, "individual class", Individual
        )

        # Reset the algorithm
        self.reset()

    @property
    def species(self) -> Species:
        """Get and set the species.

        :getter: Return the species
        :setter: Set a new species
        :type: :py:class:`~base.Species`
        :raises TypeError: If set to a value which is not a
            :py:class:`~base.Species` instance
        """
        return self._species

    @species.setter
    def species(self, value: Species) -> None:
        """Set a new species.

        :param value: The new species
        :type value: :py:class:`~base.Species`
        :raises TypeError: If *value* is not a :py:class:`~base.Species`
            instance
        """
        # Check the value
        self._species = check_instance(value, "species", Species)

        # Reset the algorithm
        self.reset()

    @property
    def pop_size(self) -> int:
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size. If set to
            :py:data:`None`, :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE`
            is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        return DEFAULT_POP_SIZE if self._pop_size is None else self._pop_size

    @pop_size.setter
    def pop_size(self, size: int | None) -> None:
        """Set the population size.

        :param size: The new population size. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE` is chosen
        :type size: :py:class:`int`, greater than zero. If set to
            :py:data:`None`, :py:attr:`~wrapper.single_pop.DEFAULT_POP_SIZE`
            is used
        :raises TypeError: If *size* is not an :py:class:`int`
        :raises ValueError: If *size* is not an integer greater than zero
        """
        # Check the value
        self._pop_size = (
            None if size is None else check_int(size, "population size", gt=0)
        )

        # Reset the algorithm
        self.reset()

    @property
    def crossover_func(self) -> Callable[
            [Individual, Individual],
            Tuple[Individual, Individual]
    ]:
        """Get and set the crossover function.

        :getter: Return the current crossover function
        :setter: Set a new crossover function. If set to :py:data:`None`,
            the :py:meth:`~base.Individual.crossover` method of the individual
            class evolved by the wrapper is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            self.individual_cls.crossover
            if self._crossover_func is None
            else self._crossover_func
        )

        # Reset the algorithm
        self.reset()

    @crossover_func.setter
    def crossover_func(
        self,
        func: Callable[
            [Individual, Individual],
            Tuple[Individual, Individual]
        ] | None
    ) -> None:
        """Set the crossover function.

        :param func: The new crossover function. If set to :py:data:`None`,
            the :py:meth:`~base.Individual.crossover` method of the individual
            class evolved by the wrapper is chosen
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._crossover_func = (
            None if func is None else check_func(func, "crossover function")
        )

        # Reset the algorithm
        self.reset()

    @property
    def mutation_func(self) -> Callable[
        [Individual, Individual],
        Tuple[Individual]
    ]:
        """Get and set the mutation function.

        :getter: Return the current mutation function
        :setter: Set a new mutation function. If set to :py:data:`None`, the
            :py:meth:`~base.Individual.mutate` method of the individual
            class evolved by the wrapper is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            self.individual_cls.mutate
            if self._mutation_func is None
            else self._mutation_func
        )

    @mutation_func.setter
    def mutation_func(
        self,
        func: Callable[
            [Individual, Individual],
            Tuple[Individual]
        ] | None
    ) -> None:
        """Set the mutation function.

        :param func: The new mutation function. If set to :py:data:`None`, the
            :py:meth:`~base.Individual.mutate` method of the individual
            class evolved by the wrapper is chosen
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._mutation_func = (
            None if func is None else check_func(func, "mutation function")
        )

        # Reset the algorithm
        self.reset()

    @property
    def selection_func(
        self
    ) -> Callable[[List[Individual], int, Any], List[Individual]]:
        """Get and set the selection function.

        :getter: Return the current selection function
        :setter: Set the new selection function. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            DEFAULT_SELECTION_FUNC
            if self._selection_func is None
            else self._selection_func
        )

    @selection_func.setter
    def selection_func(
        self,
        func: Callable[
            [List[Individual], int, Any],
            List[Individual]
        ] | None
    ) -> None:
        """Set a new selection function.

        :param func: The new selection function. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC` is chosen
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._selection_func = (
            None if func is None else check_func(func, "selection function")
        )

        # Reset the algorithm
        self.reset()

    @property
    def crossover_prob(self) -> float:
        """Get and set the crossover probability.

        :getter: Return the current crossover probability
        :setter: Set the new crossover probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` is chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return (
            DEFAULT_CROSSOVER_PROB
            if self._crossover_prob is None
            else self._crossover_prob)

    @crossover_prob.setter
    def crossover_prob(self, prob: float | None) -> None:
        """Set a new crossover probability.

        :param prob: The new probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_CROSSOVER_PROB` is chosen
        :type prob: :py:class:`float` in (0, 1)
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._crossover_prob = (
            None if prob is None else check_float(
                prob, "crossover probability", gt=0, lt=1)
        )

        # Reset the algorithm
        self.reset()

    @property
    def mutation_prob(self) -> float:
        """Get and set the mutation probability.

        :getter: Return the current mutation probability
        :setter: Set the new mutation probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` is chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return (
            DEFAULT_MUTATION_PROB
            if self._mutation_prob is None
            else self._mutation_prob
        )

    @mutation_prob.setter
    def mutation_prob(self, prob: float | None) -> None:
        """Set a new mutation probability.

        :param prob: The new probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_MUTATION_PROB` is chosen
        :type prob: :py:class:`float` in (0, 1)
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._mutation_prob = (
            None if prob is None else check_float(
                prob, "mutation probability", gt=0, lt=1
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def gene_ind_mutation_prob(self) -> float:
        """Get and set the gene independent mutation probability.

        :getter: Return the current gene independent mutation probability
        :setter: Set the new gene independent mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB`
            is chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return (
            DEFAULT_GENE_IND_MUTATION_PROB
            if self._gene_ind_mutation_prob is None
            else self._gene_ind_mutation_prob)

    @gene_ind_mutation_prob.setter
    def gene_ind_mutation_prob(self, prob: float | None) -> None:
        """Set a new gene independent mutation probability.

        :param prob: The new probability. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_GENE_IND_MUTATION_PROB` is
            chosen
        :type prob: :py:class:`float` in (0, 1)
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._gene_ind_mutation_prob = (
            None if prob is None else check_float(
                prob, "gene independent mutation probability", gt=0, lt=1
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def selection_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the selection function.

        :getter: Return the current parameters for the selection function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` is
            chosen
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return (
            DEFAULT_SELECTION_FUNC_PARAMS
            if self._selection_func_params is None
            else self._selection_func_params)

    @selection_func_params.setter
    def selection_func_params(self, params: Dict[str, Any] | None) -> None:
        """Set the parameters for the selection function.

        :param params: The new parameters. If set to :py:data:`None`,
            :py:attr:`~wrapper.single_pop.DEFAULT_SELECTION_FUNC_PARAMS` is
            chosen
        :type params: :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        """
        # Check params
        self._selection_func_params = (
            None if params is None else check_func_params(
                params, "selection function parameters"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def pop(self) -> List[Individual] | None:
        """Get the population.

        :type: :py:class:`list` of :py:class:`~base.Individual`
        """
        return self._pop

    def _evaluate_pop(self, pop: List[Individual]) -> None:
        """Evaluate the individuals of *pop* that have an invalid fitness.

        :param pop: A population
        :type pop: :py:class:`list` of :py:class:`~base.Individual`
        """
        # Select the individuals with an invalid fitness
        invalid_inds = [ind for ind in pop if not ind.fitness.valid]

        for ind in invalid_inds:
            self.evaluate(ind)

        # Number of evaluations performed
        self._current_gen_evals += (len(invalid_inds) * (
            len(self.representatives)
            if self.representatives is not None
            else 1
        )
        )

    def _do_generation_stats(self, pop: List[Individual]) -> None:
        """Perform the generation stats.

        :param pop: A population
        :type pop: :py:class:`list` of :py:class:`~base.Individual`
        """
        # Perform some stats
        record = self._stats.compile(pop) if self._stats else {}
        self._logbook.record(
            Gen=self._current_gen,
            Pop=self.index,
            NEvals=self._current_gen_evals,
            **record)
        if self.verbose:
            print(self._logbook.stream)

    def _generate_initial_pop(self) -> None:
        """Generate the initial population."""
        self._pop = []
        for _ in repeat(None, self.pop_size):
            self.pop.append(
                self.individual_cls(
                    species=self.species,
                    fitness_cls=self.fitness_function.Fitness)
            )

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this wrapper.

        Overriden to add the current population to the wrapper's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = Generational._state.fget(self)

        # Get the state of this class
        state["pop"] = self.pop

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this wrapper.

        Overriden to add the current population to the wrapper's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        Generational._state.fset(self, state)

        # Set the state of this class
        self._pop = state["pop"]

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        Overriden to generate and evaluate the initial population.
        """
        super()._new_state()

        # Generate the initial population
        self._generate_initial_pop()

        # Not really in the state,
        # but needed to evaluate the initial population
        self._current_gen_evals = 0

        # Evaluate the initial population and append its
        # statistics to the logbook
        self._evaluate_pop(self.pop)
        self._do_generation_stats(self.pop)

    def _reset_state(self) -> None:
        """Reset the wrapper state.

        Overriden to reset the initial population.
        """
        super()._reset_state()
        self._pop = None

    def best_solutions(self) -> List[HallOfFame]:
        """Get the best individuals found for each species.

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = HallOfFame(maxsize=1)
        if self.pop is not None:
            hof.update(self.pop)
        return [hof]

    def __copy__(self) -> SinglePop:
        """Shallow copy the wrapper."""
        cls = self.__class__
        result = cls(
            self.individual_cls, self.species, self.fitness_function
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> SinglePop:
        """Deepcopy the wrapper.

        :param memo: Wrapper attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the wrapper
        :rtype: :py:class:`~base.Wrapper`
        """
        cls = self.__class__
        result = cls(
            self.individual_cls, self.species, self.fitness_function
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the wrapper.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (self.individual_cls, self.species, self.fitness_function),
                self.__dict__)


# Exported symbols for this module
__all__ = [
    'SinglePop',
    'DEFAULT_POP_SIZE',
    'DEFAULT_CROSSOVER_PROB',
    'DEFAULT_MUTATION_PROB',
    'DEFAULT_GENE_IND_MUTATION_PROB',
    'DEFAULT_SELECTION_FUNC',
    'DEFAULT_SELECTION_FUNC_PARAMS'
]
