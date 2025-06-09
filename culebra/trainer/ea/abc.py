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

"""Abstract base classes for different evolutionary trainers.

This module provides several abstract classes for different kind of
evolutionary trainers.

Regarding the homogeneity of their operators:

  * :py:class:`~culebra.trainer.ea.abc.HomogeneousEA`: Defines an EA model in
    which all the sub-populations have the same configuration
  * :py:class:`~culebra.trainer.ea.abc.HeterogeneousEA`: Allows a different
    configuration for each sub-population in multi-population approaches

With respect to the number of populations being trained:

  * :py:class:`~culebra.trainer.ea.abc.SinglePopEA`: A base class for all
    the single population evolutionary trainers
  * :py:class:`~culebra.trainer.ea.abc.MultiPopEA`: A base class for all
    the multiple population evolutionary trainers

Different multi-population approaches are also provided:

  * :py:class:`~culebra.trainer.ea.abc.IslandsEA`: Abstract base class for
    island-based evolutionary approaches
  * :py:class:`~culebra.trainer.ea.abc.CooperativeEA`: Abstract base class
    for cooperative co-evolutionary trainers

Finally, two types of islands-based models are also defined:

  * :py:class:`~culebra.trainer.ea.abc.HomogeneousIslandsEA`: Abstract base
    class for island-based evolutionary approaches where all the islands share
    the same hyperparameters
  * :py:class:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA`: Abstract base
    class for island-based evolutionary approaches where each island can have
    each own configuration
"""

from __future__ import annotations

from typing import (
    Any,
    Type,
    Callable,
    Tuple,
    List,
    Dict,
    Optional,
    Sequence
)
from functools import partial, partialmethod
from itertools import repeat

from deap.tools import HallOfFame, ParetoFront

from culebra.abc import Species, FitnessFunction
from culebra.checker import (
    check_subclass,
    check_int,
    check_float,
    check_func,
    check_func_params,
    check_sequence
)
from culebra.solution.abc import Individual
from culebra.trainer.abc import (
    SingleSpeciesTrainer,
    DistributedTrainer,
    IslandsTrainer,
    CooperativeTrainer
)

from .constants import (
    DEFAULT_POP_SIZE,
    DEFAULT_CROSSOVER_PROB,
    DEFAULT_MUTATION_PROB,
    DEFAULT_GENE_IND_MUTATION_PROB,
    DEFAULT_SELECTION_FUNC,
    DEFAULT_SELECTION_FUNC_PARAMS
)

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DISTRIBUTED_EA_STATS_NAMES = ('Iter', 'Pop', 'NEvals')
"""Statistics calculated for each iteration of the
:py:class:`~culebra.trainer.ea.abc.SinglePopEA`.
"""


class HomogeneousEA(SingleSpeciesTrainer):
    """Base class for all the homogeneous evolutionary algorithms."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [HomogeneousEA],
                bool
            ]
        ] = None,
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
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new homogeneous evolutionary trainer.

        :param solution_cls: The individual class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Individual`
            subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_size: The populaion size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
        :param crossover_func: The crossover function. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :py:data:`None`
        :type crossover_func: :py:class:`~collections.abc.Callable`, optional
        :param mutation_func: The mutation function. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :py:data:`None`
        :type mutation_func: :py:class:`~collections.abc.Callable`, optional
        :param selection_func: The selection function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_func: :py:class:`~collections.abc.Callable`, optional
        :param crossover_prob: The crossover probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_prob: :py:class:`float` in (0, 1), optional
        :param mutation_prob: The mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :py:data:`None`
        :type mutation_prob: :py:class:`float` in (0, 1), optional
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_prob: :py:class:`float` in (0, 1), optional
        :param selection_func_params: The parameters for the selection
            function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_func_params: :py:class:`dict`, optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
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
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.pop_size = pop_size
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.selection_func = selection_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_ind_mutation_prob = gene_ind_mutation_prob
        self.selection_func_params = selection_func_params

    @property
    def solution_cls(self) -> Type[Individual]:
        """Get and set the individual class.

        :getter: Return the individual class
        :setter: Set a new individual class
        :type: An :py:class:`~culebra.solution.abc.Individual` subclass
        :raises TypeError: If set to a value which is not an
            :py:class:`~culebra.solution.abc.Individual` subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: Type[Individual]) -> None:
        """Set a new individual class.

        :param cls: The new class
        :type cls: An :py:class:`~culebra.solution.abc.Individual` subclass
        :raises TypeError: If *cls* is not an
        :py:class:`~culebra.solution.abc.Individual`
        """
        # Check cls
        self._solution_cls = check_subclass(
            cls, "solution class", Individual
        )

        # Reset the algorithm
        self.reset()

    @property
    def pop_size(self) -> int:
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int`, greater than zero
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        return DEFAULT_POP_SIZE if self._pop_size is None else self._pop_size

    @pop_size.setter
    def pop_size(self, size: int | None) -> None:
        """Set the population size.

        :param size: The new population size. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is chosen
        :type size: :py:class:`int`, greater than zero
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
            the :py:meth:`~culebra.solution.abc.Individual.crossover` method
            of the individual class evolved by the trainer is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            self.solution_cls.crossover
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
            the :py:meth:`~culebra.solution.abc.Individual.crossover` method
            of the individual class evolved by the trainer is chosen
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
        [Individual, float],
        Tuple[Individual]
    ]:
        """Get and set the mutation function.

        :getter: Return the current mutation function
        :setter: Set a new mutation function. If set to :py:data:`None`, the
            :py:meth:`~culebra.solution.abc.Individual.mutate` method of the
            individual class evolved by the trainer is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            self.solution_cls.mutate
            if self._mutation_func is None
            else self._mutation_func
        )

    @mutation_func.setter
    def mutation_func(
        self,
        func: Callable[
            [Individual, float],
            Tuple[Individual]
        ] | None
    ) -> None:
        """Set the mutation function.

        :param func: The new mutation function. If set to :py:data:`None`, the
            :py:meth:`~culebra.solution.abc.Individual.mutate` method of the
            individual class evolved by the trainer is chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` is
            chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` is
            chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` is
            chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return (
            DEFAULT_CROSSOVER_PROB
            if self._crossover_prob is None
            else self._crossover_prob
        )

    @crossover_prob.setter
    def crossover_prob(self, prob: float | None) -> None:
        """Set a new crossover probability.

        :param prob: The new probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` is
            chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` is
            chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` is chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            is chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            is chosen
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
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            is chosen
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


class SinglePopEA(HomogeneousEA):
    """Base class for all the single population evolutionary algorithms."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool
            ]
        ] = None,
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
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new single-population evolutionary trainer.

        :param solution_cls: The individual class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Individual`
            subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_size: The populaion size. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE`
            will be used. Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
        :param crossover_func: The crossover function. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :py:data:`None`
        :type crossover_func: :py:class:`~collections.abc.Callable`, optional
        :param mutation_func: The mutation function. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :py:data:`None`
        :type mutation_func: :py:class:`~collections.abc.Callable`, optional
        :param selection_func: The selection function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_func: :py:class:`~collections.abc.Callable`, optional
        :param crossover_prob: The crossover probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_prob: :py:class:`float` in (0, 1), optional
        :param mutation_prob: The mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :py:data:`None`
        :type mutation_prob: :py:class:`float` in (0, 1), optional
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_prob: :py:class:`float` in (0, 1), optional
        :param selection_func_params: The parameters for the selection
            function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_func_params: :py:class:`dict`, optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
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
            solution_cls=solution_cls,
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

    @property
    def pop(self) -> List[Individual] | None:
        """Get the population.

        :type: :py:class:`list` of :py:class:`~culebra.abc.Solution`
        """
        return self._pop

    def _evaluate_pop(self, pop: List[Individual]) -> None:
        """Evaluate the individuals of *pop* that have an invalid fitness.

        :param pop: A population
        :type pop: :py:class:`list` of
            :py:class:`~culebra.solution.abc.Individual`
        """
        # Select the individuals with an invalid fitness
        if self.fitness_function.is_noisy:
            invalid_inds = pop
        else:
            invalid_inds = [ind for ind in pop if not ind.fitness.valid]

        for ind in invalid_inds:
            self.evaluate(ind)

    def _generate_initial_pop(self) -> None:
        """Generate the initial population.

        The population is filled with random generated individuals.
        """
        self._pop = []
        for _ in repeat(None, self.pop_size):
            self.pop.append(
                self.solution_cls(
                    species=self.species,
                    fitness_cls=self.fitness_function.Fitness)
            )

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["pop"] = self.pop

        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
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
        self._generate_initial_pop()

        # Not really in the state,
        # but needed to evaluate the initial population
        self._current_iter_evals = 0

        # Evaluate the initial population and append its
        # statistics to the logbook
        # Since the evaluation of the initial population is performed
        # before the first iteration, fix self.current_iter = -1
        self._current_iter = -1
        self._evaluate_pop(self.pop)
        self._do_iteration_stats()
        self._num_evals += self._current_iter_evals
        self._current_iter += 1

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the initial population.
        """
        super()._reset_state()
        self._pop = None

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # Perform some stats
        record = self._stats.compile(self.pop) if self._stats else {}
        record["Iter"] = self._current_iter
        record["NEvals"] = self._current_iter_evals
        if self.container is not None:
            record["Pop"] = self.index
        self._logbook.record(**record)
        if self.verbose:
            print(self._logbook.stream)

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        Return the best single solution found for each species

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return [hof]


class MultiPopEA(DistributedTrainer):
    """Base class for all the multiple population evolutionary algorithms."""

    stats_names = DISTRIBUTED_EA_STATS_NAMES
    """Statistics calculated each iteration."""

    @property
    def subtrainer_cls(self) -> Type[SinglePopEA]:
        """Get and set the trainer class to handle the subpopulations.

        Each subpopulation will be handled by a single-population evolutionary
        trainer.

        :getter: Return the trainer class
        :setter: Set new trainer class
        :type: A :py:class:`~culebra.trainer.ea.abc.SinglePopEA` subclass
        :raises TypeError: If set to a value which is not a
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA` subclass
        """
        return self._subtrainer_cls

    @subtrainer_cls.setter
    def subtrainer_cls(self, cls: Type[SinglePopEA]) -> None:
        """Set a new trainer class to handle the subpopulations.

        Each subpopulation will be handled by a single-population evolutionary
        trainer.

        :param cls: The new class
        :type cls: A :py:class:`~culebra.trainer.ea.abc.SinglePopEA` subclass
        :raises TypeError: If *cls* is not a
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA` subclass
        """
        # Check cls
        self._subtrainer_cls = check_subclass(
            cls, "trainer class for subpopulations", SinglePopEA
        )

        # Reset the algorithm
        self.reset()


# Change the docstring of the MultiPopEA constructor to indicate that the
# subtrainer_cls must be a subclass of SinglePopEA
MultiPopEA.__init__.__doc__ = (
    MultiPopEA.__init__.__doc__.replace(
        ':py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`',
        ':py:class:`~culebra.trainer.ea.abc.SinglePopEA`'
    )
)


class IslandsEA(IslandsTrainer, MultiPopEA):
    """Base class for all the islands-based evolutionary algorithms."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA], bool]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Individual`
            subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subpopulations.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param num_subtrainers: The number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` will be
            used. Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        IslandsTrainer.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    # Copy the culebra.trainer.ea.abc.HomogeneousEA.solution_cls property
    solution_cls = HomogeneousEA.solution_cls

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        Return the best single solution found for each species

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = ParetoFront()
        if self.subtrainers is not None:
            for subtrainer in self.subtrainers:
                if subtrainer.pop is not None:
                    hof.update(subtrainer.pop)

        return [hof]

    @staticmethod
    def receive_representatives(subtrainer) -> None:
        """Receive representative solutions.

        :param subtrainer: The subtrainer receiving
            representatives
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        """
        container = subtrainer.container

        # Receive all the solutions in the queue
        queue = container._communication_queues[subtrainer.index]
        while not queue.empty():
            subtrainer._pop.extend(queue.get())

    @staticmethod
    def send_representatives(subtrainer) -> None:
        """Send representatives.

        :param subtrainer: The sender subtrainer
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        """
        container = subtrainer.container

        # Check if sending should be performed
        if subtrainer._current_iter % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subtrainer.index,
                container.num_subtrainers,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                sols = container.representation_selection_func(
                    subtrainer.pop,
                    container.representation_size,
                    **container.representation_selection_func_params
                )
                container._communication_queues[dest].put(sols)


class HomogeneousIslandsEA(IslandsEA, HomogeneousEA):
    """Abstract island-based model with homogeneous islands."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
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
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Individual`
            subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subpopulations.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_size: The populaion size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
        :param crossover_func: The crossover function. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :py:data:`None`
        :type crossover_func: :py:class:`~collections.abc.Callable`, optional
        :param mutation_func: The mutation function. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :py:data:`None`
        :type mutation_func: :py:class:`~collections.abc.Callable`, optional
        :param selection_func: The selection function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_func: :py:class:`~collections.abc.Callable`, optional
        :param crossover_prob: The crossover probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_prob: :py:class:`float` in (0, 1), optional
        :param mutation_prob: The mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :py:data:`None`
        :type mutation_prob: :py:class:`float` in (0, 1), optional
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_prob: :py:class:`float` in (0, 1), optional
        :param selection_func_params: The parameters for the selection
            function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_func_params: :py:class:`dict`, optional
        :param num_subtrainers: The number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` will be
            used. Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        HomogeneousEA.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            pop_size=pop_size,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            selection_func=selection_func,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            gene_ind_mutation_prob=gene_ind_mutation_prob,
            selection_func_params=selection_func_params,
        )
        IslandsEA.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    @HomogeneousEA.pop_size.getter
    def pop_size(self) -> int:
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int`, greater than zero
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        if self._pop_size is not None:
            return self._pop_size
        elif self.subtrainers is not None:
            return self.subtrainers[0].pop_size
        else:
            return None

    @HomogeneousEA.crossover_func.getter
    def crossover_func(self) -> Callable[
            [Individual, Individual],
            Tuple[Individual, Individual]
    ]:
        """Get and set the crossover function.

        :getter: Return the current crossover function
        :setter: Set a new crossover function. If set to :py:data:`None`,
            the :py:meth:`~culebra.solution.abc.Individual.crossover` method
            of the individual class evolved by the trainer is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        if self._crossover_func is not None:
            return self._crossover_func
        elif self.subtrainers is not None:
            return self.subtrainers[0].crossover_func
        else:
            return None

    @HomogeneousEA.mutation_func.getter
    def mutation_func(self) -> Callable[
        [Individual, float],
        Tuple[Individual]
    ]:
        """Get and set the mutation function.

        :getter: Return the current mutation function
        :setter: Set a new mutation function. If set to :py:data:`None`, the
            :py:meth:`~culebra.solution.abc.Individual.mutate` method of the
            individual class evolved by the trainer is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        if self._mutation_func is not None:
            return self._mutation_func
        elif self.subtrainers is not None:
            return self.subtrainers[0].mutation_func
        else:
            return None

    @HomogeneousEA.selection_func.getter
    def selection_func(
        self
    ) -> Callable[[List[Individual], int, Any], List[Individual]]:
        """Get and set the selection function.

        :getter: Return the current selection function
        :setter: Set the new selection function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` is
            chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        if self._selection_func is not None:
            return self._selection_func
        elif self.subtrainers is not None:
            return self.subtrainers[0].selection_func
        else:
            return None

    @HomogeneousEA.crossover_prob.getter
    def crossover_prob(self) -> float:
        """Get and set the crossover probability.

        :getter: Return the current crossover probability
        :setter: Set the new crossover probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` is
            chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        if self._crossover_prob is not None:
            return self._crossover_prob
        elif self.subtrainers is not None:
            return self.subtrainers[0].crossover_prob
        else:
            return None

    @HomogeneousEA.mutation_prob.getter
    def mutation_prob(self) -> float:
        """Get and set the mutation probability.

        :getter: Return the current mutation probability
        :setter: Set the new mutation probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` is
            chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        if self._mutation_prob is not None:
            return self._mutation_prob
        elif self.subtrainers is not None:
            return self.subtrainers[0].mutation_prob
        else:
            return None

    @HomogeneousEA.gene_ind_mutation_prob.getter
    def gene_ind_mutation_prob(self) -> float:
        """Get and set the gene independent mutation probability.

        :getter: Return the current gene independent mutation probability
        :setter: Set the new gene independent mutation probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            is chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        if self._gene_ind_mutation_prob is not None:
            return self._gene_ind_mutation_prob
        elif self.subtrainers is not None:
            return self.subtrainers[0].gene_ind_mutation_prob
        else:
            return None

    @HomogeneousEA.selection_func_params.getter
    def selection_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the selection function.

        :getter: Return the current parameters for the selection function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            is chosen
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        if self._selection_func_params is not None:
            return self._selection_func_params
        elif self.subtrainers is not None:
            return self.subtrainers[0].selection_func_params
        else:
            return None

    def _generate_subtrainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        subpopulation :py:class:`~culebra.trainer.ea.abc.SinglePopEA` trainer,
        change the subpopulation trainers'
        :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subpopulation trainers, if necessary
        """

        def subtrainers_properties() -> Dict[str, Any]:
            """Return the subpopulation trainers' properties."""
            # Get the attributes from the container trainer
            cls = self.subtrainer_cls
            properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            properties.update(self.subtrainer_params)

            return properties

        # Get the subpopulations properties
        properties = subtrainers_properties()

        # Generate the subpopulations
        self._subtrainers = []

        for (
            index,
            checkpoint_filename
        ) in enumerate(self.subtrainer_checkpoint_filenames):
            subtrainer = self.subtrainer_cls(**properties)
            subtrainer.checkpoint_filename = checkpoint_filename
            subtrainer.index = index
            subtrainer.container = self
            subtrainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subtrainer.__class__._postprocess_iteration = (
                self.send_representatives
            )
            self._subtrainers.append(subtrainer)


class HeterogeneousEA(MultiPopEA):
    """Base class for all the heterogeneous evolutionary algorithms."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
        pop_sizes: Optional[int | Sequence[int]] = None,
        crossover_funcs: Optional[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual],
                    Tuple[Individual, Individual]
                ]
            ]
        ] = None,
        mutation_funcs: Optional[
            Callable[
                [Individual, float],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, float],
                    Tuple[Individual]
                ]
            ]
        ] = None,
        selection_funcs: Optional[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ] | Sequence[
                Callable[
                    [List[Individual], int, Any],
                    List[Individual]
                ]
            ]
        ] = None,
        crossover_probs: Optional[float | Sequence[float]] = None,
        mutation_probs: Optional[float | Sequence[float]] = None,
        gene_ind_mutation_probs: Optional[float | Sequence[float]] = None,
        selection_funcs_params: Optional[
            Dict[str, Any] | Sequence[Dict[str, Any]]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subpopulations.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_sizes: The population size for each subpopulation.
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_sizes: :py:class:`int` or
            :py:class:`~collections.abc.Sequence` of :py:class:`int`, optional
        :param crossover_funcs: The crossover function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :py:data:`None`
        :type crossover_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param mutation_funcs: The mutation function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :py:data:`None`
        :type mutation_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param selection_funcs: The selection function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param crossover_probs: The crossover probability for each
            subpopulation. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param mutation_probs: The mutation probability for each subpopulation.
            If only a single value is provided, the same probability will be
            used for all the subpopulations. Different probabilities can be
            provided in a :py:class:`~collections.abc.Sequence`. All the
            probabilities must be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :py:data:`None`
        :type mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation. If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_funcs_params: :py:class:`dict` or
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`, optional
        :param num_subtrainers: The number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` will be
            used. Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

        # Get the parameters
        self.pop_sizes = pop_sizes
        self.crossover_funcs = crossover_funcs
        self.mutation_funcs = mutation_funcs
        self.selection_funcs = selection_funcs
        self.crossover_probs = crossover_probs
        self.mutation_probs = mutation_probs
        self.gene_ind_mutation_probs = gene_ind_mutation_probs
        self.selection_funcs_params = selection_funcs_params

    @property
    def pop_sizes(self) -> Sequence[int | None]:
        """Get and set the population size for each subtrainer.

        :getter: Return the current size of each subtrainer
        :setter: Set a new size for each subtrainer. If only a single value
            is provided, the same size will be used for all the subtrainers.
            Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int` or :py:class:`~collections.abc.Sequence`
            of :py:class:`int`
        :raises TypeError: If set to a value which is not an :py:class:`int`
            or a :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If any population size is not greater than zero
        """
        if self.subtrainers is not None:
            the_pop_sizes = [
                subtrainer.pop_size
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._pop_sizes, Sequence):
            the_pop_sizes = self._pop_sizes
        else:
            the_pop_sizes = list(repeat(self._pop_sizes, self.num_subtrainers))

        return the_pop_sizes

    @pop_sizes.setter
    def pop_sizes(self, sizes: int | Sequence[int] | None) -> None:
        """Set the population size for each subtrainer.

        :param sizes: The new population sizes. If only a single value
            is provided, the same size will be used for all the subtrainers.
            Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
        :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int` or :py:class:`~collections.abc.Sequence`
            of :py:class:`int`
        :raises TypeError: If *sizes* is not an :py:class:`int`
            or a :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If any value in *size* is not greater than zero
        """
        # If None is provided ...
        if sizes is None:
            self._pop_sizes = None
        # If a sequence is provided ...
        elif isinstance(sizes, Sequence):
            self._pop_sizes = check_sequence(
                sizes,
                "population sizes",
                item_checker=partial(check_int, gt=0)
            )
        # If a scalar value is provided ...
        else:
            self._pop_sizes = check_int(sizes, "population size", gt=0)

        # Reset the algorithm
        self.reset()

    @property
    def crossover_funcs(self) -> Sequence[
        Callable[[Individual, Individual], Tuple[Individual, Individual]] |
        None
    ]:
        """Get and set the crossover function for each subpopulation.

        :getter: Return the current crossover functions
        :setter: Set the new crossover functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~culebra.solution.abc.Individual.crossover` method
            of the individual class evolved by the trainer is chosen
        :type: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if self.subtrainers is not None:
            the_funcs = [
                subtrainer.crossover_func
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._crossover_funcs, Sequence):
            the_funcs = self._crossover_funcs
        else:
            the_funcs = list(
                repeat(self._crossover_funcs, self.num_subtrainers)
            )

        return the_funcs

    @crossover_funcs.setter
    def crossover_funcs(
        self,
        funcs: Callable[
            [Individual, Individual],
            Tuple[Individual, Individual]
        ] | Sequence[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ]
        ] | None
    ) -> None:
        """Set the crossover function for each subpopulation.

        :param funcs: The new crossover functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~culebra.solution.abc.Individual.crossover` method
            of the individual class evolved by the trainer is chosen
        :type funcs: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If *funcs* is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._crossover_funcs = None
        elif isinstance(funcs, Sequence):
            self._crossover_funcs = check_sequence(
                funcs,
                "crossover functions",
                item_checker=check_func
            )
        else:
            self._crossover_funcs = check_func(funcs, "crossover function")

        # Reset the algorithm
        self.reset()

    @property
    def mutation_funcs(self) -> Sequence[
        Callable[[Individual, float], Tuple[Individual]] | None
    ]:
        """Get and set the mutation function for each subpopulation.

        :getter: Return the current mutation functions
        :setter: Set the new mutation functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~culebra.solution.abc.Individual.mutate` method of
            the individual class evolved by the trainer is chosen
        :type: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if self.subtrainers is not None:
            the_funcs = [
                subtrainer.mutation_func
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._mutation_funcs, Sequence):
            the_funcs = self._mutation_funcs
        else:
            the_funcs = list(
                repeat(self._mutation_funcs, self.num_subtrainers)
            )

        return the_funcs

    @mutation_funcs.setter
    def mutation_funcs(
        self,
        funcs: Callable[
            [Individual, float],
            Tuple[Individual]
        ] | Sequence[
            Callable[
                [Individual, float],
                Tuple[Individual]
            ]
        ] | None
    ) -> None:
        """Set the mutation function for each subpopulation.

        :param funcs: The new mutation functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the :py:meth:`~culebra.solution.abc.Individual.mutate` method of
            the individual class evolved by the trainer is chosen
        :type funcs: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If *funcs* is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._mutation_funcs = None
        elif isinstance(funcs, Sequence):
            self._mutation_funcs = check_sequence(
                funcs,
                "mutation functions",
                item_checker=check_func
            )
        else:
            self._mutation_funcs = check_func(funcs, "mutation function")

        # Reset the algorithm
        self.reset()

    @property
    def crossover_probs(self) -> Sequence[float | None]:
        """Get and set the crossover probability for each subpopulation.

        :getter: Return the current crossover probabilities
        :setter: Set the new crossover probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` is
            chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If set to a value which is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any probability is not in (0, 1)
        """
        if self.subtrainers is not None:
            the_probs = [
                subtrainer.crossover_prob
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._crossover_probs, Sequence):
            the_probs = self._crossover_probs
        else:
            the_probs = list(
                repeat(self._crossover_probs, self.num_subtrainers)
            )

        return the_probs

    @crossover_probs.setter
    def crossover_probs(self, probs: float | Sequence[float] | None) -> None:
        """Set the crossover probability for each subpopulation.

        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` is
            chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *probs* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any probability is not in (0, 1)
        """
        if probs is None:
            self._crossover_probs = None
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._crossover_probs = check_sequence(
                probs,
                "crossover probabilities",
                item_checker=partial(check_float, gt=0, lt=1)
            )
        else:
            self._crossover_probs = check_float(
                probs, "crossover probability", gt=0, lt=1
            )

        # Reset the algorithm
        self.reset()

    @property
    def mutation_probs(self) -> Sequence[float | None]:
        """Get and set the mutation probability for each subpopulation.

        :getter: Return the current mutation probabilities
        :setter: Set the new mutation probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If set to a value which is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if self.subtrainers is not None:
            the_probs = [
                subtrainer.mutation_prob
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._mutation_probs, Sequence):
            the_probs = self._mutation_probs
        else:
            the_probs = list(
                repeat(self._mutation_probs, self.num_subtrainers)
            )

        return the_probs

    @mutation_probs.setter
    def mutation_probs(self, probs: float | Sequence[float] | None):
        """Set the mutation probability for each subpopulation.

        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *probs* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if probs is None:
            self._mutation_probs = None
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._mutation_probs = check_sequence(
                probs,
                "mutation probabilities",
                item_checker=partial(check_float, gt=0, lt=1)
            )
        else:
            self._mutation_probs = check_float(
                probs, "mutation probability", gt=0, lt=1
            )

        # Reset the algorithm
        self.reset()

    @property
    def gene_ind_mutation_probs(self) -> Sequence[float | None]:
        """Get and set the gene independent mutation probabilities.

        :getter: Return the current gene independent mutation probability for
            each subpopulation
        :setter: Set new values for the gene independent mutation
            probabilities. If only a single value is provided, the same
            probability will be used for all the subpopulations. Different
            probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If set to a value which is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if self.subtrainers is not None:
            the_probs = [
                subtrainer.gene_ind_mutation_prob
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._gene_ind_mutation_probs, Sequence):
            the_probs = self._gene_ind_mutation_probs
        else:
            the_probs = list(
                repeat(self._gene_ind_mutation_probs, self.num_subtrainers)
            )

        return the_probs

    @gene_ind_mutation_probs.setter
    def gene_ind_mutation_probs(self, probs: float | Sequence[float] | None):
        """Set the subpopulations gene independent mutation probability.

        :param probs: The new probabilities. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            is chosen
        :type: :py:class:`float` or :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If *probs* is not a real number or a
            :py:class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If probabilities are not in (0, 1)
        """
        if probs is None:
            self._gene_ind_mutation_probs = None
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._gene_ind_mutation_probs = check_sequence(
                probs,
                "gene independent mutation probabilities",
                item_checker=partial(check_float, gt=0, lt=1)
            )
        else:
            self._gene_ind_mutation_probs = check_float(
                probs,
                "gene independent mutation probability",
                gt=0,
                lt=1
            )

        # Reset the algorithm
        self.reset()

    @property
    def selection_funcs(self) -> Sequence[
        Callable[[List[Individual], int, Any], List[Individual]] | None
    ]:
        """Get and set the selection function for each subpopulation.

        :getter: Return the current selection functions
        :setter: Set the new selection functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` is
            chosen
        :type: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if self.subtrainers is not None:
            the_funcs = [
                subtrainer.selection_func
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._selection_funcs, Sequence):
            the_funcs = self._selection_funcs
        else:
            the_funcs = list(
                repeat(self._selection_funcs, self.num_subtrainers)
            )

        return the_funcs

    @selection_funcs.setter
    def selection_funcs(self, funcs: Callable[
            [List[Individual], int, Any],
            List[Individual]
        ] | Sequence[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ]
    ] | None
    ) -> None:
        """Set the selection function for each subpopulation.

        :param funcs: The new selection functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` is
            chosen
        :type funcs: :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        :raises TypeError: If *funcs* is not
            :py:class:`~collections.abc.Callable` or a
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._selection_funcs = None
        # If a sequence is provided ...
        elif isinstance(funcs, Sequence):
            self._selection_funcs = check_sequence(
                funcs,
                "selection functions",
                item_checker=check_func
            )
        else:
            self._selection_funcs = check_func(
                funcs,
                "selection function"
            )

        # Reset the algorithm
        self.reset()

    @property
    def selection_funcs_params(self) -> Sequence[Dict[str, Any] | None]:
        """Get and set the parameters of the selection functions.

        :getter: Return the current parameters for the selection function of
            each subpopulation
        :setter: Set new parameters. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            is chosen
        :type: :py:class:`dict` or :py:class:`~collections.abc.Sequence`
            of :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
            or a :py:class:`~collections.abc.Sequence`
            of :py:class:`dict`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`dict`
        """
        if self.subtrainers is not None:
            the_params = [
                subtrainer.selection_func_params
                for subtrainer in self.subtrainers
            ]
        elif isinstance(self._selection_funcs_params, Sequence):
            the_params = self._selection_funcs_params
        else:
            the_params = list(
                repeat(self._selection_funcs_params, self.num_subtrainers)
            )

        return the_params

    @selection_funcs_params.setter
    def selection_funcs_params(
        self,
        param_dicts: Dict[str, Any] | Sequence[Dict[str, Any]] | None
    ) -> None:
        """Set the parameters for the selection function of each subpopulation.

        :param param_dicts: The new parameters. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            is chosen
        :type param_dicts: A :py:class:`dict` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`
        :raises TypeError: If *param_dicts* is not a :py:class:`dict`
            or a :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`dict`
        """
        if param_dicts is None:
            self._selection_funcs_params = None
        # If a sequence is provided ...
        elif isinstance(param_dicts, Sequence):
            self._selection_funcs_params = check_sequence(
                param_dicts,
                "selection functions parameters",
                item_checker=check_func_params
            )
        else:
            self._selection_funcs_params = check_func_params(
                param_dicts,
                "selection function parameters"
            )

        # Reset the algorithm
        self.reset()


class HeterogeneousIslandsEA(IslandsEA, HeterogeneousEA):
    """Abstract island-based model with heterogeneous islands."""

    _subpop_properties_mapping = {
        "pop_sizes": "pop_size",
        "crossover_funcs": "crossover_func",
        "mutation_funcs": "mutation_func",
        "crossover_probs": "crossover_prob",
        "mutation_probs": "mutation_prob",
        "gene_ind_mutation_probs": "gene_ind_mutation_prob",
        "selection_funcs": "selection_func",
        "selection_funcs_params": "selection_func_params"
    }
    """Map the container trainer names of properties sequences to the different
    subpopulation trainer property names."""

    def __init__(
        self,
        solution_cls: Type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool]
        ] = None,
        pop_sizes: Optional[int | Sequence[int]] = None,
        crossover_funcs: Optional[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual],
                    Tuple[Individual, Individual]
                ]
            ]
        ] = None,
        mutation_funcs: Optional[
            Callable[
                [Individual, float],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, float],
                    Tuple[Individual]
                ]
            ]
        ] = None,
        selection_funcs: Optional[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ] | Sequence[
                Callable[
                    [List[Individual], int, Any],
                    List[Individual]
                ]
            ]
        ] = None,
        crossover_probs: Optional[float | Sequence[float]] = None,
        mutation_probs: Optional[float | Sequence[float]] = None,
        gene_ind_mutation_probs: Optional[float | Sequence[float]] = None,
        selection_funcs_params: Optional[
            Dict[str, Any] | Sequence[Dict[str, Any]]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Individual`
            subclass
        :param species: The species for all the individuals
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subpopulations.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_sizes: The population size for each subpopulation.
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_sizes: :py:class:`int` or
            :py:class:`~collections.abc.Sequence` of :py:class:`int`, optional
        :param crossover_funcs: The crossover function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :py:data:`None`
        :type crossover_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param mutation_funcs: The mutation function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :py:data:`None`
        :type mutation_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param selection_funcs: The selection function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param crossover_probs: The crossover probability for each
            subpopulation. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param mutation_probs: The mutation probability for each subpopulation.
            If only a single value is provided, the same probability
            will be used for all the subpopulations. Different probabilities
            can be provided in a :py:class:`~collections.abc.Sequence`. All
            the probabilities must be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :py:data:`None`
        :type mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation. If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_funcs_params: :py:class:`dict` or
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`, optional
        :param num_subtrainers: The number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` will be
            used. Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        HeterogeneousEA.__init__(
            self,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params
        )
        IslandsEA.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    def _generate_subtrainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        subpopulation :py:class:`~culebra.trainer.ea.abc.SinglePopEA` trainer,
        change the subpopulation trainers'
        :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subpopulation trainers, if necessary

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subpopulations.
        """

        def subtrainers_properties() -> List[Dict[str, Any]]:
            """Obtain the properties of each subpopulation trainer.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation trainer.
            :rtype: :py:class:`list`
            """
            # Get the common attributes from the container trainer
            cls = self.subtrainer_cls
            common_properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            common_properties.update(self.subtrainer_params)

            # List with the common properties. Equal for all the subpopulations
            properties = []
            for _ in range(self.num_subtrainers):
                subpop_properties = {}
                for key, value in common_properties.items():
                    subpop_properties[key] = value
                properties.append(subpop_properties)

            # Particular properties for each subpop
            cls = self.__class__
            for (
                    property_sequence_name,
                    subpop_property_name
            ) in self._subpop_properties_mapping.items():

                # Values of the sequence
                property_sequence_values = getattr(
                    cls, property_sequence_name
                ).fget(self)

                # Check the properties' length
                if len(property_sequence_values) != self.num_subtrainers:
                    raise RuntimeError(
                        f"The length of {property_sequence_name} does not "
                        "match the number of subpopulations"
                    )
                for (
                        subpop_properties, subpop_property_value
                ) in zip(properties, property_sequence_values):
                    subpop_properties[
                        subpop_property_name] = subpop_property_value

            return properties

        # Get the subpopulations properties
        properties = subtrainers_properties()

        # Generate the subpopulations
        self._subtrainers = []

        for (
            index, (
                checkpoint_filename,
                subpop_properties
            )
        ) in enumerate(
            zip(self.subtrainer_checkpoint_filenames, properties)
        ):
            subtrainer = self.subtrainer_cls(**subpop_properties)
            subtrainer.checkpoint_filename = checkpoint_filename
            subtrainer.index = index
            subtrainer.container = self
            subtrainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subtrainer.__class__._postprocess_iteration = (
                self.send_representatives
            )
            self._subtrainers.append(subtrainer)


class CooperativeEA(CooperativeTrainer, HeterogeneousEA):
    """Abstract cooperative co-evolutionary model."""

    _subpop_properties_mapping = {
        "solution_classes": "solution_cls",
        "species": "species",
        "pop_sizes": "pop_size",
        "crossover_funcs": "crossover_func",
        "mutation_funcs": "mutation_func",
        "crossover_probs": "crossover_prob",
        "mutation_probs": "mutation_prob",
        "gene_ind_mutation_probs": "gene_ind_mutation_prob",
        "selection_funcs": "selection_func",
        "selection_funcs_params": "selection_func_params"
    }
    """Map the container names of properties sequences to the different
    subpop property names."""

    def __init__(
        self,
        solution_classes: Sequence[Type[Individual]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SinglePopEA],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopEA],
                bool
            ]
        ] = None,
        pop_sizes: Optional[int | Sequence[int]] = None,
        crossover_funcs: Optional[
            Callable[
                [Individual, Individual],
                Tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual],
                    Tuple[Individual, Individual]
                ]
            ]
        ] = None,
        mutation_funcs: Optional[
            Callable[
                [Individual, float],
                Tuple[Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, float],
                    Tuple[Individual]
                ]
            ]
        ] = None,
        selection_funcs: Optional[
            Callable[
                [List[Individual], int, Any],
                List[Individual]
            ] | Sequence[
                Callable[
                    [List[Individual], int, Any],
                    List[Individual]
                ]
            ]
        ] = None,
        crossover_probs: Optional[float | Sequence[float]] = None,
        mutation_probs: Optional[float | Sequence[float]] = None,
        gene_ind_mutation_probs: Optional[float | Sequence[float]] = None,
        selection_funcs_params: Optional[
            Dict[str, Any] | Sequence[Dict[str, Any]]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_selection_func: Optional[
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        Each species is evolved in a different subpopulation.

        :param solution_classes: The individual class for each species.
        :type solution_classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.solution.abc.Individual` subclasses
        :param species: The species to be evolved
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subpopulations.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.ea.abc.SinglePopEA`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_sizes: The population size for each subpopulation (species).
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_sizes: :py:class:`int` or
            :py:class:`~collections.abc.Sequence` of :py:class:`int`, optional
        :param crossover_funcs: The crossover function for each subpopulation
            (species). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.crossover` method will
            be used. Defaults to :py:data:`None`
        :type crossover_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param mutation_funcs: The mutation function for each subpopulation
            (species). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`, the *solution_cls*
            :py:meth:`~culebra.solution.abc.Individual.mutate` method will be
            used. Defaults to :py:data:`None`
        :type mutation_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param selection_funcs: The selection function for each subpopulation
            (species). If only a single value is provided, the same function
            will be used for all the subpopulations. Different functions can
            be provided in a :py:class:`~collections.abc.Sequence`. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC` will be
            used. Defaults to :py:data:`None`
        :type selection_funcs: :py:class:`~collections.abc.Callable` or
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Callable`, optional
        :param crossover_probs: The crossover probability for each
            subpopulation (species). If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_CROSSOVER_PROB` will be
            used. Defaults to :py:data:`None`
        :type crossover_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param mutation_probs: The mutation probability for each subpopulation
            (species). If only a single value is provided, the same probability
            will be used for all the subpopulations. Different probabilities
            can be provided in a :py:class:`~collections.abc.Sequence`. All
            the probabilities must be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_MUTATION_PROB` will be
            used. Defaults to :py:data:`None`
        :type mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation (species). If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_GENE_IND_MUTATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type gene_ind_mutation_probs: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation (species). If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :py:class:`~collections.abc.Sequence`. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type selection_funcs_params: :py:class:`dict` or
            :py:class:`~collections.abc.Sequence` of :py:class:`dict`, optional
        :param num_subtrainers: The number of subpopulations (species). If set
            to :py:data:`None`, the number of species  evolved by the trainer
            is will be used, otherwise it must match the number of species.
            Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (species). If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :param subtrainer_params: Custom parameters for the subpopulations
            (species) trainer
        :type subtrainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        CooperativeTrainer.__init__(
            self,
            solution_classes=solution_classes,
            species=species,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
        )
        HeterogeneousEA.__init__(
            self,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            pop_sizes=pop_sizes,
            crossover_funcs=crossover_funcs,
            mutation_funcs=mutation_funcs,
            selection_funcs=selection_funcs,
            crossover_probs=crossover_probs,
            mutation_probs=mutation_probs,
            gene_ind_mutation_probs=gene_ind_mutation_probs,
            selection_funcs_params=selection_funcs_params,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    def _generate_subtrainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        subpopulation :py:class:`~culebra.trainer.ea.abc.SinglePopEA` trainer,
        change the subpopulation trainers'
        :py:attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.ea.abc.CooperativeEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subpopulation trainers, if necessary

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subpopulations.
        """

        def subtrainers_properties() -> List[Dict[str, Any]]:
            """Obtain the properties of each subpopulation.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation.
            :rtype: :py:class:`list`
            """
            # Get the common attributes from the container trainer
            cls = self.subtrainer_cls
            common_properties = {
                key: getattr(self, key)
                for key in cls.__init__.__code__.co_varnames
                if hasattr(self, key) and getattr(self, key) is not None
            }

            # Append subpopulation trainer custom atributes
            common_properties.update(self.subtrainer_params)

            # List with the common properties. Equal for all the subpopulations
            properties = []
            for _ in range(self.num_subtrainers):
                subpop_properties = {}
                for key, value in common_properties.items():
                    subpop_properties[key] = value
                properties.append(subpop_properties)

            # Particular properties for each subpop
            cls = self.__class__
            for (
                    property_sequence_name,
                    subpop_property_name
            ) in self._subpop_properties_mapping.items():

                # Values of the sequence
                property_sequence_values = getattr(
                    cls, property_sequence_name
                ).fget(self)

                # Check the properties' length
                if len(property_sequence_values) != self.num_subtrainers:
                    raise RuntimeError(
                        f"The length of {property_sequence_name} does not "
                        "match the number of subpopulations"
                    )
                for (
                        subpop_properties, subpop_property_value
                ) in zip(properties, property_sequence_values):
                    subpop_properties[
                        subpop_property_name] = subpop_property_value

            return properties

        # Get the subpopulations properties
        properties = subtrainers_properties()

        # Generate the subpopulations
        self._subtrainers = []

        for (
            index, (
                checkpoint_filename,
                subpop_properties
            )
        ) in enumerate(
            zip(self.subtrainer_checkpoint_filenames, properties)
        ):
            subtrainer = self.subtrainer_cls(**subpop_properties)
            subtrainer.checkpoint_filename = checkpoint_filename
            subtrainer.index = index
            subtrainer.container = self
            subtrainer.__class__._preprocess_iteration = (
                self.receive_representatives
            )
            subtrainer.__class__._postprocess_iteration = (
                self.send_representatives
            )

            subtrainer.__class__._init_representatives = partialmethod(
                self._init_subtrainer_representatives,
                solution_classes=self.solution_classes,
                species=self.species,
                representation_size=self.representation_size
            )

            self._subtrainers.append(subtrainer)

    @staticmethod
    def receive_representatives(subtrainer) -> None:
        """Receive representative individuals.

        :param subtrainer: The subtrainer receiving
            representatives
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        """
        container = subtrainer.container

        # Receive all the individuals in the queue
        queue = container._communication_queues[subtrainer.index]

        anything_received = False
        while not queue.empty():
            msg = queue.get()
            sender_index = msg[0]
            representatives = msg[1]
            for ind_index, ind in enumerate(representatives):
                subtrainer.representatives[ind_index][sender_index] = ind

            anything_received = True

        # If any new representatives have arrived, the fitness of all the
        # individuals in the population must be invalidated and individuals
        # must be re-evaluated
        if anything_received:
            # Re-evaluate all the individuals
            for sol in subtrainer.pop:
                subtrainer.evaluate(sol)

    @staticmethod
    def send_representatives(subtrainer) -> None:
        """Send representatives.

        :param subtrainer: The sender subtrainer
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        """
        container = subtrainer.container

        # Check if sending should be performed
        if subtrainer._current_iter % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subtrainer.index,
                container.num_subtrainers,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                inds = container.representation_selection_func(
                    subtrainer.pop,
                    container.representation_size,
                    **container.representation_selection_func_params
                )

                # Send the following msg:
                # (index of sender subpop, representatives)
                container._communication_queues[dest].put(
                    (subtrainer.index, inds)
                )


# Exported symbols for this module
__all__ = [
    'HomogeneousEA',
    'SinglePopEA',
    'MultiPopEA',
    'IslandsEA',
    'HeterogeneousEA',
    'HomogeneousIslandsEA',
    'HeterogeneousIslandsEA',
    'CooperativeEA'
]
