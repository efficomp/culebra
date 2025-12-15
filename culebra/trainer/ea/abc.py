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

* :class:`~culebra.trainer.ea.abc.HeterogeneousEA`: Allows a different
  configuration for each sub-population in multi-population approaches
* :class:`~culebra.trainer.ea.abc.HomogeneousEA`: Defines an EA model in
  which all the sub-populations have the same configuration

With respect to the number of populations being trained:

* :class:`~culebra.trainer.ea.abc.MultiPopEA`: A base class for all the
  multiple population evolutionary trainers
* :class:`~culebra.trainer.ea.abc.SinglePopEA`: A base class for all the
  single population evolutionary trainers

Different multi-population approaches are also provided:

* :class:`~culebra.trainer.ea.abc.CooperativeEA`: Abstract base class
  for cooperative co-evolutionary trainers
* :class:`~culebra.trainer.ea.abc.IslandsEA`: Abstract base class for
  island-based evolutionary approaches

Finally, two types of islands-based models are also defined:

* :class:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA`: Abstract base
  class for island-based evolutionary approaches where each island can have
  each own configuration
* :class:`~culebra.trainer.ea.abc.HomogeneousIslandsEA`: Abstract base
  class for island-based evolutionary approaches where all the islands share
  the same hyperparameters
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Callable
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
:class:`~culebra.trainer.ea.abc.SinglePopEA`.
"""


class HomogeneousEA(SingleSpeciesTrainer):
    """Base class for all the homogeneous evolutionary algorithms."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[HomogeneousEA], bool] | None = None,
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
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new homogeneous evolutionary trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.abc.HomogeneousEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_crossover_func`
            is used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_mutation_func`
            is used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_selection_func`
            is used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_crossover_prob`
            is used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_mutation_prob`
            is used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_gene_ind_mutation_prob`
            is used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_selection_func_params`
            is used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
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

    @SingleSpeciesTrainer.solution_cls.setter
    def solution_cls(self, cls: type[Individual]) -> None:
        """Set a new solution class.

        :param cls: The new class
        :type cls: type[~culebra.solution.abc.Individual]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.abc.Individual`
        """
        # Check cls
        check_subclass(cls, "solution class", Individual)
        SingleSpeciesTrainer.solution_cls.fset(self, cls)

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

        :rtype: int
        :setter: Set a new population size
        :param size: The new population size. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_pop_size`
            is chosen
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        return self._pop_size

    @pop_size.setter
    def pop_size(self, size: int | None) -> None:
        """Set a new population size.

        :param size: The new population size. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_pop_size`
            is chosen
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        # Check the value
        self._pop_size = (
            self._default_pop_size
            if size is None else check_int(size, "population size", gt=0)
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_crossover_func(self) -> Callable[
        [Individual, Individual],
        tuple[Individual, Individual]
    ]:
        """Default crossover function.

        :return: The :meth:`~culebra.solution.abc.Individual.crossover` method
            of :attr:`~culebra.trainer.ea.abc.HomogeneousEA.solution_cls`
        :rtype: ~collections.abc.Callable
        """
        return self.solution_cls.crossover

    @property
    def crossover_func(self) -> Callable[
        [Individual, Individual],
        tuple[Individual, Individual]
    ]:
        """Crossover function.

        :rtype: ~collections.abc.Callable
        :setter: Set a new crossover function
        :param func: The new crossover function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_crossover_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._crossover_func

    @crossover_func.setter
    def crossover_func(
        self,
        func: Callable[
            [Individual, Individual],
            tuple[Individual, Individual]
        ] | None
    ) -> None:
        """Set a new crossover function.

        :param func: The new crossover function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_crossover_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._crossover_func = (
            self._default_crossover_func
            if func is None else check_func(func, "crossover function")
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_mutation_func(self) -> Callable[
        [Individual, float],
        tuple[Individual]
    ]:
        """Default mutation function.

        :return: The :meth:`~culebra.solution.abc.Individual.mutate` method
            of :attr:`~culebra.trainer.ea.abc.HomogeneousEA.solution_cls`
        :rtype: ~collections.abc.Callable
        """
        return self.solution_cls.mutate

    @property
    def mutation_func(self) -> Callable[
        [Individual, float],
        tuple[Individual]
    ]:
        """Mutation function.

        :rtype: ~collections.abc.Callable
        :setter: Set a new mutation function
        :param func: The new mutation function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_mutation_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._mutation_func

    @mutation_func.setter
    def mutation_func(
        self,
        func: Callable[
            [Individual, float],
            tuple[Individual]
        ] | None
    ) -> None:
        """Set a new mutation function.

        :param func: The new mutation function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_mutation_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._mutation_func = (
            self._default_mutation_func
            if func is None else check_func(func, "mutation function")
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_selection_func(
        self
    ) -> Callable[[list[Individual], int, Any], list[Individual]]:
        """Default selection function.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_SELECTION_FUNC

    @property
    def selection_func(
        self
    ) -> Callable[[list[Individual], int, Any], list[Individual]]:
        """Selection function.

        :rtype: ~collections.abc.Callable
        :setter: Set a new selection function
        :param func: The new selection function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._selection_func

    @selection_func.setter
    def selection_func(
        self,
        func: Callable[
            [list[Individual], int, Any],
            list[Individual]
        ] | None
    ) -> None:
        """Set a new selection function.

        :param func: The new selection function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._selection_func = (
            self._default_selection_func
            if func is None else check_func(func, "selection function")
        )

        # Reset the algorithm
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
        :setter: Set a new crossover probability
        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_crossover_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        return self._crossover_prob

    @crossover_prob.setter
    def crossover_prob(self, prob: float | None) -> None:
        """Set a new crossover probability.

        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_crossover_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._crossover_prob = (
            self._default_crossover_prob if prob is None else check_float(
                prob, "crossover probability", gt=0, lt=1)
        )

        # Reset the algorithm
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
        :setter: Set a new mutation probability
        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_mutation_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        return self._mutation_prob

    @mutation_prob.setter
    def mutation_prob(self, prob: float | None) -> None:
        """Set a new mutation probability.

        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_mutation_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._mutation_prob = (
            self._default_mutation_prob if prob is None else check_float(
                prob, "mutation probability", gt=0, lt=1
            )
        )

        # Reset the algorithm
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
        :setter: Set a new gene independent mutation probability
        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_gene_ind_mutation_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        return self._gene_ind_mutation_prob

    @gene_ind_mutation_prob.setter
    def gene_ind_mutation_prob(self, prob: float | None) -> None:
        """Set a new gene independent mutation probability.

        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_gene_ind_mutation_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._gene_ind_mutation_prob = (
            self._default_gene_ind_mutation_prob
            if prob is None else check_float(
                prob, "gene independent mutation probability", gt=0, lt=1
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_selection_func_params(self) -> dict[str, Any]:
        """Parameters of the default selection function.

        :return: :attr:`~culebra.trainer.ea.DEFAULT_SELECTION_FUNC_PARAMS`
        :rtype: float
        """
        return DEFAULT_SELECTION_FUNC_PARAMS

    @property
    def selection_func_params(self) -> dict[str, Any]:
        """Parameters of the selection function.

        :rtype: dict
        :setter: Set new parameters for the selection function
        :param params: The new parameters. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_selection_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        return self._selection_func_params

    @selection_func_params.setter
    def selection_func_params(self, params: dict[str, Any] | None) -> None:
        """Set new parameters for the selection function.

        :param params: The new parameters. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousEA._default_selection_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        # Check params
        self._selection_func_params = (
            self._default_selection_func_params
            if params is None else check_func_params(
                params, "selection function parameters"
            )
        )

        # Reset the algorithm
        self.reset()


class SinglePopEA(HomogeneousEA):
    """Base class for all the single population evolutionary algorithms."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[SinglePopEA], bool] | None = None,
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
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new single-population evolutionary trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.abc.SinglePopEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_crossover_func`
            is used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_mutation_func`
            is used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_selection_func`
            is used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_crossover_prob`
            is used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_mutation_prob`
            is used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_gene_ind_mutation_prob`
            is used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_selection_func_params`
            is used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.SinglePopEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

    @property
    def pop(self) -> list[Individual] | None:
        """Population.

        :rtype: list[~culebra.solution.abc.Individual]
        """
        return self._pop

    def _evaluate_pop(self, pop: list[Individual]) -> None:
        """Evaluate the individuals of *pop* that have an invalid fitness.

        :param pop: A population
        :type pop: list[~culebra.solution.abc.Individual]
        """
        invalid_inds = [ind for ind in pop if not ind.fitness.is_valid]

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
                    fitness_cls=self.fitness_function.fitness_cls)
            )

    def _get_state(self) -> dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :getter: Return the state
        :setter: Set a new state
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
        if self.verbosity:
            print(self._logbook.stream)

    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return (hof,)


class MultiPopEA(DistributedTrainer):
    """Base class for all the multiple population evolutionary algorithms."""

    stats_names = DISTRIBUTED_EA_STATS_NAMES
    """Statistics calculated each iteration."""

    @DistributedTrainer.subtrainer_cls.setter
    def subtrainer_cls(self, cls: type[SinglePopEA]) -> None:
        """Set a new trainer class to handle the subpopulations.

        Each subpopulation will be handled by a single-population evolutionary
        trainer.

        :param cls: The new class
        :type cls: type[~culebra.trainer.ea.abc.SinglePopEA]
        :raises TypeError: If *cls* is not a
            :class:`~culebra.trainer.ea.abc.SinglePopEA` subclass
        """
        # Check cls
        check_subclass(cls, "trainer class for subpopulations", SinglePopEA)
        DistributedTrainer.subtrainer_cls.fset(self, cls)


class IslandsEA(IslandsTrainer, MultiPopEA):
    """Base class for all the islands-based evolutionary algorithms."""

    def __init__(
        self,
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[IslandsEA], bool] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-population EA trainer class to handle
            the subpopulations
        :type subtrainer_cls: type[~culebra.trainer.ea.abc.SinglePopEA]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If omitted,
            :meth:`~culebra.trainer.ea.abc.IslandsEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param num_subtrainers: The number of subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.IslandsEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: dict
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        MultiPopEA.__init__(
            self,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls
        )
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    # Copy the culebra.trainer.ea.abc.HomogeneousEA.solution_cls property
    solution_cls = HomogeneousEA.solution_cls

    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        hof = ParetoFront()
        if self.subtrainers is not None:
            for subtrainer in self.subtrainers:
                if subtrainer.pop is not None:
                    hof.update(subtrainer.pop)

        return (hof,)

    @staticmethod
    def receive_representatives(subtrainer: SinglePopEA) -> None:
        """Receive representative solutions.

        :param subtrainer: The subtrainer receiving representatives
        :type subtrainer: ~culebra.trainer.ea.abc.SinglePopEA
        """
        container = subtrainer.container

        # Receive all the solutions in the queue
        queue = container._communication_queues[subtrainer.index]
        while not queue.empty():
            subtrainer._pop.extend(queue.get())

    @staticmethod
    def send_representatives(subtrainer: SinglePopEA) -> None:
        """Send representatives.

        :param subtrainer: The sender subtrainer
        :type subtrainer: ~culebra.trainer.ea.abc.SinglePopEA
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
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HomogeneousIslandsEA], bool] | None = None,
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
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-population EA trainer class to handle
            the subpopulations
        :type subtrainer_cls: type[~culebra.trainer.ea.abc.SinglePopEA]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_pop_size`
            is used. Defaults to :data:`None`
        :type pop_size: int
        :param crossover_func: The crossover function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_crossover_func`
            is used. Defaults to :data:`None`
        :type crossover_func: ~collections.abc.Callable
        :param mutation_func: The mutation function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_mutation_func`
            is used. Defaults to :data:`None`
        :type mutation_func: ~collections.abc.Callable
        :param selection_func: The selection function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_selection_func`
            is used. Defaults to :data:`None`
        :type selection_func: ~collections.abc.Callable
        :param crossover_prob: The crossover probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_crossover_prob`
            is used. Defaults to :data:`None`
        :type crossover_prob: float
        :param mutation_prob: The mutation probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_mutation_prob`
            is used. Defaults to :data:`None`
        :type mutation_prob: float
        :param gene_ind_mutation_prob: The gene independent mutation
            probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_gene_ind_mutation_prob`
            is used. Defaults to :data:`None`
        :type gene_ind_mutation_prob: float
        :param selection_func_params: The parameters for the selection
            function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_selection_func_params`
            is used. Defaults to :data:`None`
        :type selection_func_params: dict
        :param num_subtrainers: The number of subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: dict
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    @property
    def _default_pop_size(self) -> int:
        """Default population size.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: int
        """
        return None

    @HomogeneousEA.pop_size.getter
    def pop_size(self) -> int:
        """Population size.

        :return: If subtrainers have been generated, the subtrainers population
            size. Otherwise, the population size value used to call the
            constructor
        :rtype: int
        :setter: Set a new population size
        :param size: The new population size. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_pop_size`
            is chosen
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].pop_size
        return HomogeneousEA.pop_size.fget(self)

    @property
    def _default_crossover_func(self) -> Callable[
        [Individual, Individual],
        tuple[Individual, Individual]
    ]:
        """Default crossover function.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: ~collections.abc.Callable
        """
        return None

    @HomogeneousEA.crossover_func.getter
    def crossover_func(self) -> Callable[
        [Individual, Individual],
        tuple[Individual, Individual]
    ]:
        """Crossover function.

        :return: If subtrainers have been generated, the subtrainers crossover
            function. Otherwise, the crossover function value used to call the
            constructor
        :rtype: ~collections.abc.Callable
        :setter: Set a new crossover function
        :param func: The new crossover function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_crossover_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].crossover_func
        return HomogeneousEA.crossover_func.fget(self)

    @property
    def _default_mutation_func(self) -> Callable[
        [Individual, float],
        tuple[Individual]
    ]:
        """Default mutation function.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: ~collections.abc.Callable
        """
        return None

    @HomogeneousEA.mutation_func.getter
    def mutation_func(self) -> Callable[
        [Individual, float],
        tuple[Individual]
    ]:
        """Mutation function.

        :return: If subtrainers have been generated, the subtrainers mutation
            function. Otherwise, the mutation function value used to call the
            constructor
        :rtype: ~collections.abc.Callable
        :setter: Set a new mutation function
        :param func: The new mutation function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_mutation_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].mutation_func
        return HomogeneousEA.mutation_func.fget(self)

    @property
    def _default_selection_func(
        self
    ) -> Callable[[list[Individual], int, Any], list[Individual]]:
        """Default selection function.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: ~collections.abc.Callable
        """
        return None

    @HomogeneousEA.selection_func.getter
    def selection_func(
        self
    ) -> Callable[[list[Individual], int, Any], list[Individual]]:
        """Selection function.

        :return: If subtrainers have been generated, the subtrainers selection
            function. Otherwise, the selection function value used to call the
            constructor
        :rtype: ~collections.abc.Callable
        :setter: Set a new selection function
        :param func: The new selection function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].selection_func
        return HomogeneousEA.selection_func.fget(self)

    @property
    def _default_crossover_prob(self) -> float:
        """Default crossover probability.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: float
        """
        return None

    @HomogeneousEA.crossover_prob.getter
    def crossover_prob(self) -> float:
        """Crossover probability.

        :return: If subtrainers have been generated, the subtrainers crossover
            probability. Otherwise, the crossover probability value used to
            call the constructor
        :rtype: float
        :setter: Set a new crossover probability
        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_crossover_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].crossover_prob
        return HomogeneousEA.crossover_prob.fget(self)

    @property
    def _default_mutation_prob(self) -> float:
        """Default mutation probability.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: float
        """
        return None

    @HomogeneousEA.mutation_prob.getter
    def mutation_prob(self) -> float:
        """Mutation probability.

        :return: If subtrainers have been generated, the subtrainers mutation
            probability. Otherwise, the mutation probability value used to
            call the constructor
        :rtype: float
        :setter: Set a new mutation probability
        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_mutation_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].mutation_prob
        return HomogeneousEA.mutation_prob.fget(self)

    @property
    def _default_gene_ind_mutation_prob(self) -> float:
        """Default gene independent mutation probability.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: float
        """
        return None

    @HomogeneousEA.gene_ind_mutation_prob.getter
    def gene_ind_mutation_prob(self) -> float:
        """Gene independent mutation probability.

        :return: If subtrainers have been generated, the subtrainers gene
            independent mutation probability. Otherwise, the gene independent
            mutation probability value used to call the constructor
        :rtype: float
        :setter: Set a new gene independent mutation probability
        :param prob: The new probability. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_gene_ind_mutation_prob`
            is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].gene_ind_mutation_prob
        return HomogeneousEA.gene_ind_mutation_prob.fget(self)

    @property
    def _default_selection_func_params(self) -> dict[str, Any]:
        """Parameters of the default selection function.

        :return: :data:`None` to allow subtrainers use their default value
        :rtype: float
        """
        return None

    @HomogeneousEA.selection_func_params.getter
    def selection_func_params(self) -> dict[str, Any]:
        """Parameters of the selection function.

        :return: If subtrainers have been generated, the subtrainers parameters
            for the selection function. Otherwise, the parameters for the
            selection function used to call the constructor
        :rtype: dict
        :setter: Set new parameters for the selection function
        :param params: The new parameters. If omitted,
            :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA._default_selection_func_params`
            is chosen
        :type func: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].selection_func_params
        return HomogeneousEA.selection_func_params.fget(self)

    def _generate_subtrainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        :class:`~culebra.trainer.ea.abc.SinglePopEA` subtrainer, and change
        the subpopulation subtrainers'
        :attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subtrainers, if necessary.
        """

        def subtrainers_properties() -> dict[str, Any]:
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
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HeterogeneousEA], bool] | None = None,
        pop_sizes: int | Sequence[int] | None = None,
        crossover_funcs:
            Callable[
                [Individual, Individual], tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual], tuple[Individual, Individual]
                ]
            ] |
            None = None,
        mutation_funcs:
            Callable[[Individual, float], tuple[Individual]] |
            Sequence[Callable[[Individual, float], tuple[Individual]]] |
            None = None,
        selection_funcs:
            Callable[[list[Individual], int, Any], list[Individual]] |
            Sequence[
                Callable[[list[Individual], int, Any], list[Individual]]
            ] |
            None = None,
        crossover_probs: float | Sequence[float] | None = None,
        mutation_probs: float | Sequence[float] | None = None,
        gene_ind_mutation_probs: float | Sequence[float] | None = None,
        selection_funcs_params:
            dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-population EA trainer class to handle
            the subpopulations
        :type subtrainer_cls: type[~culebra.trainer.ea.abc.SinglePopEA]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.abc.HeterogeneousEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_sizes: The population size for each subpopulation.
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_pop_sizes`
            will be used. Defaults to :data:`None`
        :type pop_sizes: int | ~collections.abc.Sequence[int]
        :param crossover_funcs: The crossover function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_crossover_funcs`
            will be used. Defaults to :data:`None`
        :type crossover_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param mutation_funcs: The mutation function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_mutation_funcs`
            will be used. Defaults to :data:`None`
        :type mutation_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param selection_funcs: The selection function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_selection_funcs`
            will be used. Defaults to :data:`None`
        :type selection_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param crossover_probs: The crossover probability for each
            subpopulation. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_crossover_probs`
            will be used. Defaults to :data:`None`
        :type crossover_probs: float | ~collections.abc.Sequence[float]
        :param mutation_probs: The mutation probability for each subpopulation.
            If only a single value is provided, the same probability will be
            used for all the subpopulations. Different probabilities can be
            provided in a :class:`~collections.abc.Sequence`. All the
            probabilities must be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_mutation_probs`
            will be used. Defaults to :data:`None`
        :type mutation_probs: float | ~collections.abc.Sequence[float]
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation. If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_gene_ind_mutation_probs`
            will be used. Defaults to :data:`None`
        :type gene_ind_mutation_probs: float | ~collections.abc.Sequence[float]
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_selection_funcs_params`
            will be used. Defaults to :data:`None`
        :type selection_funcs_params: dict | ~collections.abc.Sequence[dict]
        :param num_subtrainers: The number of subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: dict
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
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
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
    def _default_pop_sizes(self) -> tuple[None]:
        """Default population size for each subtrainer.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def pop_sizes(self) -> tuple[int | None]:
        """Population size for each subtrainer.

        :return: If subtrainers have been generated, the subtrainers population
            size. Otherwise, the population size values used to call the
            constructor
        :rtype: tuple[int]
        :setter: Set the population size for each subtrainer
        :param sizes: The new population sizes. If only a single value
            is provided, the same size will be used for all the subtrainers.
            Different sizes can be provided in a
            :class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_pop_sizes`
            is chosen
        :type sizes: int | ~collections.abc.Sequence[int]
        :raises TypeError: If *sizes* is not an :class:`int`
            or a :class:`~collections.abc.Sequence` of :class:`int`
        :raises ValueError: If any value in *sizes* is not greater than zero
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.pop_size for subtrainer in self.subtrainers
            )
        return self._pop_sizes

    @pop_sizes.setter
    def pop_sizes(self, sizes: int | Sequence[int] | None) -> None:
        """Set the population size for each subtrainer.

        :param sizes: The new population sizes. If only a single value
            is provided, the same size will be used for all the subtrainers.
            Different sizes can be provided in a
            :class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_pop_sizes`
            is chosen
        :type sizes: int | ~collections.abc.Sequence[int]
        :raises TypeError: If *sizes* is not an :class:`int`
            or a :class:`~collections.abc.Sequence` of :class:`int`
        :raises ValueError: If any value in *sizes* is not greater than zero
        """
        # If None is provided ...
        if sizes is None:
            self._pop_sizes = self._default_pop_sizes
        # If a sequence is provided ...
        elif isinstance(sizes, Sequence):
            self._pop_sizes = tuple(
                check_sequence(
                    sizes,
                    "population sizes",
                    size=self.num_subtrainers,
                    item_checker=partial(check_int, gt=0)
                )
            )
        # If a scalar value is provided ...
        else:
            self._pop_sizes = (
                check_int(sizes, "population size", gt=0),
            ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_crossover_funcs(self) -> tuple[None]:
        """Default crossover function for each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def crossover_funcs(self) -> tuple[
        Callable[[Individual, Individual], tuple[Individual, Individual]] |
        None
    ]:
        """Crossover function for each subpopulation.

        :return: If subtrainers have been generated, the subtrainers crossover
            function. Otherwise, the crossover functions used to call the
            constructor
        :rtype: tuple[~collections.abc.Callable]
        :setter: Set the crossover function for each subpopulation
        :param funcs: The new crossover functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_crossover_funcs`
            is chosen
        :type funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :raises TypeError: If *funcs* is not
            :class:`~collections.abc.Callable` or a
            :class:`~collections.abc.Sequence` of
            :class:`~collections.abc.Callable`
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.crossover_func for subtrainer in self.subtrainers
            )
        return self._crossover_funcs

    @crossover_funcs.setter
    def crossover_funcs(
        self,
        funcs: Callable[
            [Individual, Individual],
            tuple[Individual, Individual]
        ] | Sequence[
            Callable[
                [Individual, Individual],
                tuple[Individual, Individual]
            ]
        ] | None
    ) -> None:
        """Set the crossover function for each subpopulation.

        :param funcs: The new crossover functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_crossover_funcs`
            is chosen
        :type funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :raises TypeError: If *funcs* is not
            :class:`~collections.abc.Callable` or a
            :class:`~collections.abc.Sequence` of
            :class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._crossover_funcs = self._default_crossover_funcs
        elif isinstance(funcs, Sequence):
            self._crossover_funcs = tuple(
                    check_sequence(
                    funcs,
                    "crossover functions",
                    size=self.num_subtrainers,
                    item_checker=check_func
                )
            )
        else:
            self._crossover_funcs = (
                check_func(funcs, "crossover function"),
            ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_mutation_funcs(self) -> tuple[None]:
        """Default mutation function for each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def mutation_funcs(self) -> tuple[
        Callable[[Individual, float], tuple[Individual]] | None
    ]:
        """Mutation function for each subpopulation.

        :return: If subtrainers have been generated, the subtrainers mutation
            function. Otherwise, the mutation functions used to call the
            constructor
        :rtype: tuple[~collections.abc.Callable]
        :setter: Set the mutation function for each subpopulation
        :param funcs: The new mutation functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_mutation_funcs`
            is chosen
        :type funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :raises TypeError: If *funcs* is not
            :class:`~collections.abc.Callable` or a
            :class:`~collections.abc.Sequence` of
            :class:`~collections.abc.Callable`
        """
        if self.subtrainers is not None:
            return tuple (
                subtrainer.mutation_func for subtrainer in self.subtrainers
            )
        return self._mutation_funcs

    @mutation_funcs.setter
    def mutation_funcs(
        self,
        funcs: Callable[
            [Individual, float],
            tuple[Individual]
        ] | Sequence[
            Callable[
                [Individual, float],
                tuple[Individual]
            ]
        ] | None
    ) -> None:
        """Set the mutation function for each subpopulation.

        :param funcs: The new mutation functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_mutation_funcs`
            is chosen
        :type funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :raises TypeError: If *funcs* is not
            :class:`~collections.abc.Callable` or a
            :class:`~collections.abc.Sequence` of
            :class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._mutation_funcs = self._default_mutation_funcs
        elif isinstance(funcs, Sequence):
            self._mutation_funcs = tuple(
                check_sequence(
                    funcs,
                    "mutation functions",
                    size=self.num_subtrainers,
                    item_checker=check_func
                )
            )
        else:
            self._mutation_funcs = (
                check_func(funcs, "mutation function"),
            ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_selection_funcs(self) -> tuple[None]:
        """Default selection function for each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def selection_funcs(self) -> tuple[
        Callable[[list[Individual], int, Any], list[Individual]] | None
    ]:
        """Selection function for each subpopulation.

        :return: If subtrainers have been generated, the subtrainers selection
            function. Otherwise, the selection functions used to call the
            constructor
        :rtype: tuple[float]
        :setter: Set a new selection function for each subpopulation
        :param funcs: The new selection functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_selection_funcs`
            is chosen
        :type funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :raises TypeError: If *funcs* is not
            :class:`~collections.abc.Callable` or a
            :class:`~collections.abc.Sequence` of
            :class:`~collections.abc.Callable`
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.selection_func for subtrainer in self.subtrainers
            )
        return self._selection_funcs

    @selection_funcs.setter
    def selection_funcs(self, funcs: Callable[
            [list[Individual], int, Any],
            list[Individual]
        ] | Sequence[
            Callable[
                [list[Individual], int, Any],
                list[Individual]
            ]
    ] | None
    ) -> None:
        """Set a new selection function for each subpopulation.

        :param funcs: The new selection functions. If only a single value is
            provided, the same function will be used for all the
            subpopulations. Different functions can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_selection_funcs`
            is chosen
        :type funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :raises TypeError: If *funcs* is not
            :class:`~collections.abc.Callable` or a
            :class:`~collections.abc.Sequence` of
            :class:`~collections.abc.Callable`
        """
        if funcs is None:
            self._selection_funcs = self._default_selection_funcs
        # If a sequence is provided ...
        elif isinstance(funcs, Sequence):
            self._selection_funcs = tuple(
                check_sequence(
                    funcs,
                    "selection functions",
                    size=self.num_subtrainers,
                    item_checker=check_func
                )
            )
        else:
            self._selection_funcs = (
                check_func(funcs, "selection function"),
        ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_crossover_probs(self) -> tuple[None]:
        """Default crossover probability for each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def crossover_probs(self) -> tuple[float | None]:
        """Crossover probability for each subpopulation.

        :return: If subtrainers have been generated, the subtrainers crossover
            probability. Otherwise, the crossover probabilities used to call
            the constructor
        :rtype: tuple[float]
        :setter: Set a new crossover probability for each subpopulation
        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_crossover_probs`
            is chosen
        :type probs: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *probs* is not a real number or a
            :class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any value in *probs* is not in (0, 1)
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.crossover_prob for subtrainer in self.subtrainers
            )
        return self._crossover_probs

    @crossover_probs.setter
    def crossover_probs(self, probs: float | Sequence[float] | None) -> None:
        """Set a new crossover probability for each subpopulation.

        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_crossover_probs`
            is chosen
        :type probs: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *probs* is not a real number or a
            :class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any value in *probs* is not in (0, 1)
        """
        if probs is None:
            self._crossover_probs = self._default_crossover_probs
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._crossover_probs = tuple(
                check_sequence(
                    probs,
                    "crossover probabilities",
                    size=self.num_subtrainers,
                    item_checker=partial(check_float, gt=0, lt=1)
                )
            )
        else:
            self._crossover_probs = (
                check_float(probs, "crossover probability", gt=0, lt=1),
            ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_mutation_probs(self) -> tuple[None]:
        """Default mutation probability for each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def mutation_probs(self) -> tuple[float | None]:
        """Mutation probability for each subpopulation.

        :return: If subtrainers have been generated, the subtrainers mutation
            probability. Otherwise, the mutation probabilities used to call
            the constructor
        :rtype: tuple[float]
        :setter: Set a new mutation probability for each subpopulation
        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_mutation_probs`
            is chosen
        :type probs: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *probs* is not a real number or a
            :class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any value in *probs* is not in (0, 1)
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.mutation_prob for subtrainer in self.subtrainers
            )
        return self._mutation_probs

    @mutation_probs.setter
    def mutation_probs(self, probs: float | Sequence[float] | None):
        """Set a new mutation probability for each subpopulation.

        :param probs: The new probabilities. If only a single value
            is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_mutation_probs`
            is chosen
        :type probs: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *probs* is not a real number or a
            :class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any value in *probs* is not in (0, 1)
        """
        if probs is None:
            self._mutation_probs = self._default_mutation_probs
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._mutation_probs = tuple(
                check_sequence(
                    probs,
                    "mutation probabilities",
                    size=self.num_subtrainers,
                    item_checker=partial(check_float, gt=0, lt=1)
                )
            )
        else:
            self._mutation_probs = (
                check_float(probs, "mutation probability", gt=0, lt=1),
            ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_gene_ind_mutation_probs(self) -> tuple[None]:
        """Default gene independent mutation probability for each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def gene_ind_mutation_probs(self) -> tuple[float | None]:
        """Gene independent mutation probabilities.

        :return: If subtrainers have been generated, the subtrainers gene
            independent mutation probability. Otherwise, the gene independent
            mutation probabilities used to call the constructor
        :rtype: tuple[float]
        :setter: Set new gene independent mutation probabilities
        :param probs: The new probabilities. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_gene_ind_mutation_probs`
            is chosen
        :type probs: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *probs* is not a real number or a
            :class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any value in *probs* is not in (0, 1)
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.gene_ind_mutation_prob for subtrainer in self.subtrainers
            )
        return self._gene_ind_mutation_probs

    @gene_ind_mutation_probs.setter
    def gene_ind_mutation_probs(self, probs: float | Sequence[float] | None):
        """Set new gene independent mutation probabilities.

        :param probs: The new probabilities. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_gene_ind_mutation_probs`
            is chosen
        :type probs: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *probs* is not a real number or a
            :class:`~collections.abc.Sequence` of real numbers
        :raises ValueError: If any value in *probs* is not in (0, 1)
        """
        if probs is None:
            self._gene_ind_mutation_probs = (
                self._default_gene_ind_mutation_probs
            )
        # If a sequence is provided ...
        elif isinstance(probs, Sequence):
            self._gene_ind_mutation_probs = tuple(
                check_sequence(
                    probs,
                    "gene independent mutation probabilities",
                    size=self.num_subtrainers,
                    item_checker=partial(check_float, gt=0, lt=1)
                )
            )
        else:
            self._gene_ind_mutation_probs = (
                check_float(
                    probs,
                    "gene independent mutation probability",
                    gt=0,
                    lt=1
                ),
            ) * self.num_subtrainers

        # Reset the algorithm
        self.reset()

    @property
    def _default_selection_funcs_params(self) -> tuple[None]:
        """Default parameters for the selection function of each subpopulation.

        :return: :data:`None` for each subtrainer, to allow subtrainers use
            their default value
        :rtype: tuple[None]
        """
        return (None, ) * self.num_subtrainers

    @property
    def selection_funcs_params(self) -> tuple[dict[str, Any] | None]:
        """Parameters for the selection function of each subpopulation.

        :return: The parameters for the selection function of each subtrainer.
            If subtrainers have not been initialized, the parameters of the
            selection function defined for each subtrainer in the trainer
            constructor is returned
        :rtype: tuple[dict]
        :setter: Set new parameters for the selection function of each
            subpopulation
        :param param_dicts: The new parameters. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_selection_funcs_params`
            is chosen
        :type param_dicts: dict | ~collections.abc.Sequence[dict]
        :raises TypeError: If *param_dicts* is not a :class:`dict`
            or a :class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :class:`dict`
        """
        if self.subtrainers is not None:
            return tuple(
                subtrainer.selection_func_params
                for subtrainer in self.subtrainers
            )
        return self._selection_funcs_params

    @selection_funcs_params.setter
    def selection_funcs_params(
        self,
        param_dicts: dict[str, Any] | Sequence[dict[str, Any]] | None
    ) -> None:
        """Set new parameters for the selection function of each subpopulation.

        :param param_dicts: The new parameters. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. If set to :data`None`,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousEA._default_selection_funcs_params`
            is chosen
        :type param_dicts: dict | ~collections.abc.Sequence[dict]
        :raises TypeError: If *param_dicts* is not a :class:`dict`
            or a :class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :class:`dict`
        """
        if param_dicts is None:
            self._selection_funcs_params = self._default_selection_funcs_params
        # If a sequence is provided ...
        elif isinstance(param_dicts, Sequence):
            self._selection_funcs_params = tuple(
                check_sequence(
                    param_dicts,
                    "selection functions parameters",
                    size=self.num_subtrainers,
                    item_checker=check_func_params
                )
            )
        else:
            self._selection_funcs_params = (
                check_func_params(param_dicts,"selection function parameters"),
            ) * self.num_subtrainers

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
        solution_cls: type[Individual],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[HeterogeneousIslandsEA], bool] | None = None,
        pop_sizes: int | Sequence[int] | None = None,
        crossover_funcs:
            Callable[
                [Individual, Individual], tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual], tuple[Individual, Individual]
                ]
            ] |
            None = None,
        mutation_funcs:
            Callable[[Individual, float], tuple[Individual]] |
            Sequence[Callable[[Individual, float], tuple[Individual]]] |
            None = None,
        selection_funcs:
            Callable[[list[Individual], int, Any], list[Individual]] |
            Sequence[
                Callable[[list[Individual], int, Any], list[Individual]]
            ] |
            None = None,
        crossover_probs: float | Sequence[float] | None = None,
        mutation_probs: float | Sequence[float] | None = None,
        gene_ind_mutation_probs: float | Sequence[float] | None = None,
        selection_funcs_params:
            dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The individual class
        :type solution_cls: type[~culebra.solution.abc.Individual]
        :param species: The species for all the individuals
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-population EA trainer class to handle
            the subpopulations
        :type subtrainer_cls: type[~culebra.trainer.ea.abc.SinglePopEA]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_sizes: The population size for each subpopulation.
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If omitted,
            :attr:`~culebra.trainer.ea.DEFAULT_POP_SIZE` will be used.
            Defaults to :data:`None`
        :param pop_sizes: The population size for each subpopulation.
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_pop_sizes`
            will be used. Defaults to :data:`None`
        :type pop_sizes: int | ~collections.abc.Sequence[int]
        :param crossover_funcs: The crossover function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_crossover_funcs`
            will be used. Defaults to :data:`None`
        :type crossover_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param mutation_funcs: The mutation function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_mutation_funcs`
            will be used. Defaults to :data:`None`
        :type mutation_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param selection_funcs: The selection function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_selection_funcs`
            will be used. Defaults to :data:`None`
        :type selection_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param crossover_probs: The crossover probability for each
            subpopulation. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_crossover_probs`
            will be used. Defaults to :data:`None`
        :type crossover_probs: float | ~collections.abc.Sequence[float]
        :param mutation_probs: The mutation probability for each subpopulation.
            If only a single value is provided, the same probability will be
            used for all the subpopulations. Different probabilities can be
            provided in a :class:`~collections.abc.Sequence`. All the
            probabilities must be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_mutation_probs`
            will be used. Defaults to :data:`None`
        :type mutation_probs: float | ~collections.abc.Sequence[float]
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation. If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_gene_ind_mutation_probs`
            will be used. Defaults to :data:`None`
        :type gene_ind_mutation_probs: float | ~collections.abc.Sequence[float]
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_selection_funcs_params`
            will be used. Defaults to :data:`None`
        :type selection_funcs_params: dict | ~collections.abc.Sequence[dict]
        :param num_subtrainers: The number of subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.HeterogeneousIslandsEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subpopulations
            trainer
        :type subtrainer_params: dict
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        HeterogeneousEA.__init__(
            self,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            num_subtrainers=num_subtrainers,
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    def _generate_subtrainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        :class:`~culebra.trainer.ea.abc.SinglePopEA` subtrainer, and change
        the subpopulation subtrainers'
        :attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subtrainers, if necessary.

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subpopulations.
        """

        def subtrainers_properties() -> list[dict[str, Any]]:
            """Obtain the properties of each subpopulation trainer.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation trainer.
            :rtype: list[dict]
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

            # list with the common properties. Equal for all the subpopulations
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
        solution_classes: Sequence[type[Individual]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SinglePopEA],
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[CooperativeEA], bool] | None = None,
        pop_sizes: int | Sequence[int] | None = None,
        crossover_funcs:
            Callable[
                [Individual, Individual], tuple[Individual, Individual]
            ] |
            Sequence[
                Callable[
                    [Individual, Individual], tuple[Individual, Individual]
                ]
            ] |
            None = None,
        mutation_funcs:
            Callable[[Individual, float], tuple[Individual]] |
            Sequence[Callable[[Individual, float], tuple[Individual]]] |
            None = None,
        selection_funcs:
            Callable[[list[Individual], int, Any], list[Individual]] |
            Sequence[
                Callable[[list[Individual], int, Any], list[Individual]]
            ] |
            None = None,
        crossover_probs: float | Sequence[float] | None = None,
        mutation_probs: float | Sequence[float] | None = None,
        gene_ind_mutation_probs: float | Sequence[float] | None = None,
        selection_funcs_params:
            dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Individual], Any], Individual] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        Each species is evolved in a different subpopulation.

        :param solution_classes: The individual class for each species
        :type solution_classes:
            ~collections.abc.Sequence[type[~culebra.abc.Solution]]
        :param species: The species to be evolved
        :type species: ~collections.abc.Sequence[~culebra.abc.Species]
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-population EA trainer class to handle
            the subpopulations
        :type subtrainer_cls: type[~culebra.trainer.ea.abc.SinglePopEA]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.ea.abc.CooperativeEA._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param pop_sizes: The population size for each subpopulation.
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_pop_sizes`
            will be used. Defaults to :data:`None`
        :type pop_sizes: int | ~collections.abc.Sequence[int]
        :param crossover_funcs: The crossover function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_crossover_funcs`
            will be used. Defaults to :data:`None`
        :type crossover_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param mutation_funcs: The mutation function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_mutation_funcs`
            will be used. Defaults to :data:`None`
        :type mutation_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param selection_funcs: The selection function for each subpopulation.
            If only a single value is provided, the same function will be used
            for all the subpopulations. Different functions can be provided in
            a :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_selection_funcs`
            will be used. Defaults to :data:`None`
        :type selection_funcs: ~collections.abc.Callable |
            ~collections.abc.Sequence[~collections.abc.Callable]
        :param crossover_probs: The crossover probability for each
            subpopulation. If only a single value is provided, the
            same probability will be used for all the subpopulations.
            Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_crossover_probs`
            will be used. Defaults to :data:`None`
        :type crossover_probs: float | ~collections.abc.Sequence[float]
        :param mutation_probs: The mutation probability for each subpopulation.
            If only a single value is provided, the same probability will be
            used for all the subpopulations. Different probabilities can be
            provided in a :class:`~collections.abc.Sequence`. All the
            probabilities must be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_mutation_probs`
            will be used. Defaults to :data:`None`
        :type mutation_probs: float | ~collections.abc.Sequence[float]
        :param gene_ind_mutation_probs: The gene independent mutation
            probability for each subpopulation. If only a single
            value is provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. All the probabilities must
            be in (0, 1). If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_gene_ind_mutation_probs`
            will be used. Defaults to :data:`None`
        :type gene_ind_mutation_probs: float | ~collections.abc.Sequence[float]
        :param selection_funcs_params: The parameters for the selection
            function of each subpopulation. If only a single value is
            provided, the same probability will be used for all the
            subpopulations. Different probabilities can be provided in a
            :class:`~collections.abc.Sequence`. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_selection_funcs_params`
            will be used. Defaults to :data:`None`
        :type selection_funcs_params: dict | ~collections.abc.Sequence[dict]
        :param num_subtrainers: The number of subtrainers (species). If
            omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (species). If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.ea.abc.CooperativeEA._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subpopulations
            (species) trainer
        :type subtrainer_params: dict
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
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    def _generate_subtrainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :attr:`~culebra.trainer.ea.abc.SinglePopEA.index`
        and a :attr:`~culebra.trainer.ea.abc.SinglePopEA.container` to each
        :class:`~culebra.trainer.ea.abc.SinglePopEA` subtrainer, and change
        the subpopulation subtrainers'
        :attr:`~culebra.trainer.ea.abc.SinglePopEA.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._preprocess_iteration`
        and
        :meth:`~culebra.trainer.ea.abc.SinglePopEA._postprocess_iteration`
        methods of the
        :attr:`~culebra.trainer.ea.abc.HomogeneousIslandsEA.subtrainer_cls`
        class are dynamically overridden, in order to allow individuals
        exchange between subtrainers, if necessary.

        :raises RuntimeError: If the length of any properties sequence does
            not match the number of subpopulations.
        """

        def subtrainers_properties() -> list[dict[str, Any]]:
            """Obtain the properties of each subpopulation.

            :raises RuntimeError: If the length of any properties sequence
                does not match the number of subpopulations.

            :return: The properties of each subpopulation.
            :rtype: list[dict]
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
    def receive_representatives(subtrainer: SinglePopEA) -> None:
        """Receive representative individuals.

        :param subtrainer: The subtrainer receiving
            representatives
        :type subtrainer: ~culebra.trainer.ea.abc.SinglePopEA
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
    def send_representatives(subtrainer: SinglePopEA) -> None:
        """Send representatives.

        :param subtrainer: The sender subtrainer
        :type subtrainer: ~culebra.trainer.ea.abc.SinglePopEA
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
