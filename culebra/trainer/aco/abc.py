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

"""Abstract base classes for the ACO-based trainers.

This module provides several abstract classes for different kind of
Ant Colony Optimization based trainers. The base class for all the single
colony ACO-based trainers is :class:`~culebra.trainer.aco.abc.SingleColACO`.
Elitism is also supported via the :class:`~culebra.trainer.aco.abc.ElitistACO`.

* Regarding the matrices structure of the ACO-based approach:

  * :class:`~culebra.trainer.aco.abc.MultipleHeuristicMatricesACO`: A base
    class for all the single colony ACO-based trainers which use multiple
    heuristic matrices
  * :class:`~culebra.trainer.aco.abc.MultiplePheromoneMatricesACO`: A base
    class for all the single colony ACO-based trainers which use multiple
    phermone matrices
  * :class:`~culebra.trainer.aco.abc.SingleHeuristicMatrixACO`: A base class
    for all the single colony ACO-based trainers which rely on a single
    heuristic matrix
  * :class:`~culebra.trainer.aco.abc.SingleObjACO`: A base class for all
    the single colony ACO-based trainers which optimize only a single
    objective, that is, with a single pheromone matrix and also a single
    heuristic matrix
  * :class:`~culebra.trainer.aco.abc.SinglePheromoneMatrixACO`: A base class
    for all the single colony ACO-based trainers which rely on a single
    pheromone matrix

* With respect to the pheromone updating procedure:

  * Approaches relying on pheromone evaporation:

    * :class:`~culebra.trainer.aco.abc.PheromoneBasedACO`: A base class for
      all pheromone-based single objective ACO trainers
    * :class:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO`: A base
      class for all the elitist, pheromone-based, and reseteable single colony
      ACO trainers

  * Population-based approaches:

    * :class:`~culebra.trainer.aco.abc.MaxPheromonePACO`: A base class for all
      the population-based single colony ACO trainers using a maximum pheromone
      amount
    * :class:`~culebra.trainer.aco.abc.PACO`: A base class for all
      the population-based single colony ACO trainers
    * :class:`~culebra.trainer.aco.abc.SingleObjPACO`: A base class for all
      the population-based single colony and single objective ACO trainers

* Finally, regarding the kind of problem:

  * :class:`~culebra.trainer.aco.abc.ACOFS`: A base class for all the
    ACO-based approaches for FS
  * :class:`~culebra.trainer.aco.abc.ACOTSP`: A base class for all the
    ACO-based approaches for TSP
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    Tuple,
    Type,
    List,
    Callable,
    Dict,
    Optional
)
from collections.abc import Sequence
from random import random
from math import comb
from functools import partial
from itertools import repeat, combinations
from copy import deepcopy

import numpy as np
from deap.tools import HallOfFame, ParetoFront


from culebra.abc import Species, FitnessFunction
from culebra.checker import (
    check_subclass,
    check_instance,
    check_sequence,
    check_int,
    check_float,
    check_matrix
)
from culebra.solution.abc import Ant
from culebra.solution.tsp import Ant as TSPAnt, Species as TSPSpecies
from culebra.solution.feature_selection import (
    Ant as FSAnt,
    Species as FSSpecies
)
from culebra.fitness_function.tsp.abc import TSPFitnessFunction

from culebra.trainer.abc import SingleSpeciesTrainer

from .constants import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_EXPLOITATION_PROB,
    DEFAULT_PHEROMONE_DEPOSIT_WEIGHT,
    DEFAULT_PHEROMONE_EVAPORATION_RATE,
    DEFAULT_CONVERGENCE_CHECK_FREQ,
    DEFAULT_ACOFS_INITIAL_PHEROMONE,
    DEFAULT_ACOFS_HEURISTIC_INFLUENCE,
    DEFAULT_ACOFS_EXPLOITATION_PROB,
    DEFAULT_ACOFS_DISCARD_PROB
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class SingleColACO(SingleSpeciesTrainer):
    """Base class for all the single colony ACO algorithms."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SingleColACO],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new single-colony ACO trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclass
        SingleSpeciesTrainer.__init__(
            self,
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
        self.initial_pheromone = initial_pheromone
        self.heuristic = heuristic
        self.pheromone_influence = pheromone_influence
        self.heuristic_influence = heuristic_influence
        self.exploitation_prob = exploitation_prob
        self.col_size = col_size

        # The phermonne matrices have not been initialized yet
        self._pheromone = None

    @property
    @abstractmethod
    def num_pheromone_matrices(self) -> int:
        """Number of pheromone matrices used by this trainer.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The num_pheromone_matrices property has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def num_heuristic_matrices(self) -> int:
        """Number of heuristic matrices used by this trainer.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The num_heuristic_matrices property has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def pheromone_shapes(self) -> Sequence[Tuple[int, int], ...]:
        """Shape of the pheromone matrices.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Sequence[tuple[int]]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The pheromone_shapes property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def heuristic_shapes(self) -> Sequence[Tuple[int, int], ...]:
        """Shape of the heuristic matrices.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Sequence[tuple[int]]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The heuristic_shapes property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    def solution_cls(self) -> Type[Ant]:
        """Solution class.

        :rtype: type[~culebra.solution.abc.Ant]
        :setter: Set a new solution class
        :param cls: The new class
        :type cls: type[~culebra.solution.abc.Ant]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.abc.Ant` subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: Type[Ant]) -> None:
        """Set a new solution class.

        :param cls: The new class
        :type cls: type[~culebra.solution.abc.Ant]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.abc.Ant` subclass
        """
        # Check cls
        self._solution_cls = check_subclass(cls, "solution class", Ant)

        # Reset the trainer
        self.reset()

    @property
    def initial_pheromone(self) -> Sequence[float, ...]:
        """Initial value for each pheromone matrix.

        :return: One initial value for each pheromone matrix
        :rtype: ~collections.abc.Sequence[float]
        :setter: Set the initial value for each pheromone matrix
        :param values: New initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the pheromone matrices
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
        """
        return self._initial_pheromone

    @initial_pheromone.setter
    def initial_pheromone(self, values: float | Sequence[float, ...]) -> None:
        """Set the initial value for each pheromone matrix.

        :param values: New initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the pheromone matrices
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
        """
        # Check the values
        if isinstance(values, Sequence):
            self._initial_pheromone = check_sequence(
                    values,
                    "initial pheromone",
                    item_checker=partial(check_float, gt=0)
                )

            if len(self._initial_pheromone) == 1:
                if self.num_pheromone_matrices > 1:
                    self._initial_pheromone = list(
                        repeat(
                            self._initial_pheromone[0],
                            self.num_pheromone_matrices
                        )
                    )
            # Check the length
            elif len(self._initial_pheromone) != self.num_pheromone_matrices:
                raise ValueError(
                    "Incorrect number of initial pheromone values"
                )
        else:
            self._initial_pheromone = list(
                repeat(
                    check_float(
                        values,
                        "initial pheromone", gt=0
                    ),
                    self.num_pheromone_matrices
                )
            )

        # Reset the trainer
        self.reset()

    @property
    def heuristic(self) -> Sequence[np.ndarray[float], ...]:
        """Heuristic matrices.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Sequence[~numpy.ndarray[float]]
        :setter: Set new heuristic matrices
        :param values: The new heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the heuristic matrices.
            If set to :data:`None`, the default heuristic matrices for the
            problem are assumed
        :type values:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The heuristic property has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @heuristic.setter
    def heuristic(
        self,
        values: Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...] | None
    ) -> None:
        """Set new heuristic matrices.

        This property must be overridden by subclasses to return a correct
        value.

        :param values: The new heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the heuristic matrices.
            If set to :data:`None`, the default heuristic matrices for the
            problem are assumed
        :type values:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The heuristic property has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @property
    def pheromone_influence(self) -> Sequence[float, ...]:
        r"""Relative influence of pheromone (:math:`{\alpha}`).

        :return: One value for each pheromone matrix
        :rtype: ~collections.abc.Sequence[float]

        :getter: Return the relative influence of each pheromone matrix.
        :setter: Set new values for the relative influence of each pheromone
            matrix
        :param values: New value for the relative influence of each pheromone
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            pheromone matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` is
            chosen
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
        """
        return (
            [DEFAULT_PHEROMONE_INFLUENCE] * self.num_pheromone_matrices
            if self._pheromone_influence is None
            else self._pheromone_influence
        )

    @pheromone_influence.setter
    def pheromone_influence(
        self, values: float | Sequence[float, ...]
    ) -> None:
        r"""Set the relative influence of pheromone (:math:`{\alpha}`).

        :param values: New value for the relative influence of each pheromone
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            pheromone matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` is
            chosen
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
        """
        if values is None:
            self._pheromone_influence = None
        else:
            if isinstance(values, Sequence):
                self._pheromone_influence = check_sequence(
                        values,
                        "pheromone influence",
                        item_checker=partial(check_float, ge=0)
                    )

                if len(self._pheromone_influence) == 1:
                    if self.num_pheromone_matrices > 1:
                        self._pheromone_influence = list(
                            repeat(
                                self._pheromone_influence[0],
                                self.num_pheromone_matrices
                            )
                        )
                # Check the length
                elif (
                    len(self._pheromone_influence) !=
                    self.num_pheromone_matrices
                ):
                    raise ValueError(
                        "Incorrect number of pheromone influence values"
                    )
            else:
                self._pheromone_influence = list(
                    repeat(
                        check_float(
                            values,
                            "pheromone influence", ge=0
                        ),
                        self.num_pheromone_matrices
                    )
                )

        # Reset the trainer
        self.reset()

    @property
    def heuristic_influence(self) -> Sequence[float, ...]:
        r"""Relative influence of heuristic (:math:`{\beta}`).

        :return: One value for each heuristic matrix
        :rtype: ~collections.abc.Sequence[float]
        :setter: Set new values for the relative influence of each heuristic
            matrix
        :param values: New value for the relative influence of each heuristic
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            heuristic matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` is
            chosen.
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        """
        return (
            [DEFAULT_HEURISTIC_INFLUENCE] * self.num_heuristic_matrices
            if self._heuristic_influence is None
            else self._heuristic_influence
        )

    @heuristic_influence.setter
    def heuristic_influence(
        self, values: float | Sequence[float, ...]
    ) -> None:
        r"""Set the relative influence of heuristic (:math:`{\beta}`).

        :param values: New value for the relative influence of each heuristic
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            heuristic matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` is
            chosen.
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        """
        if values is None:
            self._heuristic_influence = None
        else:
            if isinstance(values, Sequence):
                self._heuristic_influence = check_sequence(
                        values,
                        "heuristic influence",
                        item_checker=partial(check_float, ge=0)
                    )

                if len(self._heuristic_influence) == 1:
                    if self.num_heuristic_matrices > 1:
                        self._heuristic_influence = list(
                            repeat(
                                self._heuristic_influence[0],
                                self.num_heuristic_matrices
                            )
                        )
                # Check the length
                elif (
                    len(self._heuristic_influence) !=
                    self.num_heuristic_matrices
                ):
                    raise ValueError(
                        "Incorrect number of heuristic influence values"
                    )
            else:
                self._heuristic_influence = list(
                    repeat(
                        check_float(
                            values,
                            "heuristic influence", ge=0
                        ),
                        self.num_heuristic_matrices
                    )
                )

        # Reset the trainer
        self.reset()

    @property
    def exploitation_prob(self) -> float:
        """Exploitation probability (:math:`{q_0}`).

        :rtype: float
        :setter: Set a new value for the exploitation probability
        :param prob: The new probability. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in [0, 1]
        """
        return (
            DEFAULT_EXPLOITATION_PROB
            if self._exploitation_prob is None
            else self._exploitation_prob
        )

    @exploitation_prob.setter
    def exploitation_prob(self, prob: float | None) -> None:
        """Set a new exploitation probability (:math:`{q_0}`).

        :param prob: The new probability. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` is chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in [0, 1]
        """
        # Check prob
        self._exploitation_prob = (
            None if prob is None else check_float(
                prob, "exploitation probability", ge=0, le=1)
        )

        # Reset the trainer
        self.reset()

    @property
    def col_size(self) -> int:
        """Colony size.

        This property must be overridden by subclasses to return a correct
        default value.

        :rtype: int
        :setter: Set a new value for the colony size
        :param size: The new colony size
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        raise NotImplementedError(
            "The col_size property getter has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @col_size.setter
    def col_size(self, size: int | None) -> None:
        """Set a new value for the colony size.

        :param size: The new colony size
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        # Check the value
        self._col_size = (
            None if size is None else check_int(size, "colony size", gt=0)
        )

        # Reset the trainer
        self.reset()

    @property
    def col(self) -> List[Ant] | None:
        """Colony.

        :return: The colony or :data:`None` if it has not been generated yet
        :rtype: list[~culebra.solution.abc.Ant]
        """
        return self._col

    @property
    def pheromone(self) -> Sequence[np.ndarray[float], ...] | None:
        """Pheromone matrices.

        :return: The pheromone matrices or :data:`None` if the search process
            has not begun
        :rtype: ~collections.abc.Sequence[~numpy.ndarray[float]]
        """
        return self._pheromone

    @property
    def choice_info(self) -> np.ndarray[float] | None:
        """Choice information for all the graph's arcs.

        The choice information is generated from both the pheromone and the
        heuristic matrices, modified by other parameters (depending on the ACO
        approach) and is used to obtain the probalility of following the next
        feasible arc for the node.

        :return: The choice information or :data:`None` if the search process
            has not begun
        :rtype: ~numpy.ndarray[float]
        """
        return self._choice_info

    def _ant_choice_info(self, ant: Ant) -> np.ndarray[float]:
        """Return the choice info to obtain the next node the ant will visit.

        All the previously visited nodes are discarded. Subclasses should
        override this method if the :class:`~culebra.abc.Species`
        constraining the solutions of the problem supports node banning.

        :param ant: The ant
        :type ant: ~culebra.solution.abc.Ant
        :rtype: ~numpy.ndarray[float]
        """
        if ant.current is None:
            ant_choice_info = np.sum(self.choice_info, axis=1)
        else:
            ant_choice_info = np.copy(self.choice_info[ant.current])

        ant_choice_info[ant.path] = 0
        ant_choice_info[ant.discarded] = 0

        return ant_choice_info

    def _next_choice(self, ant: Ant) -> int | None:
        """Choose the next node for an ant.

        The election is made from the feasible neighborhood of the current
        node, which is composed of those nodes neither discarded nor visited
        yet by the ant and connected to its current node.

        The best possible node is selected with probability
        :attr:`~culebra.trainer.aco.abc.SingleColACO.exploitation_prob`. In
        case the best node is not chosen, the next node is selected
        probabilistically according to the
        :attr:`~culebra.trainer.aco.abc.SingleColACO.choice_info` matrix.

        :param ant: The ant
        :type ant: ~culebra.solution.abc.Ant
        :return: The index of the chosen node or :data:`None` if there
            isn't any feasible node
        :rtype: int
        """
        # Choice info for the feasible neighborhood of the current node
        ant_choice_info = self._ant_choice_info(ant)

        # If there is any feasible node ...
        if np.any(ant_choice_info > 0):
            if random() < self.exploitation_prob:
                # Exploitation
                return np.random.choice(
                    np.where(
                        ant_choice_info == np.max(ant_choice_info)
                    )[0]
                )
            else:
                # Exploration
                node_list = np.arange(
                    0, len(ant_choice_info), dtype=int
                )

                return np.random.choice(
                    node_list,
                    p=ant_choice_info/np.sum(ant_choice_info)
                )

        # If there isn't any feasible node
        return None

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: list[~deap.tools.HallOfFame]
        """
        hof = ParetoFront()
        if self.col is not None:
            hof.update(self.col)
        return [hof]

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :class:`~culebra.trainer.aco.abc.SingleColACO` class, the colony, the
        choice_info matrix and the node list are created. Subclasses which
        need more objects or data structures should override this method.
        """
        super()._init_internals()
        self._col = []
        self._choice_info = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the colony, the choice_info matrix and the node
        list. If subclasses overwrite the
        :meth:`~culebra.trainer.aco.abc.SingleColACO._init_internals`
        method to add any new internal object, this method should also be
        overridden to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._col = None
        self._choice_info = None

    @abstractmethod
    def _calculate_choice_info(self) -> None:
        """Calculate the choice information.

        The choice information is generated from both the pheromone and the
        heuristic matrices, modified by other parameters (depending on the ACO
        approach) and is used to obtain the probalility of following the next
        feasible arc for the node.

        This method should be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _calculate_choice_info method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the iteration metrics (number of evaluations, execution time)
        before each iteration is run and create an empty ant colony.
        Overridden to calculate the choice information before executing the
        next iteration.
        """
        super()._start_iteration()
        self._col = []
        self._calculate_choice_info()

    def _generate_ant(self) -> Ant:
        """Generate a new ant.

        The ant makes its path and gets evaluated.

        :return: The new ant
        :rtype: ~culebra.solution.abc.Ant
        """
        ant = self.solution_cls(
            self.species,
            self.fitness_function.fitness_cls
        )
        choice = self._next_choice(ant)

        while choice is not None:
            ant.append(choice)
            choice = self._next_choice(ant)

        self.evaluate(ant)

        return ant

    def _generate_col(self) -> None:
        """Fill the colony with evaluated ants."""
        # Fill the colony
        while len(self.col) < self.col_size:
            self.col.append(self._generate_ant())

    def _init_pheromone(self) -> None:
        """Init the pheromone matrix(ces) according to the initial value(s)."""
        self._pheromone = [
            np.full(
                shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone, shape in zip(
                self.initial_pheromone, self.pheromone_shapes
            )
        ]

    def _pheromone_amount (self, ant: Ant) -> Tuple[float, ...]:
        """Return the amount of pheromone to be deposited by an ant.

        The reciprocal of an objective fitness value will be used for
        minimization objectives, while the objective's fitness value is used
        for maximization objectives.

        :param ant: The ant
        :type ant: ~culebra.solution.abc.Ant
        :return: The amount of pheromone to be deposited for each objective
        :rtype: tuple[float]
        """
        return tuple(
            1/value if weight < 0 else value
            for (value, weight) in zip(ant.fitness.values, ant.fitness.weights)
        )

    def _deposit_pheromone(
        self,
        ants: Sequence[Ant],
        weight: float = DEFAULT_PHEROMONE_DEPOSIT_WEIGHT
    ) -> None:
        """Make some ants deposit weighted pheromone.

        This method must be overridden by subclasses to take into account
        the correct number and shape of the pheromone matrices.

        :param ants: The ants
        :type ants: ~collections.abc.Sequence[~culebra.solution.abc.Ant]
        :param weight: Weight for the pheromone. Defaults to
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_DEPOSIT_WEIGHT`
        :type weight: float
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _deposit_pheromone method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @abstractmethod
    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone.

        This method should be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _increase_pheromone method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone.

        This method should be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _decrease_pheromone method has not been implemented in the "
            f"{self.__class__.__name__} class")

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        First, pheromone is decreased by
        :meth:`~culebra.trainer.aco.abc.SingleColACO._decrease_pheromone`, and
        then it is increased by
        :meth:`~culebra.trainer.aco.abc.SingleColACO._increase_pheromone`
        """
        self._decrease_pheromone()
        self._increase_pheromone()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the pheromone
        self._update_pheromone()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        record = self._stats.compile(self.col) if self._stats else {}
        record["Iter"] = self._current_iter
        record["NEvals"] = self._current_iter_evals
        if self.container is not None:
            record["Col"] = self.index
        self._logbook.record(**record)
        if self.verbose:
            print(self._logbook.stream)

    def __copy__(self) -> SingleColACO:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.aco.abc.SingleColACO
        """
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.initial_pheromone
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> SingleColACO:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.aco.abc.SingleColACO
        """
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.initial_pheromone
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__,
                (
                    self.solution_cls,
                    self.species,
                    self.fitness_function,
                    self.initial_pheromone
                ),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> SingleColACO:
        """Return a single colony ACO trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.aco.abc.SingleColACO
        """
        obj = cls(
            state['_solution_cls'],
            state['_species'],
            state['_fitness_function'],
            state['_initial_pheromone']
        )
        obj.__setstate__(state)
        return obj


class SinglePheromoneMatrixACO(SingleColACO):
    """Base class for all the single pheromone matrix ACO algorithms."""

    @property
    def num_pheromone_matrices(self) -> int:
        """Number of pheromone matrices used by this trainer.

        :rtype: int
        """
        return 1


class SingleHeuristicMatrixACO(SingleColACO):
    """Base class for all the single heuristic matrix ACO algorithms."""

    @property
    def num_heuristic_matrices(self) -> int:
        """Number of heuristic matrices used by this trainer.

        :rtype: int
        """
        return 1


class MultiplePheromoneMatricesACO(SingleColACO):
    """Base class for all the multiple pheromone matrices ACO algorithms."""

    @property
    def num_pheromone_matrices(self) -> int:
        """Number of pheromone matrices used by this trainer.

        :rtype: int
        """
        return self.fitness_function.num_obj


class MultipleHeuristicMatricesACO(SingleColACO):
    """Base class for all the multiple heuristic matrices ACO algorithms."""

    @property
    def num_heuristic_matrices(self) -> int:
        """Number of heuristic matrices used by this trainer.

        :rtype: int
        """
        return self.fitness_function.num_obj


class SingleObjACO(SinglePheromoneMatrixACO, SingleHeuristicMatrixACO):
    """Base class for all the single-objective ACO algorithms."""

    @property
    def fitness_function(self) -> FitnessFunction:
        """Training fitness function.

        :rtype: ~culebra.abc.FitnessFunction

        :setter: Set a new fitness function
        :param func: The new training fitness function
        :type func: ~culebra.abc.FitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        :raises ValueError: If *func* has more than one objective
        """
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, func: FitnessFunction) -> None:
        """Set a new training fitness function.

        :param func: The new training fitness function
        :type func: ~culebra.abc.FitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        :raises ValueError: If *func* has more than one objective
        """
        SinglePheromoneMatrixACO.fitness_function.fset(self, func)
        if self.fitness_function.num_obj != 1:
            raise ValueError("Incorrect number of objectives")


class PheromoneBasedACO(SingleColACO):
    """Base class for all the ACO approaches guided by pheromone matrices.

    This kind of ACO approach relies on the pheromone matrices to guide the
    search, modified by the colony's ants at each iteration. Thus, the
    pheromone matrices must be part of the trainer's state.
    """

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        pheromone_evaporation_rate: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [PheromoneBasedACO],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new single-colony ACO trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.PheromoneBasedACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.PheromoneBasedACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.PheromoneBasedACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.PheromoneBasedACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param pheromone_evaporation_rate: Pheromone evaluation rate
            (:math:`{\rho}`). If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            will be used. Defaults to :data:`None`
        :type pheromone_evaporation_rate: float
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclasses
        SingleColACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )
        self.pheromone_evaporation_rate = pheromone_evaporation_rate

    @property
    def pheromone_evaporation_rate(self) -> float:
        r"""Pheromone evaporation rate (:math:`{\rho}`).

        :rtype: float
        :setter: Set a new value for the pheromone evaporation rate
        :param value: The new value for the pheromone evaporation rate. If set
            to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            is chosen
        :type value: float
        :raises TypeError: If *value* is not a floating point number
        :raises ValueError: If *value* is outside (0, 1]
        """
        return (
            DEFAULT_PHEROMONE_EVAPORATION_RATE
            if self._pheromone_evaporation_rate is None
            else self._pheromone_evaporation_rate
        )

    @pheromone_evaporation_rate.setter
    def pheromone_evaporation_rate(self, value: float | None) -> None:
        r"""Set a new value for the pheromone evaporation rate (:math:`{\rho}`).

        :param value: The new value for the pheromone evaporation rate. If set
            to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            is chosen
        :type value: float
        :raises TypeError: If *value* is not a floating point number
        :raises ValueError: If *value* is outside (0, 1]
        """
        # Check the value
        self._pheromone_evaporation_rate = (
            None if value is None else check_float(
                value, "pheromone evaporation rate", gt=0, le=1
            )
        )

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current pheromone matrices.

        :rtype: dict
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["pheromone"] = self._pheromone

        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current pheromone matrices to the trainer's
        state.

        :param state: The last loaded state
        :type state: dict
        """
        # Set the state of the superclass
        super()._set_state(state)

        # Set the state of this class
        self._pheromone = state["pheromone"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to generate the initial pheromone matrices.
        """
        super()._new_state()
        self._init_pheromone()

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the pheromone matrices.
        """
        super()._reset_state()
        self._pheromone = None

    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone."""
        for pher in self._pheromone:
            pher *= (1 - self.pheromone_evaporation_rate)

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone."""
        self._deposit_pheromone(self.col)


class ElitistACO(SingleColACO):
    """Base class for all the elitist single colony ACO algorithms."""

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

        # Create the elite container
        self._elite = ParetoFront()

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the elite.
        """
        super()._reset_state()
        self._elite = None

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: list[~deap.tools.HallOfFame]
        """
        best_ones = self._elite if self._elite is not None else ParetoFront()
        return [best_ones]

    def _update_elite(self) -> None:
        """Update the elite (best-so-far) ant(s)."""
        self._elite.update(self.col)


class ReseteablePheromoneBasedACO(ElitistACO, PheromoneBasedACO):
    """Base class for the reseteable elitist pheromone-based ACO approaches."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        pheromone_evaporation_rate: Optional[float] = None,
        convergence_check_freq: Optional[int] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [ReseteablePheromoneBasedACO],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a reseteable elitist pheromone-based ACO trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param pheromone_evaporation_rate: Pheromone evaluation rate
            (:math:`{\rho}`). If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            will be used. Defaults to :data:`None`
        :type pheromone_evaporation_rate: float
        :param convergence_check_freq: Convergence assessment frequency. If
            set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            will be used. Defaults to :data:`None`
        :type convergence_check_freq: int
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclasses
        PheromoneBasedACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            pheromone_evaporation_rate=pheromone_evaporation_rate
        )
        ElitistACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob = exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.convergence_check_freq = convergence_check_freq

    @property
    def convergence_check_freq(self) -> int:
        """Convergence assessment frequency.

        :rtype: int
        :setter: Set a value for the convergence assessment frequency
        :param value: New value for the convergence assessment frequency. If
            set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is non-positive
        """
        return (
            DEFAULT_CONVERGENCE_CHECK_FREQ
            if self._convergence_check_freq is None
            else self._convergence_check_freq
        )

    @convergence_check_freq.setter
    def convergence_check_freq(self, value: int | None) -> None:
        """Set a value for the convergence assessment frequency.

        :param value: New value for the convergence assessment frequency. If
            set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is non-positive
        """
        # Check the value
        self._convergence_check_freq = (
            None if value is None else check_int(
                value, "convergence assessment frequency", gt=0
            )
        )

    def _has_converged(self) -> bool:
        """Detect if the trainer has converged.

        :return: :data:`True` if the trainer has converged
        :rtype: bool
        """
        convergence = True

        for pher in self.pheromone:
            num_rows =  pher.shape[0]
            for row in pher:
                min_pher_count = np.isclose(row, 0).sum()

                if (
                    min_pher_count != num_rows and
                    min_pher_count != num_rows - 2
                ):
                    convergence = False
                    break

        return convergence

    def _should_reset_pheromone(self) -> bool:
        """Detect if the trainer should be reseted.

        :return: :data:`True` if pheromone should be reset
        :rtype: bool
        """
        if (
            self._current_iter > 0 and
            self._current_iter % self.convergence_check_freq == 0 and
            self._has_converged()
        ):
            return True

        return False

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the elite
        self._update_elite()

        # Update the pheromone
        self._update_pheromone()

        # Reset pheromone if convergence is reached
        if self._should_reset_pheromone():
            self._init_pheromone()


class PACO(SingleColACO):
    """Base class for all the population-based single colony ACO algorithms.

    This kind of ACO approach relies on a population of ants that generate the
    pheromone matrices. Thus, the pheromone matrices are now an internal data
    structure of the algorithm, with the population being part of the trainer's
    state.
    """

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [PACO],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        pop_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new population-based ACO trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.PACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.PACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param pop_size: The population size. If set to :data:`None`,
            *col_size* will be used. Defaults to :data:`None`
        :type pop_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclasses
        SingleColACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.pop_size = pop_size

    @property
    def pop_size(self) -> int:
        """Population size.

        :rtype: int
        :setter: Set the population size
        :param size: The new population size. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.abc.SingleColACO.col_size` is chosen
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        return self.col_size if self._pop_size is None else self._pop_size

    @pop_size.setter
    def pop_size(self, size: int | None) -> None:
        """Set the population size.

        :param size: The new population size. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.abc.SingleColACO.col_size` is chosen
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        # Check the value
        self._pop_size = (
            None if size is None else check_int(size, "population size", gt=0)
        )

        # Reset the trainer
        self.reset()

    @property
    def pop(self) -> List[Ant] | None:
        """Population.

        :return: The population or :data:`None` if it has not been generated
        :rtype: list[~culebra.solution.abc.Ant]
        """
        return self._pop

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :rtype: dict
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["pop"] = self._pop

        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :param state: The last loaded state
        :type state: dict
        """
        # Set the state of the superclass
        super()._set_state(state)

        # Set the state of this class
        self._pop = state["pop"]

        # Generate the pheromone matrix with the current population
        self._update_pheromone()

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to create an empty population.
        """
        super()._new_state()

        # Create an empty population
        self._pop = []

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the population.
        """
        super()._reset_state()
        self._pop = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :class:`~culebra.trainer.aco.abc.PACO` class, the pheromone
        matrices are created. Subclasses which need more objects or data
        structures should override this method.
        """
        super()._init_internals()
        self._init_pheromone()

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the pheromone matrices. If subclasses overwrite
        the :meth:`~culebra.trainer.aco.abc.PACO._init_internals` method to
        add any new internal object, this method should also be overridden to
        reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._pheromone = None

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: list[~deap.tools.HallOfFame]
        """
        hof = ParetoFront()
        if self.pop is not None:
            hof.update(self.pop)
        return [hof]

    @abstractmethod
    def _update_pop(self) -> None:
        """Update the population.

        This method should be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _update_pop method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    def _pheromone_amount (
        self, ant: Optional[Ant] = None
    ) -> Tuple[float, ...]:
        """Return the amount of pheromone to be deposited by an ant.

        All the ants deposit/remove the same amount of pheromone,
        :attr:`~culebra.trainer.aco.abc.PACO.initial_pheromone`.

        :param ant: The ant, optional (it is ignored)
        :type ant: ~culebra.solution.abc.Ant
        :return: The amount of pheromone to be deposited for each objective
        :rtype: tuple[float]
        """
        return tuple(self.initial_pheromone)

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        The pheromone trails are updated according to the current population.
        """
        # Init the pheromone matrix
        self._init_pheromone()

        # Update the pheromone matrix with the current population
        self._deposit_pheromone(self.pop)

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the population
        self._update_pop()

        # Update the pheromone
        self._update_pheromone()


class MaxPheromonePACO(PACO):
    """Base class for the PACO approaches with a maximum amount of pheromone."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...],
        max_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [MaxPheromonePACO],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        pop_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new population-based ACO trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param max_pheromone: Maximum amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type max_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param pop_size: The population size. If set to :data:`None`,
            *col_size* will be used. Defaults to :data:`None`
        :type pop_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclasses
        PACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            pop_size=pop_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.max_pheromone = max_pheromone

    @property
    def max_pheromone(self) -> Sequence[float, ...]:
        """Maximum value for each pheromone matrix.

        :rtype: ~collections.abc.Sequence[float]
        :setter: Set the maximum value for each pheromone matrix
        :param values: New maximum value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If any element in *values* is lower than or
            equal to its corresponding initial pheromone value
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
        """
        return self._max_pheromone

    @max_pheromone.setter
    def max_pheromone(self, values: float | Sequence[float, ...]) -> None:
        """Set the maximum value for each pheromone matrix.

        :param values: New maximum value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If any element in *values* is lower than or
            equal to its corresponding initial pheromone value
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
        """
        # Check the values
        if isinstance(values, Sequence):
            self._max_pheromone = check_sequence(
                    values,
                    "maximum pheromone",
                    item_checker=partial(check_float, gt=0)
                )

            if len(self._max_pheromone) == 1:
                if self.num_pheromone_matrices > 1:
                    self._max_pheromone = list(
                        repeat(
                            self._max_pheromone[0],
                            self.num_pheromone_matrices
                        )
                    )
            # Check the length
            elif len(self._max_pheromone) != self.num_pheromone_matrices:
                raise ValueError(
                    "Incorrect number of maximum pheromone values"
                )
        else:
            self._max_pheromone = list(
                repeat(
                    check_float(
                        values,
                        "maximum pheromone", gt=0
                    ),
                    self.num_pheromone_matrices
                )
            )

        # Check that each max value is not lower than its corresponding
        # initial pheromone value
        for (
            val, max_val
        ) in zip(
            self._initial_pheromone, self._max_pheromone
        ):
            if val >= max_val:
                raise ValueError(
                    "Each maximum pheromone value must be higher than "
                    "its corresponding initial pheromone value"
                )

        # Reset the trainer
        self.reset()

    def _pheromone_amount (
        self, ant: Optional[Ant] = None
    ) -> Tuple[float, ...]:
        """Return the amount of pheromone to be deposited by an ant.

        All the ants deposit/remove the same amount of pheromone, which
        is obtained as
        (:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.initial_pheromone` -
        :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.initial_pheromone`) /
        :attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.pop_size`.

        :param ant: The ant, optional (it is ignored)
        :type ant: ~culebra.solution.abc.Ant
        :return: The amount of pheromone to be deposited for each objective
        :rtype: tuple[float]
        """
        return tuple(
            max_pher - init_pher / self.pop_size
            for (init_pher, max_pher) in
            zip(self.initial_pheromone, self.max_pheromone)
        )

    def __copy__(self) -> MaxPheromonePACO:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.aco.abc.MaxPheromonePACO
        """
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.initial_pheromone,
            self.max_pheromone
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> MaxPheromonePACO:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.aco.abc.MaxPheromonePACO
        """
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.initial_pheromone,
            self.max_pheromone
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__,
                (
                    self.solution_cls,
                    self.species,
                    self.fitness_function,
                    self.initial_pheromone,
                    self.max_pheromone
                ),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> MaxPheromonePACO:
        """Return a max pheromone PACO trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.aco.abc.MaxPheromonePACO
        """
        obj = cls(
            state['_solution_cls'],
            state['_species'],
            state['_fitness_function'],
            state['_initial_pheromone'],
            state['_max_pheromone']
        )
        obj.__setstate__(state)
        return obj


class SingleObjPACO(MaxPheromonePACO, SingleObjACO):
    """Base class for the single colony and single objective PACO approaches."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...],
        max_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SingleObjPACO],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        pop_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new population-based ACO trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param max_pheromone: Maximum amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices`
            pheromone matrices.
        :type max_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param pop_size: The population size. If set to :data:`None`,
            *col_size* will be used. Defaults to :data:`None`
        :type pop_size: int
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclasses
        SingleObjACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone
        )

        MaxPheromonePACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            max_pheromone=max_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            pop_size=pop_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

    @abstractmethod
    def _update_pop(self) -> None:
        """Update the population.

        The population may be updated with the current iteration's colony,
        depending on the updation criterion implemented. Ants entering and
        leaving the population are placed, respectively, in the
        *_pop_ingoing* and *_pop_outgoing* lists, to be taken into account in
        the pheromone updation process.

        This method should be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _update_pop method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :class:`~culebra.trainer.aco.abc.SingleObjPACO` class, the
        *_pop_ingoing* and *_pop_outgoing* lists of ants for the population
        are created. Subclasses which need more objects or data structures
        should override this method.
        """
        super()._init_internals()
        self._pop_ingoing = []
        self._pop_outgoing = []

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the *_pop_ingoing* and *_pop_outgoing* lists of
        ants for the population. If subclasses overwrite the
        :meth:`~culebra.trainer.aco.abc.SingleObjPACO._init_internals`
        method to add any new internal object, this method should also be
        overridden to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._pop_ingoing = None
        self._pop_outgoing = None

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone.

        All the ants in the *_pop_ingoing* list increment pheromone on their
        paths.
        """
        self._deposit_pheromone(self._pop_ingoing)

    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone.

        All the ants in the *_pop_outgoing* list decrement pheromone on their
        paths.
        """
        # Use a negative weight to remove pheromone
        self._deposit_pheromone(self._pop_outgoing, weight=-1)


class ACOTSP(SingleColACO):
    """Abstract base class for all the ACO trainers for TSP problems."""

    @property
    def solution_cls(self) -> Type[TSPAnt]:
        """Solution class.

        :rtype: type[~culebra.solution.tsp.Ant]
        :setter: Set a new solution class
        :param cls: The new class
        :type cls: type[~culebra.solution.tsp.Ant]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.tsp.Ant` subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: Type[TSPAnt]) -> None:
        """Set a new solution class.

        :param cls: The new class
        :type cls: type[~culebra.solution.tsp.Ant]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.tsp.Ant` subclass
        """
        # Check cls
        self._solution_cls = check_subclass(
            cls, "solution class", TSPAnt
        )

        # Reset the trainer
        self.reset()

    @property
    def species(self) -> TSPSpecies:
        """Species.

        :rtype: ~culebra.solution.tsp.Species
        :setter: Set a new species
        :param value: The new species
        :type value: ~culebra.solution.tsp.Species
        :raises TypeError: If *value* is not a
            :class:`~culebra.solution.tsp.Species` instance
        """
        return self._species

    @species.setter
    def species(self, value: TSPSpecies) -> None:
        """Set a new species.

        :param value: The new species
        :type value: ~culebra.solution.tsp.Species
        :raises TypeError: If *value* is not a
            :class:`~culebra.solution.tsp.Species` instance
        """
        # Check the value
        self._species = check_instance(value, "species", TSPSpecies)

        # Reset the algorithm
        self.reset()

    @property
    def fitness_function(self) -> TSPFitnessFunction:
        """Training fitness function.

        :rtype: ~culebra.fitness_function.tsp.abc.TSPFitnessFunction
        :setter: Set a new training fitness function
        :param func: The new training fitness function
        :type func: ~culebra.fitness_function.tsp.abc.TSPFitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        """
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, func: TSPFitnessFunction) -> None:
        """Set a new training fitness function.

        :param func: The new training fitness function
        :type func: ~culebra.fitness_function.tsp.abc.TSPFitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        """
        # Check the function
        self._fitness_function = check_instance(
            func, "fitness_function", TSPFitnessFunction
        )

        # Reset the algorithm
        self.reset()

    @property
    def pheromone_shapes(self) -> Sequence[Tuple[int, int], ...]:
        """Shape of the pheromone matrices.

        :rtype: ~collections.abc.Sequence[tuple[int]]
        """
        return [(self.species.num_nodes, ) * 2] * self.num_pheromone_matrices

    @property
    def heuristic_shapes(self) -> Sequence[Tuple[int, int], ...]:
        """Shape of the heuristic matrices.

        :rtype: ~collections.abc.Sequence[tuple[int]]
        """
        return [(self.species.num_nodes, ) * 2] * self.num_heuristic_matrices

    @property
    def heuristic(self) -> Sequence[np.ndarray[float], ...]:
        """Heuristic matrices.

        :rtype: ~collections.abc.Sequence[~numpy.ndarray[float]]
        :setter: Set new heuristic matrices
        :param values: The new heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the heuristic matrices.
            If set to :data:`None`, the default heuristic matrices (provided
            by the :attr:`~culebra.trainer.aco.abc.ACOTSP.fitness_function`
            property) are assumed
        :type values:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :raises TypeError: If *values* is neither an array-like object nor a
            :class:`~collections.abc.Sequence` of array-like objects
        :raises ValueError: If *values* is a sequence of array-like objects
            and it length is different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object has not the correct shape
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any element in any array-like object is negative
        """
        return (
            list(self.fitness_function.heuristic)
            if self._heuristic is None
            else self._heuristic
        )

    @heuristic.setter
    def heuristic(
        self,
        values: Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...] | None
    ) -> None:
        """Set new heuristic matrices.

        :param values: The new heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the heuristic matrices.
            If set to :data:`None`, the default heuristic matrices (provided
            by the :attr:`~culebra.trainer.aco.abc.ACOTSP.fitness_function`
            property) are assumed
        :type values:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :raises TypeError: If *values* is neither an array-like object nor a
            :class:`~collections.abc.Sequence` of array-like objects
        :raises ValueError: If *values* is a sequence of array-like objects
            and it length is different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object has not the correct shape
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any element in any array-like object is negative
        """
        if values is None:
            self._heuristic = None
        # If only one two-dimensional array-like object is provided
        else:
            if (
                isinstance(values, np.ndarray) or
                (
                    isinstance(values, Sequence) and
                    len(values) == 2 and
                    isinstance(values[0], Sequence) and not
                    isinstance(values[0][0], Sequence)
                )
            ):
                self._heuristic = list(
                    repeat(
                        check_matrix(
                            values,
                            "heuristic matrix",
                            square=True,
                            ge=0
                        ),
                        self.num_heuristic_matrices
                    )
                )
            # If a sequence is provided
            else:
                # Check the values
                self._heuristic = check_sequence(
                    values,
                    "heuristic matrices",
                    item_checker=partial(check_matrix, square=True, ge=0)
                )

                if len(self._heuristic) == 1:
                    if self.num_heuristic_matrices > 1:
                        self._heuristic = list(
                            repeat(
                                self._heuristic[0],
                                self.num_heuristic_matrices
                            )
                        )
                # Check the length
                else:
                    if len(self._heuristic) != self.num_heuristic_matrices:
                        raise ValueError(
                            "Incorrect number of heuristic matrices"
                        )

            # Check that all the matrices have the same shape
            for heur, shape in zip(self.heuristic, self.heuristic_shapes):
                if heur.shape != shape:
                    raise ValueError(
                        "Incorrect shape for the heuristic matrix/matrices"
                    )

    @SingleColACO.col_size.getter
    def col_size(self) -> int:
        """Colony size.

        :rtype: int
        :setter: Set a new value for the colony size. If set to
            :data:`None`,
            :attr:`~culebra.trainer.aco.abc.ACOTSP.species`'s
            :attr:`~culebra.solution.tsp.Species.num_nodes`
            is chosen
        :param size: The new colony size
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        return (
            self.species.num_nodes
            if self._col_size is None
            else self._col_size
        )

    def _calculate_choice_info(self) -> None:
        """Calculate the choice info matrix."""
        self._choice_info = np.ones(self.pheromone_shapes[0])

        for (
            pher,
            pher_influence
        ) in zip(
            self.pheromone,
            self.pheromone_influence
        ):
            if pher_influence > 0:
                if pher_influence == 1:
                    self._choice_info *= pher
                else:
                    self._choice_info *= np.power(pher, pher_influence)

        for (
            heur,
            heur_influence
        ) in zip(
            self.heuristic,
            self.heuristic_influence
        ):
            if heur_influence > 0:
                if heur_influence == 1:
                    self._choice_info *= heur
                else:
                    self._choice_info *= np.power(heur, heur_influence)

    def _ant_choice_info(self, ant: TSPAnt) -> np.ndarray[float]:
        """Return the choice info to obtain the next node the ant will visit.

        All the nodes banned by the
        :attr:`~culebra.trainer.aco.abc.ACOTSP.species`, along with all
        the previously visited nodes are discarded.

        :param ant: The ant
        :type ant: ~culebra.solution.tsp.Ant
        :rtype: ~numpy.ndarray[float]
        """
        ant_choice_info = super()._ant_choice_info(ant)
        ant_choice_info[self.species.banned_nodes] = 0

        return ant_choice_info

    def _deposit_pheromone(
        self,
        ants: Sequence[TSPAnt],
        weight: float = DEFAULT_PHEROMONE_DEPOSIT_WEIGHT
    ) -> None:
        """Make some ants deposit weighted pheromone.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants: ~collections.abc.Sequence[~culebra.solution.tsp.Ant]
        :param weight: Weight for the pheromone. Defaults to
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_DEPOSIT_WEIGHT`
        :type weight: float
        """
        for ant in ants:
            for pher, pher_amount in zip(
                self.pheromone, self._pheromone_amount(ant)
            ):

                weighted_pher_amount = pher_amount * weight
                org = ant.path[-1]
                for dest in ant.path:
                    pher[org][dest] += weighted_pher_amount
                    pher[dest][org] += weighted_pher_amount
                    org = dest


class ACOFS(
    SinglePheromoneMatrixACO,
    SingleHeuristicMatrixACO
):
    """Abstract base class for all the ACO-FS trainers."""

    def __init__(
        self,
        solution_cls: Type[FSAnt],
        species: FSSpecies,
        fitness_function: FitnessFunction,
        initial_pheromone: Optional[float | Sequence[float, ...]] = None,
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
        exploitation_prob: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [ACOFS],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        discard_prob: Optional[float] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new ACO-FS trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.feature_selection.Ant]
        :param species: The species for all the ants
        :type species: feature_selection.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices`
            pheromone matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_INITIAL_PHEROMONE`
            will be used. Defaults to :data:`None`
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices,
            according to the kind of problem being solved, are assumed.
            Defaults to :data:`None`
        :type heuristic:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_HEURISTIC_INFLUENCE`
            will be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_EXPLOITATION_PROB`
            will be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted, the default size,
            according to the kind of problem being solved, is assumed. Defaults
            to :data:`None`
        :type col_size: int
        :param discard_prob: Probability of discarding a node (feature). If
            set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_DISCARD_PROB` will be
            used. Defaults to :data:`None`
        :type discard_prob: float
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME`
            will be used. Defaults to :data:`None`
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
        # Init the superclasses
        SinglePheromoneMatrixACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone
        )
        SingleHeuristicMatrixACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        self.discard_prob = discard_prob

    @property
    def solution_cls(self) -> Type[FSAnt]:
        """Solution class.

        :rtype: type[~culebra.solution.feature_selection.Ant]
        :setter: Set a new solution class
        :param cls: The new class
        :type cls: type[~culebra.solution.feature_selection.Ant]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.feature_selection.Ant` subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: Type[FSAnt]) -> None:
        """Set a new solution class.

        :param cls: The new class
        :type cls: type[~culebra.solution.feature_selection.Ant]
        :raises TypeError: If *cls* is not an
            :class:`~culebra.solution.feature_selection.Ant` subclass
        """
        # Check cls
        self._solution_cls = check_subclass(
            cls, "solution class", FSAnt
        )

        # Reset the trainer
        self.reset()

    @property
    def species(self) -> FSSpecies:
        """Species.

        :rtype: ~culebra.solution.feature_selection.Species
        :setter: Set a new species
        :param value: The new species
        :type value: ~culebra.solution.feature_selection.Species
        :raises TypeError: If *value* is not a
            :class:`~culebra.solution.feature_selection.Species` instance
        """
        return self._species

    @species.setter
    def species(self, value: FSSpecies) -> None:
        """Set a new species.

        :param value: The new species
        :type value: ~culebra.solution.feature_selection.Species
        :raises TypeError: If *value* is not a
            :class:`~culebra.solution.feature_selection.Species` instance
        """
        # Check the value
        self._species = check_instance(value, "species", FSSpecies)

        # Reset the algorithm
        self.reset()

    @property
    def initial_pheromone(self) -> Sequence[float, ...]:
        """Initial value for each pheromone matrix.

        :return: One initial value for each pheromone matrix
        :rtype: ~collections.abc.Sequence[float]
        :setter: Set the initial value for each pheromone matrix
        :param values: New initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices`
            pheromone matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_INITIAL_PHEROMONE`
            is chosen
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices`
        """
        return self._initial_pheromone

    @initial_pheromone.setter
    def initial_pheromone(
        self, values: float | Sequence[float, ...] | None
    ) -> None:
        """Set the initial value for each pheromone matrix.

        :param values: New initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices`
            pheromone matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_INITIAL_PHEROMONE`
            is chosen
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_pheromone_matrices`
        """
        if values is None:
            SinglePheromoneMatrixACO.initial_pheromone.fset(
                self,
                DEFAULT_ACOFS_INITIAL_PHEROMONE
            )
        else:
            SinglePheromoneMatrixACO.initial_pheromone.fset(
                self,
                values
            )

    @property
    def pheromone_shapes(self) -> Sequence[Tuple[int, int], ...]:
        """Shape of the pheromone matrices.

        :rtype: ~collections.abc.Sequence[tuple[int]]
        """
        return [(self.species.num_feats, ) * 2] * self.num_pheromone_matrices

    @property
    def heuristic_shapes(self) -> Sequence[Tuple[int, int], ...]:
        """Shape of the heuristic matrices.

        :rtype: ~collections.abc.Sequence[tuple[int]]
        """
        return [(self.species.num_feats, ) * 2] * self.num_heuristic_matrices

    def heuristic_getter(self) -> Sequence[np.ndarray[float], ...]:
        """Heuristic matrices.

        :rtype: ~collections.abc.Sequence[~numpy.ndarray[float]]
        :setter: Set new heuristic matrices
        :param values: The new heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the heuristic matrices.
            If set to :data:`None`, the default
            heuristic (all-ones matrix with a zero diagonal) is assumed.
        :type values:
            ~collections.abc.Sequence[~collections.abc.Sequence[float]] |
            ~collections.abc.Sequence[~collections.abc.Sequence[~collections.abc.Sequence[float]]]
        :raises TypeError: If *values* is neither an array-like object nor a
            :class:`~collections.abc.Sequence` of array-like objects
        :raises ValueError: If *values* is a sequence of array-like objects
            and it length is different from
            :attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object has not the correct shape
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any element in any array-like object is negative
        """
        return (
            [
                np.ones(shape) - np.identity(shape[0])
                for shape in self.heuristic_shapes
            ]
            if self._heuristic is None
            else self._heuristic
        )

    # Use the ACOSTP heuristic setter
    heuristic = property(fget=heuristic_getter, fset=ACOTSP.heuristic.fset)

    @SinglePheromoneMatrixACO.heuristic_influence.getter
    def heuristic_influence(self) -> Sequence[float, ...]:
        r"""Relative influence of heuristic (:math:`{\beta}`).

        :return: One value for each heuristic matrix
        :rtype: ~collections.abc.Sequence[float]
        :setter: Set new values for the relative influence of each heuristic
            matrix
        :param values: New value for the relative influence of each heuristic
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            heuristic matrices. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_HEURISTIC_INFLUENCE` is
            chosen.
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :attr:`~culebra.trainer.aco.abc.ACOFS.num_heuristic_matrices`
        """
        return (
            [DEFAULT_ACOFS_HEURISTIC_INFLUENCE] * self.num_heuristic_matrices
            if self._heuristic_influence is None
            else self._heuristic_influence
        )

    @SinglePheromoneMatrixACO.exploitation_prob.getter
    def exploitation_prob(self) -> float:
        """Exploitation probability (:math:`{q_0}`).

        :rtype: float
        :setter: Set a new value for the exploitation probability
        :param prob: The new probability. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_EXPLOITATION_PROB` is
            chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in [0, 1]
        """
        return (
            DEFAULT_ACOFS_EXPLOITATION_PROB
            if self._exploitation_prob is None
            else self._exploitation_prob
        )

    @SingleColACO.col_size.getter
    def col_size(self) -> int:
        """Colony size.

        :rtype: int
        :setter: Set a new value for the colony size. If set to
            :data:`None`,
            :attr:`~culebra.trainer.aco.abc.ACOFS.species`'s
            :attr:`~culebra.solution.feature_selection.Species.num_feats`
            is chosen
        :param size: The new colony size
        :type size: int
        :raises TypeError: If *size* is not an :class:`int`
        :raises ValueError: If *size* is not greater than zero
        """
        return (
            self.species.num_feats
            if self._col_size is None
            else self._col_size
        )

    @property
    def discard_prob(self) -> float:
        """Probability of discarding a node.

        :rtype: float
        :setter: Set a new discard probability
        :param prob: The new probability. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_DISCARD_PROB` is
            chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        return (
            DEFAULT_ACOFS_DISCARD_PROB
            if self._discard_prob is None
            else self._discard_prob
        )

    @discard_prob.setter
    def discard_prob(self, prob: float | None) -> None:
        """Set a new discard probability.

        :param prob: The new probability. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ACOFS_DISCARD_PROB` is
            chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in (0, 1)
        """
        # Check prob
        self._discard_prob = (
            None if prob is None else check_float(
                prob, "discard probability", gt=0, lt=1)
        )

        # Reset the algorithm
        self.reset()

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :class:`~culebra.trainer.aco.abc.ACOFS` class, the pheromone
        matrices are created and also the choice info matrix is calculated.
        Subclasses which need more objects or data structures should override
        this method.
        """
        super()._init_internals()
        self._init_pheromone()
        self._calculate_choice_info()

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the pheromone matrices. If subclasses overwrite
        the :meth:`~culebra.trainer.aco.abc.ACOFS._init_internals` method
        to add any new internal object, this method should also be overridden
        to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._pheromone = None

    # Use the same implementation for _calculate_choice_info as ACOTSP
    _calculate_choice_info = ACOTSP._calculate_choice_info

    def _ant_choice_info(self, ant: FSAnt) -> np.ndarray[float]:
        """Return the choice info to obtain the next node the ant will visit.

        All the nodes banned by the
        :attr:`~culebra.trainer.aco.abc.ACOFS.species`, along with all
        the previously visited nodes are discarded.

        :param ant: The ant
        :type ant: ~culebra.solution.feature_selection.Ant
        :rtype: ~numpy.ndarray[float]
        """
        # Discard all the visited feats
        ant_choice_info = super()._ant_choice_info(ant)

        # Discard also all the banned feats
        banned_feats = np.zeros((0,), dtype=int)
        if self.species.min_feat > 0:
            banned_feats = np.union1d(
                banned_feats,
                np.arange(self.species.min_feat)
            )
        if self.species.max_feat < self.species.num_feats - 1:
            banned_feats = np.union1d(
                banned_feats,
                np.arange(self.species.max_feat + 1, self.species.num_feats)
            )
        ant_choice_info[banned_feats] = 0

        return ant_choice_info

    def _generate_ant(self) -> FSAnt:
        """Create a new ant.

        The ant chooses the first node randomly, but taking into account the
        amount of pheromone deposited on each arc. Nodes belonging to arcs
        with a higher amount of pheromone are more likely to be selected.

        Then the remaining nodes also randomly selected according to the amount
        of pheromone deposited in their adjacent arcs. The ant may discard
        nodes randomly according to
        :attr:`~culebra.trainer.aco.abc.ACOFS.discard_prob`

        :return: The new ant
        :rtype: ~culebra.solution.feature_selection.Ant
        """
        correct_ant_generated = False

        while correct_ant_generated is False:
            # Start with an empty ant
            ant = self.solution_cls(
                self.species, self.fitness_function.fitness_cls
            )

            # Try choosing a feature
            choice = self._next_choice(ant)

            # The chosen node is considered if the species maximum size has not
            # yet been reached
            while choice is not None and ant.num_feats < self.species.max_size:

                # The chosen node may be discarded
                if random() < self.discard_prob:
                    ant.discard(choice)
                else:
                    ant.append(choice)

                # Try another node
                choice = self._next_choice(ant)

            if ant.species.is_member(ant):
                correct_ant_generated = True

        # Evaluate and return the ant
        self.evaluate(ant)

        return ant

    def _deposit_pheromone(
        self,
        ants: Sequence[FSAnt],
        weight: float = DEFAULT_PHEROMONE_DEPOSIT_WEIGHT
    ) -> None:
        """Make some ants deposit weighted pheromone.

        The pheromone amount deposited by each ant is equally divided across
        all possible feature pair combinations derived from its set of
        selected features.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants:
            ~collections.abc.Sequence[~culebra.solution.feature_selection.Ant]
        :param weight: Weight for the pheromone. Defaults to
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_DEPOSIT_WEIGHT`
        :type weight: float
        """
        for ant in ants:
            if len(ant.path) > 1:
                # All the combinations of two features from those in the path
                indices = combinations(ant.path, 2)

                # Divide the amount of pheromone among all the couples
                amount_per_combination = tuple(
                    (pher_amount / comb(len(ant.path), 2)) * weight
                    for pher_amount in self._pheromone_amount(ant)
                )

                # Deposit the pheromone
                for pher, pher_amount in zip(
                    self.pheromone, amount_per_combination
                ):
                    for (i, j) in indices:
                        pher[i][j] += amount_per_combination
                        pher[j][i] += amount_per_combination


# Exported symbols for this module
__all__ = [
    'SingleColACO',
    'SinglePheromoneMatrixACO',
    'SingleHeuristicMatrixACO',
    'MultiplePheromoneMatricesACO',
    'MultipleHeuristicMatricesACO',
    'SingleObjACO',
    'PheromoneBasedACO',
    'ElitistACO',
    'ReseteablePheromoneBasedACO',
    'PACO',
    'MaxPheromonePACO',
    'SingleObjPACO',
    'ACOTSP',
    'ACOFS'
]
