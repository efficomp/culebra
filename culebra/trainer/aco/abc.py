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

"""Abstract base classes for different ACO-based trainers.

This module provides several abstract classes for different kind of
Ant Colony Optimization based trainers.

By the moment:

  * :py:class:`~culebra.trainer.aco.abc.SingleColACO`: A base class for all
    the single colony ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.SinglePheromoneMatrixACO`: A base class
    for all the single colony ACO-based trainers which rely on a single
    pheromone matrix
  * :py:class:`~culebra.trainer.aco.abc.SingleHeuristicMatrixACO`: A base class
    for all the single colony ACO-based trainers which rely on a single
    heuristic matrix
  * :py:class:`~culebra.trainer.aco.abc.MultiplePheromoneMatricesACO`: A base
    class for all the single colony ACO-based trainers which use multiple
  * :py:class:`~culebra.trainer.aco.abc.MultipleHeuristicMatricesACO`: A base
    class for all the single colony ACO-based trainers which use multiple
    pheromone matrices
  * :py:class:`~culebra.trainer.aco.abc.SingleObjACO`: A base class for all
    the single objective ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.PheromoneBasedACO`: A base class for
    all pheromone-based single objective ACO trainers
  * :py:class:`~culebra.trainer.aco.abc.ElitistACO`: A base class for all
    the elitist single colony ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO`: A base
    class for all the elitist, pheromone-based, and reseteable single colony
    ACO trainers
  * :py:class:`~culebra.trainer.aco.abc.PACO`: A base class for all
    the population-based single colony ACO trainers
  * :py:class:`~culebra.trainer.aco.abc.MaxPheromonePACO`: A base class for all
    the population-based single colony ACO trainers using a maximum pheromone
    amount
  * :py:class:`~culebra.trainer.aco.abc.SingleObjPACO`: A base class for all
    the population-based single colony and single objective ACO trainers
  * :py:class:`~culebra.trainer.aco.abc.ACO_FS`: A base class for all
    the ACO-base approaches for FS

"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    Sequence,
    Tuple,
    Type,
    List,
    Callable,
    Dict,
    Optional
)
from random import random
from math import isclose, comb
from functools import partial
from itertools import repeat, combinations
from copy import copy, deepcopy

import numpy as np
from deap.tools import HallOfFame, ParetoFront


from culebra.abc import Species, FitnessFunction
from culebra.checker import (
    check_subclass,
    check_sequence,
    check_int,
    check_float,
    check_matrix
)
from culebra.solution.abc import Ant
from culebra.trainer.abc import SingleSpeciesTrainer

from .constants import (
    DEFAULT_PHEROMONE_INFLUENCE,
    DEFAULT_HEURISTIC_INFLUENCE,
    DEFAULT_EXPLOITATION_PROB,
    DEFAULT_PHEROMONE_EVAPORATION_RATE,
    DEFAULT_CONVERGENCE_CHECK_FREQ,
    DEFAULT_ACO_FS_INITIAL_PHEROMONE,
    DEFAULT_ACO_FS_HEURISTIC_INFLUENCE,
    DEFAULT_ACO_FS_EXPLOITATION_PROB,
    DEFAULT_ACO_FS_DISCARD_PROB
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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
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

    @property
    @abstractmethod
    def num_pheromone_matrices(self) -> int:
        """Get the number of pheromone matrices used by this trainer.

        This property must be overridden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overridden
        :type: :py:class:`int`
        """
        raise NotImplementedError(
            "The num_pheromone_matrices property has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def num_heuristic_matrices(self) -> int:
        """Get the number of heuristic matrices used by this trainer.

        This property must be overridden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overridden
        :type: :py:class:`int`
        """
        raise NotImplementedError(
            "The num_heuristic_matrices property has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @property
    def solution_cls(self) -> Type[Ant]:
        """Get and set the ant class.

        :getter: Return the ant class
        :setter: Set a new ant class
        :type: An :py:class:`~culebra.solution.abc.Ant` subclass
        :raises TypeError: If set to a value which is not an
            :py:class:`~culebra.solution.abc.Ant` subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: Type[Ant]) -> None:
        """Set a new ant class.

        :param cls: The new class
        :type cls: An :py:class:`~culebra.solution.abc.Ant` subclass
        :raises TypeError: If *cls* is not an
        :py:class:`~culebra.solution.abc.Ant`
        """
        # Check cls
        self._solution_cls = check_subclass(
            cls, "solution class", Ant
        )

        # Reset the trainer
        self.reset()

    @property
    def initial_pheromone(self) -> Sequence[float, ...]:
        """Get and set the initial value for each pheromone matrix.

        :getter: Return the initial value for each pheromone matrix.
        :setter: Set a new initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            is provided
        """
        return self._initial_pheromone

    @initial_pheromone.setter
    def initial_pheromone(self, values: float | Sequence[float, ...]) -> None:
        """Set the initial value for each pheromone matrix.

        :param values: New initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices.
        :type values: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
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
    def heuristic(self) -> Sequence[np.ndarray, ...]:
        """Get and set the heuristic matrices.

        :getter: Return the heuristic matrices
        :setter: Set new heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If set to :py:data:`None`, the default
            heuristic (provided by the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.fitness_function`
            property) are assumed.
        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        :raises TypeError: If neither an array-like object nor a
            :py:class:`~collections.abc.Sequence` of array-like objects is
            provided
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any array-like object has not an homogeneous
            shape
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object is not square
        :raises ValueError: If any element in any array-like object is negative
        :raises ValueError: If a sequence is provided and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        :raises ValueError: If the array-like objects have different shapes
        """
        return self._heuristic

    @heuristic.setter
    def heuristic(
        self,
        values: Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...] | None
    ) -> None:
        """Set new heuristic matrices.

        :param values: New heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If set to :py:data:`None`, the
            default heuristic (provided by the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.fitness_function`
            property) are assumed
        :type: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional
            array-like objects
        :raises TypeError: If *values* is neither an array-like object nor a
            :py:class:`~collections.abc.Sequence` of array-like objects
        :raises ValueError: If any element in any array-like object is not a
            float number
        :raises ValueError: If any array-like object has not an homogeneous
            shape
        :raises ValueError: If any array-like object has not two dimensions
        :raises ValueError: If any array-like object is not square
        :raises ValueError: If any element in any array-like object is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
        :raises ValueError: If the array-like objects have different shapes
        """
        if values is None:
            values = self.fitness_function.heuristic(self.species)

        # If only one two-dimensional array-like object is provided
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
                    raise ValueError("Incorrect number of heuristic matrices")

                # Check that all the matrices have the same shape
                the_shape = self._heuristic[0].shape
                for matrix in self._heuristic:
                    if matrix.shape != the_shape:
                        raise ValueError(
                            "All the heuristic matrices must have the same "
                            "shape"
                        )

        # Check the shape
        if self._heuristic[0].shape[0] == 0:
            raise ValueError("A heuristic matrix can not be empty")

    @property
    def pheromone_influence(self) -> Sequence[float, ...]:
        r"""Get and set the influence of pheromone (:math:`{\alpha}`).

        :getter: Return the relative influence of each pheromone matrix.
        :setter: Set a new value for the relative influence of each pheromone
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` is
            chosen.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            is provided
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
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` is
            chosen.
        :type values: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
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
        r"""Get and set the relative influence of heuristic (:math:`{\beta}`).

        :getter: Return the relative influence of each heuristic matrix.
        :setter: Set a new value for the relative influence of each heuristic
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` is
            chosen.
        :type: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            is provided
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
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` is
            chosen.
        :type values: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
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
        """Get and set the exploitation probability (:math:`{q_0}`).

        :getter: Return the current exploitation probability
        :setter: Set the new exploitation probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` is
            chosen
        :type: :py:class:`float` in [0, 1]
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in [0, 1]
        """
        return (
            DEFAULT_EXPLOITATION_PROB
            if self._exploitation_prob is None
            else self._exploitation_prob
        )

    @exploitation_prob.setter
    def exploitation_prob(self, prob: float | None) -> None:
        """Set a new exploitation probability (:math:`{q_0}`).

        :param prob: The new probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` is
            chosen
        :type prob: :py:class:`float` in [0, 1]
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
        """Get and set the colony size.

        :getter: Return the current colony size
        :setter: Set a new value for the colony size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.fitness_function`'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes` is chosen
        :type: :py:class:`int`, greater than zero
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        return (
            self.fitness_function.num_nodes
            if self._col_size is None
            else self._col_size
        )

    @col_size.setter
    def col_size(self, size: int | None) -> None:
        """Set the colony size.

        :param size: The new colony size. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.fitness_function`'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes` is chosen
        :type size: :py:class:`int`, greater than zero
        :raises TypeError: If *size* is not an :py:class:`int`
        :raises ValueError: If *size* is not an integer greater than zero
        """
        # Check the value
        self._col_size = (
            None if size is None else check_int(size, "colony size", gt=0)
        )

        # Reset the trainer
        self.reset()

    @property
    def col(self) -> List[Ant] | None:
        """Get the colony.

        :type: :py:class:`list` of :py:class:`~culebra.abc.Solution`
        """
        return self._col

    @property
    def pheromone(self) -> Sequence[np.ndarray, ...] | None:
        """Get the pheromone matrices.

        If the search process has not begun, :py:data:`None` is returned.

        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        """
        return self._pheromone

    @property
    def choice_info(self) -> np.ndarray | None:
        """Get the choice information for all the graph's arcs.

        The choice information is generated from both the pheromone and the
        heuristic matrices, modified by other parameters (depending on the ACO
        approach) and is used to obtain the probalility of following the next
        feasible arc for the node.

        If the search process has not begun, :py:data:`None` is returned.

        :type: :py:class:`~numpy.ndarray`
        """
        return self._choice_info

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        Return the best single solution found for each species

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = ParetoFront()
        if self.col is not None:
            hof.update(self.col)
        return [hof]

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :py:class:`~culebra.aco.abc.SingleColACO` class, the colony, the
        choice_info matrix and the node list are created. Subclasses which
        need more objects or data structures should override this method.
        """
        super()._init_internals()
        self._col = []
        self._choice_info = None
        self._node_list = np.arange(
            0, self.fitness_function.num_nodes, dtype=int
        )

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the colony, the choice_info matrix and the node
        list. If subclasses overwrite the
        :py:meth:`~culebra.aco.abc.SingleColACO._init_internals`
        method to add any new internal object, this method should also be
        overridden to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._col = None
        self._choice_info = None
        self._node_list = None

    @abstractmethod
    def _calculate_choice_info(self) -> None:
        """Calculate the choice information.

        The choice information is generated from both the pheromone and the
        heuristic matrices, modified by other parameters (depending on the ACO
        approach) and is used to obtain the probalility of following the next
        feasible arc for the node.

        This method should be overridden by subclasses.
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

    def _initial_choice(self, ant: Ant) -> int | None:
        """Choose the initial node for an ant.

        The selection is made randomly among all connected nodes, avoiding
        already discarded nodes.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`

        :return: The index of the chosen node or :py:data:`None` if there
            isn't any feasible node
        :rtype: :py:class:`int` or :py:data:`None`
        """
        # Get the nodes connected to any other node
        feasible_nodes = np.argwhere(
            np.sum(self.choice_info, axis=1) > 0
        ).flatten()

        # Avoid discarded nodes
        feasible_nodes = np.setdiff1d(feasible_nodes, ant.discarded)

        if len(feasible_nodes) == 0:
            return None

        return np.random.choice(feasible_nodes)

    def _next_choice(self, ant: Ant) -> int | None:
        """Choose the next node for an ant.

        The election is made from the feasible neighborhood of the current
        node, which is composed of those nodes not visited yet by the ant and
        connected to its current node. 
        
        If the ant's path is empty, the
        :py:meth:`~culebra.trainer.aco.abc.SingleColACO._initial_choice`
        method is called. Otherwise, the best possible node is selected with
        probability
        :py:attr:`~culebra.trainer.aco.abc.SingleColACO._exploitation_prob`.
        Finally, if the best node is not chosen, the next node is selected
        probabilistically according to the 
        :py:attr:`~culebra.trainer.aco.abc.SingleColACO.choice_info` matrix.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :return: The index of the chosen node or :py:data:`None` if there
            isn't any feasible node
        :rtype: :py:class:`int` or :py:data:`None`
        """
        # Return the initial choice for ants with an empty path
        if ant.current is None:
            return self._initial_choice(ant)
        
        # Choice info for the feasible neighborhood of the current node
        ant_choice_info = np.copy(self.choice_info[ant.current])
        ant_choice_info[ant.path] = 0
        ant_choice_info[ant.discarded] = 0
        
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
                return np.random.choice(
                    self._node_list,
                    p=ant_choice_info/np.sum(ant_choice_info)
                )

        # If there isn't any feasible node
        return None

    def _generate_ant(self) -> Ant:
        """Generate a new ant.

        The ant makes its path and gets evaluated.
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
        shape = self._heuristic[0].shape
        self._pheromone = [
            np.full(
                shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromone
        ]


    def _pheromone_amount (self, ant: Ant) -> Tuple[float, ...]:
        """Return the amount of pheromone to be deposited by an ant.

        The reciprocal of an objective fitness value will be used for
        minimization objectives, while the objective's fitness value is used 
        for maximization objectives.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :return: The amount of pheromone to be deposited for each objective
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return tuple(
            1/value if weight < 0 else value
            for (value, weight) in zip(ant.fitness.values, ant.fitness.weights)
        )
        
    def _deposit_pheromone(
        self, ants: Sequence[Ant], weight: Optional[float] = 1
    ) -> None:
        """Make some ants deposit weighted pheromone.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.solution.abc.Ant`
        :param weight: Weight for the pheromone. Defaults to 1
        :type weight: :py:class:`float`, optional
        """
        for ant in ants:
            for pher_index, pher_amount in enumerate(
                self._pheromone_amount(ant)
            ):

                weighted_pher_amount = pher_amount * weight
                org = ant.path[-1]
                for dest in ant.path:
                    self._pheromone[pher_index][org][dest] += (
                        weighted_pher_amount
                    )
                    self._pheromone[pher_index][dest][org] += (
                        weighted_pher_amount
                    )
                    org = dest

    @abstractmethod
    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The _increase_pheromone method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _decrease_pheromone(self) -> None:
        """Decrease the amount of pheromone.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The _decrease_pheromone method has not been implemented in the "
            f"{self.__class__.__name__} class")

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        First, the pheromone are evaporated. Then ants deposit pheromone
        according to their fitness.
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
        """Shallow copy the trainer."""
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
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
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
        :rtype: :py:class:`tuple`
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

        :param state: The state.
        :type state: :py:class:`~dict`
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
        """Get the number of pheromone matrices used by this trainer.

        :type: :py:class:`int`
        """
        return 1


class SingleHeuristicMatrixACO(SingleColACO):
    """Base class for all the single heuristic matrix ACO algorithms."""

    @property
    def num_heuristic_matrices(self) -> int:
        """Get the number of heuristic matrices used by this trainer.

        :type: :py:class:`int`
        """
        return 1


class MultiplePheromoneMatricesACO(SingleColACO):
    """Base class for all the multiple pheromone matrices ACO algorithms."""

    @property
    def num_pheromone_matrices(self) -> int:
        """Get the number of pheromone matrices used by this trainer.

        :type: :py:class:`int`
        """
        return self.fitness_function.num_obj


class MultipleHeuristicMatricesACO(SingleColACO):
    """Base class for all the multiple heuristic matrices ACO algorithms."""

    @property
    def num_heuristic_matrices(self) -> int:
        """Get the number of heuristic matrices used by this trainer.

        :type: :py:class:`int`
        """
        return self.fitness_function.num_obj


class SingleObjACO(SinglePheromoneMatrixACO, SingleHeuristicMatrixACO):
    """Base class for all the single-objective ACO algorithms."""

    @property
    def fitness_function(self) -> FitnessFunction:
        """Get and set the training fitness function.

        :getter: Return the fitness function
        :setter: Set a new fitness function
        :type: :py:class:`~culebra.abc.FitnessFunction`
        :raises TypeError: If set to a value which is not a fitness function
        :raises ValueError: If set to a fitness function with more than one
            objective
        """
        return SingleColACO.fitness_function.fget(self)

    @fitness_function.setter
    def fitness_function(self, func: FitnessFunction) -> None:
        """Set a new training fitness function.

        :param func: New training fitness function
        :type func: :py:class:`~culebra.abc.FitnessFunction`
        :raises TypeError: If set to a value which is not a fitness function
        :raises ValueError: If set to a fitness function with more than one
            objective
        """
        SingleColACO.fitness_function.fset(self, func)
        if self.fitness_function.num_obj != 1:
            raise ValueError("Incorrect number of objectives")

    def _calculate_choice_info(self) -> None:
        """Calculate the choice info matrix."""
        self._choice_info = np.ones(self.pheromone[0].shape)
        
        if self.pheromone_influence[0] > 0:
            if self.pheromone_influence[0] == 1:
                self._choice_info *= self.pheromone[0]
            else:
                self._choice_info *= np.power(
                    self.pheromone[0], self.pheromone_influence[0]
                )
        if self.heuristic_influence[0] > 0:
            if self.heuristic_influence[0] == 1:
                self._choice_info *= self.heuristic[0]
            else:
                self._choice_info *= np.power(
                    self.heuristic[0], self.heuristic_influence[0]
                )

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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param pheromone_evaporation_rate: Pheromone evaluation rate
            (:math:`{\rho}`). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            will be used. Defaults to :py:data:`None`
        :type pheromone_evaporation_rate: :py:class:`float`, optional
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
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
        r"""Get and set the pheromone evaporation rate (:math:`{\rho}`).

        :getter: Return the pheromone evaporation rate
        :setter: Set a value for the pheromone evaporation rate. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            is chosen
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a float
        :raises ValueError: If set to value outside (0, 1]
        """
        return (
            DEFAULT_PHEROMONE_EVAPORATION_RATE
            if self._pheromone_evaporation_rate is None
            else self._pheromone_evaporation_rate
        )

    @pheromone_evaporation_rate.setter
    def pheromone_evaporation_rate(self, value: float | None) -> None:
        r"""Set a value for the pheromone evaporation rate (:math:`{\rho}`).

        :param value: New value for the pheromone evaporation rate. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            is chosen
        :type value: :py:class:`float`
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

        :type: :py:class:`dict`
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
        :type state: :py:class:`dict`
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
        """Increase the amount of pheromone.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.
        """
        self._deposit_pheromone(self.col)

class ElitistACO(SingleColACO):
    """Base class for all the elitist single colony ACO algorithms."""

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current elite to the trainer's state.

        :type: :py:class:`dict`
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
        :type state: :py:class:`dict`
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

        Return the best single solution found for each species

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        best_ones = self._elite if self._elite is not None else ParetoFront()
        return [best_ones]

    def _update_elite(self) -> None:
        """Update the elite (best-so-far) ant."""
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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.ReseteablePheromoneBasedACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param pheromone_evaporation_rate: Pheromone evaluation rate
            (:math:`{\rho}`). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            will be used. Defaults to :py:data:`None`
        :type pheromone_evaporation_rate: :py:class:`float`, optional     
        :param convergence_check_freq: Convergence assessment frequency. If
            set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            will be used. Defaults to :py:data:`None`
        :type convergence_check_freq: :py:class:`int`, optional
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
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
        """Get and set the convergence assessment frequency.

        :getter: Return the convergence assessment frequency
        :setter: Set a value for the convergence assessment frequency. If set
            to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an int
        :raises ValueError: If set to a non-positive value
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
            set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            is chosen
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is non-positive
        """
        # Check the value
        self._convergence_check_freq = (
            None if value is None else check_int(
                value, "convergence assessment frequency", gt=0
            )
        )

    def _has_converged(self) -> None:
        """Detect if the trainer has converged.

        :return: :py:data:`True` if the trainer has converged
        :rtype: :py:class:`bool`
        """
        convergence = True

        for pher in self.pheromone:
            for row in range(len(pher)):
                min_pher_count = np.isclose(pher[row], 0).sum()

                if (
                    min_pher_count != self.species.num_nodes and
                    min_pher_count != self.species.num_nodes - 2
                ):
                    convergence = False
                    break

        return convergence

    def _should_reset_pheromone(self) -> bool:
        """Return :py:data:`True` if pheromone should be reset."""
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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
        :param pop_size: The population size. If set to :py:data:`None`,
            *col_size* will be used. Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
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
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.col_size` is chosen
        :type: :py:class:`int`, greater than zero
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        return self.col_size if self._pop_size is None else self._pop_size

    @pop_size.setter
    def pop_size(self, size: int | None) -> None:
        """Set the population size.

        :param size: The new population size. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.col_size` is chosen
        :type size: :py:class:`int`, greater than zero
        :raises TypeError: If *size* is not an :py:class:`int`
        :raises ValueError: If *size* is not an integer greater than zero
        """
        # Check the value
        self._pop_size = (
            None if size is None else check_int(size, "population size", gt=0)
        )

        # Reset the trainer
        self.reset()

    @property
    def pop(self) -> List[Ant] | None:
        """Get the population.

        :type: :py:class:`list` of :py:class:`~culebra.abc.Solution`
        """
        return self._pop

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :type: :py:class:`dict`
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
        :type state: :py:class:`dict`
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
        :py:class:`~culebra.trainer.aco.abc.PACO` class, the pheromone
        matrices are created. Subclasses which need more objects or data
        structures should override this method.
        """
        super()._init_internals()
        self._init_pheromone()

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the pheromone matrices. If subclasses overwrite
        the :py:meth:`~culebra.trainer.aco.abc.PACO._init_internals` method to
        add any new internal object, this method should also be overridden to
        reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._pheromone = None

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

    @abstractmethod
    def _update_pop(self) -> None:
        """Update the population.

        This method should be overridden by subclasses.
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
        :py:attr:`~culebra.trainer.aco.abc.PACO.initial_pheromone`.

        :param ant: The ant, optional (it is ignored)
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :return: The amount of pheromone to be deposited for each objective
        :rtype: :py:class:`tuple` of :py:class:`float`
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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param max_pheromone: Maximum amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type max_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
        :param pop_size: The population size. If set to :py:data:`None`,
            *col_size* will be used. Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
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
        """Get and set the maximum value for each pheromone matrix.

        :getter: Return the maximum value for each pheromone matrix.
        :setter: Set a new maximum value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`

        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If any value is lower than or equal to its
            corresponding initial pheromone value
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            is provided
        """
        return self._max_pheromone

    @max_pheromone.setter
    def max_pheromone(self, values: float | Sequence[float, ...]) -> None:
        """Set the maximum value for each pheromone matrix.

        :param values: New maximum value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
            pheromone matrices.
        :type values: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If any element in *values* is lower than or
            equal to its corresponding initial pheromone value
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.num_pheromone_matrices`
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
        (:py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.initial_pheromone` -
        :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.initial_pheromone`) /
        :py:attr:`~culebra.trainer.aco.abc.MaxPheromonePACO.pop_size`.

        :param ant: The ant, optional (it is ignored)
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :return: The amount of pheromone to be deposited for each objective
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return tuple(
            max_pher - init_pher / self.pop_size
            for (init_pher, max_pher) in
            zip(self.initial_pheromone, self.max_pheromone)
        )

    def __copy__(self) -> MaxPheromonePACO:
        """Shallow copy the trainer."""
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
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
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
        :rtype: :py:class:`tuple`
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

        :param state: The state.
        :type state: :py:class:`~dict`
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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param max_pheromone: Maximum amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices`
            pheromone matrices.
        :type max_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.SingleObjPACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_EXPLOITATION_PROB` will
            be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
        :param pop_size: The population size. If set to :py:data:`None`,
            *col_size* will be used. Defaults to :py:data:`None`
        :type pop_size: :py:class:`int`, greater than zero, optional
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
        """
        raise NotImplementedError(
            "The _update_pop method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :py:class:`~culebra.trainer.aco.abc.SingleObjPACO` class, the
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
        :py:meth:`~culebra.trainer.aco.abc.SingleObjPACO._init_internals`
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


class ACO_FS(
    SinglePheromoneMatrixACO,
    SingleHeuristicMatrixACO
):
    """Abstract base class for all the ACO-FS trainers."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...] = None,
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
                [ACO_FS],
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

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.ACO_FS.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_INITIAL_PHEROMONE`
            will be used. Defaults to :py:data:`None`
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.ACO_FS.num_heuristic_matrices`
            heuristic matrices. If omitted, the default heuristic matrices
            provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristic: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional array-like
            objects, optional
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.ACO_FS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.ACO_FS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_HEURISTIC_INFLUENCE`
            will be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_EXPLOITATION_PROB`
            will be used. Defaults to :py:data:`None`
        :type exploitation_prob: :py:class:`float`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param col_size: The colony size. If set to :py:data:`None`,
            *fitness_function*'s
            :py:attr:`~culebra.abc.FitnessFunction.num_nodes`
            will be used. Defaults to :py:data:`None`
        :type col_size: :py:class:`int`, greater than zero, optional
        :param discard_prob: Probability of discarding a node (feature). If
            set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_DISCARD_PROB` will be
            used. Defaults to :py:data:`None`
        :type discard_prob: :py:class:`float` in (0, 1), optional
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
    def initial_pheromone(self) -> Sequence[float, ...]:
        """Get and set the initial value for each pheromone matrix.

        :getter: Return the initial value for each pheromone matrix.
        :setter: Set a new initial value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.ACO_FS.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_INITIAL_PHEROMONE`
            is chosen
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.ACO_FS.num_pheromone_matrices`
            is provided
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
            :py:attr:`~culebra.trainer.aco.ACO_FS.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
                :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_INITIAL_PHEROMONE`
                is chosen
        :type values: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If *values* is neither a float nor a Sequence of
            float values
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.ACO_FS.num_pheromone_matrices`
        """
        if values is None:
            SinglePheromoneMatrixACO.initial_pheromone.fset(
                self,
                DEFAULT_ACO_FS_INITIAL_PHEROMONE
            )
        else:
            SinglePheromoneMatrixACO.initial_pheromone.fset(
                self,
                values
            )

    @SinglePheromoneMatrixACO.heuristic_influence.getter
    def heuristic_influence(self) -> Sequence[float, ...]:
        r"""Get and set the relative influence of heuristic (:math:`{\beta}`).

        :getter: Return the relative influence of each heuristic matrix.
        :setter: Set a new value for the relative influence of each heuristic
            matrix. Both a scalar value or a sequence of values are allowed.
            If a scalar value is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.ACO_FS.num_heuristic_matrices`
            heuristic matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_HEURISTIC_INFLUENCE`
            is chosen.
        :type: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.ACO_FS.num_heuristic_matrices`
            is provided
        """
        return (
            [DEFAULT_ACO_FS_HEURISTIC_INFLUENCE] * self.num_heuristic_matrices
            if self._heuristic_influence is None
            else self._heuristic_influence
        )
    
    @SinglePheromoneMatrixACO.exploitation_prob.getter
    def exploitation_prob(self) -> float:
        """Get and set the exploitation probability (:math:`{q_0}`).

        :getter: Return the current exploitation probability
        :setter: Set the new exploitation probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_EXPLOITATION_PROB` is
            chosen
        :type: :py:class:`float` in [0, 1]
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in [0, 1]
        """
        return (
            DEFAULT_ACO_FS_EXPLOITATION_PROB
            if self._exploitation_prob is None
            else self._exploitation_prob
        )

    @property
    def discard_prob(self) -> float:
        """Get and set the probability of discarding a node.

        :getter: Return the current discard probability
        :setter: Set a new value for the discard probability. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_DISCARD_PROB` is
            chosen
        :type: :py:class:`float` in (0, 1)
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value which is not in (0, 1)
        """
        return (
            DEFAULT_ACO_FS_DISCARD_PROB
            if self._discard_prob is None
            else self._discard_prob
        )

    @discard_prob.setter
    def discard_prob(self, prob: float | None) -> None:
        """Set a new discard probability.

        :param prob: The new probability. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_DISCARD_PROB` is
            chosen
        :type prob: :py:class:`float` in (0, 1)
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
        :py:class:`~culebra.trainer.aco.abc.ACO_FS` class, the pheromone
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
        the :py:meth:`~culebra.trainer.aco.abc.ACO_FS._init_internals` method
        to add any new internal object, this method should also be overridden
        to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._pheromone = None

    def _calculate_choice_info(self) -> None:
        """Calculate the choice info matrix."""
        self._choice_info = np.ones(self.pheromone[0].shape)
        
        if self.pheromone_influence[0] > 0:
            if self.pheromone_influence[0] == 1:
                self._choice_info *= self.pheromone[0]
            else:
                self._choice_info *= np.power(
                    self.pheromone[0], self.pheromone_influence[0]
                )
        if self.heuristic_influence[0] > 0:
            if self.heuristic_influence[0] == 1:
                self._choice_info *= self.heuristic[0]
            else:
                self._choice_info *= np.power(
                    self.heuristic[0], self.heuristic_influence[0]
                )

    def _initial_choice(self, ant: Ant) -> int | None:
        """Choose the initial node for an ant.

        The selection is made randomly among all connected nodes but
        proportionally to the pheromone amount accumulated in the arcs
        connected to each node and avoiding already discarded nodes.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`

        :return: The index of the chosen node or :py:data:`None` if there
            isn't any feasible node
        :rtype: :py:class:`int` or :py:data:`None`
        """
        # Make zeros in choice _info for discarded nodes
        ant_choice_info = copy(self.choice_info)
        ant_choice_info[ant.discarded, :] = 0

        # Get the nodes connected to any other node and their selection probs
        feasible_nodes = list(range(self.fitness_function.num_nodes))

        temp = np.sum(ant_choice_info, axis=1)
        temp_sum = temp.sum()

        if isclose(temp_sum, 0):
            return None
        else:
            feasible_node_probs = temp / temp_sum
            return np.random.choice(feasible_nodes, p=feasible_node_probs)

    def _generate_ant(self) -> Ant:
        """Create a new ant.

        The ant chooses the first node randomly, but taking into account the
        amount of pheromone deposited on each arc. Nodes belonging to arcs
        with a higher amount of pheromone are more likely to be selected.

        Then the remaining nodes also randomly selected according to the amount
        of pheromone deposited in their adjacent arcs. The ant may discard
        nodes randomly according to
        :py:attr:`~culebra.trainer.aco.PACO_FS.discard_prob`
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
        self, ants: Sequence[Ant], weight: Optional[float] = 1
    ) -> None:
        """Make some ants deposit weighted pheromone.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.solution.abc.Ant`
        :param weight: Weight for the pheromone. Defaults to 1
        :type weight: :py:class:`float`, optional
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
    'ACO_FS'
]
