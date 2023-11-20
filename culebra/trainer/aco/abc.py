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
    pheromone matrices
  * :py:class:`~culebra.trainer.aco.abc.MultipleHeuristicMatricesACO`: A base
    class for all the single colony ACO-based trainers which use multiple
    pheromone matrices
  * :py:class:`~culebra.trainer.aco.abc.SingleObjACO`: A base class for all
    the single objective ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.ElitistACO`: A base class for all
    the elitist single colony ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.WeightedElitistACO`: A base class for
    all the elitist single colony ACO-based trainers where the elite deposit
    a weighted amount of pheromone
  * :py:class:`~culebra.trainer.aco.abc.PACO`: A base class for all
    the population-based single colony ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.AgeBasedPACO`: A base class for all
    the population-based single colony ACO-based trainers with an age-based
    population update strategy
  * :py:class:`~culebra.trainer.aco.abc.QualityBasedPACO`: A base class for all
    the population-based single colony ACO-based trainers with a quality-based
    population update strategy
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    Sequence,
    Type,
    List,
    Callable,
    Dict,
    Optional
)
from functools import partial
from itertools import repeat
from random import sample

import numpy as np
from deap.tools import HallOfFame, ParetoFront, sortNondominated


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
    DEFAULT_CONVERGENCE_CHECK_FREQ,
    DEFAULT_ELITE_WEIGHT
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
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
        :raises TypeError: If any value is not an array
        :raises ValueError: If any matrix has a wrong shape or contain any
            negative value.
        :raises ValueError: If a sequence of matrices with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
            is provided
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
            property) are assumed.
        :type: A two-dimensional array-like object or a
            :py:class:`~collections.abc.Sequence` of two-dimensional
            array-like objects
        :raises TypeError: If *values* neither a two-dimensional array-like
            object nor a :py:class:`~collections.abc.Sequence` of
            two-dimensional array-like objects
        :raises ValueError: If any element in any two-dimensional array-like
            object is not a float number
        :raises ValueError: If any any two-dimensional array-like
            object has not an homogeneous shape
        :raises ValueError: If any array-like object in *values* has not two
            dimensions
        :raises ValueError: If any element in any two-dimensional array-like
            object is negative
        :raises ValueError: If *values* is a sequence and it length is
            different from
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.num_heuristic_matrices`
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
        the_shape = self._heuristic[0].shape
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
    def col_size(self) -> int:
        """Get and set the colony size.

        :getter: Return the current colony size
        :setter: Set a new value for the colony size. If set to
            :py:data:`None`,
            If set to :py:data:`None`,
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
    @abstractmethod
    def pheromone(self) -> Sequence[np.ndarray, ...] | None:
        """Get the pheromone matrices.

        If the search process has not begun, :py:data:`None` is returned.

        This property must be overridden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overridden
        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        """
        raise NotImplementedError(
            "The pheromone property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

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

    def _initial_choice(self) -> int | None:
        """Choose the initial node for an ant.

        The selection is made randomly among all connected nodes.

        :return: The index of the chosen node or :py:data:`None` if there
            isn't any feasible node
        :rtype: :py:class:`int` or :py:data:`None`
        """
        # Get the nodes connected to any other node
        feasible_nodes = np.argwhere(
            np.sum(self.choice_info, axis=0) > 0
        ).flatten()

        if len(feasible_nodes) == 0:
            return None

        return np.random.choice(feasible_nodes)

    def _feasible_neighborhood_probs(self, ant: Ant) -> np.ndarray:
        """Return the probabilities of the feasible neighborhood of an ant.

        The feasible neighborhood is composed of those nodes not visited yet
        by the ant and connected to its current node. Each of these nodes has
        a probability of being visited from the current node, which is
        calculated from the
        :py:attr:`~culebra.trainer.aco.abc.SingleColACO.choice_info` matrix.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :raises ValueError: If *ant* has not a current node.

        :return: An array with the probability of following each feasible node
            from the current node of *ant*
        :rtype: :py:class:`~numpy.ndarray`
        """
        if ant.current is None:
            raise ValueError("The ant has not a current node")

        probs = np.copy(self.choice_info[ant.current])
        probs[ant.path] = 0
        probs_sum = np.sum(probs)
        if probs_sum == 0:
            return probs

        return probs / probs_sum

    def _next_choice(self, ant: Ant) -> int | None:
        """Choose the next node for an ant.

        The election is based on the feasible neighborhood probabilities of the
        ant's current node. If the ant's path is empty, the
        :py:meth:`~culebra.trainer.aco.abc.SingleColACO._initial_choice`
        method is called.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :return: The index of the chosen node or :py:data:`None` if there
            isn't any feasible node
        :rtype: :py:class:`int` or :py:data:`None`
        """
        # Return the initial choice for ants with an empty path
        if ant.current is None:
            return self._initial_choice()

        # Return the next feasible node, if available
        probs = self._feasible_neighborhood_probs(ant)
        if np.any(probs > 0):
            return np.random.choice(self._node_list, p=probs)

        # If there isn't any feasible node
        return None

    def _generate_ant(self) -> Ant:
        """Generate a new ant.

        The ant makes its path and gets evaluated.
        """
        ant = self.solution_cls(self.species, self.fitness_function.Fitness)
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
                ant.fitness.pheromone_amount
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
        self._choice_info = (
            np.power(self.pheromone[0], self.pheromone_influence[0]) *
            np.power(self.heuristic[0], self.heuristic_influence[0])
        )


class PheromoneBasedACO(SingleColACO):
    """Base class for al the ACO approaches guided by pheromone matrices.

    This kind of ACO approach relies on the pheromone matrices to guide the
    search, modified by the colony's ants at each iteration. Thus, the
    pheromone matrices must be part of the trainer's state.
    """

    @property
    def pheromone(self) -> Sequence[np.ndarray, ...] | None:
        """Get the pheromone matrices.

        If the search process has not begun, :py:data:`None` is returned.

        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        """
        return self._pheromone

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
        heuristic_shape = self._heuristic[0].shape
        self._pheromone = [
            np.full(
                heuristic_shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromone
        ]

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the pheromone matrices.
        """
        super()._reset_state()
        self._pheromone = None


class ElitistACO(SingleColACO):
    """Base class for all the elitist single colony ACO algorithms."""

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
        convergence_check_freq: Optional[int] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [ElitistACO],
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
        r"""Create a new elitist single-colony ACO trainer.

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
            :py:attr:`~culebra.trainer.aco.abc.ElitistACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.ElitistACO.num_heuristic_matrices`
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
            :py:attr:`~culebra.trainer.aco.abc.ElitistACO.num_pheromone_matrices`
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
            :py:attr:`~culebra.trainer.aco.abc.ElitistACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
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
        # Init the superclass
        SingleColACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
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

    def _reset_pheromone(self) -> None:
        """Reset the pheromone matrices."""
        heuristic_shape = self._heuristic[0].shape
        self._pheromone = [
            np.full(
                heuristic_shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromone
        ]

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
            self._reset_pheromone()


class WeightedElitistACO(ElitistACO):
    """Base class for all the weighted elitist single colony ACO algorithms."""

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
        convergence_check_freq: Optional[int] = None,
        elite_weight: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [WeightedElitistACO],
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
        r"""Create a new weighted elitist single-colony ACO trainer.

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
            :py:attr:`~culebra.trainer.aco.abc.WeightedElitistACO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.abc.WeightedElitistACO.num_heuristic_matrices`
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
            :py:attr:`~culebra.trainer.aco.abc.WeightedElitistACO.num_pheromone_matrices`
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
            :py:attr:`~culebra.trainer.aco.abc.WeightedElitistACO.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :py:data:`None`
        :type heuristic_influence: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`,
            optional
        :param convergence_check_freq: Convergence assessment frequency. If
            set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            will be used. Defaults to :py:data:`None`
        :type convergence_check_freq: :py:class:`int`, optional
        :param elite_weight: Weight for the elite ants (best-so-far ants)
            respect to the iteration-best ants.
            If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ELITE_WEIGHT` will be used.
            Defaults to :py:data:`None`
        :type elite_weight: :py:class:`float` in [0, 1], optional
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
        ElitistACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            convergence_check_freq=convergence_check_freq,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        self.elite_weight = elite_weight

    @property
    def elite_weight(self) -> float:
        """Get and set the elite weigth.

        :getter: Return the current elite weigth
        :setter: Set the new elite weigth.  If set to :py:data:`None`,
         :py:attr:`~culebra.trainer.aco.DEFAULT_ELITE_WEIGHT` is chosen
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a real number
        :raises ValueError: If set to a value outside [0, 1]
        """
        return (
            DEFAULT_ELITE_WEIGHT
            if self._elite_weight is None
            else self._elite_weight)

    @elite_weight.setter
    def elite_weight(self, weight: float | None) -> None:
        """Set a new elite weigth.

        :param weight: The new weight. If set to :py:data:`None`,
         :py:attr:`~culebra.trainer.aco.DEFAULT_ELITE_WEIGHT` is chosen
        :type weight: :py:class:`float`
        :raises TypeError: If *weight* is not a real number
        :raises ValueError: If *weight* is outside [0, 1]
        """
        # Check prob
        self._elite_weight = (
            None if weight is None else check_float(
                weight, "elite weigth", ge=0, le=1
            )
        )

        # Reset the trainer
        self.reset()


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
        max_pheromone: float | Sequence[float, ...],
        heuristic: Optional[
            Sequence[Sequence[float], ...] |
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float | Sequence[float, ...]] = None,
        heuristic_influence: Optional[float | Sequence[float, ...]] = None,
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
        :param max_pheromone: Maximum amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            pheromone matrices.
        :type max_pheromone: :py:class:`float` or
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
        self.max_pheromone = max_pheromone
        self.pop_size = pop_size

    @property
    def max_pheromone(self) -> Sequence[float, ...]:
        """Get and set the maximum value for each pheromone matrix.

        :getter: Return the maximum value for each pheromone matrix.
        :setter: Set a new maximum value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            pheromone matrices.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`

        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If any value is lower than or equal to its
            corresponding initial pheromone value
        :raises ValueError: If a sequence of values with a length different
            from
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
            is provided
        """
        return self._max_pheromone

    @max_pheromone.setter
    def max_pheromone(self, values: float | Sequence[float, ...]) -> None:
        """Set the maximum value for each pheromone matrix.

        :param values: New maximum value for each pheromone matrix. Both a
            scalar value or a sequence of values are allowed. If a scalar value
            is provided, it will be used for all the
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
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
            :py:attr:`~culebra.trainer.aco.abc.PACO.num_pheromone_matrices`
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

    @property
    def pheromone(self) -> Sequence[np.ndarray, ...] | None:
        """Get the pheromone matrices.

        If the search process has not begun, :py:data:`None` is returned.

        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        """
        return self._pheromone

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

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to initialize the population.
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
        matrices and the *_pop_ingoing* and *_pop_outgoing* lists of ants for
        the population are created. Subclasses which need more objects or data
        structures should override this method.
        """
        super()._init_internals()
        heuristic_shape = self._heuristic[0].shape
        self._pheromone = [
            np.full(
                heuristic_shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromone
        ]
        self._pop_ingoing = []
        self._pop_outgoing = []

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the pheromone matrices and the *_pop_ingoing* and
        *_pop_outgoing* lists of ants for the population. If subclasses
        overwrite the :py:meth:`~culebra.trainer.aco.abc.PACO._init_internals`
        method to add any new internal object, this method should also be
        overridden to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._pheromone = None
        self._pop_ingoing = None
        self._pop_outgoing = None

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

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the population
        self._update_pop()

        # Update the pheromone
        self._update_pheromone()

    def _deposit_pheromone(
        self, ants: Sequence[Ant], weight: Optional[float] = 1
    ) -> None:
        """Make some ants deposit weighted pheromone.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.solution.abc.Ant`
        :param weight: Weight for the pheromone. Negative weights remove
            pheromone. Defaults to 1
        :type weight: :py:class:`float`, optional
        """
        for ant in ants:
            for pher_index, (init_pher, max_pher) in enumerate(
                zip(self.initial_pheromone, self.max_pheromone)
            ):
                pher_delta = (
                    (max_pher - init_pher) / self.pop_size
                ) * weight
                org = ant.path[-1]
                for dest in ant.path:
                    self._pheromone[pher_index][org][dest] += pher_delta
                    self._pheromone[pher_index][dest][org] += pher_delta
                    org = dest

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


class AgeBasedPACO(PACO):
    """Base class for PACO with an age-based population update strategy."""

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :py:class:`~culebra.trainer.aco.abc.AgeBasedPACO` class, the youngest
        ant index is created. Subclasses which need more objects or data
        structures should override this method.
        """
        super()._init_internals()
        self._youngest_index = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the youngest ant index. If subclasses overwrite the
        :py:meth:`~culebra.trainer.aco.abc.AgeBasedPACO._init_internals`
        method to add any new internal object, this method should also be
        overridden to reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._youngest_index = None

    def _update_pop(self) -> None:
        """Update the population.

        The population is updated with the current iteration's colony. The best
        ants in the current colony, which are put in the *_pop_ingoing* list,
        will replace the eldest ants in the population, put in the
        *_pop_outgoing* list.

        These lists will be used later within the
        :py:meth:`~culebra.trainer.aco.abc.AgeBasedPACO._increase_pheromone`
        and
        :py:meth:`~culebra.trainer.aco.abc.AgeBasedPACO._decrease_pheromone`
        methods, respectively.
        """
        # Ingoing ants
        self._pop_ingoing = ParetoFront()
        self._pop_ingoing.update(self.col)

        # Outgoing ants
        self._pop_outgoing = []

        # Remaining room in the population
        remaining_room_in_pop = self.pop_size - len(self.pop)

        # For all the ants in the ingoing list
        for ant in self._pop_ingoing:
            # If there is still room in the population, just append it
            if remaining_room_in_pop > 0:
                self._pop.append(ant)
                remaining_room_in_pop -= 1

                # If the population is full, start with ants replacement
                if remaining_room_in_pop == 0:
                    self._youngest_index = 0
            # The eldest ant is replaced
            else:
                self._pop_outgoing.append(self.pop[self._youngest_index])
                self.pop[self._youngest_index] = ant
                self._youngest_index = (
                    (self._youngest_index + 1) % self.pop_size
                )


class QualityBasedPACO(PACO):
    """Base class for PACO with a quality-based population update strategy."""

    def _update_pop(self) -> None:
        """Update the population.

        The population now keeps the best ants found ever (non-dominated ants).
        The best ants in the current colony may enter the population (and also
        the *_pop_ingoing* list) and may also replace any ant (which will be
        appended to the *_pop_outgoing* list).

        Besides, if the number of non-dominated ants exceeds the population
        size, some ants will be randomly removed (and appended to the
        *_pop_outgoing* list).

        The *_pop_ingoing* and *_pop_outgoing* lists will be used later within
        the
        :py:meth:`~culebra.trainer.aco.abc.QualityBasedPACO._increase_pheromone`
        and
        :py:meth:`~culebra.trainer.aco.abc.QualityBasedPACO._decrease_pheromone`
        methods, respectively.
        """
        # Best ants in current colony
        best_in_col = ParetoFront()
        best_in_col.update(self.col)

        # Split the best ants into nondominated fronts
        nondominated_fronts = sortNondominated(
            self.pop + list(best_in_col),
            len(self.pop) + len(best_in_col)
        )

        # Keep the best ants for the next population
        new_pop = []
        remaining_room_in_pop = self.pop_size
        for front in nondominated_fronts:
            if len(front) <= remaining_room_in_pop:
                new_pop.extend(front)
                remaining_room_in_pop -= len(front)
            else:
                new_pop.extend(sample(front, remaining_room_in_pop))

        # Obtain the ingoing ants
        self._pop_ingoing = []
        for col_best_ant in best_in_col:
            for new_pop_ant in new_pop:
                if id(col_best_ant) == id(new_pop_ant):
                    self._pop_ingoing.append(col_best_ant)
                    break

        # Obtain the outgoing ants
        self._pop_outgoing = []
        for old_pop_ant in self.pop:
            remains = False
            for new_pop_ant in new_pop:
                if id(old_pop_ant) == id(new_pop_ant):
                    remains = True
                    break
            if not remains:
                self._pop_outgoing.append(old_pop_ant)

        # Update the population
        self._pop = new_pop


# Exported symbols for this module
__all__ = [
    'SingleColACO',
    'SinglePheromoneMatrixACO',
    'SingleHeuristicMatrixACO',
    'MultiplePheromoneMatricesACO',
    'MultipleHeuristicMatricesACO',
    'SingleObjACO',
    'ElitistACO',
    'WeightedElitistACO',
    'PACO',
    'AgeBasedPACO',
    'QualityBasedPACO'
]
