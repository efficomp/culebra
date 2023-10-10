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

"""Implementation of single colony ACO algorithms."""

from __future__ import annotations

from typing import (
    Type,
    Callable,
    Optional,
    Sequence
)

import numpy as np

from culebra.abc import Species, FitnessFunction
from culebra.checker import check_float
from culebra.solution.abc import Ant
from culebra.trainer.aco.abc import SinglePopACO


DEFAULT_PHEROMONE_INFLUENCE = 1
r"""Default pheromone influence (:math:`{\alpha}`)."""

DEFAULT_HEURISTIC_INFLUENCE = 2
r"""Default heuristic influence (:math:`{\beta}`)."""

DEFAULT_PHEROMONE_EVAPORATION_RATE = 0.5
r"""Default pheromone evaporation rate (:math:`{\rho}`)."""


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class AntSystem(SinglePopACO):
    """Implement the Ant System algorithm."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromones: Sequence[float, ...],
        heuristics: Optional[
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        pheromone_influence: Optional[float] = None,
        heuristic_influence: Optional[float] = None,
        pheromone_evaporation_rate: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopACO],
                bool
            ]
        ] = None,
        pop_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new Ant System trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function. Since the
        original Ant System algorithm was proposed to solve single-objective
        problems, only the first objective of the function is taken into
        account.
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromones: Initial amount of pheromone for the paths
            of each pheromones matrix. Since the original Ant System algorithm
            was proposed to solve single-objective problems, only one
            pheromones matrix is used. Thus, only the first value in the
            sequence is taken into account.
        :type initial_pheromones: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :param heuristics: Heuristics matrices. Since the original Ant System
            algorithm was proposed to solve single-objective problems, only one
            heuristics matrix is used. Thus, only the first value in the
            sequence is taken into account. If omitted, the default heuristics
            provided *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristics: :py:class:`~collections.abc.Sequence` of
            two-dimensional array-like objects, optional
        :param pheromone_influence: Relative influence of pheromones
            (:math:`{\alpha}`) for the choice of an arc. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used. Defaults to :py:data:`None`
        :type pheromone_influence: :py:class:`float`, optional
        :param heuristic_influence: Relative influence of heuristics
            (:math:`{\beta}`) for the choice of an arc. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTICS_INFLUENCE` will
            be used. Defaults to :py:data:`None`
        :type heuristic_influence: :py:class:`float`, optional
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
        :param pop_size: The population (colony) size. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_POP_SIZE`
            will be used. Defaults to :py:data:`None`
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
        SinglePopACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromones=initial_pheromones,
            heuristics=heuristics,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            pop_size=pop_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )
        self.pheromone_influence = pheromone_influence
        self.heuristic_influence = heuristic_influence
        self.pheromone_evaporation_rate = pheromone_evaporation_rate

    @property
    def initial_pheromones(self) -> Sequence[float, ...]:
        """Get and set the initial value for the pheromones matrix.

        Since the original Ant System algorithm was proposed to solve
        single-objective problems, only one pheromones matrix is used. Thus,
        only the first value in the sequence is taken into account.

        :getter: Return the initial value for the pheromones matrix.
        :setter: Set a new initial value for the pheromones matricx.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If the sequence is empty
        """
        return SinglePopACO.initial_pheromones.fget(self)

    @initial_pheromones.setter
    def initial_pheromones(self, values: Sequence[float, ...]) -> None:
        """Set the initial value for the pheromones matrix.

        Since the original Ant System algorithm was proposed to solve
        single-objective problems, only one pheromones matrix is used. Thus,
        only the first element in *values* is taken into account.

        :type values: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If any element in *values* is not a float value
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is empty
        """
        SinglePopACO.initial_pheromones.fset(self, values)
        self._initial_pheromones = [self._initial_pheromones[0]]

    @property
    def heuristics(self) -> Sequence[np.ndarray, ...]:
        """Get and set the heuristic matrix.

        Since the original Ant System algorithm was proposed to solve
        single-objective problems, only one heuristics matrix is used. Thus,
        only the first value in the sequence is taken into account.

        :getter: Return the heuristics matrices
        :setter: Set new heuristics matrices. If set to :py:data:`None`, the
            default heuristics provided by the
            :py:attr:`~culebra.trainer.aco.SinglePopACO.fitness_function`
            property) are assumed.
        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        :raises TypeError: If any value is not an array
        :raises ValueError: If the matrix has a wrong shape or contain any
            negative value.
        """
        return SinglePopACO.heuristics.fget(self)

    @heuristics.setter
    def heuristics(
        self,
        values: Sequence[Sequence[Sequence[float], ...], ...] | None
    ) -> None:
        """Set new heuristic matrices.

        Since the original Ant System algorithm was proposed to solve
        single-objective problems, only one heuristics matrix is used. Thus,
        only the first element in *values* is taken into account.

        :param values: New heuristics matrices. If set to :py:data:`None`, the
            default heuristics provided by the
            :py:attr:`~culebra.trainer.aco.SinglePopACO.fitness_function`
            property) are assumed.
        :type: :py:class:`~collections.abc.Sequence` of two-dimensional
            array-like objects
        :raises TypeError: If *values* is not a
            :py:class:`~collections.abc.Sequence`

        :raises TypeError: If any element in *values* is not an array-like
            object
        :raises ValueError: If any element in *values* has any not floatç
            element
        :raises ValueError: If any element in *values* has not an homogeneous
            shape
        :raises ValueError: If any element in *values* has not two dimensions
        :raises ValueError: If any element in *values* has negative values
        """
        SinglePopACO.heuristics.fset(self, values)
        self._heuristics = [self._heuristics[0]]

    @property
    def pheromone_influence(self) -> float:
        r"""Get and set the relative influence of pheromones (:math:`{\alpha}`).

        :getter: Return the relative influence of pheromones
        :setter: Set a value for the relative influence of pheromones. If set
            to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` is
            chosen
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a float
        :raises ValueError: If set to a negative value
        """
        return (
            DEFAULT_PHEROMONE_INFLUENCE if self._pheromone_influence is None
            else self._pheromone_influence
        )

    @pheromone_influence.setter
    def pheromone_influence(self, value: float | None) -> None:
        r"""Set a value for relative influence of pheromones (:math:`{\alpha}`).

        :param value: New value for relative influence of pheromones. If set
            to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` is
            chosen
        :type value: :py:class:`float`
        :raises TypeError: If *value* is not a floating point number
        :raises ValueError: If *value* is negative
        """
        # Check the value
        self._pheromone_influence = (
            None if value is None else check_float(
                value, "pheromone influence", ge=0
            )
        )

    @property
    def heuristic_influence(self) -> float:
        r"""Get and set the relative influence of heuristics (:math:`{\beta}`).

        :getter: Return the relative influence of heuristics
        :setter: Set a value for the relative influence of heuristics. If set
            to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` is
            chosen
        :type: :py:class:`float`
        :raises TypeError: If set to a value which is not a float
        :raises ValueError: If set to a negative value
        """
        return (
            DEFAULT_HEURISTIC_INFLUENCE if self._heuristic_influence is None
            else self._heuristic_influence
        )

    @heuristic_influence.setter
    def heuristic_influence(self, value: float | None) -> None:
        r"""Set a value for relative influence of heuristics (:math:`{\beta}`).

        :param value: New value for relative influence of heuristics. If set
            to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` is
            chosen
        :type value: :py:class:`float`
        :raises TypeError: If *value* is not a floating point number
        :raises ValueError: If *value* is negative
        """
        # Check the value
        self._heuristic_influence = (
            None if value is None else check_float(
                value, "heuristic influence", ge=0
            )
        )

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
        r"""Set a value for pheromone evaporation rate (:math:`{\rho}`).

        :param value: New value for pheromone evaporation rate. If set to
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

    def _calculate_choice_info(self) -> None:
        """Calculate the choice info matrix."""
        self._choice_info = (
            np.power(self.pheromones[0], self.pheromone_influence) *
            np.power(self.heuristics[0], self.heuristic_influence)
        )

    def _evaporate_pheromones(self) -> None:
        """Evaporate pheromones."""
        self._pheromones[0] *= (1 - self.pheromone_evaporation_rate)

    def _deposit_pheromones(self) -> None:
        """Deposit pheromones.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also modified with the same increment
        """
        for ant in self.pop:
            pheromones_amount = ant.fitness.pheromones_amount[0]
            org = ant.path[-1]
            for dest in ant.path:
                self._pheromones[0][org][dest] += pheromones_amount
                self._pheromones[0][dest][org] += pheromones_amount
                org = dest


# Exported symbols for this module
__all__ = [
    'AntSystem',
    'DEFAULT_PHEROMONE_INFLUENCE',
    'DEFAULT_HEURISTIC_INFLUENCE',
    'DEFAULT_PHEROMONE_EVAPORATION_RATE'
]