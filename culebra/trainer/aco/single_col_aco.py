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
    Sequence,
    Dict,
    Any
)
from math import ceil

import numpy as np

from deap.tools import ParetoFront

from culebra.abc import Species, FitnessFunction
from culebra.checker import check_float, check_int
from culebra.solution.abc import Ant
from culebra.trainer.aco.abc import SingleColACO, ElitistACO


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


DEFAULT_PHEROMONE_INFLUENCE = 1.0
r"""Default pheromone influence (:math:`{\alpha}`)."""

DEFAULT_HEURISTIC_INFLUENCE = 2.0
r"""Default heuristic influence (:math:`{\beta}`)."""

DEFAULT_PHEROMONE_EVAPORATION_RATE = 0.1
r"""Default pheromone evaporation rate (:math:`{\rho}`)."""

DEFAULT_ELITE_WEIGHT = 0.3
"""Default weight for the elite ant (best-so-far ant) respect to the
iteration-best ant."""

DEFAULT_MMAS_ITER_BEST_USE_LIMIT = 250
r"""Default limit for the number of iterations for the
:math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS to give up using the
iteration-best ant to deposit pheromones. Iterations above this limit will use
only the global-best ant."""


class AntSystem(SingleColACO):
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
        r"""Create a new Ant System trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function. Since the
            original Ant System algorithm was proposed to solve
            single-objective problems, only the first objective of the
            function is taken into account.
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
            provided by *fitness_function* are assumed. Defaults to
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
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
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
            initial_pheromones=initial_pheromones,
            heuristics=heuristics,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
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
        return SingleColACO.initial_pheromones.fget(self)

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
        SingleColACO.initial_pheromones.fset(self, values)
        self._initial_pheromones = [self._initial_pheromones[0]]

    @property
    def heuristics(self) -> Sequence[np.ndarray, ...]:
        """Get and set the heuristic matrix.

        Since the original Ant System algorithm was proposed to solve
        single-objective problems, only one heuristics matrix is used. Thus,
        only the first value in the sequence is taken into account.

        :getter: Return the heuristics matrices
        :setter: Set new heuristics matrices. If set to :py:data:`None`, the
            default heuristics (provided by the
            :py:attr:`~culebra.trainer.aco.AntSystem.fitness_function`
            property) are assumed.
        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        :raises TypeError: If any value is not an array
        :raises ValueError: If the matrix has a wrong shape or contain any
            negative value.
        """
        return SingleColACO.heuristics.fget(self)

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
            default heuristics (provided by the
            :py:attr:`~culebra.trainer.aco.AntSystem.fitness_function`
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
        SingleColACO.heuristics.fset(self, values)
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
        for ant in self.col:
            pheromones_amount = ant.fitness.pheromones_amount[0]
            org = ant.path[-1]
            for dest in ant.path:
                self._pheromones[0][org][dest] += pheromones_amount
                self._pheromones[0][dest][org] += pheromones_amount
                org = dest


class ElitistAntSystem(AntSystem, ElitistACO):
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
        convergence_check_freq: Optional[int] = None,
        elite_weight: Optional[float] = None,
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
        r"""Create a new Elitist Ant System trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function. Since the
            original Ant System algorithm was proposed to solve
            single-objective problems, only the first objective of the
            function is taken into account.
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
            provided by *fitness_function* are assumed. Defaults to
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
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used. Defaults to :py:data:`None`
        :type heuristic_influence: :py:class:`float`, optional
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
        :param elite_weight: Weight for the elite ant (best-so-far ant)
            respect to the iteration-best ant.
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
        AntSystem.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromones=initial_pheromones,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            pheromone_evaporation_rate=pheromone_evaporation_rate
        )
        ElitistACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromones=initial_pheromones,
            heuristics=heuristics,
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

        # Reset the algorithm
        self.reset()

    def _deposit_pheromones(self) -> None:
        """Deposit pheromones.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also modified with the same increment
        """
        # Iteration-best ants
        iter_best = ParetoFront()
        iter_best.update(self.col)
        for ant in iter_best:
            self._deposit_ant_weighted_pheromones(
                ant,
                (1 - self.elite_weight) / len(iter_best)
            )

        # Elite ants
        for ant in self._elite:
            self._deposit_ant_weighted_pheromones(
                ant,
                self.elite_weight / len(self._elite)
            )


class MMAS(AntSystem, ElitistACO):
    r""":math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` Ant System algorithm."""

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
        iter_best_use_limit: Optional[int] = None,
        convergence_check_freq: Optional[int] = None,
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
        r"""Create a new :math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function. Since the
            original Ant System algorithm was proposed to solve
            single-objective problems, only the first objective of the
            function is taken into account.
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
            provided by *fitness_function* are assumed. Defaults to
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
            :py:attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used. Defaults to :py:data:`None`
        :type heuristic_influence: :py:class:`float`, optional
        :param pheromone_evaporation_rate: Pheromone evaluation rate
            (:math:`{\rho}`). If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            will be used. Defaults to :py:data:`None`
        :type pheromone_evaporation_rate: :py:class:`float`, optional
        :param iter_best_use_limit: Limit for the number of iterations to give
            up using the iteration-best ant to deposit pheromones. Iterations
            above this limit will use only the global-best ant. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_MMAS_ITER_BEST_USE_LIMIT`
            will be used. Defaults to :py:data:`None`
        :type iter_best_use_limit: :py:class:`int`, optional
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
        AntSystem.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromones=initial_pheromones,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            pheromone_evaporation_rate=pheromone_evaporation_rate
        )
        ElitistACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromones=initial_pheromones,
            heuristics=heuristics,
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
        self.iter_best_use_limit = iter_best_use_limit
        self.convergence_check_freq = convergence_check_freq

    @property
    def iter_best_use_limit(self) -> int:
        """Get and set the iteration-best use limit.

        Iterations above this limit will use only the global-best ant.

        :getter: Return the iteration-best limit
        :setter: Set a value for the iteration-best limit. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_MMAS_ITER_BEST_USE_LIMIT`
            is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an int
        :raises ValueError: If set to a non-positive value
        """
        return (
            DEFAULT_MMAS_ITER_BEST_USE_LIMIT
            if self._iter_best_use_limit is None
            else self._iter_best_use_limit
        )

    @iter_best_use_limit.setter
    def iter_best_use_limit(self, value: int | None) -> None:
        """Set a value for the iteration-best use limit.

        Iterations above this limit will use only the global-best ant.

        :param value: New value for the iteration-best limit. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_MMAS_ITER_BEST_USE_LIMIT`
            is chosen
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is non-positive
        """
        # Check the value
        self._iter_best_use_limit = (
            None if value is None else check_int(
                value, "iteration-best use limit", gt=0
            )
        )

    @property
    def _global_best_freq(self):
        """Use frequency of the global-best solution to deposit pheromones.

        Implement the schedule to choose between the iteration-best and the
        global-best ant. The global-best use frequency will vary according to
        :py:attr:`~culebra.trainer.aco.MMAS._current_iter` and
        :py:attr:`~culebra.trainer.aco.MMAS.iter_best_use_limit`.

        :type: :py:class:`int`
        """
        freq = self.iter_best_use_limit
        if self._current_iter is not None:
            freq = ceil(self.iter_best_use_limit / (self._current_iter + 1))

        return freq

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this trainer.

        Overridden to add the pheromones limits and the last iteration number
        when the elite was updated to the trainer's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = ElitistACO._state.fget(self)

        # Get the state of this class
        state["max_pheromone"] = self._max_pheromone
        state["min_pheromone"] = self._min_pheromone
        state["last_elite_iter"] = self._last_elite_iter

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the pheromones limits and the last iteration number
        when the elite was updated to the trainer's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        ElitistACO._state.fset(self, state)

        # Set the state of this class
        self._max_pheromone = state["max_pheromone"]
        self._min_pheromone = state["min_pheromone"]
        self._last_elite_iter = state["last_elite_iter"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to initialize the pheromones limits and the last iteration
        number when the elite was updated.
        """
        super()._new_state()

        # Init the pheromone limits
        self._max_pheromone = self.initial_pheromones[0]
        self._min_pheromone = (
            self._max_pheromone / (2 * self.fitness_function.num_nodes)
        )
        self._last_elite_iter = None

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the pheromones limits and the last iteration number
        when the elite was updated.
        """
        super()._reset_state()
        self._max_pheromone = None
        self._min_pheromone = None
        self._last_elite_iter = None

    def _deposit_pheromones(self) -> None:
        """Deposit pheromones.

        Overridden to choose between the iteration-best and global-best ant
        depending on the :py:attr:`~culebra.trainer.aco.MMAS._global_best_freq`
        frequency.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also modified with the same increment
        """
        if (self._current_iter + 1) % self._global_best_freq == 0:
            # Use the global-best ant
            for ant in self._elite:
                self._deposit_ant_weighted_pheromones(
                    ant,
                    1 / len(self._elite)
                )
        else:
            # Use the iteration-best ant
            iter_best = ParetoFront()
            iter_best.update(self.col)
            for ant in iter_best:
                self._deposit_ant_weighted_pheromones(
                    ant,
                    1 / len(iter_best)
                )

    def _update_pheromones(self) -> None:
        """Update the pheromone trails.

        First, the pheromones are evaporated. Then ants deposit pheromones
        according to their fitness. Finally, the max and min limits for
        the pheromones are checked.
        """
        super()._update_pheromones()

        # Check the max and min limits
        self.pheromones[0][
            self.pheromones[0] < self._min_pheromone
        ] = self._min_pheromone

        self.pheromones[0][
            self.pheromones[0] > self._max_pheromone
        ] = self._max_pheromone

    def _update_elite(self) -> None:
        """Update the elite (best-so-far) ant.

        The pheromone limits and the elite updation iteration number are
        modified accordingly.
        """
        last_elite = self._elite[0] if len(self._elite) > 0 else None
        super()._update_elite()
        current_elite = self._elite[0]

        # If a better solution has been found
        if last_elite is None or current_elite > last_elite:
            self._max_pheromone = (
                current_elite.fitness.pheromones_amount[0] /
                self.pheromone_evaporation_rate)
            self._min_pheromone = (
                self._max_pheromone / (
                    2 * self.fitness_function.num_nodes
                )
            )
            self._last_elite_iter = self._current_iter

    def _has_converged(self) -> None:
        """Detect if the trainer has converged.

        :return: :py:data:`True` if the trainer has converged
        :rtype: :py:class:`bool`
        """
        convergence = True
        # Check the pheromones matrix
        for row in range(len(self.pheromones[0])):
            max_pher_count = np.isclose(
                self.pheromones[0][row], self._max_pheromone
            ).sum()
            min_pher_count = np.isclose(
                self.pheromones[0][row], self._min_pheromone
            ).sum()

            if (
                (max_pher_count != 0 and max_pher_count != 2) or
                max_pher_count + min_pher_count != self.species.num_nodes
            ):
                convergence = False
                break

        # Check the last time the elite improved
        if (
            convergence and
            self._current_iter - self._last_elite_iter <
            self.convergence_check_freq / 2
        ):
            convergence = False

        return convergence

    def _reset_pheromones(self) -> None:
        """Reset the pheromones matrices."""
        self._pheromones[0] = np.full(
            self._pheromones[0].shape,
            self._max_pheromone,
            dtype=float
        )


# Exported symbols for this module
__all__ = [
    'AntSystem',
    'ElitistAntSystem',
    'MMAS',
    'DEFAULT_PHEROMONE_INFLUENCE',
    'DEFAULT_HEURISTIC_INFLUENCE',
    'DEFAULT_PHEROMONE_EVAPORATION_RATE',
    'DEFAULT_ELITE_WEIGHT',
    'DEFAULT_MMAS_ITER_BEST_USE_LIMIT'
]
