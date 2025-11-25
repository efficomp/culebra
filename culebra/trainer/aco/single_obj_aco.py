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

"""Implementation of single colony ACO algorithms."""

from __future__ import annotations

from typing import (
    Type,
    Callable,
    Optional,
    Dict,
    Any
)
from collections.abc import Sequence
from math import ceil
import bisect

import numpy as np

from deap.tools import ParetoFront, selBest

from culebra.abc import Species, FitnessFunction
from culebra.checker import check_float, check_int
from culebra.solution.abc import Ant
from culebra.trainer.aco.abc import (
    PheromoneBasedACO,
    SingleObjACO,
    ReseteablePheromoneBasedACO,
    SingleObjPACO
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


DEFAULT_AS_EXPLOITATION_PROB = 0
r"""Default exploitation probability (:math:`{q_0}`) for the Ant System trainer."""

DEFAULT_ELITE_WEIGHT = 0.3
"""Default weight for the elite ants (best-so-far ants) respect to the
iteration-best ant."""

DEFAULT_MMAS_ITER_BEST_USE_LIMIT = 250
r"""Default limit for the number of iterations for the
:math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS to give up using the
iteration-best ant to deposit pheromone. Iterations above this limit will use
only the global-best ant."""


class AntSystem(PheromoneBasedACO, SingleObjACO):
    """Implement the Ant System algorithm."""

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
                [AntSystem],
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

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function. Since the
            original Ant System algorithm was proposed to solve
            single-objective problems, only the first objective of the
            function is taken into account.
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.AntSystem.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.AntSystem.num_heuristic_matrices`
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
            :attr:`~culebra.trainer.aco.AntSystem.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.AntSystem.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_AS_EXPLOITATION_PROB` will
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
        SingleObjACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
        )
        PheromoneBasedACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            pheromone_evaporation_rate=pheromone_evaporation_rate,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

    @PheromoneBasedACO.exploitation_prob.getter
    def exploitation_prob(self) -> float:
        """Exploitation probability (:math:`{q_0}`).

        :rtype: float
        :setter: Set a new value for the exploitation probability
        :param prob: The new probability. If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_AS_EXPLOITATION_PROB` is
            chosen
        :type prob: float
        :raises TypeError: If *prob* is not a real number
        :raises ValueError: If *prob* is not in [0, 1]
        """
        return (
            DEFAULT_AS_EXPLOITATION_PROB
            if self._exploitation_prob is None
            else self._exploitation_prob
        )

class ElitistAntSystem(ReseteablePheromoneBasedACO, SingleObjACO):
    """Implement the Ant System algorithm."""

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
        elite_weight: Optional[float] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [ElitistAntSystem],
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

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function. Since the
            original Ant System algorithm was proposed to solve
            single-objective problems, only the first objective of the
            function is taken into account.
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.ElitistAntSystem.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.ElitistAntSystem.num_heuristic_matrices`
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
            :attr:`~culebra.trainer.aco.ElitistAntSystem.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.ElitistAntSystem.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_AS_EXPLOITATION_PROB` will
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
        :param elite_weight: Weight for the elite ant (best-so-far ant)
            respect to the iteration-best ant.
            If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_ELITE_WEIGHT` will be used.
            Defaults to :data:`None`
        :type elite_weight: float
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
        SingleObjACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
        )
        ReseteablePheromoneBasedACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            pheromone_evaporation_rate=pheromone_evaporation_rate,
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

    # Copy this property
    exploitation_prob = AntSystem.exploitation_prob

    @property
    def elite_weight(self) -> float:
        """Elite weigth.

        :rtype: float
        :setter: Set a new elite weigth
        :param weight: The new weight. If set to :data:`None`,
         :attr:`~culebra.trainer.aco.DEFAULT_ELITE_WEIGHT` is chosen
        :type weight: float
        :raises TypeError: If *weight* is not a real number
        :raises ValueError: If *weight* is outside [0, 1]
        """
        return (
            DEFAULT_ELITE_WEIGHT
            if self._elite_weight is None
            else self._elite_weight)

    @elite_weight.setter
    def elite_weight(self, weight: float | None) -> None:
        """Set a new elite weigth.

        :param weight: The new weight. If set to :data:`None`,
         :attr:`~culebra.trainer.aco.DEFAULT_ELITE_WEIGHT` is chosen
        :type weight: float
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

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone."""
        # Iteration-best ants
        iter_best = ParetoFront()
        iter_best.update(self.col)
        if len(iter_best) > 0:
            self._deposit_pheromone(
                iter_best,
                (1 - self.elite_weight) / len(iter_best)
            )

        # Elite ants
        if len(self._elite) > 0:
            self._deposit_pheromone(
                self._elite,
                self.elite_weight / len(self._elite)
            )


class MMAS(ReseteablePheromoneBasedACO, SingleObjACO):
    r""":math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` Ant System algorithm."""

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
        iter_best_use_limit: Optional[int] = None,
        convergence_check_freq: Optional[int] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [MMAS],
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

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.abc.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function. Since the
            original Ant System algorithm was proposed to solve
            single-objective problems, only the first objective of the
            function is taken into account.
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.MMAS.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.MMAS.num_heuristic_matrices`
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
            :attr:`~culebra.trainer.aco.MMAS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_INFLUENCE` will
            be used for all the pheromone matrices. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.MMAS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_HEURISTIC_INFLUENCE` will
            be used for all the heuristic matrices. Defaults to
            :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_AS_EXPLOITATION_PROB` will
            be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param pheromone_evaporation_rate: Pheromone evaluation rate
            (:math:`{\rho}`). If set to :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_EVAPORATION_RATE`
            will be used. Defaults to :data:`None`
        :type pheromone_evaporation_rate: float
        :param iter_best_use_limit: Limit for the number of iterations to give
            up using the iteration-best ant to deposit pheromone. Iterations
            above this limit will use only the global-best ant. If set to
            :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_MMAS_ITER_BEST_USE_LIMIT`
            will be used. Defaults to :data:`None`
        :type iter_best_use_limit: int
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
        SingleObjACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
        )
        ReseteablePheromoneBasedACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            heuristic=heuristic,
            pheromone_influence=pheromone_influence,
            heuristic_influence=heuristic_influence,
            exploitation_prob=exploitation_prob,
            pheromone_evaporation_rate=pheromone_evaporation_rate,
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

    # Copy this property
    exploitation_prob = AntSystem.exploitation_prob

    @property
    def iter_best_use_limit(self) -> int:
        """Iteration-best use limit.

        Iterations above this limit will use only the global-best ant.

        :rtype: int
        :setter: Set a value for the iteration-best use limit
        :param value: New value for the iteration-best limit. If set to
            :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_MMAS_ITER_BEST_USE_LIMIT`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is non-positive
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
            :data:`None`,
            :attr:`~culebra.trainer.aco.DEFAULT_MMAS_ITER_BEST_USE_LIMIT`
            is chosen
        :type value: int
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
    def _global_best_freq(self) -> int:
        """Use frequency of the global-best solution to deposit pheromone.

        Implement the schedule to choose between the iteration-best and the
        global-best ant. The global-best use frequency will vary according to
        :attr:`~culebra.trainer.aco.MMAS.current_iter` and
        :attr:`~culebra.trainer.aco.MMAS.iter_best_use_limit`.

        :rtype: int
        """
        freq = self.iter_best_use_limit
        if self.current_iter is not None:
            freq = ceil(self.iter_best_use_limit / (self.current_iter + 1))

        return freq

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the pheromone limits and the last iteration number
        when the elite was updated to the trainer's state.

        :rtype: dict
        """
        # Get the state of the superclass
        state = super()._get_state()

        # Get the state of this class
        state["max_pheromone"] = self._max_pheromone
        state["min_pheromone"] = self._min_pheromone
        state["last_elite_iter"] = self._last_elite_iter

        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the pheromone limits and the last iteration number
        when the elite was updated to the trainer's state.

        :param state: The last loaded state
        :type state: dict
        """
        # Set the state of the superclass
        super()._set_state(state)

        # Set the state of this class
        self._max_pheromone = state["max_pheromone"]
        self._min_pheromone = state["min_pheromone"]
        self._last_elite_iter = state["last_elite_iter"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to initialize the pheromone limits and the last iteration
        number when the elite was updated.
        """
        # Init the pheromone limits
        dimension = self.pheromone_shapes[0][0]
        self._max_pheromone = self.initial_pheromone[0]
        self._min_pheromone = self._max_pheromone / (2 * dimension)

        super()._new_state()

        self._last_elite_iter = None

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the pheromone limits and the last iteration number
        when the elite was updated.
        """
        super()._reset_state()
        self._max_pheromone = None
        self._min_pheromone = None
        self._last_elite_iter = None

    def _increase_pheromone(self) -> None:
        """Increase the amount of pheromone.

        Overridden to choose between the iteration-best and global-best ant
        depending on the :attr:`~culebra.trainer.aco.MMAS._global_best_freq`
        frequency.
        """
        if (self._current_iter + 1) % self._global_best_freq == 0:
            # Use the global-best ant
            if len(self._elite) > 0:
                self._deposit_pheromone(self._elite, 1 / len(self._elite))
        else:
            # Use the iteration-best ant
            iter_best = ParetoFront()
            iter_best.update(self.col)
            if len(iter_best) > 0:
                self._deposit_pheromone(iter_best, 1 / len(iter_best))

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        First, the pheromone are evaporated. Then ants deposit pheromone
        according to their fitness. Finally, the max and min limits for
        the pheromone are checked.
        """
        super()._update_pheromone()

        self.pheromone[0][
            self.pheromone[0] < self._min_pheromone
        ] = self._min_pheromone
        self.pheromone[0][
            self.pheromone[0] > self._max_pheromone
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
        if last_elite is None or current_elite.dominates(last_elite):
            dimension = self.pheromone_shapes[0][0]
            self._max_pheromone = (
                self._pheromone_amount(current_elite)[0] /
                self.pheromone_evaporation_rate
            )
            self._min_pheromone = (
                self._max_pheromone / (
                    2 * dimension
                )
            )
            self._last_elite_iter = self._current_iter

    def _has_converged(self) -> bool:
        """Detect if the trainer has converged.

        :return: :data:`True` if the trainer has converged
        :rtype: bool
        """
        convergence = True

        num_rows = self.pheromone[0].shape[0]
        for row in self.pheromone[0]:
            max_pher_count = np.isclose(row, self._max_pheromone).sum()
            min_pher_count = np.isclose(row, self._min_pheromone).sum()

            if (
                (max_pher_count != 0 and max_pher_count != 2) or
                max_pher_count + min_pher_count != num_rows
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

    def _init_pheromone(self) -> None:
        """Init the pheromone matrix."""
        self._pheromone = [
            np.full(
                shape,
                self._max_pheromone,
                dtype=float
            ) for shape in self.pheromone_shapes
        ]


class AgeBasedPACO(SingleObjPACO):
    """Single-objective PACO with an age-based population update strategy."""

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :class:`~culebra.trainer.aco.AgeBasedPACO` class, the youngest
        ant index is created. Subclasses which need more objects or data
        structures should override this method.
        """
        super()._init_internals()
        self._youngest_index = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the youngest ant index. If subclasses overwrite the
        :meth:`~culebra.trainer.aco.AgeBasedPACO._init_internals`
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
        :meth:`~culebra.trainer.aco.AgeBasedPACO._increase_pheromone`
        and
        :meth:`~culebra.trainer.aco.AgeBasedPACO._decrease_pheromone`
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


class QualityBasedPACO(SingleObjPACO):
    """Single-objective PACO with a quality-based population update strategy."""

    def _update_pop(self) -> None:
        """Update the population.

        The population now keeps the best ants found ever. The best ant in the
        current colony will enter the population (and also the *_pop_ingoing*
        list) only if is better than the worst ant in the population, which
        will be put in the *_pop_outgoing* list).

        These lists will be used later within
        the
        :meth:`~culebra.trainer.aco.QualityBasedPACO._increase_pheromone`
        and
        :meth:`~culebra.trainer.aco.QualityBasedPACO._decrease_pheromone`
        methods, respectively.
        """
        # Best ant in current colony
        [best_in_col] = selBest(self.col, 1)
        self._pop_ingoing = []
        self._pop_outgoing = []

        # If the best ant in the current colony
        # is better than the worst ant in the population
        if (
            len(self.pop) < self.pop_size or
            best_in_col.fitness > self.pop[-1].fitness
        ):
            # The best ant in the current colony will enter the population
            self._pop_ingoing.append(best_in_col)

            # If there isn't room in the population,
            # the worst ant in the population must go out
            if len(self.pop) == self.pop_size:
                self._pop_outgoing.append(self.pop[-1])
                self.pop.pop()

            # Insert the best ant keeping the population sorted
            bisect.insort(
                self.pop,
                best_in_col,
                key=lambda x: x.fitness.wvalues[0] * -1
            )


# Exported symbols for this module
__all__ = [
    'AntSystem',
    'ElitistAntSystem',
    'MMAS',
    'AgeBasedPACO',
    'QualityBasedPACO',
    'DEFAULT_AS_EXPLOITATION_PROB',
    'DEFAULT_ELITE_WEIGHT',
    'DEFAULT_MMAS_ITER_BEST_USE_LIMIT'
]
