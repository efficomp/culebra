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

"""Implementation of ACO algorithms for feature selection problems."""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Callable
from math import isclose

import numpy as np
from deap.tools import sortNondominated, selNSGA2

from culebra.abc import Base, FitnessFunction
from culebra.solution.feature_selection import Ant, Species

from culebra.trainer.aco.abc import (
    ReseteablePheromoneBasedACO,
    ElitistACO,
    PACO,
    ACOFS
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class ACOFSConvergenceDetector(Base):
    """Detect the convergence of an :attr:`~culebra.trainer.aco.abc.ACOFS`
    instance.
    """

    def __init__(
        self, convergence_check_freq: int | None = None
    ) -> None:
        """Create a convergence detector.

        :param convergence_check_freq: Convergence assessment frequency. If
            omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            will be used. Defaults to :data:`None`
        :type convergence_check_freq: int
        """
        self.convergence_check_freq = convergence_check_freq
        self.last_pheromone = None

    convergence_check_freq = (
        ReseteablePheromoneBasedACO.convergence_check_freq
    )

    def has_converged(self, trainer) -> bool:
        """Detect if the trainer has converged.

        :param trainer: The trainer
        :type trainer: ~culebra.trainer.aco.abc.ACOFS

        :return: :data:`True` if the trainer has converged
        :rtype: bool
        """
        convergence = False
        if trainer.current_iter % self.convergence_check_freq == 0:
            if trainer.current_iter != 0:
                diff = np.sum(
                    np.abs(
                        trainer.pheromone[0] - self.last_pheromone[0]
                    )
                )
                if isclose(diff, 0):
                    convergence = True

            self.last_pheromone = trainer.pheromone

        return convergence


class PACOFS(
    PACO,
    ACOFS
):
    """Implement a population-based ACO for FS algorithm."""

    def __init__(
        self,
        solution_cls: type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...] | None = None,
        heuristic:
            np.ndarray[float] | Sequence[np.ndarray[float], ...] | None = None,
        pheromone_influence: float | Sequence[float, ...] | None = None,
        heuristic_influence: float | Sequence[float, ...] | None = None,
        exploitation_prob: float | None = None,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[PACOFS], bool] | None = None,
        col_size: int | None = None,
        pop_size: int | None = None,
        discard_prob: float | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        r"""Create a new population-based ACO-FS trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.feature_selection.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.solution.feature_selection.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.PACOFS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_initial_pheromone`
            will be used. Defaults to :data:`None`
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.PACOFS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_heuristic`
            will be used. Defaults to :data:`None`
        :type heuristic:
            ~numpy.ndarray[float] |
            ~collections.abc.Sequence[~numpy.ndarray[float], ...]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.PACOFS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_pheromone_influence`
            will be used. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.PACOFS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_heuristic_influence`
            will be used. Defaults to :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_exploitation_prob`
            will be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.aco.PACOFS._default_termination_func` is
            used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_col_size` will be
            used. Defaults to :data:`None`
        :type col_size: int
        :param pop_size: The population size. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_pop_size` will be
            used. Defaults to :data:`None`
        :type pop_size: int
        :param discard_prob: Probability of discarding a node (feature). If
            omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_discard_prob` is used.
            Defaults to :data:`None`
        :type discard_prob: float
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.aco.PACOFS._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclasses
        ACOFS.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            discard_prob=discard_prob
        )
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
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )
        self._pareto_fronts = None

    def _update_pop(self) -> None:
        """Update the population."""
        # Update the population according to the NSGA-II selection
        # procedure
        self.pop[:] = selNSGA2(self.pop[:] + self.col, self.pop_size)

    def _pheromone_amount (self, ant: Ant) -> tuple[float, ...]:
        """Return the amount of pheromone to be deposited by an ant.

        Each ant deposits an amount of pheromone calculated as its rank + 1.

        :param ant: The ant
        :type ant: ~culebra.solution.feature_selection.Ant
        :return: The amount of pheromone to be deposited for each objective
        :rtype: tuple[float]
        """
        pher_amount = None
        for rank, front in enumerate(self._pareto_fronts):
            if ant in front:
                pher_amount = rank + 1
                break

        return (pher_amount,)

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        Pheromone trails are reinitialized and updated according to the
        current population.
        """
        # Reinitialize the pheromone trails
        self._init_pheromone()

        # Sort the population into nondomination levels
        # to allow the _pheromone_amount method to assign a pheromone
        # amount to each ant according to its rank + 1
        self._pareto_fronts = sortNondominated(self.pop, self.pop_size)

        # Update the pheromone matrices with the current population
        self._deposit_pheromone(self.pop)


class ElitistACOFS(
    ElitistACO,
    ACOFS
):
    """Implement an elitist ACO for FS algorithm."""

    def __init__(
        self,
        solution_cls: type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromone: float | Sequence[float, ...] | None = None,
        heuristic:
            np.ndarray[float] | Sequence[np.ndarray[float], ...] | None = None,
        pheromone_influence: float | Sequence[float, ...] | None = None,
        heuristic_influence: float | Sequence[float, ...] | None = None,
        exploitation_prob: float | None = None,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[ElitistACOFS], bool] | None = None,
        col_size: int | None = None,
        discard_prob: float | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        r"""Create a new elitist ACO-FS trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.solution.feature_selection.Ant]
        :param species: The species for all the ants
        :type species: ~culebra.solution.feature_selection.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param initial_pheromone: Initial amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.ElitistACOFS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_initial_pheromone`
            will be used. Defaults to :data:`None`
        :type initial_pheromone: float | ~collections.abc.Sequence[float]
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :attr:`~culebra.trainer.aco.ElitistACOFS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_heuristic`
            will be used. Defaults to :data:`None`
        :type heuristic:
            ~numpy.ndarray[float] |
            ~collections.abc.Sequence[~numpy.ndarray[float], ...]
        :param pheromone_influence: Relative influence of each pheromone
            matrix (:math:`{\alpha}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.ElitistACOFS.num_pheromone_matrices`
            pheromone matrices. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_pheromone_influence`
            will be used. Defaults to :data:`None`
        :type pheromone_influence: float | ~collections.abc.Sequence[float]
        :param heuristic_influence: Relative influence of each heuristic
            (:math:`{\beta}`). Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :attr:`~culebra.trainer.aco.ElitistACOFS.num_heuristic_matrices`
            heuristic matrices. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_heuristic_influence`
            will be used. Defaults to :data:`None`
        :type heuristic_influence: float | ~collections.abc.Sequence[float]
        :param exploitation_prob: Probability to make the best possible move
            (:math:`{q_0}`). If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_exploitation_prob`
            will be used. Defaults to :data:`None`
        :type exploitation_prob: float
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.aco.ElitistACOFS._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param col_size: The colony size. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_col_size` will be
            used. Defaults to :data:`None`
        :type col_size: int
        :param discard_prob: Probability of discarding a node (feature). If
            omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_discard_prob` is
            used. Defaults to :data:`None`
        :type discard_prob: float
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.aco.ElitistACOFS._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclasses
        ACOFS.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            discard_prob=discard_prob
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
            exploitation_prob=exploitation_prob,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

    def _set_state(self, state: dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to update the pheromone matrix according to the elite.

        :param state: The last loaded state
        :type state: dict
        """
        # Set the state of the superclass
        super()._set_state(state)

        # Generate the pheromone matrices with the current elite
        self._update_pheromone()

    def _update_elite(self) -> None:
        """Update the elite (best-so-far) ant.

        Overridden to update the pheromone matrix according to the elite.
        """
        super()._update_elite()

        # Update the pheromone
        self._update_pheromone()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the elite (and also the pheromone matrix)
        self._update_elite()

    def _pheromone_amount(self, ant: Ant) -> tuple[float, ...]:
        """Return the amount of pheromone to be deposited by an ant.

        Each ant deposits the same amount of pheromone:
        :attr:`~culebra.trainer.aco.ElitistACOFS.initial_pheromone`.

        :param ant: The ant
        :type ant: ~culebra.solution.feature_selection.Ant
        :return: The amount of pheromone to be deposited for each objective
        :rtype: tuple[float]
        """
        return self.initial_pheromone

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        Pheromone trails are reinitialized and updated according to the
        current population.
        """
        # Reinitialize the pheromone trails
        self._init_pheromone()

        # Update the pheromone trails
        self._deposit_pheromone(self._elite)


__all__ = [
    'ACOFSConvergenceDetector',
    'PACOFS',
    'ElitistACOFS'
]
