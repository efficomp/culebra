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

from typing import (
    Any,
    Type,
    Callable,
    Optional,
    Sequence,
    Dict
)
from copy import copy

import numpy as np

from deap.tools import sortNondominated, selNSGA2

from culebra.abc import Species, FitnessFunction
from culebra.solution.abc import Ant

from culebra.trainer.aco.abc import (
    ElitistACO,
    PACO,
    ACO_FS
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class PACO_FS(
    PACO,
    ACO_FS
):
    """Implement a population-based ACO for FS algorithm."""

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
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [PACO_FS],
                bool
            ]
        ] = None,
        col_size: Optional[int] = None,
        pop_size: Optional[int] = None,
        discard_prob: Optional[float] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        r"""Create a new population-based ACO-FS trainer.

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
            :py:attr:`~culebra.trainer.aco.PACO_FS.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_INITIAL_PHEROMONE`
            will be used. Defaults to :py:data:`None`
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.PACO_FS.num_heuristic_matrices`
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
            :py:attr:`~culebra.trainer.aco.PACO_FS.num_pheromone_matrices`
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
            :py:attr:`~culebra.trainer.aco.PACO_FS.num_heuristic_matrices`
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
        ACO_FS.__init__(
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

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to initialize the population with random ants.
        """
        super()._new_state()

        # Not really in the state,
        # but needed to evaluate the initial population
        self._current_iter_evals = 0

        # Save the colony
        current_colony = copy(self.col)

        # Fill the population and append its statistics to the logbook
        # Since the evaluation of the initial population is performed
        # before the first iteration, fix self.current_iter = -1
        # The poppulation is filled through a initial colony sized to pop_size
        # to enable the iterations stats
        self._current_iter = -1
        while len(self.col) < self.pop_size:
            self.col.append(self._generate_ant())
        self._do_iteration_stats()
        self._num_evals += self._current_iter_evals
        self._current_iter += 1
        self._pop = self._col
        self._col = current_colony

        # Update the pheromone matrix
        self._update_pheromone()

    def _update_pop(self) -> None:
        """Update the population."""
        # Update the population according to the NSGA-II selection
        # procedure
        self.pop[:] = selNSGA2(self.pop[:] + self.col, self.pop_size)

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        The pheromone trails are updated according to the current population.
        """
        # Sort the population into nondomination levels
        pareto_fronts = sortNondominated(self.pop, self.pop_size)

        # Init the pheromone matrix
        self._init_pheromone()

        # # Each ant increments pheromone according to its front
        for front_index, front in enumerate(pareto_fronts):
            self._deposit_pheromone(front, self.pop_size / (front_index + 1))


class ElitistACO_FS(
    ElitistACO,
    ACO_FS
):
    """Implement an elitist ACO for FS algorithm."""

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
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [ElitistACO_FS],
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
        r"""Create a new elitist ACO-FS trainer.

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
            :py:attr:`~culebra.trainer.aco.ElitistACO_FS.num_pheromone_matrices`
            pheromone matrices. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.aco.DEFAULT_ACO_FS_INITIAL_PHEROMONE`
            will be used. Defaults to :py:data:`None`
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.ElitistACO_FS.num_heuristic_matrices`
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
            :py:attr:`~culebra.trainer.aco.ElitistACO_FS.num_pheromone_matrices`
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
            :py:attr:`~culebra.trainer.aco.ElitistACO_FS.num_heuristic_matrices`
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
        # Init the superclasses
        ACO_FS.__init__(
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
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            col_size=col_size,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to update the pheromone matrix according to the elite.

        :param state: The last loaded state
        :type state: :py:class:`dict`
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

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        The pheromone trails are updated according to the current elite.
        """
        # Init the pheromone matrices
        self._init_pheromone()

        # Update the pheromone matrices with the current elite
        self._deposit_pheromone(self._elite)


__all__ = [
    'PACO_FS',
    'ElitistACO_FS',
]
