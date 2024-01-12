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
    Any,
    Type,
    Dict,
    Callable,
    Optional,
    Sequence
)
from random import randrange

import numpy as np

from deap.tools import HallOfFame

from culebra.abc import Species, FitnessFunction
from culebra.solution.abc import Ant

from culebra.trainer.aco.abc import (
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO,
    ElitistACO,
    PACO
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class PACO_MO(
    PACO,
    ElitistACO,
    MultiplePheromoneMatricesACO,
    MultipleHeuristicMatricesACO
):
    """Implement the PACO-MO algorithm."""

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
                [PACO_MO],
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
            :py:attr:`~culebra.trainer.aco.PACO_MO.num_pheromone_matrices`
            pheromone matrices.
        :type initial_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param max_pheromone: Maximum amount of pheromone for the paths
            of each pheromone matrix. Both a scalar value or a sequence of
            values are allowed. If a scalar value is provided, it will be used
            for all the
            :py:attr:`~culebra.trainer.aco.PACO_MO.num_pheromone_matrices`
            pheromone matrices.
        :type max_pheromone: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :param heuristic: Heuristic matrices. Both a single matrix or a
            sequence of matrices are allowed. If a single matrix is provided,
            it will be replicated for all the
            :py:attr:`~culebra.trainer.aco.PACO_MO.num_heuristic_matrices`
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
            :py:attr:`~culebra.trainer.aco.PACO_MO.num_pheromone_matrices`
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
            :py:attr:`~culebra.trainer.aco.PACO_MO.num_heuristic_matrices`
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
        MultiplePheromoneMatricesACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone
        )
        MultipleHeuristicMatricesACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone
        )
        ElitistACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone
        )
        PACO.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function,
            initial_pheromone=initial_pheromone,
            max_pheromone=max_pheromone,
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

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Overridden to ignore the population, which is now an internal
        structure since the elite solutions are kept within the state.

        :type: :py:class:`dict`
        """
        return super(PACO, self)._get_state()

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to ignore the population, which is now an internal
        structure since the elite solutions are kept within the state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        super(PACO, self)._set_state(state)

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to ignore the population, which is now an internal
        structure since the elite solutions are kept within the state.
        """
        super(PACO, self)._new_state()

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to ignore the population, which is now an internal
        structure since the elite solutions are kept within the state.
        """
        super(PACO, self)._reset_state()

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        Return the best single solution found for each species

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        return super(PACO, self).best_solutions()

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :py:class:`~culebra.trainer.aco.PACO_MO` class, the population
        is an internal structure, since is generated each iteration.
        """
        super()._init_internals()
        self._pop = []

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the population.
        """
        super()._reset_internals()
        self._pop = None

    def _calculate_choice_info(self) -> None:
        """Calculate the choice info matrix."""
        # Number of objectives
        num_obj = self.fitness_function.num_obj

        # Current number of ants in the population
        pop_size = len(self.pop)

        # Currenbt number of elite ants
        elite_size = len(self._elite)

        # Default weight for each objective
        obj_weight = np.ones(num_obj)

        # If the elite and the population are not empty
        if elite_size > 0 and pop_size > 0:
            # Get population's ants rank for each objective
            ant_rank = np.zeros((num_obj, pop_size))

            # For each objective ...
            for obj_idx in range(num_obj):
                # Rank the elite according to this objective,
                # from worst to best
                ranked_elite_by_obj = sorted(
                    self._elite,
                    key=lambda ant: ant.fitness.wvalues[obj_idx]
                )

                # For each ant in the population ...
                for ant_idx, ant in enumerate(self.pop):
                    ant_rank[obj_idx][ant_idx] = ranked_elite_by_obj.index(ant)

            # Obtain the population's ants weight for each objective
            ant_weight = np.zeros((num_obj, pop_size))

            # Sum of ranks of each ant for all the objectives
            rank_sum = ant_rank.sum(axis=0)

            # For each objective ...
            for obj_idx in range(num_obj):
                # For each ant in the population ...
                for ant_idx, ant in enumerate(self.pop):
                    ant_weight[obj_idx][ant_idx] = (
                        ant_rank[obj_idx][ant_idx] / rank_sum[ant_idx]
                    )

            # Obtain the average weight for each objective
            obj_weight = ant_weight.mean(axis=1)

        # Calculate the choice probabilites
        self._choice_info = np.zeros(self.heuristic[0].shape)
        for (
            weight,
            pheromone,
            heuristic,
            pheromone_influence,
            heuristic_influence
        ) in zip(
            obj_weight,
            self.pheromone,
            self.heuristic,
            self.pheromone_influence,
            self.heuristic_influence
        ):
            self._choice_info += (
                np.power(pheromone, pheromone_influence) *
                np.power(heuristic, heuristic_influence) *
                weight
            )

    def _update_pop(self) -> None:
        """Generate a new population for the current iteration.

        The new population is generated from the elite super-population.
        """

        def obj_dist(ant1: Ant, ant2: Ant) -> float:
            """Return the distance between two ants in the objective space.

            :param ant1: The first ant
            :type ant1: :py:class:`~culebra.solution.abc.Ant`
            :param ant2: The second ant
            :type ant2: :py:class:`~culebra.solution.abc.Ant`
            :return: The distance in the objective space
            :rtype: :py:class:`float`
            """
            fit1 = np.asarray(ant1.fitness.values)
            fit2 = np.asarray(ant2.fitness.values)
            diffs = np.abs(fit1 - fit2)
            return np.sum(diffs)

        # Number of elite ants
        elite_size = len(self._elite)

        # A new population is generated each iteration
        self._pop = []

        # If the number of elite ants is too small...
        if elite_size < self.pop_size:
            # Use all the elite ants
            for ant in self._elite:
                self._pop.append(ant)
        else:
            # Candidate ants for the new population
            candidate_ants = []
            for ant in self._elite:
                candidate_ants.append(ant)

            # Remaining room in the population
            remaining_room_in_pop = self.pop_size

            # Select one elite ant randomly to generate the new population
            pop_generator_ant_index = randrange(elite_size)
            pop_generator_ant = candidate_ants[pop_generator_ant_index]

            # Append it to the new population
            self._pop.append(pop_generator_ant)
            del candidate_ants[pop_generator_ant_index]
            remaining_room_in_pop -= 1

            # While the pop is not complete
            while remaining_room_in_pop > 0:
                # Look for the nearest ant
                nearest_ant_index = None
                nearest_ant_dist = None
                for index, ant in enumerate(candidate_ants):
                    # Init the nearest ant distance and index
                    if nearest_ant_index is None:
                        nearest_ant_index = index
                        nearest_ant_dist = obj_dist(ant, pop_generator_ant)
                    # Update the nearest ant distance and index
                    else:
                        ant_dist = obj_dist(ant, pop_generator_ant)
                        if ant_dist < nearest_ant_dist:
                            nearest_ant_index = index
                            nearest_ant_dist = ant_dist

                # Append the nearest ant to pop_generator_ant
                self._pop.append(candidate_ants[nearest_ant_index])
                del candidate_ants[nearest_ant_index]
                remaining_room_in_pop -= 1

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        The pheromone trails are updated according to the current population.
        """
        # Init the pheromone matrices
        shape = self._heuristic[0].shape
        self._pheromone = [
            np.full(
                shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromone
        ]

        # Update the pheromone matrices with the current population
        self._deposit_pheromone(self.pop)

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths from the current
        # pheromone matrix
        self._generate_col()

        # Update the elite
        self._update_elite()

        # Create a new population from the elite
        # and also the pheromone matrices
        self._update_pop()

        # Generate the pheromone matrix according to the new population
        self._update_pheromone()


# Exported symbols for this module
__all__ = [
    'PACO_MO'
]
