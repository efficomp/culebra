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
from copy import copy

import numpy as np

from deap.tools import HallOfFame, sortNondominated

from culebra.abc import Species, FitnessFunction
from culebra.solution.abc import Ant

from culebra.trainer.aco.abc import (
    SingleColACO,
    SinglePheromoneMatrixACO,
    MultipleHeuristicMatricesACO,
    ElitistACO,
    PACO,
    MaxPheromonePACO
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2024, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class PACO_MO(
    MaxPheromonePACO,
    ElitistACO,
    SinglePheromoneMatrixACO,
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
        SinglePheromoneMatrixACO.__init__(
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

        # Current number of elite ants
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
            heuristic,
            heuristic_influence
        ) in zip(
            obj_weight,
            self.heuristic,
            self.heuristic_influence
        ):
            self._choice_info += (
                np.power(self.pheromone[0], self.pheromone_influence[0]) *
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


class CPACO(
    PACO,
    SinglePheromoneMatrixACO,
    MultipleHeuristicMatricesACO
):
    """Implement the Crowding PACO algorithm."""

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

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :py:class:`~culebra.trainer.aco.CPACO` class, the heuristic influence
        correction factors are created. Subclasses which need more objects or
        data structures should override this method.
        """
        super()._init_internals()
        self._heuristic_influence_correction = np.ones(len(self.heuristic))

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the pheromone matrices. If subclasses overwrite
        the :py:meth:`~culebra.trainer.aco.CPACO._init_internals` method to
        add any new internal object, this method should also be overridden to
        reset all the internal objects of the trainer.
        """
        super()._reset_internals()
        self._heuristic_influence_correction = None

    def _calculate_choice_info(self) -> None:
        """Calculate the choice info matrix.

        The choice info matrix is re-calculated every time an ant is generated
        since in CPACO each ant uses its own heuristic influence correction
        factors.
        """
        # Heuristic info for the current ant
        heuristic_info = np.ones(self.heuristic[0].shape)
        for (
            heuristic,
            heuristic_influence,
            heuristic_influence_correction
        ) in zip(
            self.heuristic,
            self.heuristic_influence,
            self._heuristic_influence_correction

        ):
            heuristic_info *= np.power(
                heuristic,
                heuristic_influence * heuristic_influence_correction
            )

        self._choice_info = (
            np.power(self.pheromone[0], self.pheromone_influence[0]) *
            heuristic_info
        )

    def _generate_ant(self) -> Ant:
        """Generate a new ant.

        A new set of heuristic influence correction factors is generated for
        the ant. Then, the choice info for the ant is generated according to
        these correction factors. Finally, the ant makes its path and gets
        evaluated.
        """
        # Set the heuristic influence correction factors for this ant
        temp = np.random.random(len(self.heuristic))
        self._heuristic_influence_correction = temp / temp.sum()
        # Calculate the choice info for this ant
        self._calculate_choice_info()
        # generate the ant
        return super()._generate_ant()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the iteration metrics (number of evaluations, execution time)
        before each iteration is run and create an empty ant colony.
        Overridden to avoid the calculation of the choice information before
        executing the next iteration because CPACO applies a different choice
        probability for each ant.
        """
        super(SingleColACO, self)._start_iteration()
        self._col = []

    def _update_pop(self) -> None:
        """Update the population."""

        def num_common_arcs(ant1: Ant, ant2: Ant) -> int:
            """Return the number of common arcs between two ants.

            A symmetric problem is assumed. Thus, (*i*, *j*) and (*j*, *i*)
            are considered the same arc.

            :param ant1:  The first ant
            :type ant1: :py:class:`culebra.solution.abc.Ant`
            :param ant2:  The second ant
            :type ant2: :py:class:`culebra.solution.abc.Ant`
            """
            # Number of detected common arcs
            detected_common_arcs = 0

            # For each node in ant1's path
            for node1_index, node1 in enumerate(ant1.path):
                # For each node in ant2's path
                for node2_index, node2 in enumerate(ant2.path):
                    # If nodes match, try to detect a common arc
                    if node1 == node2:
                        # ant1's next node
                        ant1_next_node = ant1.path[
                            (node1_index + 1) % len(ant1.path)
                        ]
                        # ant2's next node
                        ant2_next_node = ant2.path[
                            (node2_index + 1) % len(ant2.path)
                        ]
                        # ant2's previous node
                        ant2_prev_node = ant2.path[node2_index - 1]
                        # If there is an arc match
                        if (
                            ant1_next_node == ant2_next_node or
                            ant1_next_node == ant2_prev_node
                        ):
                            # Increment the number of detected common arcs
                            detected_common_arcs += 1

            # Return the number of detected common arcs
            return detected_common_arcs

        # For each ant in the current colony
        for col_ant in self.col:

            # Closest ant in the population
            closest_ant_index = randrange(self.pop_size)
            closest_ant = self.pop[closest_ant_index]
            # Number of common arcs between ant and
            # its closest ant in the population
            closest_ant_num_common_arcs = num_common_arcs(
                col_ant,
                closest_ant
            )

            # For all the ants in the population
            for pop_ant_index, pop_ant in enumerate(self.pop):
                # If the population ant is not the closest one
                if pop_ant_index != closest_ant_index:
                    # Get the number of common arcs
                    pop_ant_num_common_arcs = num_common_arcs(
                        col_ant,
                        pop_ant
                    )

                    # If pop_ant is closer than closest_ant
                    if (pop_ant_num_common_arcs > closest_ant_num_common_arcs):
                        # Select pop_ant as the closest ant in the population
                        closest_ant_index = pop_ant_index
                        closest_ant = pop_ant
                        closest_ant_num_common_arcs = pop_ant_num_common_arcs

            # If the colony ant is better than its closest ant, replace it
            if col_ant > closest_ant:
                self.pop[closest_ant_index] = col_ant

    def _deposit_pheromone(
        self, ants: Sequence[Ant], amount: Optional[float] = 1
    ) -> None:
        """Make some ants deposit pheromone.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.solution.abc.Ant`
        :param amount: Amount of pheromone. Defaults to 1
        :type amount: :py:class:`float`, optional
        """
        for ant in ants:
            for pher in self.pheromone:
                org = ant.path[-1]
                for dest in ant.path:
                    pher[org][dest] += amount
                    pher[dest][org] += amount
                    org = dest

    def _update_pheromone(self) -> None:
        """Update the pheromone trails.

        The pheromone trails are updated according to the current population.
        """
        # Sort the population into nondomination levels
        pareto_fronts = sortNondominated(self.pop, self.pop_size)

        # Init the pheromone matrix
        self._init_pheromone()

        # Each ant increments pheromone according to its front
        for front_index, front in enumerate(pareto_fronts):
            self._deposit_pheromone(front, 1 / (front_index + 1))


__all__ = [
    'PACO_MO',
    'CPACO'
]
