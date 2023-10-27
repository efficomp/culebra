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

"""Abstract base classes for different ACO-based trainers.

This module provides several abstract classes for different kind of
Ant Colony Optimization based trainers.

By the moment:

  * :py:class:`~culebra.trainer.aco.abc.SingleColACO`: A base class for all
    the single colony ACO-based trainers
  * :py:class:`~culebra.trainer.aco.abc.ElitistACO`: A base class for all
    the elitist single colony ACO-based trainers

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

from .constants import DEFAULT_CONVERGENCE_CHECK_FREQ


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
        initial_pheromones: Sequence[float, ...],
        heuristics: Optional[
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
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
        """Create a new single-colony ACO trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromones: Initial amount of pheromone for the paths
            of each pheromones matrix
        :type initial_pheromones: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :param heuristics: Heuristics matrices. If omitted, the default
            heuristics provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristics: :py:class:`~collections.abc.Sequence` of
            two-dimensional array-like objects, optional
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
        self.col_size = col_size
        self.initial_pheromones = initial_pheromones
        self.heuristics = heuristics

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

        # Reset the algorithm
        self.reset()

    @property
    def initial_pheromones(self) -> Sequence[float, ...]:
        """Get and set the initial value for each pheromones matrix.

        :getter: Return the initial value for each pheromones matrix.
        :setter: Set a new initial value for each pheromones matrix.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If the sequence is empty
        """
        return self._initial_pheromones

    @initial_pheromones.setter
    def initial_pheromones(self, values: Sequence[float, ...]) -> None:
        """Set the initial value for each pheromones matrix.

        :param values: New initial value for each pheromones matrix.
        :type values: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If any element in *values* is not a float value
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If *values* is empty
        """
        # Check the values
        self._initial_pheromones = check_sequence(
            values,
            "initial pheromones",
            item_checker=partial(check_float, gt=0)
        )

        # Check the minimum size
        if len(self._initial_pheromones) == 0:
            raise ValueError("The initial pheromones sequence is empty")

        # Reset the algorithm
        self.reset()

    @property
    def heuristics(self) -> Sequence[np.ndarray, ...]:
        """Get and set the heuristic matrices.

        :getter: Return the heuristics matrices
        :setter: Set new heuristics matrices. If set to :py:data:`None`, the
            default heuristics (provided by the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.fitness_function`
            property) are assumed.
        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        :raises TypeError: If any value is not an array
        :raises ValueError: If the matrix has a wrong shape or contain any
            negative value.
        """
        return self._heuristics

    @heuristics.setter
    def heuristics(
        self,
        values: Sequence[Sequence[Sequence[float], ...], ...] | None
    ) -> None:
        """Set new heuristic matrices.

        :param values: New heuristics matrices. If set to :py:data:`None`, the
            default heuristics (provided by the
            :py:attr:`~culebra.trainer.aco.abc.SingleColACO.fitness_function`
            property) are assumed.
        :type: :py:class:`~collections.abc.Sequence` of two-dimensional
            array-like objects
        :raises TypeError: If *values* is not a
            :py:class:`~collections.abc.Sequence`

        :raises TypeError: If any element in *values* is not an array-like
            object
        :raises ValueError: If any element in *values* has any not float
            element
        :raises ValueError: If any element in *values* has not an homogeneous
            shape
        :raises ValueError: If any element in *values* has not two dimensions
        :raises ValueError: If any element in *values* has negative values
        """
        if values is None:
            self._heuristics = list(
                self.fitness_function.heuristics(self.species)
            )
        else:
            # Check the values
            self._heuristics = check_sequence(
                values,
                "heuristics matrices",
                item_checker=partial(check_matrix, square=True, ge=0)
            )

            # Check the minimum size
            if len(self._heuristics) == 0:
                raise ValueError("The heuristics matrices sequence is empty")

            # Check the shape
            the_shape = self._heuristics[0].shape
            if the_shape[0] == 0:
                raise ValueError("The heuristics matrices can not be empty")

            # Check that all the matrices have the same shape
            for matrix in self._heuristics:
                if matrix.shape != the_shape:
                    raise ValueError(
                        "All the heuristics matrices must have the same shape"
                    )

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

        # Reset the algorithm
        self.reset()

    @property
    def col(self) -> List[Ant] | None:
        """Get the colony.

        :type: :py:class:`list` of :py:class:`~culebra.abc.Solution`
        """
        return self._col

    @property
    def pheromones(self) -> Sequence[np.ndarray, ...] | None:
        """Get the pheromones matrices.

        If the search process has not begun, :py:data:`None` is returned.

        :type: :py:class:`~collections.abc.Sequence`
            of :py:class:`~numpy.ndarray`
        """
        return self._pheromones

    @property
    def choice_info(self) -> np.ndarray | None:
        """Get the choice information for all the graph's arcs.

        The choice information is generated from both the pheromones and
        the heuristics, modified by other parameters (depending on the ACO
        approach) and is used to obtain the probalility of following the next
        feasible arc for the node.

        If the search process has not begun, :py:data:`None` is returned.

        :type: :py:class:`~numpy.ndarray`
        """
        return self._choice_info

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this trainer.

        Overridden to add the current pheromones matrices.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = SingleSpeciesTrainer._state.fget(self)

        # Get the state of this class
        state["pheromones"] = self._pheromones

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current pheromones matrices to the trainer's
        state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        SingleSpeciesTrainer._state.fset(self, state)

        # Set the state of this class
        self._pheromones = state["pheromones"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to generate the initial pheromones matrices.
        """
        super()._new_state()
        heuristics_shape = self._heuristics[0].shape
        self._pheromones = [
            np.full(
                heuristics_shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromones
        ]

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the pheromones matrices.
        """
        super()._reset_state()
        self._pheromones = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the
        :py:class:`~culebra.aco.abc.SingleColACO` class, the colony, the
        choice_info matrix and the node list are a created. Subclasses which
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

        If subclasses overwrite the :py:class:`~culebra.aco.abc.SingleColACO`
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

        The choice information is generated from both the pheromones and
        the heuristics, modified by other parameters (depending on the ACO
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

    @abstractmethod
    def _evaporate_pheromones(self) -> None:
        """Evaporate pheromones.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The _evaporate_pheromones method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _deposit_pheromones(self) -> None:
        """Deposit pheromones.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError(
            "The _deposit_pheromones method has not been implemented in the "
            f"{self.__class__.__name__} class")

    def _update_pheromones(self) -> None:
        """Update the pheromone trails.

        First, the pheromones are evaporated. Then ants deposit pheromones
        according to their fitness.
        """
        self._evaporate_pheromones()
        self._deposit_pheromones()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the pheromones
        self._update_pheromones()

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


class ElitistACO(SingleColACO):
    """Base class for all the elitist single colony ACO algorithms."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromones: Sequence[float, ...],
        heuristics: Optional[
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
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
        """Create a new single-colony ACO trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromones: Initial amount of pheromone for the paths
            of each pheromones matrix
        :type initial_pheromones: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :param heuristics: Heuristics matrices. If omitted, the default
            heuristics provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristics: :py:class:`~collections.abc.Sequence` of
            two-dimensional array-like objects, optional
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

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this trainer.

        Overridden to add the current elite and the last iteration number
        when the elite was updated to the trainer's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = SingleColACO._state.fget(self)

        # Get the state of this class
        state["elite"] = self._elite

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current elite to the trainer's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        SingleColACO._state.fset(self, state)

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

    def _deposit_ant_weighted_pheromones(
        self, ant: Ant, weight: float
    ) -> None:
        """Make an ant deposit weighted pheromones.

        :param ant: The ant
        :type ant: :py:class:`~culebra.solution.abc.Ant`
        :param weight: The weight
        :type ant: :py:class:`float`
        """
        pheromones_amount = ant.fitness.pheromones_amount[0] * weight
        org = ant.path[-1]
        for dest in ant.path:
            self._pheromones[0][org][dest] += pheromones_amount
            self._pheromones[0][dest][org] += pheromones_amount
            org = dest

    def _update_elite(self) -> None:
        """Update the elite (best-so-far) ant."""
        self._elite.update(self.col)

    def _has_converged(self) -> None:
        """Detect if the trainer has converged.

        :return: :py:data:`True` if the trainer has converged
        :rtype: :py:class:`bool`
        """
        convergence = True

        for pher in self.pheromones:
            for row in range(len(pher)):
                min_pher_count = np.isclose(pher[row], 0).sum()

                if (
                    min_pher_count != self.species.num_nodes and
                    min_pher_count != self.species.num_nodes - 2
                ):
                    convergence = False
                    break

        return convergence

    def _reset_pheromones(self) -> None:
        """Reset the pheromones matrices."""
        heuristics_shape = self._heuristics[0].shape
        self._pheromones = [
            np.full(
                heuristics_shape,
                initial_pheromone,
                dtype=float
            ) for initial_pheromone in self.initial_pheromones
        ]

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process."""
        # Create the ant colony and the ants' paths
        self._generate_col()

        # Update the elite
        self._update_elite()

        # Update the pheromones
        self._update_pheromones()

        # Reset pheromones if convergence is reached
        if (
            self._current_iter % self.convergence_check_freq == 0 and
            self._has_converged()
        ):
            self._reset_pheromones()


class PACO(SingleColACO):
    """Base class for all the population-based single colony ACO algorithms."""

    def __init__(
        self,
        solution_cls: Type[Ant],
        species: Species,
        fitness_function: FitnessFunction,
        initial_pheromones: Sequence[float, ...],
        max_pheromones: Sequence[float, ...],
        heuristics: Optional[
            Sequence[Sequence[Sequence[float], ...], ...]
        ] = None,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SingleColACO],
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
        """Create a new single-colony ACO trainer.

        :param solution_cls: The ant class
        :type solution_cls: An :py:class:`~culebra.solution.abc.Ant`
            subclass
        :param species: The species for all the ants
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param initial_pheromones: Initial amount of pheromone for the paths
            of each pheromones matrix
        :type initial_pheromones: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :param max_pheromones: Maximum amount of pheromone for the paths
            of each pheromones matrix
        :type max_pheromones: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :param heuristics: Heuristics matrices. If omitted, the default
            heuristics provided by *fitness_function* are assumed. Defaults to
            :py:data:`None`
        :type heuristics: :py:class:`~collections.abc.Sequence` of
            two-dimensional array-like objects, optional
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

        # Get the parameters
        self.max_pheromones = max_pheromones
        self.pop_size = pop_size

    @property
    def max_pheromones(self) -> Sequence[float, ...]:
        """Get and set the maximum value for each pheromones matrix.

        :getter: Return the maximum value for each pheromones matrix.
        :setter: Set a new maximum value for each pheromones matrix.
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If any value is not a float
        :raises ValueError: If any value is negative or zero
        :raises ValueError: If any value is lower than or equal to its
            corresponding initial pheromone value
        :raises ValueError: If the sequence is empty
        """
        return self._max_pheromones

    @max_pheromones.setter
    def max_pheromones(self, values: Sequence[float, ...]) -> None:
        """Set the initial value for each pheromones matrix.

        :param values: New initial value for each pheromones matrix.
        :type values: :py:class:`~collections.abc.Sequence` of
            :py:class:`float`
        :raises TypeError: If any element in *values* is not a float value
        :raises ValueError: If any element in *values* is negative or zero
        :raises ValueError: If any element in *values* is lower than or
            equal to its corresponding initial pheromone value
        :raises ValueError: If *values* is empty
        """
        # Check the values
        self._max_pheromones = check_sequence(
            values,
            "maximum pheromones",
            item_checker=partial(check_float, gt=0)
        )

        # Check the minimum size
        if len(self._max_pheromones) == 0:
            raise ValueError("The maximum pheromones sequence is empty")

        # Check that each max value is not lower than its corresponding
        # initial pheromone value
        for (
            val, max_val
        ) in zip(
            self._initial_pheromones, self._max_pheromones
        ):
            if val >= max_val:
                raise ValueError(
                    "Each maximum pheromones value must be higher than "
                    "its corresponding initial pheromones value"
                )

        # Reset the algorithm
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

        # Reset the algorithm
        self.reset()

    @property
    def pop(self) -> List[Ant] | None:
        """Get the population.

        :type: :py:class:`list` of :py:class:`~culebra.abc.Solution`
        """
        return self._pop

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = SingleColACO._state.fget(self)

        # Get the state of this class
        state["pop"] = self._pop

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the current population to the trainer's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        SingleColACO._state.fset(self, state)

        # Set the state of this class
        self._pop = state["pop"]

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to initialize the elite.
        """
        super()._new_state()

        # Create an empty population
        self._pop = []

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overridden to reset the elite.
        """
        super()._reset_state()
        self._pop = None

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


# Exported symbols for this module
__all__ = [
    'SingleColACO',
    'ElitistACO',
    'PACO'
]
