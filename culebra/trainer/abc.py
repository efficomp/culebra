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

"""Abstract base classes for different trainers.

This module provides several abstract classes for different kind of trainers.

Regarding the number of species that are simultaneously being trained:

  * :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`: Provides a base
    class for trainers for solutions of only a single species
  * :py:class:`~culebra.trainer.abc.MultiSpeciesTrainer`: Provides a base
    class for trainers that find solutions for multiple species

With respect to the number of populations being trained:

  * :py:class:`~culebra.trainer.abc.SinglePopTrainer`: A base class for all
    the single population trainers
  * :py:class:`~culebra.trainer.abc.MultiPopTrainer`: A base class for all
    the multiple population trainers

Regarding :py:class:`~culebra.trainer.abc.MultiPopTrainer` implementations,
they can be executed sequentially or in parallel, with the aid of the following
classes:

  * :py:class:`~culebra.trainer.abc.SequentialMultiPopTrainer`: Abstract base
    class for sequential multi-population trainers
  * :py:class:`~culebra.trainer.abc.ParallelMultiPopTrainer`: Abstract base
    class for parallel multi-population trainers

Finally, different multi-population approaches are also provided:

  * :py:class:`~culebra.trainer.abc.IslandsTrainer`: Abstract base class for
    island-based approaches
  * :py:class:`~culebra.trainer.abc.CooperativeTrainer`: Abstract base class
    for cooperative trainers
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    Type,
    Optional,
    Callable,
    List,
    Dict,
    Generator,
    Sequence)
from copy import deepcopy
from functools import partial
from itertools import repeat
from multiprocessing import (
    cpu_count,
    Queue,
    Process,
    Manager)
from os import path

from deap.tools import Logbook, HallOfFame, ParetoFront

from culebra import DEFAULT_POP_SIZE
from culebra.abc import (
    Solution,
    Species,
    Trainer,
    FitnessFunction
)
from culebra.checker import (
    check_instance,
    check_int,
    check_subclass,
    check_func,
    check_func_params,
    check_sequence
)

from .constants import (
    DEFAULT_NUM_SUBPOPS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
    DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class SingleSpeciesTrainer(Trainer):
    """Base class for trainers for solutions of only a single species."""

    def __init__(
        self,
        solution_cls: Type[Solution],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[[Trainer], bool]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The solution class
        :type solution_cls: A :py:class:`~culebra.abc.Solution` subclass
        :param species: The species for all the solutions
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`,
            :py:meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
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
        Trainer.__init__(
            self,
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
        self.solution_cls = solution_cls
        self.species = species

    @property
    def solution_cls(self) -> Type[Solution]:
        """Get and set the solution class.

        :getter: Return the solution class
        :setter: Set a new solution class
        :type: A :py:class:`~culebra.abc.Solution` subclass
        :raises TypeError: If set to a value which is not a
            :py:class:`~culebra.abc.Solution` subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: Type[Solution]) -> None:
        """Set a new solution class.

        :param cls: The new class
        :type cls: A :py:class:`~culebra.abc.Solution` subclass
        :raises TypeError: If *cls* is not a
        :py:class:`~culebra.abc.Solution`
        """
        # Check cls
        self._solution_cls = check_subclass(
            cls, "solution class", Solution
        )

        # Reset the algorithm
        self.reset()

    @property
    def species(self) -> Species:
        """Get and set the species.

        :getter: Return the species
        :setter: Set a new species
        :type: :py:class:`~culebra.abc.Species`
        :raises TypeError: If set to a value which is not a
            :py:class:`~culebra.abc.Species` instance
        """
        return self._species

    @species.setter
    def species(self, value: Species) -> None:
        """Set a new species.

        :param value: The new species
        :type value: :py:class:`~culebra.abc.Species`
        :raises TypeError: If *value* is not a :py:class:`~culebra.abc.Species`
            instance
        """
        # Check the value
        self._species = check_instance(value, "species", Species)

        # Reset the algorithm
        self.reset()

    def __copy__(self) -> SingleSpeciesTrainer:
        """Shallow copy the trainer."""
        cls = self.__class__
        result = cls(
            self.solution_cls, self.species, self.fitness_function
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> SingleSpeciesTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
        """
        cls = self.__class__
        result = cls(
            self.solution_cls, self.species, self.fitness_function
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (self.solution_cls, self.species, self.fitness_function),
                self.__dict__)


class SinglePopTrainer(SingleSpeciesTrainer):
    """Base class for all the single population trainers."""

    def __init__(
        self,
        solution_cls: Type[Solution],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopTrainer],
                bool]
        ] = None,
        pop_size: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The solution class
        :type solution_cls: A :py:class:`~culebra.abc.Solution` subclass
        :param species: The species for all the solutions
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`,
            :py:meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_size: The populaion size. If set to :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_POP_SIZE`
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
        # Init the superclass
        super().__init__(
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
        self.pop_size = pop_size

    @property
    def pop_size(self) -> int:
        """Get and set the population size.

        :getter: Return the current population size
        :setter: Set a new value for the population size. If set to
            :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an :py:class:`int`
        :raises ValueError: If set to a value which is not greater than zero
        """
        return DEFAULT_POP_SIZE if self._pop_size is None else self._pop_size

    @pop_size.setter
    def pop_size(self, size: int | None) -> None:
        """Set the population size.

        :param size: The new population size. If set to :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_POP_SIZE` is chosen
        :type size: :py:class:`int`, greater than zero. If set to
            :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_POP_SIZE` is used
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
    def pop(self) -> List[Solution] | None:
        """Get the population.

        :type: :py:class:`list` of :py:class:`~culebra.abc.Solution`
        """
        return self._pop

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats."""
        # Perform some stats
        record = self._stats.compile(self.pop) if self._stats else {}
        self._logbook.record(
            Iter=self._current_iter,
            Pop=self.index,
            NEvals=self._current_iter_evals,
            **record)
        if self.verbose:
            print(self._logbook.stream)

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this trainer.

        Overriden to add the current population to the trainer's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = Trainer._state.fget(self)

        # Get the state of this class
        state["pop"] = self.pop

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overriden to add the current population to the trainer's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        Trainer._state.fset(self, state)

        # Set the state of this class
        self._pop = state["pop"]

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Overriden to reset the initial population.
        """
        super()._reset_state()
        self._pop = None

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overriden to generate an empty population.
        """
        super()._new_state()

        self._pop = []

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


class MultiPopTrainer(Trainer):
    """Base class for all the multiple population trainers."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        subpop_trainer_cls: Type[SinglePopTrainer],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopTrainer],
                bool]
        ] = None,
        num_subpops: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Solution], Any], Solution]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subpop_trainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subpop_trainer_cls: Single-population trainer class to handle
            the subpopulations.
        :type subpop_trainer_cls: Any subclass of
            :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`,
            :py:meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param num_subpops: The number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBPOPS` will be
            used. Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative solutions that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
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
        :param subpop_trainer_params: Custom parameters for the subpopulations
            trainer
        :type subpop_trainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__(
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
        self.subpop_trainer_cls = subpop_trainer_cls
        self.num_subpops = num_subpops
        self.representation_size = representation_size
        self.representation_freq = representation_freq
        self.representation_topology_func = representation_topology_func
        self.representation_topology_func_params = (
            representation_topology_func_params
        )
        self.representation_selection_func = representation_selection_func
        self.representation_selection_func_params = (
            representation_selection_func_params
        )
        self.subpop_trainer_params = subpop_trainer_params

    @property
    def subpop_trainer_cls(self) -> Type[SinglePopTrainer]:
        """Get and set the trainer class to handle the subpopulations.

        Each subpopulation will be handled by a single-population trainer.

        :getter: Return the trainer class
        :setter: Set new trainer class
        :type: A :py:class:`~culebra.trainer.abc.SinglePopTrainer` subclass
        :raises TypeError: If set to a value which is not a
            :py:class:`~culebra.trainer.abc.SinglePopTrainer` subclass
        """
        return self._subpop_trainer_cls

    @subpop_trainer_cls.setter
    def subpop_trainer_cls(self, cls: Type[SinglePopTrainer]) -> None:
        """Set a new trainer class to handle the subpopulations.

        Each subpopulation will be handled by a single-population trainer.

        :param cls: The new class
        :type cls: A :py:class:`~culebra.trainer.abc.SinglePopTrainer` subclass
        :raises TypeError: If *cls* is not a
            :py:class:`~culebra.trainer.abc.SinglePopTrainer` subclass
        """
        # Check cls
        self._subpop_trainer_cls = check_subclass(
            cls, "trainer class for subpopulations", SinglePopTrainer
        )

        # Reset the algorithm
        self.reset()

    @property
    def num_subpops(self) -> int:
        """Get and set the number of subpopulations.

        :getter: Return the current number of subpopulations
        :setter: Set a new value for the number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBPOPS` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            DEFAULT_NUM_SUBPOPS if self._num_subpops is None
            else self._num_subpops
        )

    @num_subpops.setter
    def num_subpops(self, value: int | None) -> None:
        """Set the number of subpopulations.

        :param value: The new number of subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBPOPS` is chosen
        :type value: An integer value
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._num_subpops = (
            None if value is None else check_int(
                value, "number of subpopulations", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_size(self) -> int:
        """Get and set the representation size.

        The representation size is the number of representatives sent to the
        other subpopulations

        :getter: Return the current representation size
        :setter: Set the new representation size. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` is
            chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not positive
        """
        return (
            DEFAULT_REPRESENTATION_SIZE if self._representation_size is None
            else self._representation_size
        )

    @representation_size.setter
    def representation_size(self, size: int | None) -> None:
        """Set a new representation size.

        :param size: The new size. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` is
            chosen
        :type size: :py:class:`int`
        :raises TypeError: If *size* is not an integer number
        :raises ValueError: If *size* is not positive
        """
        # Check size
        self._representation_size = (
            None if size is None else check_int(
                size, "representation size", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_freq(self) -> int:
        """Get and set the number of iterations between representatives sendings.

        :getter: Return the current frequency
        :setter: Set a new value for the frequency. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` is
            chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            DEFAULT_REPRESENTATION_FREQ if self._representation_freq is None
            else self._representation_freq
        )

    @representation_freq.setter
    def representation_freq(self, value: int | None) -> None:
        """Set the number of iterations between representatives sendings.

        :param value: The new frequency. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` is
            chosen
        :type value: An integer value
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._representation_freq = (
            None if value is None else check_int(
                value, "representation frequency", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get and set the representation topology function.

        :getter: Return the representation topology function
        :setter: Set new representation topology function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
            if self._representation_topology_func is None
            else self._representation_topology_func
        )

    @representation_topology_func.setter
    def representation_topology_func(
        self,
        func: Callable[[int, int, Any], List[int]] | None
    ) -> None:
        """Set new representation topology function.

        :param func: The new function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not a callable
        """
        # Check func
        self._representation_topology_func = (
            None if func is None else check_func(
                func, "representation topology function"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the representation topology function.

        :getter: Return the current parameters for the representation topology
            function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return (
            DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            if self._representation_topology_func_params is None
            else self._representation_topology_func_params
        )

    @representation_topology_func_params.setter
    def representation_topology_func_params(
        self, params: Dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation topology function.

        :param params: The new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen
        :type params: :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        """
        # Check params
        self._representation_topology_func_params = (
            None if params is None else check_func_params(
                params, "representation topology function parameters"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_selection_func(
        self
    ) -> Callable[[List[Solution], Any], Solution]:
        """Get and set the representation selection policy function.

        The representation selection policy func chooses which solutions are
        selected as representatives of each subpopulation.

        :getter: Return the representation selection policy function
        :setter: Set new representation selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            DEFAULT_REPRESENTATION_SELECTION_FUNC
            if self._representation_selection_func is None
            else self._representation_selection_func
        )

    @representation_selection_func.setter
    def representation_selection_func(
        self,
        func: Callable[[List[Solution], Any], Solution] | None
    ) -> None:
        """Set new representation selection policy function.

        :param func: The new function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            is chosen
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._representation_selection_func = (
            None if func is None else check_func(
                func, "representation selection policy function"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_selection_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the representation selection function.

        :getter: Return the current parameters for the representation selection
            policy function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            is chosen
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return (
            DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
            if self._representation_selection_func_params is None
            else self._representation_selection_func_params
        )

    @representation_selection_func_params.setter
    def representation_selection_func_params(
        self, params: Dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation selection policy function.

        :param params: The new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            is chosen
        :type params: :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        """
        # Check that params is a valid dict
        self._representation_selection_func_params = (
            None if params is None else check_func_params(
                params, "representation selection policy function parameters"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def logbook(self) -> Logbook | None:
        """Get the training logbook.

        Return a logbook with the statistics of the search or :py:data:`None`
        if the search has not been done yet.

        :type: :py:class:`~deap.tools.Logbook`
        """
        the_logbook = None

        if self._logbook is not None:
            the_logbook = self._logbook
        elif self.subpop_trainers is not None:
            # Create the logbook
            the_logbook = Logbook()
            # Init the logbook
            the_logbook.header = list(self.stats_names) + \
                (self._stats.fields if self._stats else [])

            for subpop_trainer in self.subpop_trainers:
                if subpop_trainer.logbook is not None:
                    the_logbook.extend(subpop_trainer.logbook)

            if self._search_finished:
                self._logbook = the_logbook

        return the_logbook

    @property
    def subpop_trainer_params(self) -> Dict[str, Any]:
        """Get and set the custom parameters of the subpopulation trainers.

        :getter: Return the current parameters for the subpopulation trainers
        :setter: Set new parameters
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return self._subpop_trainer_params

    @subpop_trainer_params.setter
    def subpop_trainer_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters for the subpopulation trainers.

        :param params: The new parameters
        :type params: A :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        """
        # Check that params is a valid dict
        self._subpop_trainer_params = check_func_params(
            params, "subpopulation trainers parameters"
        )

        # Reset the algorithm
        self.reset()

    @property
    def subpop_trainer_checkpoint_filenames(
        self
    ) -> Generator[str, None, None]:
        """Checkpoint file name of all the subpopulation trainers."""
        base_name = path.splitext(self.checkpoint_filename)[0]
        extension = path.splitext(self.checkpoint_filename)[1]

        # Generator for the subpop trainer checkpoint file names
        return (
            base_name + f"_{suffix}" + extension
            for suffix in self._subpop_suffixes
        )

    @property
    def subpop_trainers(self) -> List[SinglePopTrainer] | None:
        """Return the subpopulation trainers.

        One single-population trainer for each subpopulation

        :type: :py:class:`list` of
            :py:class:`~culebra.trainer.abc.SinglePopTrainer` trainers
        """
        return self._subpop_trainers

    @property
    def _subpop_suffixes(self) -> Generator[str, None, None]:
        """Return the suffixes for the different subpopulations.

        Can be used to generate the subpopulations' names, checkpoint files,
        etc.

        :return: A generator of the suffixes
        :rtype: A generator of :py:class:`str`
        """
        # Suffix length for subpopulations checkpoint files
        suffix_len = len(str(self.num_subpops-1))

        # Generator for the subpopulations checkpoint files
        return (f"{i:0{suffix_len}d}" for i in range(self.num_subpops))

    @staticmethod
    @abstractmethod
    def receive_representatives(subpop_trainer) -> None:
        """Receive representative solutions.

        This method must be overriden by subclasses.

        :param subpop_trainer: The subpopulation trainer receiving
            representatives
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        """
        raise NotImplementedError(
            "The receive_representatives method has not been implemented in "
            f"the {subpop_trainer.container.__class__.__name__} class")

    @staticmethod
    @abstractmethod
    def send_representatives(subpop_trainer) -> None:
        """Send representatives.

        This method must be overriden by subclasses.

        :param subpop_trainer: The sender subpopulation trainer
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        """
        raise NotImplementedError(
            "The send_representatives method has not been implemented in "
            f"the {subpop_trainer.container.__class__.__name__} class")

    @abstractmethod
    def _generate_subpop_trainers(self) -> None:
        """Generate the subpopulation trainers.

        Also assign an :py:attr:`~culebra.trainer.abc.SinglePopTrainer.index`
        and a :py:attr:`~culebra.trainer.abc.SinglePopTrainer.container` to
        each subpopulation :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        trainer, change the subpopulation trainers'
        :py:attr:`~culebra.trainer.abc.SinglePopTrainer.checkpoint_filename`
        according to the container checkpointing file name and each
        subpopulation index.

        Finally, the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.abc.MultiPopTrainer.subpop_trainer_cls` class
        are dynamically overriden, in order to allow solutions exchange
        between subpopulation trainers, if necessary

        This method must be overriden by subclasses.

        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError(
            "The _generate_subpop_trainers method has not been implemented "
            f"in the {self.__class__.__name__} class")

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overriden to set the logbook to :py:data:`None`, since the final
        logbook will be generated from the subpopulation trainers' logbook,
        once the trainer has finished.
        """
        super()._new_state()

        # The logbook will be generated from the subpopulation trainers
        self._logbook = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Overriden to create the subpopulation trainers and communication
        queues.
        """
        super()._init_internals()

        # Generate the subpopulation trainers
        self._generate_subpop_trainers()

        # Set up the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._init_internals()

        # Init the communication queues
        self._communication_queues = []
        for _ in range(self.num_subpops):
            self._communication_queues.append(Queue())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overriden to reset the subpopulation trainers and communication queues.
        """
        super()._reset_internals()
        self._subpop_trainers = None
        self._communication_queues = None

    def __copy__(self) -> MultiPopTrainer:
        """Shallow copy the trainer."""
        cls = self.__class__
        result = cls(
            self.fitness_function,
            self.subpop_trainer_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> MultiPopTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
        """
        cls = self.__class__
        result = cls(
            self.fitness_function,
            self.subpop_trainer_cls
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
                    self.fitness_function,
                    self.subpop_trainer_cls
                ),
                self.__dict__)


class SequentialMultiPopTrainer(MultiPopTrainer):
    """Abstract base class for sequential multi-population trainers."""

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._new_state` method
        of each subpopulation trainer.
        """
        super()._new_state()

        # Generate the state of all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._new_state()

    def _load_state(self) -> None:
        """Load the state of the last checkpoint.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._load_state` method
        of each subpopulation trainer.

        :raises Exception: If the checkpoint file can't be loaded
        """
        # Load the state of this trainer
        super()._load_state()

        # Load the subpopulation trainers' state
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._load_state()

    def _save_state(self) -> None:
        """Save the state at a new checkpoint.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._save_state` method
        of each subpopulation trainer.

        :raises Exception: If the checkpoint file can't be written
        """
        # Save the state of this trainer
        super()._save_state()

        # Save the state of all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._save_state()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the metrics before each iteration is run. Overriden to call
        also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._start_iteration`
        method of each subpopulation trainer.

        """
        super()._start_iteration()
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            # Fix the current iteration
            subpop_trainer._current_iter = self._current_iter
            # Start the iteration
            subpop_trainer._start_iteration()

    def _preprocess_iteration(self) -> None:
        """Preprocess the population of all the subtrainers.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._preprocess_iteration`
        method of each subpopulation trainer.
        """
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._preprocess_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._do_iteration`
        method of each subpopulation trainer.
        """
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._do_iteration_stats`
        method of each subpopulation trainer.
        """
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._do_iteration_stats()

    def _postprocess_iteration(self) -> None:
        """Postprocess the population of all the subtrainers.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._postprocess_iteration`
        method of each subpopulation trainer.
        """
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:
            subpop_trainer._postprocess_iteration()

    def _finish_iteration(self) -> None:
        """Finish an iteration.

        Close the metrics after each iteration is run. Overriden to call also
        the :py:meth:`~culebra.trainer.abc.SinglePopTrainer._finish_iteration`
        method of each subpopulation trainer and accumulate the current number
        of evaluations of all the subpopulations.
        """
        # For all the subpopulation trainers
        for subpop_trainer in self.subpop_trainers:

            # Finish the iteration of all the subpopulation trainers
            subpop_trainer._finish_iteration()
            # Get the number of evaluations
            self._current_iter_evals += subpop_trainer._current_iter_evals

        # Finish the iteration
        super()._finish_iteration()

    def _finish_search(self) -> None:
        """Finish the search process.

        Overriden to call also the
        :py:meth:`~culebra.trainer.abc.SinglePopTrainer._finish_search`
        method of each subpopulation trainer.
        """
        for subpop_trainer in self.subpop_trainers:
            # Fix the current iteration
            subpop_trainer._current_iter = self._current_iter

        # Finish the iteration
        super()._finish_search()


class ParallelMultiPopTrainer(MultiPopTrainer):
    """Abstract base class for parallel multi-population trainers."""

    @MultiPopTrainer.num_subpops.getter
    def num_subpops(self) -> int:
        """Get and set the number of subpopulations.

        :getter: Return the current number of subpopulations
        :setter: Set a new value for the number of subpopulations. If set to
            :py:data:`None`, the number of CPU cores is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            cpu_count() if self._num_subpops is None
            else self._num_subpops
        )

    @property
    def num_evals(self) -> int | None:
        """Get the number of evaluations performed while training.

        Return the number of evaluations or :py:data:`None` if the search has
        not been done yet.

        :type: :py:class:`int`
        """
        n_evals = None

        if self._num_evals is not None:
            n_evals = self._num_evals
        elif self.subpop_trainers is not None:
            n_evals = 0
            for subpop_trainer in self.subpop_trainers:
                if subpop_trainer.num_evals is not None:
                    n_evals += subpop_trainer.num_evals

            if self._search_finished:
                self._num_evals = n_evals

        return n_evals

    @property
    def runtime(self) -> float | None:
        """Get the training runtime.

        Return the training runtime or :py:data:`None` if the search has not
        been done yet.

        :type: :py:class:`float`
        """
        the_runtime = None

        if self._runtime is not None:
            the_runtime = self._runtime
        elif self.subpop_trainers is not None:
            the_runtime = 0
            for subpop_trainer in self.subpop_trainers:
                if (
                        subpop_trainer.runtime is not None
                        and subpop_trainer.runtime > the_runtime
                ):
                    the_runtime = subpop_trainer.runtime

            if self._search_finished:
                self._runtime = the_runtime

        return the_runtime

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overriden to set the overall runtime and number of evaluations to
        :py:data:`None`, since their final values will be generated from the
        subpopulation trainers' state, once the trainer has finished.
        """
        super()._new_state()

        # The runtime and number of evaluations will be generated
        # from the subpopulation trainers
        self._runtime = None
        self._num_evals = None

        # Each subpopultion handles its own current iteration
        self._current_iter = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Overriden to create a multiprocessing manager and proxies to
        communicate with the processes running the subpopulation trainers.
        """
        super()._init_internals()

        # Initialize the manager to receive the subpopulation trainers' state
        self._manager = Manager()
        self._subpop_state_proxies = []
        for _ in range(self.num_subpops):
            self._subpop_state_proxies.append(self._manager.dict())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overriden to reset the multiprocessing manager and proxies.
        """
        super()._reset_internals()
        self._manager = None
        self._subpop_state_proxies = None

    def _search(self) -> None:
        """Apply the search algorithm.

        Each subpopulation trainer runs in a different process.
        """
        # Run all the iterations
        for subpop_trainer, state_proxy, suffix in zip(
                self.subpop_trainers,
                self._subpop_state_proxies,
                self._subpop_suffixes):
            subpop_trainer.process = Process(
                target=subpop_trainer.train,
                kwargs={"state_proxy": state_proxy},
                name=f"subpop_{suffix}",
                daemon=True
            )
            subpop_trainer.process.start()

        for subpop_trainer, state_proxy in zip(
                self.subpop_trainers, self._subpop_state_proxies):
            subpop_trainer.process.join()
            subpop_trainer._state = state_proxy


# Change the docstring of the ParallelMultiPop constructor to indicate that
# the default number of subpopulations is the number of CPU cores for parallel
# multi-population approaches
ParallelMultiPopTrainer.__init__.__doc__ = (
    ParallelMultiPopTrainer.__init__.__doc__.replace(
        ':py:attr:`~culebra.trainer.DEFAULT_NUM_SUBPOPS`',
        'the number of CPU cores'
    )
)


class IslandsTrainer(SingleSpeciesTrainer, MultiPopTrainer):
    """Abstract island-based trainer."""

    def __init__(
        self,
        solution_cls: Type[Solution],
        species: Species,
        fitness_function: FitnessFunction,
        subpop_trainer_cls: Type[SinglePopTrainer],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopTrainer], bool]
        ] = None,
        num_subpops: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Solution], Any], Solution]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subpop_trainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The solution class
        :type solution_cls: A :py:class:`~culebra.abc.Solution` subclass
        :param species: The species for all the solutions
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subpop_trainer_cls: Single-population trainer class to handle
            the subpopulations (islands).
        :type subpop_trainer_cls: Any subclass of
            :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`,
            :py:meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param num_subpops: The number of subpopulations (islands). If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBPOPS` will be
            used. Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative solutions that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (island). If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
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
        :param subpop_trainer_params: Custom parameters for the subpopulations
            (islands) trainer
        :type subpop_trainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        SingleSpeciesTrainer.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function
        )

        MultiPopTrainer.__init__(
            self,
            fitness_function=fitness_function,
            subpop_trainer_cls=subpop_trainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subpops=num_subpops,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subpop_trainer_params
        )

    @MultiPopTrainer.representation_topology_func.getter
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get and set the representation topology function.

        :getter: Return the representation topology function
        :setter: Set new representation topology function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        """
        return (
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
            if self._representation_topology_func is None
            else self._representation_topology_func
        )

    @MultiPopTrainer.representation_topology_func_params.getter
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the representation topology function.

        :getter: Return the current parameters for the representation topology
            function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return (
            DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            if self._representation_topology_func_params is None
            else self._representation_topology_func_params
        )

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        Return the best single solution found for each species

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        hof = None
        # If the search hasn't been initialized an empty HoF is returned
        if self.subpop_trainers is None:
            hof = ParetoFront()
        else:
            for subpop_trainer in self.subpop_trainers:
                if hof is None:
                    hof = subpop_trainer.best_solutions()[0]
                else:
                    if subpop_trainer.pop is not None:
                        hof.update(subpop_trainer.pop)

        return [hof]

    @staticmethod
    def receive_representatives(subpop_trainer) -> None:
        """Receive representative solutions.

        :param subpop_trainer: The subpopulation trainer receiving
            representatives
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        """
        container = subpop_trainer.container

        # Receive all the solutions in the queue
        queue = container._communication_queues[subpop_trainer.index]
        while not queue.empty():
            subpop_trainer._pop.extend(queue.get())

    @staticmethod
    def send_representatives(subpop_trainer) -> None:
        """Send representatives.

        :param subpop_trainer: The sender subpopulation trainer
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        """
        container = subpop_trainer.container

        # Check if sending should be performed
        if subpop_trainer._current_iter % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subpop_trainer.index,
                container.num_subpops,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                sols = container.representation_selection_func(
                    subpop_trainer.pop,
                    container.representation_size,
                    **container.representation_selection_func_params
                )
                container._communication_queues[dest].put(sols)

    def __copy__(self) -> IslandsTrainer:
        """Shallow copy the trainer."""
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.subpop_trainer_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> IslandsTrainer:
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
            self.subpop_trainer_cls
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
                    self.subpop_trainer_cls
                ),
                self.__dict__)


class MultiSpeciesTrainer(Trainer):
    """Base class for trainers that find solutions for multiple species."""

    def __init__(
        self,
        solution_classes: Sequence[Type[Solution]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[[Trainer], bool]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new trainer.

        :param solution_classes: The solution class for each species.
        :type solution_classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution` subclasses
        :param species: The species to be evolved
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`,
            :py:meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
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
        Trainer.__init__(
            self,
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
        self.solution_classes = solution_classes
        self.species = species

    @property
    def solution_classes(self) -> Sequence[Type[Solution]]:
        """Get and set the solution classes.

        :getter: Return the current solution classes
        :setter: Set the new solution classes
        :type: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution` subclasses
        :raises TypeError: If set to a value which is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`~culebra.abc.Solution` subclass
        """
        return self._solution_classes

    @solution_classes.setter
    def solution_classes(self, classes: Sequence[Type[Solution]]) -> None:
        """Set the new solution classes.

        :param classes: The classes.
        :type classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution` subclasses
        :raises TypeError: If *classes* is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *classes* is not a
            :py:class:`~culebra.abc.Solution` subclass
        """
        self._solution_classes = check_sequence(
            classes,
            "solution classes",
            item_checker=partial(check_subclass, cls=Solution)
        )

        # Reset the algorithm
        self.reset()

    @property
    def species(self) -> Sequence[Species]:
        """Get and set the species for each subpopulation.

        :getter: Return the current species
        :setter: Set the new species
        :type: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :raises TypeError: If set to a value which is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of the sequence is not a
            :py:class:`~culebra.abc.Species`
        """
        return self._species

    @species.setter
    def species(self, value: Sequence[Species]) -> None:
        """Set the new species.

        :param value: The species.
        :type value: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :raises TypeError: If *value* is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *value* is not a
            :py:class:`~culebra.abc.Species`
        """
        self._species = check_sequence(
            value,
            "species",
            item_checker=partial(check_instance, cls=Species)
        )

        # Reset the algorithm
        self.reset()

    def __copy__(self) -> SingleSpeciesTrainer:
        """Shallow copy the trainer."""
        cls = self.__class__
        result = cls(
            self.solution_classes, self.species, self.fitness_function
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> SingleSpeciesTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
        """
        cls = self.__class__
        result = cls(
            self.solution_classes, self.species, self.fitness_function
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (self.solution_classes, self.species, self.fitness_function),
                self.__dict__)


class CooperativeTrainer(MultiSpeciesTrainer, MultiPopTrainer):
    """Abstract cooperative trainer model."""

    def __init__(
        self,
        solution_classes: Sequence[Type[Solution]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        subpop_trainer_cls: Type[SinglePopTrainer],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SinglePopTrainer],
                bool
            ]
        ] = None,
        pop_sizes: Optional[int | Sequence[int]] = None,
        num_subpops: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
        representation_topology_func: Optional[
            Callable[[int, int, Any], List[int]]
        ] = None,
        representation_topology_func_params: Optional[
            Dict[str, Any]
        ] = None,
        representation_selection_func: Optional[
            Callable[[List[Solution], Any], Solution]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subpop_trainer_params: Any
    ) -> None:
        """Create a new trainer.

        Each species is evolved in a different subpopulation.

        :param solution_classes: The individual class for each species.
        :type solution_classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution` subclasses
        :param species: The species to be evolved
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subpop_trainer_cls: Single-population trainer class to handle
            the subpopulations.
        :type subpop_trainer_cls: Any subclass of
            :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        :param max_num_iters: Maximum number of iterations. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :py:data:`None`
        :type max_num_iters: :py:class:`int`, optional
        :param custom_termination_func: Custom termination criterion. If set to
            :py:data:`None`,
            :py:meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :py:data:`None`
        :type custom_termination_func: :py:class:`~collections.abc.Callable`,
            optional
        :param pop_sizes: The population size for each subpopulation (species).
            If only a single value is provided, the same size will be used for
            all the subpopulations. Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_POP_SIZE` will be used.
            Defaults to :py:data:`None`
        :type pop_sizes: :py:class:`int` or
            :py:class:`~collections.abc.Sequence` of :py:class:`int`, optional
        :param num_subpops: The number of subpopulations (species). If set to
            :py:data:`None`, the number of species  evolved by the trainer is
            will be used, otherwise it must match the number of species.
            Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation (species). If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
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
        :param subpop_trainer_params: Custom parameters for the subpopulations
            (species) trainer
        :type subpop_trainer_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        MultiSpeciesTrainer.__init__(
            self,
            solution_classes=solution_classes,
            species=species,
            fitness_function=fitness_function
        )

        MultiPopTrainer.__init__(
            self,
            fitness_function=fitness_function,
            subpop_trainer_cls=subpop_trainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subpops=num_subpops,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=(
                representation_topology_func_params
            ),
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subpop_trainer_params
        )

        # Get the rest of parameters
        self.pop_sizes = pop_sizes

    @property
    def representatives(self) -> Sequence[Sequence[Solution | None]] | None:
        """Return the representatives of all the species."""
        # Default value
        the_representatives = None

        # If the representatives have been gathered before
        if self._representatives is not None:
            the_representatives = self._representatives
        elif self.subpop_trainers is not None:
            # Create the container for the representatives
            the_representatives = [
                [None] * self.num_subpops
                for _ in range(self.representation_size)
            ]
            for (
                    subpop_index,
                    subpop_trainer
                    ) in enumerate(self.subpop_trainers):
                for (context_index,
                     _
                     ) in enumerate(subpop_trainer.representatives):
                    the_representatives[
                        context_index][
                            subpop_index - 1
                    ] = subpop_trainer.representatives[
                            context_index][
                                subpop_index - 1
                    ]

            if self._search_finished is True:
                self._representatives = the_representatives

        return the_representatives

    @property
    def num_subpops(self) -> int:
        """Get and set the number of subpopulations.

        :getter: Return the current number of subpopulations
        :setter: Set a new value for the number of subpopulations. If set to
            :py:data:`None`, the number of species  evolved by the trainer is
            chosen, otherwise *it must match the number of species*
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is different of the number
            of species
        """
        return (
            len(self.species) if self._num_subpops is None
            else self._num_subpops
        )

    @num_subpops.setter
    def num_subpops(self, value: int | None) -> None:
        """Set the number of subpopulations.

        :param value: The new number of subpopulations. If set to
            :py:data:`None`, the number of species  evolved by the trainer is
            chosen, otherwise *it must match the number of species*
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is different of the number of species
        """
        # Check the value
        if value is not None and value != len(self.species):
            raise ValueError(
                "The number of subpopulations must match the number of "
                f"species: {self.species}"
            )

        self._num_subpops = value

        # Reset the algorithm
        self.reset()

    @property
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get and set the representation topology function.

        :getter: Return the representation topology function
        :setter: Set new representation topology function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen, otherwise it must match
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
        :raises ValueError: If set to a value different of
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
        """
        return (
            DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC
            if self._representation_topology_func is None
            else self._representation_topology_func
        )

    @representation_topology_func.setter
    def representation_topology_func(
        self,
        func: Callable[[int, int, Any], List[int]] | None
    ) -> None:
        """Set new representation topology function.

        :param func: The new function. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
            is chosen, otherwise it must match
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
        :type func: :py:class:`~collections.abc.Callable`
        :raises TypeError: If *func* is not a callable
        :raises ValueError: If *func* is different of
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
        """
        # Check func
        if (
            func is not None and
            func != DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC
        ):
            raise ValueError(
                "The representation topology function must be "
                f"{DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC.__name__}"
            )

        self._representation_topology_func = func

        # Reset the algorithm
        self.reset()

    @property
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get and set the parameters of the representation topology function.

        :getter: Return the current parameters for the representation topology
            function
        :setter: Set new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen, otherwise it must match
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        :raises ValueError: If set to a value different of
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        """
        return (
            DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
            if self._representation_topology_func_params is None
            else self._representation_topology_func_params
        )

    @representation_topology_func_params.setter
    def representation_topology_func_params(
        self, params: Dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation topology function.

        :param params: The new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            is chosen, otherwise it must match
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        :type params: :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        :raises ValueError: If *params* is different of
            :py:attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        """
        # Check params
        if (
            params is not None and
            params != DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
        ):
            raise ValueError(
                "The representation topology function parameters must be "
                f"{DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS}"
            )

        self._representation_topology_func_params = params

        # Reset the algorithm
        self.reset()

    @property
    def pop_sizes(self) -> Sequence[int | None]:
        """Get and set the population size for each subpopulation.

        :getter: Return the current size of each subpopulation
        :setter: Set a new size for each subpopulation. If only a single value
            is provided, the same size will be used for all the subpopulations.
            Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int` or :py:class:`~collections.abc.Sequence`
            of :py:class:`int`
        :raises TypeError: If set to a value which is not an :py:class:`int`
            or a :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If any population size is not greater than zero
        """
        if self.subpop_trainers is not None:
            the_pop_sizes = [
                subpop_trainer.pop_size
                for subpop_trainer in self.subpop_trainers
            ]
        elif isinstance(self._pop_sizes, Sequence):
            the_pop_sizes = self._pop_sizes
        else:
            the_pop_sizes = list(repeat(self._pop_sizes, self.num_subpops))

        return the_pop_sizes

    @pop_sizes.setter
    def pop_sizes(self, sizes: int | Sequence[int] | None) -> None:
        """Set the population size for each subpopulation.

        :param sizes: The new population sizes. If only a single value
            is provided, the same size will be used for all the subpopulations.
            Different sizes can be provided in a
            :py:class:`~collections.abc.Sequence`. All the sizes must be
            greater then zero. If set to :py:data:`None`,
        :py:attr:`~culebra.DEFAULT_POP_SIZE` is chosen
        :type: :py:class:`int` or :py:class:`~collections.abc.Sequence`
            of :py:class:`int`
        :raises TypeError: If *sizes* is not an :py:class:`int`
            or a :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises ValueError: If any value in *size* is not greater than zero
        """
        # If None is provided ...
        if sizes is None:
            self._pop_sizes = None
        # If a sequence is provided ...
        elif isinstance(sizes, Sequence):
            self._pop_sizes = check_sequence(
                sizes,
                "population sizes",
                item_checker=partial(check_int, gt=0)
            )
        # If a scalar value is provided ...
        else:
            self._pop_sizes = check_int(sizes, "population size", gt=0)

        # Reset the algorithm
        self.reset()

    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best individuals found for each species.

        :return: A sequence containing :py:class:`~deap.tools.HallOfFame` of
            individuals. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
        """
        # List of hofs
        hofs = []
        # If the search hasn't been initialized a list of empty hofs is
        # returned, one hof per species
        if self.subpop_trainers is None:
            for _ in range(self.num_subpops):
                hofs.append(ParetoFront())
        # Else, the best solutions of each species are returned
        else:
            for subpop_trainer in self.subpop_trainers:
                hofs.append(subpop_trainer.best_solutions()[0])

        return hofs

    def best_representatives(self) -> List[List[Solution]] | None:
        """Return a list of representatives from each species.

        :return: A list of representatives lists. One representatives list for
            each one of the evolved species or :py:data:`None` if the search
            has nos finished
        :rtype: :py:class:`list` of :py:class:`list` of
            :py:class:`~culebra.abc.Solution` or :py:data:`None`
        """
        # Check if the trianing has finished
        # self._search_finished could be None or False...
        if self._search_finished is not True:
            the_representatives = None
        else:
            # Create the container for the representatives
            the_representatives = [
                [None] * self.num_subpops
                for _ in range(self.representation_size)
            ]

            # Get the best solutions for each species
            best_ones = self.best_solutions()

            # Select the representatives
            for species_index, hof in enumerate(best_ones):
                species_representatives = self.representation_selection_func(
                    hof,
                    self.representation_size,
                    **self.representation_selection_func_params
                )

                # Insert the representsatives
                for ind_index, ind in enumerate(species_representatives):
                    the_representatives[ind_index][species_index] = ind

        return the_representatives

    @staticmethod
    def receive_representatives(subpop_trainer) -> None:
        """Receive representative individuals.

        :param subpop_trainer: The subpopulation trainer receiving
            representatives
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        """
        container = subpop_trainer.container

        # Receive all the individuals in the queue
        queue = container._communication_queues[subpop_trainer.index]

        anything_received = False
        while not queue.empty():
            msg = queue.get()
            sender_index = msg[0]
            representatives = msg[1]
            for ind_index, ind in enumerate(representatives):
                subpop_trainer.representatives[ind_index][sender_index] = ind

            anything_received = True

        # If any new representatives have arrived, the fitness of all the
        # individuals in the population must be invalidated and individuals
        # must be re-evaluated
        if anything_received:
            # Re-evaluate all the individuals
            for sol in subpop_trainer.pop:
                subpop_trainer.evaluate(sol)

    @staticmethod
    def send_representatives(subpop_trainer) -> None:
        """Send representatives.

        :param subpop_trainer: The sender subpopulation trainer
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        """
        container = subpop_trainer.container

        # Check if sending should be performed
        if subpop_trainer._current_iter % container.representation_freq == 0:
            # Get the destinations according to the representation topology
            destinations = container.representation_topology_func(
                subpop_trainer.index,
                container.num_subpops,
                **container.representation_topology_func_params
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                inds = container.representation_selection_func(
                    subpop_trainer.pop,
                    container.representation_size,
                    **container.representation_selection_func_params
                )

                # Send the following msg:
                # (index of sender subpop, representatives)
                container._communication_queues[dest].put(
                    (subpop_trainer.index, inds)
                )

    @staticmethod
    def _init_subpop_trainer_representatives(
        subpop_trainer,
        solution_classes,
        species,
        representation_size
    ):
        """Init the representatives of the other species for a given
        subpopulation trainer.

        This method is used to override dynamically the
        :py:meth:`~culebra.abc.Trainer._init_representatives` of all the
        subpopulation trainers, when they are generated with the
        :py:meth:`~culebra.trainer.abc.CooperativeTrainer._generate_subpop_trainers`
        method, to let them initialize the list of representative individuals
        of the other species

        :param subpop_trainer: The subpopulation trainer. The representatives
            from the remaining subpopulation trainers will be initialized for
            this subpopulation trainer
        :type subpop_trainer: :py:class:`~culebra.trainer.abc.SinglePopTrainer`
        :param solution_classes: The individual class for each species.
        :type solution_classes: :py:class:`~collections.abc.Sequence`
            of :py:class:`~culebra.abc.Solution` subclasses
        :param species: The species to be evolved by this trainer
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :param representation_size: Number of representative individuals
            generated for each species
        :type representation_size: :py:class:`int`
        """
        subpop_trainer._representatives = []

        for _ in range(representation_size):
            subpop_trainer._representatives.append(
                [
                    ind_cls(
                        spe, subpop_trainer.fitness_function.Fitness
                    ) if i != subpop_trainer.index else None
                    for i, (ind_cls, spe) in enumerate(
                        zip(solution_classes, species)
                    )
                ]
            )

    def __copy__(self) -> CooperativeTrainer:
        """Shallow copy the trainer."""
        cls = self.__class__
        result = cls(
            self.solution_classes,
            self.species,
            self.fitness_function,
            self.subpop_trainer_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> CooperativeTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
        """
        cls = self.__class__
        result = cls(
            self.solution_classes,
            self.species,
            self.fitness_function,
            self.subpop_trainer_cls
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
                    self.solution_classes,
                    self.species,
                    self.fitness_function,
                    self.subpop_trainer_cls
                ),
                self.__dict__)


# Exported symbols for this module
__all__ = [
    'SingleSpeciesTrainer',
    'SinglePopTrainer',
    'MultiPopTrainer',
    'SequentialMultiPopTrainer',
    'ParallelMultiPopTrainer',
    'IslandsTrainer',
    'MultiSpeciesTrainer',
    'CooperativeTrainer'
]
