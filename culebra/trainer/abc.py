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

"""Abstract base classes for different trainers.

This module provides several abstract classes for different kind of trainers.

Regarding the number of species that are simultaneously being trained:

* :class:`~culebra.trainer.abc.MultiSpeciesTrainer`: Provides a base class for
  trainers that find solutions for multiple species
* :class:`~culebra.trainer.abc.SingleSpeciesTrainer`: Provides a base class
  for trainers for solutions of only a single species

Trainers can also be distributed. The
:class:`~culebra.trainer.abc.DistributedTrainer` class provides a base support
to distribute a trainer making use a several subtrainers. Two implementations
of this class are also provided:

* :class:`~culebra.trainer.abc.ParallelDistributedTrainer`: Abstract base
  class for parallel distributed trainers
* :class:`~culebra.trainer.abc.SequentialDistributedTrainer`: Abstract base
  class for sequential distributed trainers

Finally, some usual distributed approaches are also provided:

* :class:`~culebra.trainer.abc.CooperativeTrainer`: Abstract base class for
  cooperative trainers
* :class:`~culebra.trainer.abc.IslandsTrainer`: Abstract base class for
  island-based approaches
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from collections.abc import Sequence, Callable, Generator
from copy import deepcopy
from functools import partial
from multiprocess import (
    cpu_count,
    Queue,
    Process,
    Manager
)

from deap.tools import Logbook, HallOfFame, ParetoFront

from culebra import SERIALIZED_FILE_EXTENSION
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
    DEFAULT_NUM_SUBTRAINERS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
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
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'




class SingleSpeciesTrainer(Trainer):
    """Base class for trainers for solutions of only a single species."""

    def __init__(
        self,
        solution_cls: type[Solution],
        species: Species,
        fitness_function: FitnessFunction,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[Trainer], bool] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.abc.Solution]
        :param species: The species for all the solutions
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.abc.SingleSpeciesTrainer._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.abc.SingleSpeciesTrainer._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.abc.SingleSpeciesTrainer._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.abc.SingleSpeciesTrainer._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.abc.SingleSpeciesTrainer._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        Trainer.__init__(
            self,
            fitness_function=fitness_function,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

        # Get the parameters
        self.solution_cls = solution_cls
        self.species = species

    @property
    def solution_cls(self) -> type[Solution]:
        """Solution class.

        :rtype: type[~culebra.abc.Solution]

        :setter: Set a new solution class
        :param cls: The new class
        :type cls: type[~culebra.abc.Solution]
        :raises TypeError: If *cls* is not valid solution class
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, cls: type[Solution]) -> None:
        """Set a new solution class.

        :param cls: The new class
        :type cls: type[~culebra.abc.Solution]
        :raises TypeError: If *cls* is not a :class:`~culebra.abc.Solution`
            subclass
        """
        # Check cls
        self._solution_cls = check_subclass(cls, "solution class", Solution)

        # Reset the algorithm
        self.reset()

    @property
    def species(self) -> Species:
        """Species.

        :rtype: ~culebra.abc.Species

        :setter: Set a new species
        :param value: The new species
        :type value: ~culebra.abc.Species
        :raises TypeError: If *value* is not a valid species
        """
        return self._species

    @species.setter
    def species(self, value: Species) -> None:
        """Set a new species.

        :param value: The new species
        :type value: ~culebra.abc.Species
        :raises TypeError: If *value* is not a :class:`~culebra.abc.Species`
            instance
        """
        # Check the value
        self._species = check_instance(value, "species", Species)

        # Reset the algorithm
        self.reset()

    def __copy__(self) -> SingleSpeciesTrainer:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.SingleSpeciesTrainer
        """
        cls = self.__class__
        result = cls(
            self.solution_cls, self.species, self.fitness_function
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> SingleSpeciesTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.SingleSpeciesTrainer
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
        :rtype: tuple
        """
        return (self.__class__,
                (self.solution_cls, self.species, self.fitness_function),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> SingleSpeciesTrainer:
        """Return a single species trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.abc.SingleSpeciesTrainer
        """
        obj = cls(
            state['_solution_cls'],
            state['_species'],
            state['_fitness_function']
        )
        obj.__setstate__(state)
        return obj


class DistributedTrainer(Trainer):
    """Base class for all the distributed trainers."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SingleSpeciesTrainer],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[SingleSpeciesTrainer], bool] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Solution], Any], Solution] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-species trainer class to handle
            the subtrainers.
        :type subtrainer_cls: type[~culebra.trainer.abc.SingleSpeciesTrainer]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.abc.DistributedTrainer._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param num_subtrainers: The number of subtrainers. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative solutions that
            will be sent to the other subtrainers. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subtrainer. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subtrainers
            trainer
        :type subtrainer_params: dict
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__(
            fitness_function=fitness_function,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

        # Get the parameters
        self.subtrainer_cls = subtrainer_cls
        self.num_subtrainers = num_subtrainers
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
        self.subtrainer_params = subtrainer_params

    @property
    def subtrainer_cls(self) -> type[SingleSpeciesTrainer]:
        """Trainer class to handle the subtrainers.

        Each subtrainer will be handled by a single-species trainer.

        :rtype: type[~culebra.trainer.abc.SingleSpeciesTrainer]
        :setter: Set a new trainer class to handle the subtrainers
        :param cls: The new class
        :type cls: type[~culebra.trainer.abc.SingleSpeciesTrainer]
        :raises TypeError: If *cls* is not a valid trainer class
        """
        return self._subtrainer_cls

    @subtrainer_cls.setter
    def subtrainer_cls(self, cls: type[SingleSpeciesTrainer]) -> None:
        """Set a new trainer class to handle the subtrainers.

        Each subtrainer will be handled by a single-species trainer.

        :param cls: The new class
        :type cls: type[~culebra.trainer.abc.SingleSpeciesTrainer]
        :raises TypeError: If *cls* is not a
            :class:`~culebra.trainer.abc.SingleSpeciesTrainer` subclass
        """
        # Check cls
        self._subtrainer_cls = check_subclass(
            cls, "trainer class for subtrainers", SingleSpeciesTrainer
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_num_subtrainers(self) -> int:
        """Default number of subtrainers.

        :return: :attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS`
        :rtype: int
        """
        return DEFAULT_NUM_SUBTRAINERS

    @property
    def num_subtrainers(self) -> int:
        """Number of subtrainers.

        :rtype: int
        :setter: Set a new value for the number of subtrainers
        :param value: The new number of subtrainers. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_num_subtrainers`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return self._num_subtrainers

    @num_subtrainers.setter
    def num_subtrainers(self, value: int | None) -> None:
        """Set a new value for the number of subtrainers.

        :param value: The new number of subtrainers. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_num_subtrainers`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._num_subtrainers = (
            self._default_num_subtrainers if value is None else check_int(
                value, "number of subtrainers", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_representation_size(self) -> int:
        """Default number of representatives sent to the other subtrainers.

        :return: :attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE`
        :rtype: int
        """
        return DEFAULT_REPRESENTATION_SIZE

    @property
    def representation_size(self) -> int:
        """Representation size.

        :return: The number of representatives sent to the other subtrainers
        :rtype: int
        :setter: Set a new representation size
        :param size: The new size. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_size`
            is chosen
        :type size: int
        :raises TypeError: If *size* is not an integer number
        :raises ValueError: If *size* is not positive
        """
        return self._representation_size

    @representation_size.setter
    def representation_size(self, size: int | None) -> None:
        """Set a new representation size.

        :param size: The new size. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_size`
            is chosen
        :type size: int
        :raises TypeError: If *size* is not an integer number
        :raises ValueError: If *size* is not positive
        """
        # Check size
        self._representation_size = (
            self._default_representation_size if size is None else check_int(
                size, "representation size", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_representation_freq(self) -> int:
        """Default number of iterations between representatives sending.

        :return: :attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ`
        :rtype: int
        """
        return DEFAULT_REPRESENTATION_FREQ

    @property
    def representation_freq(self) -> int:
        """Number of iterations between representatives sendings.

        :rtype: int
        :setter: Set a new value for the frequency
        :param value: The new frequency. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_freq`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return self._representation_freq

    @representation_freq.setter
    def representation_freq(self, value: int | None) -> None:
        """Set the number of iterations between representatives sendings.

        :param value: The new frequency. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_freq`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._representation_freq = (
            self._default_representation_freq if value is None else check_int(
                value, "representation frequency", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_representation_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Default topology function.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Callable
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The representation_topology_func property has not been "
            f"implemented in the {self.__class__.__name__} class"
        )

    @property
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Representation topology function.

        :rtype: ~collections.abc.Callable
        :setter: Set new representation topology function
        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_topology_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._representation_topology_func

    @representation_topology_func.setter
    def representation_topology_func(
        self,
        func: Callable[[int, int, Any], list[int]] | None
    ) -> None:
        """Set new representation topology function.

        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_topology_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._representation_topology_func = (
            self._default_representation_topology_func
            if func is None else check_func(
                func, "representation topology function"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_representation_topology_func_params(self) -> dict[str, Any]:
        """Default parameters for the default topology function.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: dict
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The representation_topology_func_params property has not been "
            f"implemented in the {self.__class__.__name__} class"
        )

    @property
    def representation_topology_func_params(self) -> dict[str, Any]:
        """Parameters of the representation topology function.

        :rtype: dict
        :setter: Set new parameters
        :param params: The new parameters. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_topology_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        return self._representation_topology_func_params

    @representation_topology_func_params.setter
    def representation_topology_func_params(
        self, params: dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation topology function.

        :param params: The new parameters. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_topology_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        # Check params
        self._representation_topology_func_params = (
            self._default_representation_topology_func_params
            if params is None else check_func_params(
                params, "representation topology function parameters"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_representation_selection_func(
        self
    ) -> Callable[[list[Solution], Any], Solution]:
        """Default selection policy function to choose the representatives.

        :return: :attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_REPRESENTATION_SELECTION_FUNC

    @property
    def representation_selection_func(
        self
    ) -> Callable[[list[Solution], Any], Solution]:
        """Representation selection policy function.

        :return: A function that chooses which solutions are selected as
            representatives of each subtrainer
        :rtype: ~collections.abc.Callable
        :setter: Set new representation selection policy function.
        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._representation_selection_func

    @representation_selection_func.setter
    def representation_selection_func(
        self,
        func: Callable[[list[Solution], Any], Solution] | None
    ) -> None:
        """Set new representation selection policy function.

        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._representation_selection_func = (
            self._default_representation_selection_func
            if func is None else check_func(
                func, "representation selection policy function"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _default_representation_selection_func_params(self) -> dict[str, Any]:
        """Default parameters for the representatives selection policy function.

        :return: :attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
        :rtype: dict
        """
        return DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS

    @property
    def representation_selection_func_params(self) -> dict[str, Any]:
        """Parameters of the representation selection function.

        :rtype: dict
        :setter: Set new parameters
        :param params: The new parameters. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_selection_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        return self._representation_selection_func_params

    @representation_selection_func_params.setter
    def representation_selection_func_params(
        self, params: dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation selection policy function.

        :param params: The new parameters. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representation_selection_func_params`
            is chosen
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        # Check that params is a valid dict
        self._representation_selection_func_params = (
            self._default_representation_selection_func_params
            if params is None else check_func_params(
                params, "representation selection policy function parameters"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def logbook(self) -> Logbook | None:
        """Trainer logbook.

        :return: A logbook with the statistics of the search or :data:`None`
            if the search has not been done yet
        :rtype: ~deap.tools.Logbook
        """
        the_logbook = None

        if self._logbook is not None:
            the_logbook = self._logbook
        elif self.subtrainers is not None:
            # Create the logbook
            the_logbook = Logbook()
            # Init the logbook
            the_logbook.header = list(self.stats_names) + \
                (self._stats.fields if self._stats else [])

            for subtrainer in self.subtrainers:
                if subtrainer.logbook is not None:
                    the_logbook.extend(subtrainer.logbook)

            if self._search_finished:
                self._logbook = the_logbook

        return the_logbook

    @property
    def subtrainer_params(self) -> dict[str, Any]:
        """Custom parameters for the subtrainers.

        :rtype: dict
        :setter: Set new parameters
        :param params: The new parameters
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        return self._subtrainer_params

    @subtrainer_params.setter
    def subtrainer_params(self, params: dict[str, Any]) -> None:
        """Set the parameters for the subtrainers.

        :param params: The new parameters
        :type params: dict
        :raises TypeError: If *params* is not a :class:`dict`
        """
        # Check that params is a valid dict
        self._subtrainer_params = check_func_params(
            params, "subtrainers parameters"
        )

        # Reset the algorithm
        self.reset()

    @property
    def subtrainer_checkpoint_filenames(
        self
    ) -> Generator[str, None, None]:
        """Checkpoint file name for all the subtrainers.

        :return: A generator of the filenames
        :rtype: ~typing.Generator[str, None, None]
        """
        base_name = self.checkpoint_filename[:-len(SERIALIZED_FILE_EXTENSION)]
        extension = SERIALIZED_FILE_EXTENSION

        # Generator for the subtrainer checkpoint file names
        return (
            base_name + f"_{suffix}" + extension
            for suffix in self._subtrainer_suffixes
        )

    @property
    def subtrainers(self) -> list[SingleSpeciesTrainer] | None:
        """Subtrainers.

        One single-species trainer for each subtrainer.

        :rtype: list[~culebra.trainer.abc.SingleSpeciesTrainer]
        """
        return self._subtrainers

    @property
    def _subtrainer_suffixes(self) -> Generator[str, None, None]:
        """Suffixes for the different subtrainers.

        Can be used to generate the subtrainers' names, checkpoint files,
        etc.

        :return: A generator of the suffixes
        :rtype: ~typing.Generator[str, None, None]
        """
        # Suffix length for subtrainers checkpoint files
        suffix_len = len(str(self.num_subtrainers-1))

        # Generator for the subtrainers checkpoint files
        return (f"{i:0{suffix_len}d}" for i in range(self.num_subtrainers))

    @staticmethod
    @abstractmethod
    def receive_representatives(subtrainer: SingleSpeciesTrainer) -> None:
        """Receive representative solutions.

        This method must be overridden by subclasses.

        :param subtrainer: The subtrainer receiving representatives
        :type subtrainer: ~culebra.trainer.abc.SingleSpeciesTrainer
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The receive_representatives method has not been implemented in "
            f"the {subtrainer.container.__class__.__name__} class")

    @staticmethod
    @abstractmethod
    def send_representatives(subtrainer: SingleSpeciesTrainer) -> None:
        """Send representatives.

        This method must be overridden by subclasses.

        :param subtrainer: The sender subtrainer
        :type subtrainer: ~culebra.trainer.abc.SingleSpeciesTrainer
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The send_representatives method has not been implemented in "
            f"the {subtrainer.container.__class__.__name__} class")

    @abstractmethod
    def _generate_subtrainers(self) -> None:
        """Generate the subtrainers.

        Also assign an :attr:`~culebra.trainer.abc.SingleSpeciesTrainer.index`
        and a :attr:`~culebra.trainer.abc.SingleSpeciesTrainer.container` to
        each :class:`~culebra.trainer.abc.SingleSpeciesTrainer` subtrainer,
        and change the subtrainers'
        :attr:`~culebra.trainer.abc.SingleSpeciesTrainer.checkpoint_filename`
        according to the container checkpointing file name and each
        subtrainer index.

        Finally, the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._preprocess_iteration`
        and
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._postprocess_iteration`
        methods of the
        :attr:`~culebra.trainer.abc.DistributedTrainer.subtrainer_cls`
        class are dynamically overridden, in order to allow solutions exchange
        between subtrainers, if necessary.

        This method must be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _generate_subtrainers method has not been implemented "
            f"in the {self.__class__.__name__} class")

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to set the logbook to :data:`None`, since the final
        logbook will be generated from the subtrainers' logbook,
        once the trainer has finished.
        """
        super()._new_state()

        # The logbook will be generated from the subtrainers
        self._logbook = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Overridden to create the subtrainers and communication
        queues.
        """
        super()._init_internals()

        # Generate the subtrainers
        self._generate_subtrainers()

        # Set up the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._init_internals()

        # Init the communication queues
        self._communication_queues = []
        for _ in range(self.num_subtrainers):
            self._communication_queues.append(Queue())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the subtrainers and communication queues.
        """
        super()._reset_internals()
        self._subtrainers = None
        self._communication_queues = None

    def __copy__(self) -> DistributedTrainer:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.DistributedTrainer
        """
        cls = self.__class__
        result = cls(
            self.fitness_function,
            self.subtrainer_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> DistributedTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.DistributedTrainer
        """
        cls = self.__class__
        result = cls(
            self.fitness_function,
            self.subtrainer_cls
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__,
                (
                    self.fitness_function,
                    self.subtrainer_cls
                ),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> DistributedTrainer:
        """Return a distributed trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.abc.DistributedTrainer
        """
        obj = cls(
            state['_fitness_function'],
            state['_subtrainer_cls']
        )
        obj.__setstate__(state)
        return obj


class SequentialDistributedTrainer(DistributedTrainer):
    """Abstract base class for sequential distributed trainers."""

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._new_state` method
        of each subtrainer.
        """
        super()._new_state()

        # Generate the state of all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._new_state()
            self._num_evals += subtrainer._num_evals

    def _load_state(self) -> None:
        """Load the state of the last checkpoint.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._load_state` method
        of each subtrainer.

        :raises Exception: If the checkpoint file can't be loaded
        """
        # Load the state of this trainer
        super()._load_state()

        # Load the subtrainers' state
        for subtrainer in self.subtrainers:
            subtrainer._load_state()

    def _save_state(self) -> None:
        """Save the state at a new checkpoint.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._save_state` method
        of each subtrainer.

        :raises Exception: If the checkpoint file can't be written
        """
        # Save the state of this trainer
        super()._save_state()

        # Save the state of all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._save_state()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the metrics before each iteration is run. Overridden to call
        also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._start_iteration`
        method of each subtrainer.

        """
        super()._start_iteration()
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            # Fix the current iteration
            subtrainer._current_iter = self._current_iter
            # Start the iteration
            subtrainer._start_iteration()

    def _preprocess_iteration(self) -> None:
        """Preprocess the iteration of all the subtrainers.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._preprocess_iteration`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._preprocess_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._do_iteration`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._do_iteration_stats`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration_stats()

    def _postprocess_iteration(self) -> None:
        """Postprocess the iteration of all the subtrainers.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._postprocess_iteration`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._postprocess_iteration()

    def _finish_iteration(self) -> None:
        """Finish an iteration.

        Gather the metrics after each iteration is run. Overridden to call
        also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._finish_iteration`
        method of each subtrainer and accumulate the current number
        of evaluations of all the subtrainers.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:

            # Finish the iteration of all the subtrainers
            subtrainer._finish_iteration()
            # Get the number of evaluations
            self._current_iter_evals += subtrainer._current_iter_evals

        # Finish the iteration
        super()._finish_iteration()

    def _finish_search(self) -> None:
        """Finish the search process.

        Overridden to call also the
        :meth:`~culebra.trainer.abc.SingleSpeciesTrainer._finish_search`
        method of each subtrainer.
        """
        for subtrainer in self.subtrainers:
            # Fix the current iteration
            subtrainer._current_iter = self._current_iter

        # Finish the iteration
        super()._finish_search()


class ParallelDistributedTrainer(DistributedTrainer):
    """Abstract base class for parallel distributed trainers."""

    @property
    def _default_num_subtrainers(self) -> int:
        """Default number of subtrainers.

        :return: The number of CPU cores
        :rtype: int
        """
        return cpu_count()

    @property
    def current_iter(self) -> int | None:
        """Current iteration.

        :return: The current iteration or :data:`None` if the search has
            not been done yet
        :rtype: int
        """
        if self.subtrainers is not None:
            return self.subtrainers[0].current_iter
        return self._current_iter

    @property
    def num_evals(self) -> int | None:
        """Number of evaluations performed while training.

        :return: The number of evaluations or :data:`None` if the search has
            not been done yet
        :rtype: int
        """
        n_evals = None

        if self._num_evals is not None:
            n_evals = self._num_evals
        elif self.subtrainers is not None:
            n_evals = 0
            for subtrainer in self.subtrainers:
                if subtrainer.num_evals is not None:
                    n_evals += subtrainer.num_evals

            if self._search_finished:
                self._num_evals = n_evals

        return n_evals

    @property
    def runtime(self) -> float | None:
        """Training runtime.

        :return: The training runtime or :data:`None` if the search has not
            been done yet
        :rtype: float
        """
        the_runtime = None

        if self._runtime is not None:
            the_runtime = self._runtime
        elif self.subtrainers is not None:
            the_runtime = 0
            for subtrainer in self.subtrainers:
                if (
                        subtrainer.runtime is not None
                        and subtrainer.runtime > the_runtime
                ):
                    the_runtime = subtrainer.runtime

            if self._search_finished:
                self._runtime = the_runtime

        return the_runtime

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to set the overall runtime and number of evaluations to
        :data:`None`, since their final values will be generated from the
        subtrainers' state, once the trainer has finished.
        """
        super()._new_state()

        # The runtime and number of evaluations will be generated
        # from the subtrainers
        self._runtime = None
        self._num_evals = None

        # Each subtrainer handles its own current iteration
        self._current_iter = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Overridden to create a multiprocess manager and proxies to
        communicate with the processes running the subtrainers.
        """
        super()._init_internals()

        # Initialize the manager to receive the subtrainers' state
        self._manager = Manager()
        self._subtrainer_state_proxies = []
        for _ in range(self.num_subtrainers):
            self._subtrainer_state_proxies.append(self._manager.dict())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the multiprocess manager and proxies.
        """
        super()._reset_internals()
        self._manager = None
        self._subtrainer_state_proxies = None

    def _search(self) -> None:
        """Apply the search algorithm.

        Each subtrainer runs in a different process.
        """
        # Run all the iterations
        for subtrainer, state_proxy, suffix in zip(
                self.subtrainers,
                self._subtrainer_state_proxies,
                self._subtrainer_suffixes):
            subtrainer.process = Process(
                target=subtrainer.train,
                kwargs={"state_proxy": state_proxy},
                name=f"subtrainer_{suffix}",
                daemon=True
            )
            subtrainer.process.start()

        for subtrainer, state_proxy in zip(
                self.subtrainers, self._subtrainer_state_proxies):
            subtrainer.process.join()
            subtrainer._set_state(state_proxy)


class IslandsTrainer(SingleSpeciesTrainer, DistributedTrainer):
    """Abstract island-based trainer."""

    def __init__(
        self,
        solution_cls: type[Solution],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SingleSpeciesTrainer],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[SingleSpeciesTrainer], bool] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Solution], Any], Solution] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.abc.Solution]
        :param species: The species for all the solutions
        :type species: ~culebra.abc.Species
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-species trainer class to handle
            the subtrainers (islands).
        :type subtrainer_cls: type[~culebra.trainer.abc.SingleSpeciesTrainer]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.abc.IslandsTrainer._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param num_subtrainers: The number of subtrainers (islands). If
            omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative solutions that
            will be sent to the other subtrainers. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subtrainer (island). If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.abc.IslandsTrainer._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subtrainers
            (islands) trainer
        :type subtrainer_params: dict
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        SingleSpeciesTrainer.__init__(
            self,
            solution_cls=solution_cls,
            species=species,
            fitness_function=fitness_function
        )

        DistributedTrainer.__init__(
            self,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=representation_topology_func_params,
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    @property
    def _default_representation_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Default topology function.

        :return: :attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC

    @property
    def _default_representation_topology_func_params(self) -> dict[str, Any]:
        """Default parameters for the default topology function.

        :return: :attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        :rtype: dict
        """
        return DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS

    def __copy__(self) -> IslandsTrainer:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.IslandsTrainer
        """
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.subtrainer_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> IslandsTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.IslandsTrainer
        """
        cls = self.__class__
        result = cls(
            self.solution_cls,
            self.species,
            self.fitness_function,
            self.subtrainer_cls
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__,
                (
                    self.solution_cls,
                    self.species,
                    self.fitness_function,
                    self.subtrainer_cls
                ),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> IslandsTrainer:
        """Return an islands-based trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.abc.IslandsTrainer
        """
        obj = cls(
            state['_solution_cls'],
            state['_species'],
            state['_fitness_function'],
            state['_subtrainer_cls']
        )
        obj.__setstate__(state)
        return obj


class MultiSpeciesTrainer(Trainer):
    """Base class for trainers that find solutions for multiple species."""

    def __init__(
        self,
        solution_classes: Sequence[type[Solution]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        max_num_iters: int | None = None,
        custom_termination_func: Callable[[Trainer], bool] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param solution_classes: The solution class for each species.
        :type solution_classes:
            ~collections.abc.Sequence[type[~culebra.abc.Solution]]
        :param species: The species to be evolved
        :type species: ~collections.abc.Sequence[~culebra.abc.Species]
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.abc.MultiSpeciesTrainer._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.abc.MultiSpeciesTrainer._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.abc.MultiSpeciesTrainer._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.abc.MultiSpeciesTrainer._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.abc.MultiSpeciesTrainer._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.abc.MultiSpeciesTrainer._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        Trainer.__init__(
            self,
            fitness_function=fitness_function,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed
        )

        # Get the parameters
        self.solution_classes = solution_classes
        self.species = species

    @property
    def solution_classes(self) -> tuple[type[Solution]]:
        """Solution classes.

        :rtype: tuple[~culebra.abc.Solution]
        :setter: Set the new solution classes
        :param classes: The classes.
        :type classes: ~collections.abc.Sequence[~culebra.abc.Solution]
        :raises TypeError: If *classes* is not a
            :class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *classes* is not a
            :class:`~culebra.abc.Solution` subclass
        """
        return self._solution_classes

    @solution_classes.setter
    def solution_classes(self, classes: Sequence[type[Solution]]) -> None:
        """Set the new solution classes.

        :param classes: The classes.
        :type classes: ~collections.abc.Sequence[~culebra.abc.Solution]
        :raises TypeError: If *classes* is not a
            :class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *classes* is not a
            :class:`~culebra.abc.Solution` subclass
        """
        self._solution_classes = tuple(
            check_sequence(
                classes,
                "solution classes",
                item_checker=partial(check_subclass, cls=Solution)
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def species(self) -> tuple[Species]:
        """Species for each subtrainer.

        :rtype: tuple[~culebra.abc.Species]
        :setter: Set the new species
        :param value: The species
        :type value: ~collections.abc.Sequence[~culebra.abc.Species]
        :raises TypeError: If *value* is not a
            :class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *value* is not a valid species
        """
        return self._species

    @species.setter
    def species(self, value: Sequence[Species]) -> None:
        """Set the new species.

        :param value: The species
        :type value: ~collections.abc.Sequence[~culebra.abc.Species]
        :raises TypeError: If *value* is not a
            :class:`~collections.abc.Sequence`
        :raises ValueError: If any element of *value* is not a
            :class:`~culebra.abc.Species`
        """
        self._species = tuple(
            check_sequence(
                value,
                "species",
                item_checker=partial(check_instance, cls=Species)
            )
        )

        # Reset the algorithm
        self.reset()

    def __copy__(self) -> MultiSpeciesTrainer:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.MultiSpeciesTrainer
        """
        cls = self.__class__
        result = cls(
            self.solution_classes, self.species, self.fitness_function
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> MultiSpeciesTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.MultiSpeciesTrainer
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
        :rtype: tuple
        """
        return (self.__class__,
                (self.solution_classes, self.species, self.fitness_function),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> MultiSpeciesTrainer:
        """Return a single species trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.abc.MultiSpeciesTrainer
        """
        obj = cls(
            state['_solution_classes'],
            state['_species'],
            state['_fitness_function']
        )
        obj.__setstate__(state)
        return obj


class CooperativeTrainer(MultiSpeciesTrainer, DistributedTrainer):
    """Abstract cooperative trainer model."""

    def __init__(
        self,
        solution_classes: Sequence[type[Solution]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        subtrainer_cls: type[SingleSpeciesTrainer],
        max_num_iters: int | None = None,
        custom_termination_func:
            Callable[[SingleSpeciesTrainer], bool] | None = None,
        num_subtrainers: int | None = None,
        representation_size: int | None = None,
        representation_freq: int | None = None,
        representation_topology_func:
            Callable[[int, int, Any], list[int]] | None = None,
        representation_topology_func_params: dict[str, Any] | None = None,
        representation_selection_func:
            Callable[[list[Solution], Any], Solution] | None = None,
        representation_selection_func_params: dict[str, Any] | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_filename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None,
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        Each species is evolved in a different subtrainer.

        :param solution_classes: The individual class for each species
        :type solution_classes:
            ~collections.abc.Sequence[type[~culebra.abc.Solution]]
        :param species: The species to be evolved
        :type species: ~collections.abc.Sequence[~culebra.abc.Species]
        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param subtrainer_cls: Single-species trainer class to handle
            the subtrainers.
        :type subtrainer_cls: type[~culebra.trainer.abc.SingleSpeciesTrainer]
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If
            omitted,
            :meth:`~culebra.trainer.abc.CooperativeTrainer._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :type custom_termination_func: ~collections.abc.Callable
        :param num_subtrainers: The number of subtrainers (species). If
            omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_num_subtrainers`
            will be used. Defaults to :data:`None`
        :type num_subtrainers: int
        :param representation_size: Number of representative individuals that
            will be sent to the other subtrainers. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_representation_size`
            will be used. Defaults to :data:`None`
        :type representation_size: int
        :param representation_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_representation_freq`
            will be used. Defaults to :data:`None`
        :type representation_freq: int
        :param representation_topology_func: Topology function for
            representatives sending. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_representation_topology_func`
            will be used. Defaults to :data:`None`
        :type representation_topology_func: ~collections.abc.Callable
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_representation_topology_func_params`
            will be used. Defaults to :data:`None`
        :type representation_topology_func_params: dict
        :param representation_selection_func: Policy function to choose the
            representatives from each subtrainer (species). If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_representation_selection_func`
            will be used. Defaults to :data:`None`
        :type representation_selection_func: ~collections.abc.Callable
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_representation_selection_func_params`
            will be used. Defaults to :data:`None`
        :type representation_selection_func_params: dict
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_checkpoint_filename`
            will be used. Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :param subtrainer_params: Custom parameters for the subtrainers
            (species) trainer
        :type subtrainer_params: dict
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        MultiSpeciesTrainer.__init__(
            self,
            solution_classes=solution_classes,
            species=species,
            fitness_function=fitness_function
        )

        DistributedTrainer.__init__(
            self,
            fitness_function=fitness_function,
            subtrainer_cls=subtrainer_cls,
            max_num_iters=max_num_iters,
            custom_termination_func=custom_termination_func,
            num_subtrainers=num_subtrainers,
            representation_size=representation_size,
            representation_freq=representation_freq,
            representation_topology_func=representation_topology_func,
            representation_topology_func_params=representation_topology_func_params,
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_activation=checkpoint_activation,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbosity=verbosity,
            random_seed=random_seed,
            **subtrainer_params
        )

    @property
    def _default_representation_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Default representation topology function.

        :return: :attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC

    @property
    def _default_representation_topology_func_params(self) -> dict[str, Any]:
        """Default parameters of the representation topology function.

        :return: :attr:`~culebra.trainer.DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
        :rtype: dict
        """
        return DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS

    @property
    def representatives(self) -> list[list[Solution | None]] | None:
        """Representatives of the other species.

        :rtype: list[list[~culebra.abc.Solution]]
        """
        # Default value
        the_representatives = None

        # If the representatives have been gathered before
        if self._representatives is not None:
            the_representatives = self._representatives
        elif self.subtrainers is not None:
            # Create the container for the representatives
            the_representatives = [
                [None] * self.num_subtrainers
                for _ in range(self.representation_size)
            ]
            for (
                    subtrainer_index,
                    subtrainer
                    ) in enumerate(self.subtrainers):

                if subtrainer.representatives is None:
                    the_representatives = None
                    break

                for (context_index,
                     _
                     ) in enumerate(subtrainer.representatives):
                    the_representatives[
                        context_index][
                            subtrainer_index - 1
                    ] = subtrainer.representatives[
                            context_index][
                                subtrainer_index - 1
                    ]

            if self._search_finished is True:
                self._representatives = the_representatives

        return the_representatives

    @property
    def _default_num_subtrainers(self) -> int:
        """Default number of subtrainers.

        :return: The number of species
        :rtype: int
        """
        return len(self.species)

    @DistributedTrainer.num_subtrainers.setter
    def num_subtrainers(self, value: int | None) -> None:
        """Set a new value for the number of subtrainers.

        :param value: The new number of subtrainers. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_num_subtrainers`
            is chosen. Otherwise *it must match*
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_num_subtrainers`
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* does not match
            :attr:`~culebra.trainer.abc.CooperativeTrainer._default_num_subtrainers`
        """
        DistributedTrainer.num_subtrainers.fset(self, value)
        if (
            self._num_subtrainers is not None and
            self._num_subtrainers != len(self.species)
        ):
            raise ValueError(
                "The number of subtrainers must match the number of "
                f"species: {self.species}"
            )

    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        # List of hofs
        hofs = []
        # If the search hasn't been initialized a list of empty hofs is
        # returned, one hof per species
        if self.subtrainers is None:
            for _ in range(self.num_subtrainers):
                hofs.append(ParetoFront())
        # Else, the best solutions of each species are returned
        else:
            for subtrainer in self.subtrainers:
                hofs.append(subtrainer.best_solutions()[0])

        return tuple(hofs)

    def best_representatives(self) -> list[list[Solution]] | None:
        """Return a list of representatives from each species.

        :return: A list of representatives lists. One representatives list for
            each one of the evolved species or :data:`None` if the search
            has not finished
        :rtype: list[list[~culebra.abc.Solution]]
        """
        # Check if the trianing has finished
        # self._search_finished could be None or False...
        if self._search_finished is not True:
            the_representatives = None
        else:
            # Create the container for the representatives
            the_representatives = [
                [None] * self.num_subtrainers
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
    def _init_subtrainer_representatives(
        subtrainer: SingleSpeciesTrainer,
        solution_classes: Sequence[type[Solution]],
        species: Sequence[Species],
        representation_size: int
    ):
        """Init the representatives of the other species for a subtrainer.

        This method is used to override dynamically the
        :meth:`~culebra.abc.Trainer._init_representatives` of all the
        subtrainers, when they are generated with the
        :meth:`~culebra.trainer.abc.CooperativeTrainer._generate_subtrainers`
        method, to let them initialize the list of representative individuals
        of the other species

        :param subtrainer: The subtrainer. The representatives from the
            remaining subtrainers will be initialized for this subtrainer
        :type subtrainer: ~culebra.trainer.abc.SingleSpeciesTrainer
        :param solution_classes: The individual class for each species.
        :type solution_classes:
            ~collections.abc.Sequence[type[~culebra.abc.Solution]]
        :param species: The species to be evolved by this trainer
        :type species: ~collections.abc.Sequence[~culebra.abc.Species]
        :param representation_size: Number of representative individuals
            generated for each species
        :type representation_size: int
        """
        subtrainer._representatives = []

        for _ in range(representation_size):
            subtrainer._representatives.append(
                [
                    ind_cls(
                        spe, subtrainer.fitness_function.fitness_cls
                    ) if i != subtrainer.index else None
                    for i, (ind_cls, spe) in enumerate(
                        zip(solution_classes, species)
                    )
                ]
            )

    def __copy__(self) -> CooperativeTrainer:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.CooperativeTrainer
        """
        cls = self.__class__
        result = cls(
            self.solution_classes,
            self.species,
            self.fitness_function,
            self.subtrainer_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> CooperativeTrainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.CooperativeTrainer
        """
        cls = self.__class__
        result = cls(
            self.solution_classes,
            self.species,
            self.fitness_function,
            self.subtrainer_cls
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__,
                (
                    self.solution_classes,
                    self.species,
                    self.fitness_function,
                    self.subtrainer_cls
                ),
                self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> CooperativeTrainer:
        """Return a cooperative trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.abc.CooperativeTrainer
        """
        obj = cls(
            state['_solution_classes'],
            state['_species'],
            state['_fitness_function'],
            state['_subtrainer_cls']
        )
        obj.__setstate__(state)
        return obj


# Exported symbols for this module
__all__ = [
    'SingleSpeciesTrainer',
    'DistributedTrainer',
    'SequentialDistributedTrainer',
    'ParallelDistributedTrainer',
    'IslandsTrainer',
    'MultiSpeciesTrainer',
    'CooperativeTrainer'
]
