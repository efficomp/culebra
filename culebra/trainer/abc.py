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

The :class:`~culebra.trainer.abc.CentralizedTrainer` class supports the
development of a trainer to solve a problem related to a single species. On the
other hand, the :class:`~culebra.trainer.abc.DistributedTrainer` class provides
a base support to distribute a trainer making use a several centralized
subtrainers. Two implementations of this class are also provided:

* :class:`~culebra.trainer.abc.CommonFitnessFunctionDistributedTrainer`:
  Abstract base class for distributed trainers whose subtrainers share the
  same fitness function
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
from collections.abc import Sequence, Callable
import gzip
from functools import partial
from copy import deepcopy
from time import perf_counter
import random

import numpy as np
from multiprocess import (
    Queue,
    Process,
    Manager

)
from multiprocess.managers import DictProxy
import dill
from deap.tools import Logbook, Statistics, HallOfFame, ParetoFront

from culebra import DEFAULT_INDEX, SERIALIZED_FILE_EXTENSION
from culebra.abc import (
    Solution,
    Species,
    Trainer,
    FitnessFunction
)
from culebra.checker import (
    check_bool,
    check_int,
    check_instance,
    check_subclass,
    check_sequence,
    check_func,
    check_filename
)
from .constants import (
    DEFAULT_MAX_NUM_ITERS,
    DEFAULT_CHECKPOINT_ACTIVATION,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_BASENAME,
    DEFAULT_VERBOSITY,
    DEFAULT_NUM_REPRESENTATIVES,
    DEFAULT_COOPERATIVE_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_TOPOLOGY_FUNC,
    DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ,
    DEFAULT_REPRESENTATIVES_SELECTION_FUNC
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


ITER_OBJ_STATS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max,
}
"""Statistics calculated for each objective every iteration."""


class CentralizedTrainer(Trainer):
    """Base class for all the centralized trainers."""

    def __init__(
        self,
        fitness_func: FitnessFunction,
        solution_cls: type[Solution],
        species: Species,
        custom_termination_func:
            Callable[[CentralizedTrainer], bool] | None = None,
        max_num_iters: int | None = None,
        checkpoint_activation: bool | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_basename: str | None = None,
        verbosity: bool | None = None,
        random_seed: int | None = None
    ) -> None:
        """Create a new trainer.

        :param fitness_func: The training fitness function
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param solution_cls: The solution class
        :type solution_cls: type[~culebra.abc.Solution]
        :param species: The species for the solutions
        :type species: ~culebra.abc.Species
        :param custom_termination_func: Custom termination criterion. If
            omitted, :meth:`~culebra.trainer.abc.CentralizedTrainer._default_termination_func`
            is used. Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param max_num_iters: Maximum number of iterations. If omitted,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_max_num_iters`
            will be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param checkpoint_activation: Checkpoining activation. If omitted,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_activation`
            will be used. Defaults to :data:`None`
        :type checkpoint_activation: bool
        :param checkpoint_freq: The checkpoint frequency. If omitted,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_freq`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_basename: The checkpoint base file path. If omitted,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_basename`
            will be used. Defaults to :data:`None`
        :type checkpoint_basename: str
        :param verbosity: The verbosity. If omitted,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_verbosity`
            will be used. Defaults to :data:`None`
        :type verbosity: bool
        :param random_seed: The seed. Defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__()

        # Get the parameters
        self.fitness_func=fitness_func
        self.solution_cls = solution_cls
        self.species = species
        self.custom_termination_func = custom_termination_func
        self.max_num_iters = max_num_iters
        self.checkpoint_activation = checkpoint_activation
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_basename = checkpoint_basename
        self.verbosity = verbosity
        self.random_seed = random_seed

        # Index of this trainer
        self.index = None

        # Init the container
        # Only used in case of being used within a distributed configuration
        self.container = None

        # Init the state proxy
        # Only used when running within a parallel contaniner
        self.state_proxy = None

        # Init the representatives exchange functions
        # Only used in case of being used within a distributed configuration
        self.receive_representatives_func = None
        self.send_representatives_func = None

        # Init the state
        self._cooperators = None
        self._logbook = None
        self._current_iter = None
        self._num_evals = None
        self._runtime = None
        self._training_finished = False

        # Init the internals
        self._stats = None
        self._current_iter_evals = None
        self._current_iter_start_time = None

    @property
    def fitness_func(self) -> FitnessFunction:
        """Training fitness function.

        :rtype: ~culebra.abc.FitnessFunction

        :setter: Set a new fitness function
        :param value: The new training fitness function
        :type value: ~culebra.abc.FitnessFunction
        :raises TypeError: If *value* is not a valid fitness function
        """
        return self._fitness_func

    @fitness_func.setter
    def fitness_func(self, value: FitnessFunction) -> None:
        """Set a new training fitness function.

        :param value: The new training fitness function
        :type value: ~culebra.abc.FitnessFunction
        :raises TypeError: If *value* is not a valid fitness function
        """
        # Check the function
        self._fitness_func = check_instance(
            value, "fitness function", FitnessFunction
        )

        # Reset the trainer
        self.reset()

    @property
    def solution_cls(self) -> type[Solution]:
        """Solution class.

        :rtype: type[~culebra.abc.Solution]

        :setter: Set the new solution class
        :param value: The new class
        :type value: type[~culebra.abc.Solution]
        :raises TypeError: If *value* is not a :class:`~culebra.abc.Solution`
            subclass
        """
        return self._solution_cls

    @solution_cls.setter
    def solution_cls(self, value: type[Solution]) -> None:
        """Set a new solution class.

        :param value: The new class
        :type value: type[~culebra.abc.Solution]
        :raises TypeError: If *value* is not a :class:`~culebra.abc.Solution`
            subclass
        """
        self._solution_cls = check_subclass(value, "solution class", Solution)

        # Reset the trainer
        self.reset()

    @property
    def species(self) -> Species:
        """Species.

        :rtype: ~culebra.abc.Species

        :setter: Set the new species
        :param value: The new species
        :type value: ~culebra.abc.Species
        :raises TypeError: If *value* is not a :class:`~culebra.abc.Species`
        """
        return self._species

    @species.setter
    def species(self, value: Species) -> None:
        """Set a new species.

        :param value: The new species
        :type value: ~culebra.abc.Species
        :raises TypeError: If *value* is not a :class:`~culebra.abc.Species`
        """
        self._species = check_instance(value, "species", Species)

        # Reset the trainer
        self.reset()

    @property
    def custom_termination_func(self) -> Callable[
            [CentralizedTrainer],
            bool
    ] | None:
        """Custom termination criterion.

        Although the trainer will always stop when the
        :attr:`~culebra.trainer.abc.CentralizedTrainer.max_num_iters` are
        reached, a custom termination criterion can be set to detect
        convergente and stop the trainer earlier. This custom termination
        criterion must be a function which receives the trainer as its unique
        argument and returns a boolean value, :data:`True` if the training
        should terminate or :data:`False` otherwise.

        If more than one arguments are needed to define the termination
        condition, :func:`functools.partial` can be used:

        .. code-block:: python

            from functools import partial

            def my_crit(trainer, max_iters):
                if trainer.current_iter < max_iters:
                    return False
                return True

            trainer.custom_termination_func = partial(my_crit, max_iters=10)

        :setter: Set a new custom termination criterion
        :param func: The new custom termination criterion. If set to
            :data:`None`, the default termination criterion is used
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._custom_termination_func

    @custom_termination_func.setter
    def custom_termination_func(
        self,
        func: Callable[
                [CentralizedTrainer],
                bool
        ] | None
    ) -> None:
        """Set the custom termination criterion.

        :param func: The new custom termination criterion. If set to
            :data:`None`, the default termination criterion is used
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._custom_termination_func = (
            None if func is None else check_func(
                func, "custom termination criterion"
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_max_num_iters(self) -> int:
        """Default maximum number of iterations.

        :return: :attr:`~culebra.trainer.DEFAULT_MAX_NUM_ITERS`
        :rtype: int
        """
        return DEFAULT_MAX_NUM_ITERS

    @property
    def max_num_iters(self) -> int:
        """Maximum number of iterations.

        :rtype: int

        :setter: Set a new value for the maximum number of iterations
        :param value: The new maximum number of iterations. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_max_num_iters`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return (
            self._default_max_num_iters
            if self._max_num_iters is None
            else self._max_num_iters
        )

    @max_num_iters.setter
    def max_num_iters(self, value: int | None) -> None:
        """Set the maximum number of iterations.

        :param value: The new maximum number of iterations. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_max_num_iters`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._max_num_iters = (
            None if value is None else check_int(
                value, "maximum number of iterations", gt=0
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_checkpoint_activation(self) -> bool:
        """Default checkpointing activation.

        :return: :attr:`~culebra.trainer.DEFAULT_CHECKPOINT_ACTIVATION`
        :rtype: bool
        """
        return DEFAULT_CHECKPOINT_ACTIVATION

    @property
    def checkpoint_activation(self) -> bool:
        """Checkpointing activation.

        :return: :data:`True` if checkpointing is active, or :data:`False`
            otherwise
        :rtype: bool
        :setter: Modify the checkpointing activation
        :param value: New value for the checkpoint activation. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_activation`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not a boolean value
        """
        return (
            self._default_checkpoint_activation
            if self._checkpoint_activation is None
            else self._checkpoint_activation
        )

    @checkpoint_activation.setter
    def checkpoint_activation(self, value: bool | None) -> None:
        """Modify the checkpointing activation.

        :param value: New value for the checkpoint activation. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_activation`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not a boolean value
        """
        self._checkpoint_activation = (
            None if value is None else check_bool(
                value, "checkpoint activation"
            )
        )

    @property
    def _default_checkpoint_freq(self) -> int:
        """Default checkpointing frequency.

        :return: :attr:`~culebra.trainer.DEFAULT_CHECKPOINT_FREQ`
        :rtype: int
        """
        return DEFAULT_CHECKPOINT_FREQ

    @property
    def checkpoint_freq(self) -> int:
        """Checkpoint frequency.

        :rtype: int

        :setter: Modify the checkpoint frequency
        :param value: New value for the checkpoint frequency. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_freq`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return (
            self._default_checkpoint_freq
            if self._checkpoint_freq is None
            else self._checkpoint_freq
        )

    @checkpoint_freq.setter
    def checkpoint_freq(self, value: int | None) -> None:
        """Set a value for the checkpoint frequency.

        :param value: New value for the checkpoint frequency. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_freq`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._checkpoint_freq = (
            None if value is None else check_int(
                value, "checkpoint frequency", gt=0
            )
        )

    @property
    def _default_checkpoint_basename(self) -> str:
        """Default checkpointing base file name.

        :return: :attr:`~culebra.trainer.DEFAULT_CHECKPOINT_BASENAME`
        :rtype: str
        """
        return DEFAULT_CHECKPOINT_BASENAME

    @property
    def checkpoint_basename(self) -> str:
        """Checkpoint base file path.

        :rtype: str

        :setter: Modify the checkpoint base file path
        :param value: New value for the checkpoint base file path. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_basename`
            is chosen
        :type value: str
        :raises TypeError: If *value* is not a valid file name
        """
        return (
            self._default_checkpoint_basename
            if self._checkpoint_basename is None
            else self._checkpoint_basename
        )

    @checkpoint_basename.setter
    def checkpoint_basename(self, value: str | None) -> None:
        """Set a value for the checkpoint base file path.

        :param value: New value for the checkpoint base file path. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_checkpoint_basename`
            is chosen
        :type value: str
        :raises TypeError: If *value* is not a valid file name
        """
        # Check the value
        self._checkpoint_basename = (
            None if value is None else check_filename(
                value,
                name="checkpoint base file name"
            )
        )

    @property
    def _default_verbosity(self) -> bool:
        """Default verbosity.

        :return: :attr:`~culebra.trainer.DEFAULT_VERBOSITY`
        :rtype: bool
        """
        return DEFAULT_VERBOSITY

    @property
    def verbosity(self) -> bool:
        """Verbosity of this trainer.

        :rtype: bool

        :setter: Set a new value for the verbosity
        :param value: The verbosity. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_verbosity`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not boolean
        """
        return (
            self._default_verbosity
            if self._verbosity is None
            else self._verbosity
        )

    @verbosity.setter
    def verbosity(self, value: bool) -> None:
        """Set the verbosity of this trainer.

        :param value: The verbosity. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_verbosity`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not boolean
        """
        self._verbosity = (
            None if value is None else check_bool(value, "verbosity")
        )

    @property
    def random_seed(self) -> int:
        """Random seed used by this trainer.

        :rtype: int
        :setter: Set a new value for the random seed
        :param value: New value
        :type value: int
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        """Set the random seed for this trainer.

        :param value: Random seed for the random generator
        :type value: int
        """
        self._random_seed = value
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)

        # Reset the trainer
        self.reset()

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint file path.

        :return: The file path to store checkpoints
        :rtype: str
        """
        suffix = ""
        if self.container:
            suffix_len = len(str(self.container.num_subtrainers - 1))
            suffix = f"_{self.index:0{suffix_len}d}"

        return self.checkpoint_basename + suffix + SERIALIZED_FILE_EXTENSION

    @property
    def _default_index(self) -> int:
        """Default index.

        :return: :attr:`~culebra.DEFAULT_INDEX`
        :rtype: int
        """
        return DEFAULT_INDEX

    @property
    def index(self) -> int:
        """Trainer index.

        The trainer index is only used within distributed trainers.

        :rtype: int
        :setter: Set a new value for trainer index.
        :param value: New value for the trainer index. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_index` is
            chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is a negative number
        """
        return (
            self._default_index if self._index is None else self._index
        )

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a value for trainer index.

        :param value: New value for the trainer index. If set to
            :data:`None`,
            :attr:`~culebra.trainer.abc.CentralizedTrainer._default_index` is
            chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is a negative number
        """
        # Check the value
        self._index = (
            None if value is None else check_int(value, "index", ge=0)
        )

    @property
    def container(self) -> DistributedTrainer | None:
        """Container of this trainer.

        The trainer container is only used by distributed trainers. For the
        rest of trainers defaults to :data:`None`.

        :rtype: ~culebra.trainer.abc.DistributedTrainer
        :setter: Set a new value for container of this trainer
        :param value: New value for the container or :data:`None`
        :type value: ~culebra.trainer.abc.DistributedTrainer
        :raises TypeError: If *value* is not a valid trainer
        """
        return self._container

    @container.setter
    def container(self, value: DistributedTrainer | None) -> None:
        """Set a container for this trainer.

        The trainer container is only used by distributed trainers. For the
        rest of trainers defaults to :data:`None`.

        :param value: New value for the container or :data:`None`
        :type value: ~culebra.trainer.abc.DistributedTrainer
        :raises TypeError: If *value* is not a valid trainer
        """
        # Check the value
        self._container = (
            None if value is None else check_instance(
                value, "container", cls=DistributedTrainer
            )
        )

    @property
    def state_proxy(self) -> DictProxy | None:
        """Proxy for the state of this trainer.

        The proxy is used to copy the state of the trainer only when training
        is executed within a :class:`multiprocess.Process`. Defaults to
        :data:`None`

        :rtype: ~multiprocess.managers.DictProxy
        :setter: Set a new value for the state proxy of this trainer
        :param value: New value for the state proxy or :data:`None`
        :type value: ~multiprocess.managers.DictProxy
        :raises TypeError: If *value* is not a valid proxy
        """
        return self._state_proxy

    @state_proxy.setter
    def state_proxy(self, value: DictProxy | None = None) -> None:
        """Set a state proxy for this trainer.

        The proxy is used to copy the state of the trainer only when training
        is executed within a :class:`multiprocess.Process`. Defaults to
        :data:`None`

        :param value: New value for the state proxy or :data:`None`
        :type value: ~multiprocess.managers.DictProxy
        :raises TypeError: If *value* is not a valid proxy
        """
        # Check the value
        self._state_proxy = (
            None if value is None else check_instance(
                value, "state proxy", cls=DictProxy
            )
        )

    @property
    def _default_receive_representatives_func(
        self
    ) -> Callable[[CentralizedTrainer], None]:
        """Default implementation for the representatives reception function.

        It does nothing.
        """
        return lambda *a, **k: None

    @property
    def receive_representatives_func(
        self
    ) -> Callable[[CentralizedTrainer], None]:
        """Representatives reception function.

        Distributed trainers should set this property to their subtrainers
        in order to implement an adequate representatives exchange mechanism.

        :rtype: ~collections.abc.Callable

        :setter: Set a new function
        :param func: The new function
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not a valid function
        """
        return (
            self._default_receive_representatives_func
            if self._receive_representatives_func is None
            else self._receive_representatives_func
        )

    @receive_representatives_func.setter
    def receive_representatives_func(
        self, func: Callable[[CentralizedTrainer], None]
    ) -> None:
        """Set a new representatives reception function.

        Distributed trainers should set this property to their subtrainers
        in order to implement an adequate representatives exchange mechanism.

        :param func: The new function
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not a valid function
        """
        self._receive_representatives_func = (
            None if func is None else
            check_func(
                func, "representatives reception function"
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_send_representatives_func(
        self
    ) -> Callable[[CentralizedTrainer], None]:
        """Default implementation for the representatives sending function.

        It does nothing.
        """
        return lambda *a, **k: None

    @property
    def send_representatives_func(
        self
    ) -> Callable[[CentralizedTrainer], None]:
        """Representatives sending function.

        Distributed trainers should set this property to their subtrainers
        in order to implement an adequate representatives exchange mechanism.

        :rtype: ~collections.abc.Callable

        :setter: Set a new function
        :param func: The new function
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not a valid function
        """
        return (
            self._default_send_representatives_func
            if self._send_representatives_func is None
            else self._send_representatives_func
        )

    @send_representatives_func.setter
    def send_representatives_func(
        self, func: Callable[[CentralizedTrainer], None]
    ) -> None:
        """Set a new representatives sending function.

        Distributed trainers should set this property to their subtrainers
        in order to implement an adequate representatives exchange mechanism.

        :param func: The new function
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not a valid function
        """
        self._send_representatives_func = (
            None if func is None else
            check_func(
                func, "representatives sending function"
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def iteration_metric_names(self) -> tuple(str):
        """Names of the metrics recorded each iteration.

        :rtype: tuple[str]
        """
        return ('Iter',) + (
            () if self.container is None else ('SubTr',)
        ) + ('NEvals',)

    @property
    def iteration_obj_stats(self) -> dict(str, Callable):
        """Stats applied to each objective every iteration.

        :rtype: dict
        """
        return ITER_OBJ_STATS

    @property
    def training_finished(self) -> bool:
        """Check if training has finished.
        
        :return: :data`True` if training has finished
        :rtype: bool
        """
        return self._training_finished

    @property
    def logbook(self) -> Logbook | None:
        """Trainer logbook.

        :return: A logbook with the statistics of the training or :data:`None`
            if the training has not been done yet
        :rtype: ~deap.tools.Logbook
        """
        return self._logbook

    @property
    def current_iter(self) -> int | None:
        """Current iteration.

        :return: The current iteration or :data:`None` if the training has
            not been done yet
        :rtype: int
        """
        return self._current_iter

    @property
    def num_iters(self) -> int | None:
        """Number of iterations performed while training.

        :return: The number of iterations or :data:`None` if the training has
            not been done yet
        :rtype: int
        """
        return self.current_iter

    @property
    def num_evals(self) -> int | None:
        """Number of evaluations performed while training.

        :return: The number of evaluations or :data:`None` if the training has
            not been done yet
        :rtype: int
        """
        return self._num_evals

    @property
    def runtime(self) -> float | None:
        """Training runtime.

        :return: The training runtime or :data:`None` if the training has not
            been done yet.

        :rtype: float
        """
        return self._runtime

    @property
    def cooperators(self) -> list[list[Solution | None]] | None:
        """Cooperators of the other species.

        Only used by cooperative trainers. If the trainer does not use
        cooperators, :data:`None` is returned.

        :rtype: list[list[~culebra.abc.Solution]]
        """
        return self._cooperators

    def _generate_cooperators(
        self
    ) -> Sequence[Sequence[Solution | None]] | None:
        """Generate cooperators from other species.

        :return: The cooperators
        :rtype: ~collections.abc.Sequence[
            ~collections.abc.Sequence[~culebra.abc.Solution | None]] | None
        """
        the_cooperators = None
        if (self.container and isinstance(self.container, CooperativeTrainer)):
            # Create the cooperators list
            the_cooperators = [
                [None] * self.container.num_subtrainers
                for _ in range(self.container.num_representatives)
            ]

            # Fill the cooperators list
            for cooperator_idx in range(self.container.num_representatives):
                for subtr_idx, subtr in enumerate(self.container.subtrainers):
                    if subtr_idx != self.index:
                        the_cooperators[cooperator_idx][subtr_idx] = (
                            subtr.solution_cls(
                                subtr.species, subtr.fitness_func.fitness_cls
                            )
                        )

        return the_cooperators

    @abstractmethod
    def select_representatives(self) -> list[Solution]:
        """Select representative solutions.

        This abtract method is intended to be called within distributed
        trainers to make the implementation of migrations easier. It should be
        implemented by subclasses in order to achieve the desired behavior.

        :return: A list of solutions
        :rtype: list[~culebra.abc.Solution]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The select_representatives method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    @abstractmethod
    def integrate_representatives(
        self, representatives: list[Solution]
    ) -> None:
        """Integrate representative solutions.

        This abtract method is intended to be called within distributed
        trainers to make the implementation of migrations easier. It should be
        implemented by subclasses in order to achieve the desired behavior.

        :param representatives: A list of solutions
        :type representatives: list[~culebra.abc.Solution]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The integrate_representatives method has not been implemented in "
            f"the {self.__class__.__name__} class"
        )

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start training.

        Init the statistics and the number of evaluations and start time of
        the current iteration.
        """
        # Initialize statistics object
        self._stats = Statistics(key=lambda sol: sol.fitness.values)

        # Configure the stats
        for name, func in self.iteration_obj_stats.items():
            self._stats.register(name, func, axis=0)

        self._current_iter_evals = None
        self._current_iter_start_time = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Reset the statistics and the number of evaluations and start time of
        the current iteration.
        """
        self._stats = None
        self._current_iter_evals = None
        self._current_iter_start_time = None

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Init the cooperators, the logbook, and the current iteration,
        number of evaluations and runtime.
        """
        # The trainer hasn't trained yet
        self._training_finished = False

        # Init the cooperators
        self._cooperators = self._generate_cooperators()

        # Create a new logbook
        self._logbook = Logbook()

        # Init the logbook
        self._logbook.header = (
            list(self.iteration_metric_names) +
            (self._stats.fields if self._stats else [])
        )

        # Init the current iteration
        self._current_iter = 0

        # Init the number of evaluations
        self._num_evals = 0

        # Init the computing runtime
        self._runtime = 0

    def _reset_state(self) -> None:
        """Reset the trainer state.

        Reset the cooperators, the logbook, and the current iteration,
        number of evaluations and runtime.
        """
        self._training_finished = False

        self._cooperators = None
        self._logbook = None
        self._current_iter = None
        self._num_evals = None
        self._runtime = None

    def _get_state(self) -> dict[str, Any]:
        """Return the state of this trainer.

        Overridden to add the cooperators, the logbook, and the current
        number of evaluations, runtime, iteration and random state.

        :rtype: dict
        """
        return {
            "training_finished": self.training_finished,
            "cooperators": self.cooperators,
            "logbook": self.logbook,
            "num_evals": self._num_evals,
            "runtime": self._runtime,
            "current_iter": self._current_iter,
            "rnd_state": random.getstate(),
            "np_rnd_state": np.random.get_state(),
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        """Set the state of this trainer.

        Overridden to add the cooperators, the logbook, and the current
        number of evaluations, runtime, iteration and random state.

        :param state: The last loaded state
        :type state: dict
        """
        self._training_finished = state["training_finished"]
        self._cooperators = state["cooperators"]
        self._logbook = state["logbook"]
        self._num_evals = state["num_evals"]
        self._runtime = state["runtime"]
        self._current_iter = state["current_iter"]
        random.setstate(state["rnd_state"])
        np.random.set_state(state["np_rnd_state"])

    def _init_state(self) -> None:
        """Init the trainer state.

        If there is any checkpoint file, the state is initialized from it with
        the :meth:`~culebra.trainer.abc.CentralizedTrainer._load_state` method.
        Otherwise a new initial state is generated with the
        :meth:`~culebra.trainer.abc.CentralizedTrainer._new_state` method.
        """
        # Init the trainer state
        create_new_state = True
        if self.checkpoint_activation:
            # Try to load the state of the last checkpoint
            try:
                self._load_state()
                create_new_state = False
            # If a checkpoint can't be loaded, make initialization
            except Exception:
                pass

        if create_new_state:
            self._new_state()

    def _save_state(self) -> None:
        """Save the state at a new checkpoint.

        :raises Exception: If the checkpoint file can't be written
        """
        # Save the state
        with gzip.open(self.checkpoint_filename, 'wb') as f:
            dill.dump(self._get_state(), f)

    def _load_state(self) -> None:
        """Load the state of the last checkpoint.

        :raises Exception: If the checkpoint file can't be loaded
        """
        # Load the state
        with gzip.open(self.checkpoint_filename, 'rb') as f:
            self._set_state(dill.load(f))

    def reset(self) -> None:
        """Reset the trainer.

        Delete the state of the trainer (with
        :meth:`~culebra.trainer.abc.CentralizedTrainer._reset_state`) and also
        all the internal data structures needed to perform the training (with
        :meth:`~culebra.trainer.abc.CentralizedTrainer._reset_internals`).

        This method should be invoqued each time a hyper parameter is
        modified.
        """
        # Reset the trainer internals
        self._reset_internals()

        # Reset the trainer state
        self._reset_state()

    def _init_training(self) -> None:
        """Init the training process.

        Initialize the state of the trainer and all the internal data
        structures needed to perform the training.
        """
        # Init the trainer internals
        self._init_internals()

        # Init the state of the trainer
        self._init_state()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the iteration metrics (number of evaluations, execution time)
        before each iteration is run.
        """
        self._current_iter_evals = 0
        self._current_iter_start_time = perf_counter()

    @abstractmethod
    def _do_iteration(self) -> None:
        """Implement an iteration of the training process.

        This abstract method should be implemented by subclasses in order to
        implement the desired behavior.
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _do_iteration method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def _get_objective_stats(self) -> dict:
        """Gather the objective stats.

        This method should be implemented by subclasses in order to perform
        the adequate stats each iteration.

        :return: The stats
        :rtype: dict
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _get_objective_stats method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def _get_iteration_metrics(self) -> dict:
        """Collect the iteration metrics.

        :return: The metrics
        :rtype: dict
        """
        record = {}
        record['Iter'] = self._current_iter
        if self.container is not None:
            record['SubTr'] = self.index
        record['NEvals'] = self._current_iter_evals
        return record

    def _update_logbook(self) -> None:
        """Append the iteration dato to the logbook.

        If verbosity is activated, also output the log data to the console.
        """
        record = self._get_iteration_metrics()
        record.update(self._get_objective_stats())
        self.logbook.record(**record)
        if self.verbosity:
            print(self._logbook.stream)

    def _default_termination_func(self) -> bool:
        """Default termination criterion.

        :return: :data:`True`
            if :attr:`~culebra.trainer.abc.CentralizedTrainer.max_num_iters`
            iterations have been run
        :rtype: bool
        """
        if self.current_iter < self.max_num_iters:
            return False

        return True

    def _termination_criterion(self) -> bool:
        """Control the training termination.

        :return: :data:`True` if either the default termination criterion or
            a custom termination criterion is met. The default termination
            criterion is implemented by the
            :meth:`~culebra.trainer.abc.CentralizedTrainer._default_termination_func`
            method. Another custom termination criterion can be set with
            :attr:`~culebra.trainer.abc.CentralizedTrainer.custom_termination_func`
            method
        :rtype: bool
        """
        ret_val = False
        # Try the default termination
        if self._default_termination_func():
            ret_val = True
        # Try the custom termination criterion, if has been defined
        elif self.custom_termination_func is not None:
            ret_val = self.custom_termination_func(self)
            check_bool(ret_val, "custom termination func returned value")

        return ret_val

    def _finish_iteration(self) -> None:
        """Finish an iteration.

        Finish the iteration metrics (number of evaluations, execution time)
        after each iteration is run.
        """
        end_time = perf_counter()
        self._runtime += end_time - self._current_iter_start_time
        self._num_evals += self._current_iter_evals

        # Save the trainer state at each checkpoint
        if (self.checkpoint_activation and
                self._current_iter % self.checkpoint_freq == 0):
            self._save_state()

        # Increment the current iteration
        self._current_iter += 1

    def _do_training(self) -> None:
        """Apply the training algorithm.

        Execute the trainer until the termination condition is met. Each
        iteration is composed by the following steps:

        * :meth:`~culebra.trainer.abc.CentralizedTrainer._start_iteration`
        * :meth:`~culebra.trainer.abc.CentralizedTrainer.receive_representatives_func`
        * :meth:`~culebra.trainer.abc.CentralizedTrainer._do_iteration`
        * :meth:`~culebra.trainer.abc.CentralizedTrainer.send_representatives_func`
        * :meth:`~culebra.trainer.abc.CentralizedTrainer._finish_iteration`
        * :meth:`~culebra.trainer.abc.CentralizedTrainer._update_logbook`
        """
        # Run all the iterations
        while not self._termination_criterion():
            self._start_iteration()
            self.receive_representatives_func(self)
            self._do_iteration()
            self.send_representatives_func(self)
            self._update_logbook()
            self._finish_iteration()

    def _finish_training(self) -> None:
        """Finish the training process.

        This method is called after the training has finished. It can be
        overridden to perform any treatment of the solutions found.
        """
        self._training_finished = True

        # Save the last state
        if self.checkpoint_activation:
            self._save_state()

        # Update state proxy
        if self.state_proxy is not None:
            state = self._get_state()
            for key, val in state.items():
                self.state_proxy[key] = val

    def __copy__(self) -> Trainer:
        """Shallow copy the trainer.

        :return: The copied triner
        :rtype: ~culebra.trainer.abc.CentralizedTrainer
        """
        cls = self.__class__
        result = cls(self.fitness_func, self.solution_cls, self.species)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Trainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied triner
        :rtype: ~culebra.trainer.abc.CentralizedTrainer
        """
        cls = self.__class__
        kwargs = {
            "fitness_func": deepcopy(self.fitness_func, memo),
            "solution_cls": deepcopy(self.solution_cls),
            "species": deepcopy(self.species, memo)
            }
        result = cls(**kwargs)
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo | {
                    id(self.fitness_func): result.fitness_func,
                    id(self.solution_cls): result.solution_cls,
                    id(self.species): result.species
                }
            )
        )
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (
            self.__class__,
            (self.fitness_func, self.solution_cls, self.species),
            self.__dict__
        )

    @classmethod
    def __fromstate__(cls, state: dict) -> Trainer:
        """Return a trainer from a state.

        :param state: The state
        :type state: dict
        :return: The triner
        :rtype: ~culebra.trainer.abc.CentralizedTrainer
        """
        obj = cls(
            state['_fitness_func'],
            state['_solution_cls'],
            state['_species']
        )
        obj.__setstate__(state)
        return deepcopy(obj)


class DistributedTrainer(Trainer):
    """Base class for all the distributed trainers."""

    def __init__(
        self,
        *subtrainers: tuple[CentralizedTrainer],
        num_representatives: int | None = None,
        representatives_selection_func:
            Callable[[list[Solution], int], list[Solution]] | None = None,
        representatives_exchange_freq: int | None = None,
        topology_func: Callable[[int, int, Any], list[int]] | None = None
    ) -> None:
        """Create a new distributed trainer.

        :param subtrainers: The subtrainers
        :type subtrainers: tuple[~culebra.trainer.abc.CentralizedTrainer]
        :param num_representatives: Number of representative solutions that
            will be sent to the other subtrainers. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_num_representatives`
            will be used. Defaults to :data:`None`
        :type num_representatives: int
        :param representatives_selection_func: Policy function to choose the
            representatives from each subtrainer. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representatives_selection_func`
            will be used. Defaults to :data:`None`
        :type representatives_selection_func: ~collections.abc.Callable
        :param representatives_exchange_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representatives_exchange_freq`
            will be used. Defaults to :data:`None`
        :type representatives_exchange_freq: int
        :param topology_func: Topology function for representatives sending.
            If omitted,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_topology_func`
            will be used. Defaults to :data:`None`
        :type topology_func: ~collections.abc.Callable
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the subtrainer list
        self._subtrainers = tuple(
            check_sequence(
                subtrainers,
                "subtrainers",
                item_checker=partial(
                    check_instance,
                    cls=CentralizedTrainer
                )
            )
        )

        # Check the number of subtrainers
        if self.num_subtrainers < 2:
            raise ValueError("Less than two subtrainers")

        # Assign an index and a container to each subtrainer
        # Also set functions to send and receive representatives to each
        # subtrainer
        for idx, subtrainer in enumerate(self.subtrainers):
            subtrainer.index = idx
            subtrainer.container = self
            subtrainer.receive_representatives_func = (
                self.receive_representatives
            )
            subtrainer.send_representatives_func = (
                self.send_representatives
            )

        # Init the superclass
        super().__init__()

        # Get the parameters
        self.num_representatives = num_representatives
        self.representatives_selection_func = (
            representatives_selection_func
        )
        self.representatives_exchange_freq = representatives_exchange_freq
        self.topology_func = topology_func

        self._communication_queues = None

    @property
    def subtrainers(self) -> list[Trainer] | None:
        """Subtrainers.

        One single-species trainer for each subtrainer.

        :rtype: list[~culebra.trainer.abc.CentralizedTrainer]
        """
        return self._subtrainers

    @property
    def _default_num_representatives(self) -> int:
        """Default number of representatives sent to the other subtrainers.

        :return: :attr:`~culebra.trainer.DEFAULT_NUM_REPRESENTATIVES`
        :rtype: int
        """
        return DEFAULT_NUM_REPRESENTATIVES

    @property
    def num_representatives(self) -> int:
        """Number of representatives.

        :return: The number of representatives sent to the other subtrainers
        :rtype: int
        :setter: Set a new number of representatives
        :param value: The new value. If set to :data:`None`,
            :attr:`~~culebra.trainer.abc.DistributedTrainer._default_num_representatives`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is not positive
        """
        return (
            self._default_num_representatives
            if self._num_representatives is None
            else self._num_representatives
        )

    @num_representatives.setter
    def num_representatives(self, value: int | None) -> None:
        """Set a new number of representatives.

        :param value: The new value. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_num_representatives`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer number
        :raises ValueError: If *value* is not positive
        """
        # Check value
        self._num_representatives = (
            None if value is None else check_int(
                value, "number of representatives", gt=0
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_representatives_selection_func(
        self
    ) -> Callable[[list[Solution], int], list[Solution]]:
        """Default selection policy function to choose the representatives.

        :return:
                :attr:`~culebra.trainer.DEFAULT_REPRESENTATIVES_SELECTION_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_REPRESENTATIVES_SELECTION_FUNC

    @property
    def representatives_selection_func(
        self
    ) -> Callable[[list[Solution], int], list[Solution]]:
        """Representatives selection policy function.

        :return: A function that chooses which solutions are selected as
            representatives of each subtrainer
        :rtype: ~collections.abc.Callable
        :setter: Set new representatives selection policy function.
        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representatives_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return (
            self._default_representatives_selection_func
            if self._representatives_selection_func is None
            else self._representatives_selection_func
        )

    @representatives_selection_func.setter
    def representatives_selection_func(
        self,
        func: Callable[[list[Solution], int], list[Solution]] | None
    ) -> None:
        """Set new representatives selection policy function.

        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representatives_selection_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._representatives_selection_func = (
            None if func is None else check_func(
                func, "representatives selection policy function"
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def _default_representatives_exchange_freq(self) -> int:
        """Default number of iterations between representatives sending.

        :return: :attr:`~culebra.trainer.DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ`
        :rtype: int
        """
        return DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ

    @property
    def representatives_exchange_freq(self) -> int:
        """Number of iterations between representatives sendings.

        :rtype: int
        :setter: Set a new value for the frequency
        :param value: The new frequency. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representatives_exchange_freq`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return (
            self._default_representatives_exchange_freq
            if self._representatives_exchange_freq is None
            else self._representatives_exchange_freq
        )

    @representatives_exchange_freq.setter
    def representatives_exchange_freq(self, value: int | None) -> None:
        """Set the number of iterations between representatives sendings.

        :param value: The new frequency. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_representatives_exchange_freq`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._representatives_exchange_freq = (
            None if value is None else check_int(
                value, "representatives exchange frequency", gt=0
            )
        )

        # Reset the trainer
        self.reset()

    @property
    @abstractmethod
    def _default_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Default topology function.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: ~collections.abc.Callable
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The topology_func property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    def topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Topology function.

        :rtype: ~collections.abc.Callable
        :setter: Set new topology function
        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_topology_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return (
            self._default_topology_func
            if self._topology_func is None
            else self._topology_func
        )

    @topology_func.setter
    def topology_func(
        self,
        func: Callable[[int, int, Any], list[int]] | None
    ) -> None:
        """Set new topology function.

        :param func: The new function. If set to :data:`None`,
            :attr:`~culebra.trainer.abc.DistributedTrainer._default_topology_func`
            is chosen
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._topology_func = (
            None if func is None else check_func(
                func, "topology function"
            )
        )

        # Reset the trainer
        self.reset()

    @property
    def iteration_metric_names(self) -> tuple(str):
        """Names of the metrics recorded each iteration.

        :rtype: tuple[str]
        """
        return self.subtrainers[0].iteration_metric_names

    @property
    def iteration_obj_stats(self) -> dict(str, Callable):
        """Stats applied to each objective every iteration.

        :rtype: dict
        """
        return self.subtrainers[0].iteration_obj_stats

    @property
    def num_subtrainers(self) -> int:
        """Number of subtrainers.

        :rtype: int
        """
        return len(self.subtrainers)


    @property
    def training_finished(self) -> bool:
        """Check if training has finished.
        
        :return: :data`True` if training has finished
        :rtype: bool
        """
        for subtr in self.subtrainers:
            if not subtr.training_finished:
                return False

        return True

    @property
    def logbook(self) -> Logbook | None:
        """Trainer logbook.

        :return: A logbook with the statistics of the training or :data:`None`
            if the training has not been done yet
        :rtype: ~deap.tools.Logbook
        """
        the_logbook = None

        for subtr in self.subtrainers:
            if subtr.logbook is not None:
                if the_logbook is None:
                    # Create the logbook
                    the_logbook = Logbook()
                    # Init the logbook
                    the_logbook.header = subtr.logbook.header.copy()

                the_logbook.extend(subtr.logbook)

        return the_logbook

    @property
    def num_evals(self) -> int | None:
        """Number of evaluations performed while training.

        :return: The number of evaluations or :data:`None` if the training has
            not been done yet
        :rtype: int
        """
        n_evals = None

        for subtr in self.subtrainers:
            if subtr.num_evals is not None:
                if n_evals is None:
                    n_evals = 0
                n_evals += subtr.num_evals

        return n_evals

    @property
    def num_iters(self) -> int | None:
        """Number of iterations performed while training.

        :return: The number of iterations performed by the subtrainer that
            has performed more iterations or :data:`None` if the training has
            not been done yet
        :rtype: int
        """
        the_num_iters = None
        for subtr in self.subtrainers:
            if subtr.num_iters is not None:
                if the_num_iters is None or the_num_iters < subtr.num_iters:
                    the_num_iters = subtr.num_iters

        return the_num_iters

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start training.

        Create the subtrainer communication queues.
        """
        # Init the communication queues
        self._communication_queues = []
        for _ in range(self.num_subtrainers):
            self._communication_queues.append(Queue())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Reset the subtrainer communication queues.
        """
        self._communication_queues = None

    def reset(self) -> None:
        """Reset the trainer.

        Delete all the internal data structures needed to perform the training.
        """
        # Reset the trainer internals
        self._reset_internals()

    def _init_training(self) -> None:
        """Init the training process.

        Initialize all the internal data structures needed to perform the
        training.
        """
        # Init the trainer internals
        self._init_internals()

    def _finish_training(self) -> None:
        """Finish the training process.

        This method is called after the training has finished. It can be
        overridden to perform any treatment of the solutions found.
        """

    @staticmethod
    @abstractmethod
    def receive_representatives(subtrainer: CentralizedTrainer) -> None:
        """Receive representative solutions.

        This method must be overridden by subclasses.

        :param subtrainer: The subtrainer receiving representatives
        :type subtrainer: ~culebra.trainer.abc.CentralizedTrainer
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The receive_representatives method has not been implemented in "
            f"the {subtrainer.container.__class__.__name__} class")

    @staticmethod
    @abstractmethod
    def send_representatives(subtrainer: CentralizedTrainer) -> None:
        """Send representatives.

        This method must be overridden by subclasses.

        :param subtrainer: The sender subtrainer
        :type subtrainer: ~culebra.trainer.abc.CentralizedTrainer
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The send_representatives method has not been implemented in "
            f"the {subtrainer.container.__class__.__name__} class")

    def __copy__(self) -> DistributedTrainer:
        """Shallow copy the trainer.

        :return: The copied trainer
        :rtype: ~culebra.trainer.abc.DistributedTrainer
        """
        cls = self.__class__
        result = cls(*self.subtrainers)
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

        # Avoid a recursive copy of subtrainers
        subtrainers = deepcopy(
            self.subtrainers,
            memo | {id(self): None}
        )
        result = cls(*subtrainers)

        # Avoid to deepcopy subtrainers again
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo | {id(self.subtrainers): result.subtrainers}
            )
        )

        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, self.subtrainers, self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> DistributedTrainer:
        """Return a distributed trainer from a state.

        :param state: The state
        :type state: dict
        :return: The trainer
        :rtype: ~culebra.trainer.abc.DistributedTrainer
        """
        obj = cls(*state['_subtrainers'])
        obj.__setstate__(state)
        return deepcopy(obj)

class CommonFitnessFunctionDistributedTrainer(DistributedTrainer):
    """Base class for all the distributed trainers with a common training
    fitness function."""

    def __init__(
        self,
        *subtrainers: tuple[CentralizedTrainer],
        num_representatives: int | None = None,
        representatives_selection_func:
            Callable[[list[Solution], int], list[Solution]] | None = None,
        representatives_exchange_freq: int | None = None,
        topology_func: Callable[[int, int, Any], list[int]] | None = None
    ) -> None:
        """Create a new distributed trainer.

        :param subtrainers: The subtrainers
        :type subtrainers: tuple[~culebra.trainer.abc.CentralizedTrainer]
        :param num_representatives: Number of representative solutions that
            will be sent to the other subtrainers. If omitted,
            :attr:`~culebra.trainer.abc.CommonFitnessFunctionDistributedTrainer._default_num_representatives`
            will be used. Defaults to :data:`None`
        :type num_representatives: int
        :param representatives_selection_func: Policy function to choose the
            representatives from each subtrainer. If omitted,
            :attr:`~culebra.trainer.abc.CommonFitnessFunctionDistributedTrainer._default_representatives_selection_func`
            will be used. Defaults to :data:`None`
        :type representatives_selection_func: ~collections.abc.Callable
        :param representatives_exchange_freq: Number of iterations between
            representatives sendings. If omitted,
            :attr:`~culebra.trainer.abc.CommonFitnessFunctionDistributedTrainer._default_representatives_exchange_freq`
            will be used. Defaults to :data:`None`
        :type representatives_exchange_freq: int
        :param topology_func: Topology function for representatives sending.
            If omitted,
            :attr:`~culebra.trainer.abc.CommonFitnessFunctionDistributedTrainer._default_topology_func`
            will be used. Defaults to :data:`None`
        :type topology_func: ~collections.abc.Callable
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__(
            *subtrainers,
            num_representatives=num_representatives,
            representatives_selection_func=representatives_selection_func,
            representatives_exchange_freq=representatives_exchange_freq,
            topology_func=topology_func
        )

        # Check that al subtrainers use the same training fitness function
        for subtr in self.subtrainers:
            if subtr.fitness_func != self.fitness_func:
                raise ValueError(
                    "All subtrainers must use the same fitness function"
                )

    @property
    def fitness_func(self) -> FitnessFunction:
        """Training fitness function.

        :rtype: ~culebra.abc.FitnessFunction
        :rtype: tuple[str]
        """
        return self.subtrainers[0].fitness_func


class SequentialDistributedTrainer(DistributedTrainer):
    """Abstract base class for sequential distributed trainers."""

    @property
    def runtime(self) -> float | None:
        """Training runtime.

        :return: The training runtime or :data:`None` if the training has not
            been done yet
        :rtype: float
        """
        the_runtime = None

        for subtr in self.subtrainers:
            if subtr.runtime is not None:
                if the_runtime is None:
                    the_runtime = 0
                the_runtime += subtr.runtime

        return the_runtime

    def _init_training(self) -> None:
        """Init the training process.
        
        Overridden to init the training of subtrainers.
        """
        # Init the training
        super()._init_training()

        # Init the training of all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._init_training()

    def _finish_training(self) -> None:
        """Finish the training process.
        
        Overridden to finish the training of subtrainers.
        """
        # Finish the training of all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._finish_training()

        # Finish the training
        super()._finish_training()

    def _termination_criterion(self) -> bool:
        """Control the training termination.

        :return: :data:`True` if all the subtrainers have met their
            termination criteria
        :rtype: bool
        """
        for subtr in self.subtrainers:
            if not subtr._termination_criterion():
                return False

        return True

    def _do_training(self) -> None:
        """Apply the training algorithm."""
        # Run all the iterations
        while not self._termination_criterion():
            for subtr in self.subtrainers:
                if not subtr._termination_criterion():
                    subtr._start_iteration()
                    subtr.receive_representatives_func(subtr)
                    subtr._do_iteration()
                    subtr.send_representatives_func(subtr)
                    subtr._update_logbook()
                    subtr._finish_iteration()


class ParallelDistributedTrainer(DistributedTrainer):
    """Abstract base class for parallel distributed trainers."""

    @property
    def runtime(self) -> float | None:
        """Training runtime.

        :return: The training runtime or :data:`None` if the training has not
            been done yet
        :rtype: float
        """
        the_runtime = None

        for subtr in self.subtrainers:
            if subtr.runtime is not None:
                if the_runtime is None or subtr.runtime > the_runtime:
                    the_runtime = subtr.runtime

        return the_runtime

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start training.

        Overridden to create a multiprocess manager and proxies to
        communicate with the processes running the subtrainers.
        """
        super()._init_internals()

        # Initialize the manager to receive the subtrainers' state
        self._manager = Manager()
        for subtr in self.subtrainers:
            subtr.state_proxy = self._manager.dict()

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        Overridden to reset the multiprocess manager and proxies.
        """
        super()._reset_internals()
        self._manager = None
        for subtr in self.subtrainers:
            subtr.state_proxy = None

    def _do_training(self) -> None:
        """Apply the training algorithm.

        Each subtrainer runs in a different process.
        """
        # Run all the iterations
        for subtr_idx, subtr in enumerate(self.subtrainers):
            subtr.process = Process(
                target=subtr.train,
                name=f"subtrainer_{subtr_idx}",
                daemon=True
            )
            subtr.process.start()

        for subtr in self.subtrainers:
            subtr.process.join()
            subtr._set_state(subtr.state_proxy)


class IslandsTrainer(CommonFitnessFunctionDistributedTrainer):
    """Abstract island-based trainer."""

    @property
    def _default_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Default topology function.

        :return: :attr:`~culebra.trainer.DEFAULT_ISLANDS_TOPOLOGY_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_ISLANDS_TOPOLOGY_FUNC

    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        hof = ParetoFront()
        for subtrainer in self.subtrainers:
            hof.update(*subtrainer.best_solutions())

        return (hof,)

    @staticmethod
    def receive_representatives(subtrainer: CentralizedTrainer) -> None:
        """Receive representative solutions.

        :param subtrainer: The subtrainer receiving representatives
        :type subtrainer: ~culebra.trainer.abc.CentralizedTrainer
        """
        container = subtrainer.container

        # Receive all the solutions in the queue
        queue = container._communication_queues[subtrainer.index]
        while not queue.empty():
            subtrainer.integrate_representatives(queue.get())

    @staticmethod
    def send_representatives(subtrainer: CentralizedTrainer) -> None:
        """Send representatives.

        :param subtrainer: The sender subtrainer
        :type subtrainer: ~culebra.trainer.abc.CentralizedTrainer
        """
        container = subtrainer.container

        # Check if sending should be performed
        if (
            subtrainer.current_iter %
            container.representatives_exchange_freq == 0
        ):
            # Get the destinations according to the topology
            destinations = container.topology_func(
                subtrainer.index,
                container.num_subtrainers
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                sols = subtrainer.select_representatives()
                container._communication_queues[dest].put(sols)


class CooperativeTrainer(CommonFitnessFunctionDistributedTrainer):
    """Abstract cooperative trainer model."""

    @property
    def _default_topology_func(
        self
    ) -> Callable[[int, int, Any], list[int]]:
        """Default topology function.

        :return: :attr:`~culebra.trainer.DEFAULT_COOPERATIVE_TOPOLOGY_FUNC`
        :rtype: ~collections.abc.Callable
        """
        return DEFAULT_COOPERATIVE_TOPOLOGY_FUNC

    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        """
        # List of hofs
        hofs = []
        for subtrainer in self.subtrainers:
            hofs.append(subtrainer.best_solutions()[0])

        return tuple(hofs)

    def best_cooperators(self) -> list[list[Solution]] | None:
        """Return a list of cooperators from each species.

        :return: A list of cooperators lists. One cooperators list for
            each one of the evolved species or :data:`None` if the training
            has not finished
        :rtype: list[list[~culebra.abc.Solution]]
        """
        # Check if the trianing has finished
        if not self.training_finished:
            the_cooperators = None
        else:
            # Create the container for the cooperators
            the_cooperators = [
                [None] * self.num_subtrainers
                for _ in range(self.num_representatives)
            ]

            # Get the best solutions for each species
            best_ones = self.best_solutions()

            # Select the cooperators
            for species_index, hof in enumerate(best_ones):
                # Select a prrportional number of copies of each solution
                species_cooperators = []
                deck = []

                for _ in range(self.num_representatives):
                    if not deck:
                        deck = list(hof)
                        random.shuffle(deck)
                    species_cooperators.append(deck.pop())

                # Insert the representsatives
                for sol_index, sol in enumerate(species_cooperators):
                    the_cooperators[sol_index][species_index] = sol

        return the_cooperators

    @staticmethod
    def receive_representatives(subtrainer: CentralizedTrainer) -> None:
        """Receive representative solutions.

        :param subtrainer: The subtrainer receiving representatives
        :type subtrainer: ~culebra.trainer.abc.CentralizedTrainer
        """
        container = subtrainer.container

        # Receive all the solutions in the queue
        queue = container._communication_queues[subtrainer.index]

        anything_received = False
        while not queue.empty():
            msg = queue.get()
            sender_index = msg[0]
            representatives = msg[1]
            for sol_index, sol in enumerate(representatives):
                subtrainer.cooperators[sol_index][sender_index] = sol

            anything_received = True

        # If any new representatives have arrived, the fitness of all the
        # solutions in the population must be invalidated and solutions
        # must be re-evaluated
        if anything_received:
            # Re-evaluate all the solutions
            for sol in subtrainer.pop:
                subtrainer.evaluate(
                    sol,
                    subtrainer.fitness_func,
                    subtrainer.index,
                    subtrainer.cooperators
            )

    @staticmethod
    def send_representatives(subtrainer: CentralizedTrainer) -> None:
        """Send representatives.

        :param subtrainer: The sender subtrainer
        :type subtrainer: ~culebra.trainer.abc.CentralizedTrainer
        """
        container = subtrainer.container

        # Check if sending should be performed
        if (
            subtrainer.current_iter %
            container.representatives_exchange_freq == 0
        ):
            # Get the destinations according to the representation topology
            destinations = container.topology_func(
                subtrainer.index,
                container.num_subtrainers
            )

            # For each destination
            for dest in destinations:
                # Get the representatives
                sols = subtrainer.select_representatives()

                # Send the following msg:
                # (index of sender subpop, representatives)
                container._communication_queues[dest].put(
                    (subtrainer.index, sols)
                )


# Exported symbols for this module
__all__ = [
    'CentralizedTrainer',
    'DistributedTrainer',
    'CommonFitnessFunctionDistributedTrainer',
    'SequentialDistributedTrainer',
    'ParallelDistributedTrainer',
    'IslandsTrainer',
    'CooperativeTrainer'
]
