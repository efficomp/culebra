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

  * :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`: Provides a base
    class for trainers for solutions of only a single species
  * :py:class:`~culebra.trainer.abc.MultiSpeciesTrainer`: Provides a base
    class for trainers that find solutions for multiple species

Trainers can also be distributed. The
:py:class:`~culebra.trainer.abc.DistributedTrainer` class provides a base
support to distribute a trainer making use a several subtrainers. Two
implementations of this class are also provided:

  * :py:class:`~culebra.trainer.abc.SequentialDistributedTrainer`: Abstract
    base class for sequential distributed trainers
  * :py:class:`~culebra.trainer.abc.ParallelDistributedTrainer`: Abstract base
    class for parallel distributed trainers

Finally, some usual distributed approaches are also provided:

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
from multiprocessing import (
    cpu_count,
    Queue,
    Process,
    Manager)
from os import path

from deap.tools import Logbook, HallOfFame, ParetoFront

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

from .topology import full_connected_destinations
from .constants import (
    DEFAULT_NUM_SUBTRAINERS,
    DEFAULT_REPRESENTATION_SIZE,
    DEFAULT_REPRESENTATION_FREQ,
    DEFAULT_REPRESENTATION_SELECTION_FUNC,
    DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC,
    DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS,
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC = full_connected_destinations
"""Default topology function for the cooperative model."""

COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS = {}
"""Parameters for the default topology function in the cooperative model."""


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


class DistributedTrainer(Trainer):
    """Base class for all the distributed trainers."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SingleSpeciesTrainer],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SingleSpeciesTrainer],
                bool]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
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
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subtrainers.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
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
        :param num_subtrainers: The number of subtrainers. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` will be
            used. Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative solutions that
            will be sent to the other subtrainers. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subtrainer. If set to :py:data:`None`,
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
        :param subtrainer_params: Custom parameters for the subtrainers
            trainer
        :type subtrainer_params: keyworded variable-length argument list
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
        self.subtrainer_cls = subtrainer_cls
        self.num_subtrainers = num_subtrainers
        self.representation_size = representation_size
        self.representation_freq = representation_freq
        self.representation_selection_func = representation_selection_func
        self.representation_selection_func_params = (
            representation_selection_func_params
        )
        self.subtrainer_params = subtrainer_params

    @property
    def subtrainer_cls(self) -> Type[SingleSpeciesTrainer]:
        """Get and set the trainer class to handle the subtrainers.

        Each subtrainer will be handled by a single-species trainer.

        :getter: Return the trainer class
        :setter: Set new trainer class
        :type: A :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer` subclass
        :raises TypeError: If set to a value which is not a
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer` subclass
        """
        return self._subtrainer_cls

    @subtrainer_cls.setter
    def subtrainer_cls(self, cls: Type[SingleSpeciesTrainer]) -> None:
        """Set a new trainer class to handle the subtrainers.

        Each subtrainer will be handled by a single-species trainer.

        :param cls: The new class
        :type cls: A :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
            subclass
        :raises TypeError: If *cls* is not a
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer` subclass
        """
        # Check cls
        self._subtrainer_cls = check_subclass(
            cls, "trainer class for subtrainers", SingleSpeciesTrainer
        )

        # Reset the algorithm
        self.reset()

    @property
    def num_subtrainers(self) -> int:
        """Get and set the number of subtrainers.

        :getter: Return the current number of subtrainers
        :setter: Set a new value for the number of subtrainers. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            DEFAULT_NUM_SUBTRAINERS if self._num_subtrainers is None
            else self._num_subtrainers
        )

    @num_subtrainers.setter
    def num_subtrainers(self, value: int | None) -> None:
        """Set the number of subtrainers.

        :param value: The new number of subtrainers. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` is chosen
        :type value: An integer value
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._num_subtrainers = (
            None if value is None else check_int(
                value, "number of subtrainers", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def representation_size(self) -> int:
        """Get and set the representation size.

        The representation size is the number of representatives sent to the
        other subtrainers

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
    @abstractmethod
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get the representation topology function.

        This property must be overridden by subclasses to return a correct
        value.

        :type: :py:class:`~collections.abc.Callable`
        """
        raise NotImplementedError(
            "The representation_topology_func property has not been "
            f"implemented in the {self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get the parameters of the representation topology function.

        This property must be overridden by subclasses to return a correct
        value.

        :type: :py:class:`dict`
        """
        raise NotImplementedError(
            "The representation_topology_func_params property has not been "
            f"implemented in the {self.__class__.__name__} class"
        )

    @property
    def representation_selection_func(
        self
    ) -> Callable[[List[Solution], Any], Solution]:
        """Get and set the representation selection policy function.

        The representation selection policy func chooses which solutions are
        selected as representatives of each subtrainer.

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
    def subtrainer_params(self) -> Dict[str, Any]:
        """Get and set the custom parameters of the subtrainers.

        :getter: Return the current parameters for the subtrainers
        :setter: Set new parameters
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return self._subtrainer_params

    @subtrainer_params.setter
    def subtrainer_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters for the subtrainers.

        :param params: The new parameters
        :type params: A :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
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
        """Checkpoint file name of all the subtrainers."""
        base_name = path.splitext(self.checkpoint_filename)[0]
        extension = path.splitext(self.checkpoint_filename)[1]

        # Generator for the subtrainer checkpoint file names
        return (
            base_name + f"_{suffix}" + extension
            for suffix in self._subtrainer_suffixes
        )

    @property
    def subtrainers(self) -> List[SingleSpeciesTrainer] | None:
        """Return the subtrainers.

        One single-species trainer for each subtrainer

        :type: :py:class:`list` of
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer` trainers
        """
        return self._subtrainers

    @property
    def _subtrainer_suffixes(self) -> Generator[str, None, None]:
        """Return the suffixes for the different subtrainers.

        Can be used to generate the subtrainers' names, checkpoint files,
        etc.

        :return: A generator of the suffixes
        :rtype: A generator of :py:class:`str`
        """
        # Suffix length for subtrainers checkpoint files
        suffix_len = len(str(self.num_subtrainers-1))

        # Generator for the subtrainers checkpoint files
        return (f"{i:0{suffix_len}d}" for i in range(self.num_subtrainers))

    @staticmethod
    @abstractmethod
    def receive_representatives(subtrainer) -> None:
        """Receive representative solutions.

        This method must be overridden by subclasses.

        :param subtrainer: The subtrainer receiving representatives
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        """
        raise NotImplementedError(
            "The receive_representatives method has not been implemented in "
            f"the {subtrainer.container.__class__.__name__} class")

    @staticmethod
    @abstractmethod
    def send_representatives(subtrainer) -> None:
        """Send representatives.

        This method must be overridden by subclasses.

        :param subtrainer: The sender subtrainer
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        """
        raise NotImplementedError(
            "The send_representatives method has not been implemented in "
            f"the {subtrainer.container.__class__.__name__} class")

    @abstractmethod
    def _generate_subtrainers(self) -> None:
        """Generate the subtrainers.

        Also assign an
        :py:attr:`~culebra.trainer.abc.SingleSpeciesTrainer.index`
        and a :py:attr:`~culebra.trainer.abc.SingleSpeciesTrainer.container` to
        each subtrainer :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
        trainer, change the subtrainers'
        :py:attr:`~culebra.trainer.abc.SingleSpeciesTrainer.checkpoint_filename`
        according to the container checkpointing file name and each
        subtrainer index.

        Finally, the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._preprocess_iteration`
        and
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._postprocess_iteration`
        methods of the
        :py:attr:`~culebra.trainer.abc.DistributedTrainer.subtrainer_cls`
        class are dynamically overridden, in order to allow solutions exchange
        between subtrainers, if necessary

        This method must be overridden by subclasses.

        :raises NotImplementedError: if has not been overridden
        """
        raise NotImplementedError(
            "The _generate_subtrainers method has not been implemented "
            f"in the {self.__class__.__name__} class")

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to set the logbook to :py:data:`None`, since the final
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
        """Shallow copy the trainer."""
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
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
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
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (
                    self.fitness_function,
                    self.subtrainer_cls
                ),
                self.__dict__)


class SequentialDistributedTrainer(DistributedTrainer):
    """Abstract base class for sequential distributed trainers."""

    def _new_state(self) -> None:
        """Generate a new trainer state.

        Overridden to call also the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._new_state` method
        of each subtrainer.
        """
        super()._new_state()

        # Generate the state of all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._new_state()
            self._num_evals  += subtrainer._num_evals

    def _load_state(self) -> None:
        """Load the state of the last checkpoint.

        Overridden to call also the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._load_state` method
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
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._save_state` method
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
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._start_iteration`
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
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._preprocess_iteration`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._preprocess_iteration()

    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        Overridden to call also the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._do_iteration`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats.

        Overridden to call also the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._do_iteration_stats`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._do_iteration_stats()

    def _postprocess_iteration(self) -> None:
        """Postprocess the iteration of all the subtrainers.

        Overridden to call also the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._postprocess_iteration`
        method of each subtrainer.
        """
        # For all the subtrainers
        for subtrainer in self.subtrainers:
            subtrainer._postprocess_iteration()

    def _finish_iteration(self) -> None:
        """Finish an iteration.

        Close the metrics after each iteration is run. Overridden to call also
        the
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._finish_iteration`
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
        :py:meth:`~culebra.trainer.abc.SingleSpeciesTrainer._finish_search`
        method of each subtrainer.
        """
        for subtrainer in self.subtrainers:
            # Fix the current iteration
            subtrainer._current_iter = self._current_iter

        # Finish the iteration
        super()._finish_search()


class ParallelDistributedTrainer(DistributedTrainer):
    """Abstract base class for parallel distributed trainers."""

    @DistributedTrainer.num_subtrainers.getter
    def num_subtrainers(self) -> int:
        """Get and set the number of subtrainers.

        :getter: Return the current number of subtrainers
        :setter: Set a new value for the number of subtrainers. If set to
            :py:data:`None`, the number of CPU cores is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            cpu_count() if self._num_subtrainers is None
            else self._num_subtrainers
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
        """Get the training runtime.

        Return the training runtime or :py:data:`None` if the search has not
        been done yet.

        :type: :py:class:`float`
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
        :py:data:`None`, since their final values will be generated from the
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

        Overridden to create a multiprocessing manager and proxies to
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

        Overridden to reset the multiprocessing manager and proxies.
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


# Change the docstring of the ParallelMultiPop constructor to indicate that
# the default number of subtrainers is the number of CPU cores for parallel
# distributed approaches
ParallelDistributedTrainer.__init__.__doc__ = (
    ParallelDistributedTrainer.__init__.__doc__.replace(
        ':py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS`',
        'the number of CPU cores'
    )
)


class IslandsTrainer(SingleSpeciesTrainer, DistributedTrainer):
    """Abstract island-based trainer."""

    def __init__(
        self,
        solution_cls: Type[Solution],
        species: Species,
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SingleSpeciesTrainer],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SingleSpeciesTrainer], bool]
        ] = None,
        num_subtrainers: Optional[int] = None,
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
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        :param solution_cls: The solution class
        :type solution_cls: A :py:class:`~culebra.abc.Solution` subclass
        :param species: The species for all the solutions
        :type species: :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subtrainers (islands).
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
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
        :param num_subtrainers: The number of subtrainers (islands). If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_NUM_SUBTRAINERS` will be
            used. Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative solutions that
            will be sent to the other subtrainers. If set to
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
            representatives from each subtrainer (island). If set to
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
        :param subtrainer_params: Custom parameters for the subtrainers
            (islands) trainer
        :type subtrainer_params: keyworded variable-length argument list
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
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )
        self.representation_topology_func = representation_topology_func
        self.representation_topology_func_params = (
            representation_topology_func_params
        )

    @property
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

    @representation_topology_func.setter
    def representation_topology_func(
        self,
        func: Callable[[int, int, Any], List[int]] | None
    ) -> None:
        """Set new representation topology function.

        :param func: The new function. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC`
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

    @representation_topology_func_params.setter
    def representation_topology_func_params(
        self, params: Dict[str, Any] | None
    ) -> None:
        """Set the parameters for the representation topology function.

        :param params: The new parameters. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
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

    def __copy__(self) -> IslandsTrainer:
        """Shallow copy the trainer."""
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
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
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
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (
                    self.solution_cls,
                    self.species,
                    self.fitness_function,
                    self.subtrainer_cls
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
        """Get and set the species for each subtrainer.

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


class CooperativeTrainer(MultiSpeciesTrainer, DistributedTrainer):
    """Abstract cooperative trainer model."""

    def __init__(
        self,
        solution_classes: Sequence[Type[Solution]],
        species: Sequence[Species],
        fitness_function: FitnessFunction,
        subtrainer_cls: Type[SingleSpeciesTrainer],
        max_num_iters: Optional[int] = None,
        custom_termination_func: Optional[
            Callable[
                [SingleSpeciesTrainer],
                bool
            ]
        ] = None,
        num_subtrainers: Optional[int] = None,
        representation_size: Optional[int] = None,
        representation_freq: Optional[int] = None,
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
        **subtrainer_params: Any
    ) -> None:
        """Create a new trainer.

        Each species is evolved in a different subtrainer.

        :param solution_classes: The individual class for each species.
        :type solution_classes: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution` subclasses
        :param species: The species to be evolved
        :type species: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Species`
        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~culebra.abc.FitnessFunction`
        :param subtrainer_cls: Single-species trainer class to handle
            the subtrainers.
        :type subtrainer_cls: Any subclass of
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
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
        :param num_subtrainers: The number of subtrainers (species). If set to
            :py:data:`None`, the number of species  evolved by the trainer is
            will be used, otherwise it must match the number of species.
            Defaults to :py:data:`None`
        :type num_subtrainers: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subtrainers. If set to
            :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_SIZE` will
            be used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of iterations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~culebra.trainer.DEFAULT_REPRESENTATION_FREQ` will
            be used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subtrainer (species). If set to
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
        :param subtrainer_params: Custom parameters for the subtrainers
            (species) trainer
        :type subtrainer_params: keyworded variable-length argument list
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
            representation_selection_func=representation_selection_func,
            representation_selection_func_params=(
                representation_selection_func_params
            ),
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed,
            **subtrainer_params
        )

    @property
    def representation_topology_func(
        self
    ) -> Callable[[int, int, Any], List[int]]:
        """Get the representation topology function.

        :type: :py:class:`~collections.abc.Callable`
        """
        return COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC

    @property
    def representation_topology_func_params(self) -> Dict[str, Any]:
        """Get the parameters of the representation topology function.

        :type: :py:class:`dict`
        """
        return COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS

    @property
    def representatives(self) -> Sequence[Sequence[Solution | None]] | None:
        """Return the representatives of all the species."""
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
    def num_subtrainers(self) -> int:
        """Get and set the number of subtrainers.

        :getter: Return the current number of subtrainers
        :setter: Set a new value for the number of subtrainers. If set to
            :py:data:`None`, the number of species  evolved by the trainer is
            chosen, otherwise *it must match the number of species*
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is different of the number
            of species
        """
        return (
            len(self.species) if self._num_subtrainers is None
            else self._num_subtrainers
        )

    @num_subtrainers.setter
    def num_subtrainers(self, value: int | None) -> None:
        """Set the number of subtrainers.

        :param value: The new number of subtrainers. If set to
            :py:data:`None`, the number of species  evolved by the trainer is
            chosen, otherwise *it must match the number of species*
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is different of the number of species
        """
        # Check the value
        if value is not None and value != len(self.species):
            raise ValueError(
                "The number of subtrainers must match the number of "
                f"species: {self.species}"
            )

        self._num_subtrainers = value

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
        if self.subtrainers is None:
            for _ in range(self.num_subtrainers):
                hofs.append(ParetoFront())
        # Else, the best solutions of each species are returned
        else:
            for subtrainer in self.subtrainers:
                hofs.append(subtrainer.best_solutions()[0])

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
        subtrainer,
        solution_classes,
        species,
        representation_size
    ):
        """Init the representatives of the other species for a subtrainer.

        This method is used to override dynamically the
        :py:meth:`~culebra.abc.Trainer._init_representatives` of all the
        subtrainers, when they are generated with the
        :py:meth:`~culebra.trainer.abc.CooperativeTrainer._generate_subtrainers`
        method, to let them initialize the list of representative individuals
        of the other species

        :param subtrainer: The subtrainer. The representatives from the
            remaining subtrainers will be initialized for this subtrainer
        :type subtrainer:
            :py:class:`~culebra.trainer.abc.SingleSpeciesTrainer`
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
        subtrainer._representatives = []

        for _ in range(representation_size):
            subtrainer._representatives.append(
                [
                    ind_cls(
                        spe, subtrainer.fitness_function.Fitness
                    ) if i != subtrainer.index else None
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
            self.subtrainer_cls
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
            self.subtrainer_cls
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
                    self.subtrainer_cls
                ),
                self.__dict__)


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
