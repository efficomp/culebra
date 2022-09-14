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

"""Base class for multiple-population evolutionary algorithms."""

from __future__ import annotations
from abc import abstractmethod
from typing import (
    Any,
    Type,
    Optional,
    Callable,
    List,
    Dict,
    Generator)
from copy import deepcopy
from multiprocessing import (
    cpu_count,
    Queue,
    Process,
    Manager)
from os import path
from deap.tools import Logbook, selTournament
from culebra.base import (
    Individual,
    FitnessFunction,
    check_int,
    check_subclass,
    check_func,
    check_func_params
)
from culebra.wrapper import Generational
from culebra.wrapper.single_pop import SinglePop
from . import full_connected_destinations

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_NUM_SUBPOPS = 1
"""Default number of subpopulations."""

DEFAULT_REPRESENTATION_SIZE = 5
"""Default value for the number of representatives sent to the other
subpopulations.
"""

DEFAULT_REPRESENTATION_FREQ = 10
"""Default value for the number of generations between representatives
sending."""

DEFAULT_REPRESENTATION_TOPOLOGY_FUNC = full_connected_destinations
"""Default topology function for representatives sending."""

DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS = {}
"""Default parameters to obtain the destinations with the topology function."""

DEFAULT_REPRESENTATION_SELECTION_FUNC = selTournament
"""Default selection policy function to choose the representatives."""

DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS = {'tournsize': 3}
"""Default parameters for the representatives selection policy function."""


class MultiPop(Generational):
    """Base class for all the multiple population wrapper methods."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        subpop_wrapper_cls: Type[SinglePop],
        num_gens: Optional[int] = None,
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
            Callable[[List[Individual], Any], Individual]
        ] = None,
        representation_selection_func_params: Optional[
            Dict[str, Any]
        ] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None,
        **subpop_wrapper_params: Any
    ) -> None:
        """Create a new wrapper.

        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param subpop_wrapper_cls: Single-population wrapper class to handle
            the subpopulations.
        :type subpop_wrapper_cls: Any subclass of
            :py:class:`~wrapper.single_pop.SinglePop`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
        :param num_subpops: The number of subpopulations. If set to
            :py:data:`None`, :py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`
            will be used. Defaults to :py:data:`None`
        :type num_subpops: :py:class:`int`, optional
        :param representation_size: Number of representative individuals that
            will be sent to the other subpopulations. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SIZE` will be
            used. Defaults to :py:data:`None`
        :type representation_size: :py:class:`int`, optional
        :param representation_freq: Number of generations between
            representatives sendings. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` will be
            used. Defaults to :py:data:`None`
        :type representation_freq: :py:class:`int`, optional
        :param representation_topology_func: Topology function for
            representatives sending. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_topology_func_params: Parameters to obtain the
            destinations with the topology function. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_topology_func_params: :py:class:`dict`, optional
        :param representation_selection_func: Policy function to choose the
            representatives from each subpopulation. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func:
            :py:class:`~collections.abc.Callable`, optional
        :param representation_selection_func_params: Parameters to obtain the
            representatives with the selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
            will be used. Defaults to :py:data:`None`
        :type representation_selection_func_params: :py:class:`dict`,
            optional
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_ENABLE` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FREQ` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`, :py:attr:`~base.DEFAULT_CHECKPOINT_FILENAME` will
            be used. Defaults to :py:data:`None`
        :type checkpoint_filename: :py:class:`str`, optional
        :param verbose: The verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` will be used. Defaults to
            :py:data:`None`
        :type verbose: :py:class:`bool`, optional
        :param random_seed: The seed, defaults to :py:data:`None`
        :type random_seed: :py:class:`int`, optional
        :param subpop_wrapper_params: Custom parameters for the subpopulations
            wrapper
        :type subpop_wrapper_params: keyworded variable-length argument list
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__(
            fitness_function=fitness_function,
            num_gens=num_gens,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.subpop_wrapper_cls = subpop_wrapper_cls
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
        self.subpop_wrapper_params = subpop_wrapper_params

    @property
    def subpop_wrapper_cls(self) -> Type[SinglePop]:
        """Get and set the wrapper class to handle the subpopulations.

        Each subpopulation will be handled by a single-population wrapper.

        :getter: Return the wrapper class
        :setter: Set new wrapper class
        :type: A :py:class:`~wrapper.single_pop.SinglePop` subclass
        :raises TypeError: If set to a value which is not a
            :py:class:`~wrapper.single_pop.SinglePop` subclass
        """
        return self._subpop_wrapper_cls

    @subpop_wrapper_cls.setter
    def subpop_wrapper_cls(self, cls: Type[SinglePop]) -> None:
        """Set a new wrapper class to handle the subpopulations.

        Each subpopulation will be handled by a single-population wrapper.

        :param cls: The new class
        :type cls: A :py:class:`~wrapper.single_pop.SinglePop` subclass
        :raises TypeError: If *cls* is not a
            :py:class:`~wrapper.single_pop.SinglePop` subclass
        """
        # Check cls
        self._subpop_wrapper_cls = check_subclass(
            cls, "wrapper class for subpopulations", SinglePop
        )

        # Reset the algorithm
        self.reset()

    @property
    def num_subpops(self) -> int:
        """Get and set the number of subpopulations.

        :getter: Return the current number of subpopulations
        :setter: Set a new value for the number of subpopulations. If set to
            :py:data:`None`, :py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`
            is chosen
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
            :py:data:`None`, :py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`
            is chosen
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SIZE` is chosen
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SIZE` is chosen
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
        """Get and set the number of generations between representatives sendings.

        :getter: Return the current frequency
        :setter: Set a new value for the frequency. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` is chosen
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
        """Set the number of generations between representatives sendings.

        :param value: The new frequency. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_FREQ` is chosen
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC`
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS`
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
    ) -> Callable[[List[Individual], Any], Individual]:
        """Get and set the representation selection policy function.

        The representation selection policy func chooses which individuals are
        selected as representatives of each subpopulation.

        :getter: Return the representation selection policy function
        :setter: Set new representation selection policy function. If set to
            :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC`
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
        func: Callable[[List[Individual], Any], Individual] | None
    ) -> None:
        """Set new representation selection policy function.

        :param func: The new function. If set to :py:data:`None`,
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC`
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
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
            :py:attr:`~wrapper.multi_pop.DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS`
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
    def subpop_wrapper_params(self) -> Dict[str, Any]:
        """Get and set the custom parameters of the subpopulation wrappers.

        :getter: Return the current parameters for the subpopulation wrappers
        :setter: Set new parameters
        :type: :py:class:`dict`
        :raises TypeError: If set to a value which is not a :py:class:`dict`
        """
        return self._subpop_wrapper_params

    @subpop_wrapper_params.setter
    def subpop_wrapper_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters for the subpopulation wrappers.

        :param params: The new parameters
        :type params: A :py:class:`dict`
        :raises TypeError: If *params* is not a :py:class:`dict`
        """
        # Check that params is a valid dict
        self._subpop_wrapper_params = check_func_params(
            params, "subpopulation wrappers parameters"
        )

        # Reset the algorithm
        self.reset()

    @property
    def subpop_wrappers(self) -> List[SinglePop] | None:
        """Return the subpopulation wrappers.

        One single-population wrapper for each subpopulation

        :type: :py:class:`list` of :py:class:`~wrapper.single_pop.SinglePop`
            wrappers
        """
        return self._subpop_wrappers

    @property
    def _subpop_suffixes(self) -> Generator[str, None, None]:
        """Return the suffixes for the different subpopulations.

        Can be used to generate the subpopulations' names, checkpoint files,
        etc.

        :return: A generator of the suffixes
        :rtype: A generator of :py:_class:`str`
        """
        # Suffix length for subpopulations checkpoint files
        suffix_len = len((self.num_subpops-1).__str__())

        # Generator for the subpopulations checkpoint files
        return (f"{i:0{suffix_len}d}" for i in range(self.num_subpops))

    @property
    def subpop_wrapper_checkpoint_filenames(
        self
    ) -> Generator[str, None, None]:
        """Checkpoint file name of all the subpopulation wrappers."""
        base_name = path.splitext(self.checkpoint_filename)[0]
        extension = path.splitext(self.checkpoint_filename)[1]

        # Generator for the subpop wrapper checkpoint file names
        return (
            base_name + f"_{suffix}" + extension
            for suffix in self._subpop_suffixes
        )

    @abstractmethod
    def _generate_subpop_wrappers(self) -> None:
        """Generate the subpopulation wrappers.

        Also assign an :py:attr:`~wrapper.single_pop.SinglePop.index` and a
        :py:attr:`~wrapper.single_pop.SinglePop.container` to each
        subpopulation :py:class:`~wrapper.single_pop.SinglePop` wrapper,
        change the subpopulation wrappers'
        :py:attr:`~wrapper.single_pop.SinglePop.checkpoint_filename` according
        to the container checkpointing file name and each subpopulation index.

        Finally, the
        :py:meth:`~wrapper.single_pop.SinglePop._preprocess_generation` and
        :py:meth:`~wrapper.single_pop.SinglePop._postprocess_generation`
        methods of the
        :py:attr:`~wrapper.multi_pop.MultiPop.subpop_wrapper_cls` class are
        dynamically overriden, in order to allow individuals exchange between
        subpopulation wrappers, if necessary

        This method must be overriden by subclasses.

        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError(
            "The _generate_subpop_wrappers method has not been implemented "
            f"in the {self.__class__.__name__} class")

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        Overriden to set the logbook to :py:data:`None`, since the final
        logbook will be generated from the subpopulation wrappers' logbook,
        once the wrapper has finished.
        """
        super()._new_state()

        # The logbook will be generated from the subpopulation wrappers
        self._logbook = None

    def _init_internals(self) -> None:
        """Set up the wrapper internal data structures to start searching.

        Overriden to create the subpopulation wrappers and communication
        queues.
        """
        super()._init_internals()

        # Generate the subpopulation wrappers
        self._generate_subpop_wrappers()

        # Set up the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._init_internals()

        # Init the communication queues
        self._communication_queues = []
        for _ in range(self.num_subpops):
            self._communication_queues.append(Queue())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the wrapper.

        Overriden to reset the subpopulation wrappers and communication queues.
        """
        super()._reset_internals()
        self._subpop_wrappers = None
        self._communication_queues = None

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
        elif self.subpop_wrappers is not None:
            # Create the logbook
            the_logbook = Logbook()
            # Init the logbook
            the_logbook.header = list(self.stats_names) + \
                (self._stats.fields if self._stats else [])

            for subpop_wrapper in self.subpop_wrappers:
                if subpop_wrapper.logbook is not None:
                    the_logbook.extend(subpop_wrapper.logbook)

            if self._search_finished:
                self._logbook = the_logbook

        return the_logbook

    @staticmethod
    @abstractmethod
    def receive_representatives(subpop_wrapper) -> None:
        """Receive representative individuals.

        This method must be overriden by subclasses.

        :param subpop_wrapper: The subpopulation wrapper receiving
            representatives
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        """
        raise NotImplementedError(
            "The receive_representatives method has not been implemented in "
            f"the {subpop_wrapper.container.__class__.__name__} class")

    @staticmethod
    @abstractmethod
    def send_representatives(subpop_wrapper) -> None:
        """Send representatives.

        This method must be overriden by subclasses.

        :param subpop_wrapper: The sender subpopulation wrapper
        :type subpop_wrapper: :py:class:`~wrapper.single_pop.SinglePop`
        """
        raise NotImplementedError(
            "The send_representatives method has not been implemented in "
            f"the {subpop_wrapper.container.__class__.__name__} class")

    def __copy__(self) -> MultiPop:
        """Shallow copy the wrapper."""
        cls = self.__class__
        result = cls(
            self.fitness_function,
            self.subpop_wrapper_cls
        )
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> MultiPop:
        """Deepcopy the wrapper.

        :param memo: Wrapper attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the wrapper
        :rtype: :py:class:`~base.Wrapper`
        """
        cls = self.__class__
        result = cls(
            self.fitness_function,
            self.subpop_wrapper_cls
        )
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the wrapper.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__,
                (
                    self.fitness_function,
                    self.subpop_wrapper_cls
                ),
                self.__dict__)


class SequentialMultiPop(MultiPop):
    """Abstract sequential multi-population model for evolutionary wrappers."""

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        Overriden to call also the
        :py:meth:`~wrapper.single_pop.SinglePop._new_state` method
        of each subpopulation wrapper.
        """
        super()._new_state()

        # Generate the state of all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._new_state()

    def _load_state(self) -> None:
        """Load the state of the last checkpoint.

        Overriden to call also the
        :py:meth:`~wrapper.single_pop.SinglePop._load_state` method
        of each subpopulation wrapper.

        :raises Exception: If the checkpoint file can't be loaded
        """
        # Load the state of this wrapper
        super()._load_state()

        # Load the subpopulation wrappers' state
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._load_state()

    def _save_state(self) -> None:
        """Save the state at a new checkpoint.

        Overriden to call also the
        :py:meth:`~wrapper.single_pop.SinglePop._save_state` method
        of each subpopulation wrapper.

        :raises Exception: If the checkpoint file can't be written
        """
        # Save the state of this wrapper
        super()._save_state()

        # Save the state of all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._save_state()

    def _start_generation(self) -> None:
        """Start a generation.

        Prepare the metrics before each generation is run.
        """
        super()._start_generation()
        # For all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            # Fix the current generation
            subpop_wrapper._current_gen = self._current_gen
            # Start the generation
            subpop_wrapper._start_generation()

    def _preprocess_generation(self) -> None:
        """Preprocess the population of all the subwrappers."""
        # For all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._preprocess_generation()

    def _do_generation(self) -> None:
        """Implement a generation of the search process."""
        # For all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._do_generation()

    def _postprocess_generation(self) -> None:
        """Postprocess the population of all the subwrappers."""
        # For all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            subpop_wrapper._postprocess_generation()

    def _finish_generation(self) -> None:
        """Finish a generation.

        Close the metrics after each generation is run.
        """
        # For all the subpopulation wrappers
        for subpop_wrapper in self.subpop_wrappers:
            # Finish the generation of all the subpopulation wrappers
            subpop_wrapper._finish_generation()
            # Get the number of evaluations
            self._current_gen_evals += subpop_wrapper._current_gen_evals

        # Finish the generation
        super()._finish_generation()


class ParallelMultiPop(MultiPop):
    """Abstract parallel multi-population model for evolutionary wrappers."""

    @MultiPop.num_subpops.getter
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
        elif self.subpop_wrappers is not None:
            n_evals = 0
            for subpop_wrapper in self.subpop_wrappers:
                if subpop_wrapper.num_evals is not None:
                    n_evals += subpop_wrapper.num_evals

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
        elif self.subpop_wrappers is not None:
            the_runtime = 0
            for subpop_wrapper in self.subpop_wrappers:
                if (
                        subpop_wrapper.runtime is not None
                        and subpop_wrapper.runtime > the_runtime
                ):
                    the_runtime = subpop_wrapper.runtime

            if self._search_finished:
                self._runtime = the_runtime

        return the_runtime

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        Overriden to set the overall runtime and number of evaluations to
        :py:data:`None`, since their final values will be generated from the
        subpopulation wrappers' data, once the wrapper has finished.
        """
        super()._new_state()

        # The runtime and number of evaluations will be generated
        # from the subpopulation wrappers
        self._runtime = None
        self._num_evals = None

    def _init_internals(self) -> None:
        """Set up the wrapper internal data structures to start searching.

        Overriden to create a multiprocessing manager and proxies to
        communicate with the processes running the subpopulation wrappers.
        """
        super()._init_internals()

        # Initialize the manager to receive the subpopulation wrappers' state
        self._manager = Manager()
        self._subpop_state_proxies = []
        for _ in range(self.num_subpops):
            self._subpop_state_proxies.append(self._manager.dict())

    def _reset_internals(self) -> None:
        """Reset the internal structures of the wrapper.

        Overriden to reset the multiprocessing manager and proxies.
        """
        super()._reset_internals()
        self._manager = None
        self._subpop_state_proxies = None

    def _search(self) -> None:
        """Apply the search algorithm.

        Each subpopulation wrapper runs in a different process.
        """
        # Run all the generations
        for subpop_wrapper, state_proxy, suffix in zip(
                self.subpop_wrappers,
                self._subpop_state_proxies,
                self._subpop_suffixes):
            subpop_wrapper.process = Process(
                target=subpop_wrapper.train,
                kwargs={"state_proxy": state_proxy},
                name=f"subpop_{suffix}",
                daemon=True
            )
            subpop_wrapper.process.start()

        for subpop_wrapper, state_proxy in zip(
                self.subpop_wrappers, self._subpop_state_proxies):
            subpop_wrapper.process.join()
            subpop_wrapper._state = state_proxy


# Change the docstring of the ParallelMultiPop constructor to indicate that
# the default number of subpopulations is the number of CPU cores for parallel
# multi-population approaches
ParallelMultiPop.__init__.__doc__ = ParallelMultiPop.__init__.__doc__.replace(
    ':py:attr:`~wrapper.multi_pop.DEFAULT_NUM_SUBPOPS`',
    'the number of CPU cores'
)

# Exported symbols for this module
__all__ = [
    'MultiPop',
    'SequentialMultiPop',
    'ParallelMultiPop',
    'DEFAULT_NUM_SUBPOPS',
    'DEFAULT_REPRESENTATION_SIZE',
    'DEFAULT_REPRESENTATION_FREQ',
    'DEFAULT_REPRESENTATION_TOPOLOGY_FUNC',
    'DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS',
    'DEFAULT_REPRESENTATION_SELECTION_FUNC',
    'DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS'
]
