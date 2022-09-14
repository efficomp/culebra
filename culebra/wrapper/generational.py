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

"""Base class for generational algorithms."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Optional
from time import perf_counter
from culebra.base import (
    FitnessFunction,
    Wrapper,
    check_int
)

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_STATS_NAMES = ("Gen", "Pop", "NEvals")
"""Default statistics calculated for each generation."""

DEFAULT_NUM_GENS = 100
"""Default number of generations."""


class Generational(Wrapper):
    """Base class for all the generational wrapper methods."""

    stats_names = DEFAULT_STATS_NAMES
    """Statistics calculated each generation."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        num_gens: Optional[int] = None,
        checkpoint_enable: Optional[bool] = None,
        checkpoint_freq: Optional[int] = None,
        checkpoint_filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> None:
        """Create a new wrapper.

        :param fitness_function: The training fitness function
        :type fitness_function: :py:class:`~base.FitnessFunction`
        :param num_gens: The number of generations. If set to
            :py:data:`None`, :py:attr:`~wrapper.DEFAULT_NUM_GENS` will
            be used. Defaults to :py:data:`None`
        :type num_gens: :py:class:`int`, optional
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
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__(
            fitness_function=fitness_function,
            checkpoint_enable=checkpoint_enable,
            checkpoint_freq=checkpoint_freq,
            checkpoint_filename=checkpoint_filename,
            verbose=verbose,
            random_seed=random_seed
        )

        # Get the parameters
        self.num_gens = num_gens

    @property
    def num_gens(self) -> int:
        """Get and set the number of generations.

        :getter: Return the current number of generations
        :setter: Set a new value for the number of generations. If set to
            :py:data:`None`, the default number of generations,
            :py:attr:`~wrapper.DEFAULT_NUM_GENS`, is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return DEFAULT_NUM_GENS if self._num_gens is None else self._num_gens

    @num_gens.setter
    def num_gens(self, value: int | None) -> None:
        """Set the number of generations.

        :param value: The new number of generations. If set to
            :py:data:`None`, the default number of generations,
            :py:attr:`~wrapper.DEFAULT_NUM_GENS`, is chosen
        :type value: An integer value or :py:data:`None`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        # Check the value
        self._num_gens = (
            None if value is None else check_int(
                value, "number of generations", gt=0
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def _state(self) -> Dict[str, Any]:
        """Get and set the state of this wrapper.

        Overriden to add the current generation to the wrapper's state.

        :getter: Return the state
        :setter: Set a new state
        :type: :py:class:`dict`
        """
        # Get the state of the superclass
        state = Wrapper._state.fget(self)

        # Get the state of this class
        state["current_gen"] = self._current_gen

        return state

    @_state.setter
    def _state(self, state: Dict[str, Any]) -> None:
        """Set the state of this wrapper.

        Overriden to add the current generation to the wrapper's state.

        :param state: The last loaded state
        :type state: :py:class:`dict`
        """
        # Set the state of the superclass
        Wrapper._state.fset(self, state)

        # Set the state of this class
        self._current_gen = state["current_gen"]

    def _new_state(self) -> None:
        """Generate a new wrapper state.

        Overriden to set the current generation to 0.
        """
        super()._new_state()

        # Init the current generation
        self._current_gen = 0

    def _reset_state(self) -> None:
        """Reset the wrapper state.

        Overriden to reset the current generation.
        """
        super()._reset_state()
        self._current_gen = None

    def _init_internals(self) -> None:
        """Set up the wrapper internal data structures to start searching.

        Overriden to add two internal attributes to handle the current
        generation's number of evaluations and runtime.
        """
        super()._init_internals()

        # Will be initialized at _start_generation
        self._current_gen_evals = None
        self._current_gen_start_time = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the wrapper.

        Overriden to reset the internal attributes that handle the current
        generation's number of evaluations and runtime.
        """
        super()._reset_internals()

        self._current_gen_evals = None
        self._current_gen_start_time = None

    def _start_generation(self) -> None:
        """Start a generation.

        Prepare the generation metrics (number of evaluations, execution time)
        before each generation is run.
        """
        self._current_gen_evals = 0
        self._current_gen_start_time = perf_counter()

    def _preprocess_generation(self) -> None:
        """Preprocess the population(s).

        Subclasses should override this method to make any preprocessment
        before performing a generation.
        """
        pass

    @abstractmethod
    def _do_generation(self) -> None:
        """Implement a generation of the search process.

        This abstract method should be implemented by subclasses in order to
        implement the desired evolutionary algorithm.
        """

    def _postprocess_generation(self) -> None:
        """Postprocess the population(s).

        Subclasses should override this method to make any postprocessment
        after performing a generation.
        """
        pass

    def _finish_generation(self) -> None:
        """Finish a generation.

        Finish the generation metrics (number of evaluations, execution time)
        after each generation is run.
        """
        end_time = perf_counter()
        self._runtime += end_time - self._current_gen_start_time
        self._num_evals += self._current_gen_evals

        # Save the wrapper state at each checkpoint
        if (self.checkpoint_enable and
                self._current_gen % self.checkpoint_freq == 0):
            self._save_state()

    def _search(self) -> None:
        """Apply the search algorithm.

        Execute the wrapper during :py:class:`~Generational.num_gens`
        generations. Each generation is composed by the following steps:

            * :py:meth:`~Generational._start_generation`
            * :py:meth:`~Generational._preprocess_generation`
            * :py:meth:`~Generational._do_generation`
            * :py:meth:`~Generational._postprocess_generation`
            * :py:meth:`~Generational._finish_generation`
        """
        # Run all the generations
        for self._current_gen in range(self._current_gen + 1,
                                       self.num_gens + 1):
            self._start_generation()
            self._preprocess_generation()
            self._do_generation()
            self._postprocess_generation()
            self._finish_generation()


# Exported symbols for this module
__all__ = [
    'Generational',
    'DEFAULT_STATS_NAMES',
    'DEFAULT_NUM_GENS'
]
