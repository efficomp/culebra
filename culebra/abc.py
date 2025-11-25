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

"""Base classes of culebra.

Fundamental classes to solve optimization problems. The :mod:`~culebra.abc`
module defines:

* A :class:`~culebra.abc.Base` class from which all the classes of culebra
  inherit
* A :class:`~culebra.abc.Fitness` class to store the fitness values for each
  :class:`~culebra.abc.Solution`
* A :class:`~culebra.abc.FitnessFunction` class to evaluate a
  :class:`~culebra.abc.Solution` during the training process and assign it
  values for its :class:`~culebra.abc.Fitness`
* A :class:`~culebra.abc.Solution` class, which will be used within the
  :class:`~culebra.abc.Trainer` class to seek the best solution(s) for a
  problem
* A :class:`~culebra.abc.Species` class to define the characteristics of
  solutions belonging to a given domain
* A :class:`~culebra.abc.Trainer` class to find solutions for a problem
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple, List, Optional, Dict, Type, Callable
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from time import perf_counter
from multiprocess.managers import DictProxy
import random

import numpy as np
import dill
import gzip
from deap.tools import Logbook, Statistics, HallOfFame

from culebra import (
    DEFAULT_MAX_NUM_ITERS,
    SERIALIZED_FILE_EXTENSION,
    DEFAULT_CHECKPOINT_ENABLE,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_VERBOSITY,
    DEFAULT_INDEX,
)
from culebra.checker import (
    check_bool,
    check_int,
    check_float,
    check_instance,
    check_sequence,
    check_func,
    check_filename
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


TRAINER_STATS_NAMES = ('Iter', 'NEvals')
"""Statistics calculated for each iteration of the
:class:`~culebra.abc.Trainer`.
"""

TRAINER_OBJECTIVE_STATS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max,
}
"""Statistics calculated for each objective within a
:class:`~culebra.abc.Trainer`.
"""


class Base:
    """Base for all classes in culebra."""

    def dump(self, filename: str) -> None:
        """Serialize this object and save it to a file.

        :param filename: The file name.
        :type filename: str
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If the *filename* extension is not
            :attr:`~culebra.SERIALIZED_FILE_EXTENSION`
        """
        filename = check_filename(
            filename,
            name="serialized file name",
            ext=SERIALIZED_FILE_EXTENSION
        )

        with gzip.open(filename, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> Base:
        """Load a serialized object from a file.

        :param filename: The file name.
        :type filename: str
        :return: The loaded object
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If the *filename* extension is not
            :attr:`~culebra.SERIALIZED_FILE_EXTENSION`
        """
        filename = check_filename(
            filename,
            name="serialized file name",
            ext=SERIALIZED_FILE_EXTENSION
        )

        with gzip.open(filename, 'rb') as f:
            return cls.__fromstate__(dill.load(f).__dict__)

    def __copy__(self) -> Base:
        """Shallow copy the object.

        :return: The copied object
        :rtype: ~culebra.abc.Base
        """
        cls = self.__class__
        result = cls()
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Base:
        """Deepcopy the object.

        :param memo: Object attributes
        :type memo: dict
        :return:  The copied object
        :rtype: ~culebra.abc.Base
        """
        cls = self.__class__
        result = cls()
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object.

        :param state: The state
        :type state: dict
        """
        self.__dict__.update(state)

    def __reduce__(self) -> tuple:
        """Reduce the object.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, (), self.__dict__)

    def __repr__(self) -> str:
        """Object representation.

        :rtype: str
        """
        cls = self.__class__
        cls_name = cls.__name__

        properties = (
            p
            for p in dir(cls)
            if (
                isinstance(getattr(cls, p), property)
                and not p.startswith("_"))
        )

        msg = cls_name
        sep = "("
        for prop in properties:
            value = getattr(self, prop)
            value_str = (
                value.__module__ + "." + value.__name__
                if isinstance(value, type) else repr(value)
            )
            msg += sep + prop + ": " + value_str
            sep = ", "

        if sep[0] == "(":
            msg += sep
        msg += ")"
        return msg

    @classmethod
    def __fromstate__(cls, state: dict) -> Base:
        """Return an object from a state.

        :param state: The state
        :type state: dict
        :return: The object
        :rtype: ~culebra.abc.Base
        """
        obj = cls()
        obj.__setstate__(state)
        return obj


class Fitness(Base):
    """Define the base class for the fitness of a solution."""

    weights = ()
    """A :class:`tuple` containing an integer value for each objective
    being optimized. The weights are used in the fitness comparison. They are
    shared among all fitnesses of the same type. When subclassing
    :class:`~culebra.abc.Fitness`, the weights must be defined as a tuple
    where each element is associated to an objective. A negative weight element
    corresponds to the minimization of the associated objective and positive
    weight to the maximization.
    """

    names = ()
    """Names of the objectives."""

    thresholds = ()
    """A list with the similarity thresholds defined for all the objectives.
    Fitness objects are compared lexicographically. The comparison applies a
    similarity threshold to assume that two fitness values are similar (if
    their difference is lower than or equal to the similarity threshold)
    """

    def __init__(self, values: Optional[Sequence[float, ...]] = None) -> None:
        """Construct a default fitness object.

        :param values: Initial values for the fitness, optional
        :type values: ~collections.abc.Sequence[float]
        """
        # Init the superclasses
        super().__init__()
        if values is None:
            del self.values
        else:
            self.values = values

    @property
    def is_valid(self) -> bool:
        """Validness of the fitness.

        :return: :data:`True` if the fitness is valid
        :rtype: bool
        """
        if (
                self._values is None or
                len(self._values) == 0 or
                any(val is None for val in self._values)
        ):
            return False

        return True

    @property
    def values(self) -> Tuple[float | None]:
        """Fitness values.

        :rtype: tuple[float | None]

        :setter: Set the new fitness values
        :param fit_values: The new values
        :type fit_values: tuple[float]
        :raises TypeError: If any element in *fit_values* is not a real number
        """
        return tuple(self._values)

    @values.setter
    def values(self, fit_values: Sequence[float]):
        """Set the fitness values.

        :param fit_values: The new values
        :type fit_values: tuple[float]
        :raises TypeError: If any element in *fit_values* is not a real number
        """
        self._values = check_sequence(
            fit_values,
            "fitness values",
            size=self.num_obj,
            item_checker=partial(check_float)
        )

    @values.deleter
    def values(self):
        """Delete the current fitness values."""
        self._values = [None] * self.num_obj

    @property
    def wvalues(self) -> Tuple[float]:
        """Fitness weighted values.

        :rtype: tuple[float | None]
        """
        return tuple(
            v * w if v is not None else None
            for (v, w) in zip(self.values, self.weights)
        )

    @property
    def num_obj(self) -> int:
        """Number of objectives.

        :rtype: int
        """
        return len(self.weights)

    def update_value(self, value: float, obj_index: int) -> None:
        """Update the value of a fitnes objective.

        :param value: The new value
        :type value: float
        :param obj_index: Index of the objective
        :type obj_index: int
        :raises TypeError: If *value* is not a real number or *index* is not
            an integer number
        :raises ValueError: If *index* is negative or greatuer to or equal
            than the number of objectives
        """
        # Check the objective ndex
        obj_index = check_int(
            obj_index, "objective index", ge=0, lt=self.num_obj
        )

        # Check the value
        value = check_float(value, f"new value for objective {obj_index}")

        self._values[obj_index] = value

    def dominates(self, other: Fitness, which: slice = slice(None)) -> bool:
        """Check if this fitness dominates another one.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :param which: Slice indicating on which objectives the domination is
            tested. The default value is :class:`slice` (:data:`None`),
            representing every objective
        :type which: slice
        :return: :data:`True` if each objective of this fitness is not
            strictly worse than the corresponding objective of the *other* and
            at least one objective is strictly better
        :rtype: bool
        """
        not_equal = False

        for self_wval, other_wval, threshold in zip(
            self.wvalues[which], other.wvalues[which], self.thresholds[which]
        ):
            if self_wval - other_wval > threshold:
                not_equal = True
            elif other_wval - self_wval > threshold:
                not_equal = False
                break

        return not_equal

    def __hash__(self) -> int:
        """Return the hash number for this fitness.

        :rtype: int
        """
        return hash(self.values)

    def __gt__(self, other: Fitness) -> bool:
        """Lexicographic greater than operator.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :rtype: bool
        """
        return not self.__le__(other)

    def __ge__(self, other: Fitness) -> bool:
        """Lexicographic greater than or equal to operator.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :rtype: bool
        """
        return not self.__lt__(other)

    def __le__(self, other: Fitness) -> bool:
        """Lexicographic less than or equal to operator.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :rtype: bool
        """
        le = True

        for self_wval, other_wval, threshold in zip(
            self.wvalues, other.wvalues, self.thresholds
        ):
            if other_wval - self_wval > threshold:
                le = True
                break
            if self_wval - other_wval > threshold:
                le = False
                break

        return le

    def __lt__(self, other: Fitness) -> bool:
        """Lexicographic less than operator.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :rtype: bool
        """
        lt = False

        for self_wval, other_wval, threshold in zip(
            self.wvalues, other.wvalues, self.thresholds
        ):
            if other_wval - self_wval > threshold:
                lt = True
                break
            if self_wval - other_wval > threshold:
                lt = False
                break

        return lt

    def __eq__(self, other: Fitness) -> bool:
        """Lexicographic equal to operator.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :rtype: bool
        """
        eq = True

        for self_wval, other_wval, threshold in zip(
            self.wvalues, other.wvalues, self.thresholds
        ):
            if abs(self_wval - other_wval) > threshold:
                eq = False
                break

        return eq

    def __ne__(self, other: Fitness) -> bool:
        """Lexicographic not equal to operator.

        :param other: The other fitness
        :type other: ~culebra.abc.Fitness
        :rtype: bool
        """
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """Object representation.

        :rtype: str
        """
        return Base.__repr__(self)

    def __str__(self) -> str:
        """Object as a string.

        :rtype: str
        """
        return str(self.values)


class FitnessFunction(Base):
    """Base fitness function."""

    @property
    @abstractmethod
    def obj_weights(self) -> Tuple[int, ...]:
        """Objective weights.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: tuple[int]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The obj_weights property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: tuple[str]
        """
        raise NotImplementedError(
            "The obj_names property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def obj_thresholds(self) -> List[float]:
        """Objective similarity thresholds.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: list[float]
        :setter: Set new thresholds
        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        raise NotImplementedError(
            "The obj_thresholds property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @obj_thresholds.setter
    @abstractmethod
    def obj_thresholds(
        self, values: float | Sequence[float] | None
    ) -> None:
        """Set new objective similarity thresholds.

        This property setter must be overridden by subclasses to return a
        correct value.

        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        raise NotImplementedError(
            "The obj_thresholds property setter has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    def num_obj(self) -> int:
        """Number of objectives.

        :rtype: int
        """
        return len(self.obj_weights)

    @property
    def fitness_cls(self) -> Type[Fitness]:
        """Fitness class.

        :rtype: type[~culebra.abc.Fitness]
        """
        fitness_class_name = f"{self.__class__.__name__}.Fitness"
        fitness_class = type(
            fitness_class_name,
            (Fitness,),
            {
                "weights": self.obj_weights,
                "names": self.obj_names,
                "thresholds": self.obj_thresholds
            }
        )
        return fitness_class

    @abstractmethod
    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        Parameters *representatives* and *index* are used only for cooperative
        evaluations

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem, optional
        :type index: int
        :param representatives: Representative solutions of each species
            being optimized, optional
        :type representatives:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The evaluate method has not been implemented in the "
            f"{self.__class__.__name__} class")


class Species(Base):
    """Base class for all the species in culebra.

    Each solution returned by a :class:`~culebra.abc.Trainer` must belong
    to a species which constraints its parameter values.
    """

    @abstractmethod
    def is_member(self, sol: Solution) -> bool:
        """Check if a solution meets the constraints imposed by the species.

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: The solution
        :type sol: ~culebra.abc.Solution
        :return: :data:`True` if the solution belongs to the species, or
            :data:`False` otherwise
        :rtype: bool
        """
        raise NotImplementedError(
            "The check method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )


class Solution(Base):
    """Base class for all the solutions.

    All the solutions have the following attributes:

    * :attr:`~culebra.abc.Solution.species`: A species with the constraints
      that the solution must meet.
    * :attr:`~culebra.abc.Solution.fitness`: A
      :class:`~culebra.abc.Fitness` class for the solution.
    """

    species_cls = Species
    """Class for the species used by the :class:`~culebra.abc.Solution`
    class to constrain all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness]
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: ~culebra.abc.Species
        :param fitness_cls: The solutions's fitness class
        :type fitness_cls: type[~culebra.abc.Fitness]
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__()

        self._species = check_instance(
            species, "species", cls=self.species_cls
        )

        self.fitness = fitness_cls()

    @property
    def species(self) -> Species:
        """Solution's species.

        :rtype: ~culebra.abc.Species
        """
        return self._species

    @property
    def fitness(self) -> Fitness:
        """Solution's fitness.

        :rtype: ~culebra.abc.Fitness

        :setter: Set a new Fitness
        :param value: The new fitness
        :type value: ~culebra.abc.Fitness
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value: Fitness) -> None:
        """Set a new fitness for the solution.

        :param value: The new fitness
        :type value: ~culebra.abc.Fitness
        """
        self._fitness = check_instance(value, "fitness class", cls=Fitness)

    def delete_fitness(self) -> None:
        """Delete the solution's fitness."""
        del self._fitness.values

    def __hash__(self) -> int:
        """Return the hash number for this solution.

        The hash number is used for equality comparisons. Currently is
        implemented as the hash of the solution's string representation.

        :rtype: int
        """
        return hash(str(self))

    def dominates(self, other: Solution) -> bool:
        """Dominate operator.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`True` if each objective of the solution is not
            strictly worse than the corresponding objective of *other* and at
            least one objective is strictly better.
        :rtype: bool
        """
        return self.fitness.dominates(other.fitness)

    def __eq__(self, other: Solution) -> bool:
        """Equality test.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`True` if *other* codes the same solution, or
            :data:`False` otherwise
        :rtype: bool
        """
        return hash(self) == hash(other)

    def __ne__(self, other: Solution) -> bool:
        """Not equality test.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`False` if *other* codes the same solutions, or
            :data:`True` otherwise
        :rtype: bool
        """
        return not self.__eq__(other)

    def __lt__(self, other: Solution) -> bool:
        """Less than operator.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`True` if the solution's fitness is less than the
            *other*'s fitness
        :rtype: bool
        """
        return self.fitness < other.fitness

    def __gt__(self, other: Solution) -> bool:
        """Greater than operator.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`True` if the solution's fitness is greater than
            the *other*'s fitness
        :rtype: bool
        """
        return self.fitness > other.fitness

    def __le__(self, other: Solution) -> bool:
        """Less than or equal to operator.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`True` if the solution's fitness is less than or
            equal to the *other*'s fitness
        :rtype: bool
        """
        return self.fitness <= other.fitness

    def __ge__(self, other: Solution) -> bool:
        """Greater than or equal to operator.

        :param other: Other solution
        :type other: ~culebra.abc.Solution
        :return: :data:`True` if the solution's fitness is greater than
            or equal to the *other*'s fitness
        :rtype: bool
        """
        return self.fitness >= other.fitness

    def __copy__(self) -> Solution:
        """Shallow copy the solution.

        :return: The copied object
        :rtype: ~culebra.abc.Solution
        """
        cls = self.__class__
        result = cls(self.species, self.fitness.__class__)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Solution:
        """Deepcopy the solution.

        :param memo: Solution attributes
        :type memo: dict
        :return:  The copied solution
        :rtype: ~culebra.abc.Solution
        """
        cls = self.__class__
        result = cls(self.species, self.fitness.__class__)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the solution.

        :return: The reduction
        :rtype: tuple
        """
        return (
            self.__class__,
            (self.species, self.fitness.__class__),
            self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Solution:
        """Return a solution from a state.

        :param state: The state
        :type state: dict
        :return: The solution
        :rtype: ~culebra.abc.Solution
        """
        obj = cls(state['_species'], state['_fitness'].__class__)
        obj.__setstate__(state)
        return obj


class Trainer(Base):
    """Base class for all the training algorithms."""

    stats_names = TRAINER_STATS_NAMES
    """Statistics calculated each iteration."""

    objective_stats = TRAINER_OBJECTIVE_STATS
    """Statistics calculated for each objective."""

    def __init__(
        self,
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

        :param fitness_function: The training fitness function
        :type fitness_function: ~culebra.abc.FitnessFunction
        :param max_num_iters: Maximum number of iterations. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_MAX_NUM_ITERS` will
            be used. Defaults to :data:`None`
        :type max_num_iters: int
        :param custom_termination_func: Custom termination criterion. If set to
            :data:`None`,
            :meth:`~culebra.abc.Trainer._default_termination_func` is used.
            Defaults to :data:`None`
        :type custom_termination_func: ~collections.abc.Callable
        :param checkpoint_enable: Enable/disable checkpoining. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            will be used. Defaults to :data:`None`
        :type checkpoint_enable: bool
        :param checkpoint_freq: The checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            will be used. Defaults to :data:`None`
        :type checkpoint_freq: int
        :param checkpoint_filename: The checkpoint file path. If set to
            :data:`None`,
            :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` will be used.
            Defaults to :data:`None`
        :type checkpoint_filename: str
        :param verbose: The verbosity. If set to
            :data:`None`, :data:`__debug__` will be used. Defaults to
            :data:`None`
        :type verbose: bool
        :param random_seed: The seed, defaults to :data:`None`
        :type random_seed: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Init the superclass
        super().__init__()

        # Fitness function
        self.fitness_function = fitness_function

        # Maximum number of iterations
        self.max_num_iters = max_num_iters

        # Custom termination criterion
        self.custom_termination_func = custom_termination_func

        # Configure checkpointing, random seed and verbosity
        self.checkpoint_enable = checkpoint_enable
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_filename = checkpoint_filename
        self.verbose = verbose
        self.random_seed = random_seed

        # Container trainer, in case of being used in a distributed
        # configuration
        self.container = None

        # Index of this trainer, in case of being used in a distributed
        # configuration
        self.index = None

    @staticmethod
    def _get_fitness_values(sol: Solution) -> Tuple[float, ...]:
        """Return the fitness values of a solution.

        DEAP's :class:`~deap.tools.Statistics` class needs a function to
        obtain the fitness values of a solution.

        :param sol: The solution
        :type sol: ~culebra.abc.Solution
        :return: The fitness values of *sol*
        :rtype: tuple
        """
        return sol.fitness.values

    @property
    def fitness_function(self) -> FitnessFunction:
        """Training fitness function.

        :rtype: ~culebra.abc.FitnessFunction

        :setter: Set a new fitness function
        :param func: The new training fitness function
        :type func: ~culebra.abc.FitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        """
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, func: FitnessFunction) -> None:
        """Set a new training fitness function.

        :param func: The new training fitness function
        :type func: ~culebra.abc.FitnessFunction
        :raises TypeError: If *func* is not a valid fitness function
        """
        # Check the function
        self._fitness_function = check_instance(
            func, "fitness_function", FitnessFunction
        )

        # Reset the algorithm
        self.reset()

    @property
    def max_num_iters(self) -> int:
        """Maximum number of iterations.

        :rtype: int

        :setter: Set a new value for the maximum number of iterations
        :param value: The new maximum number of iterations. If set to
            :data:`None`, the default maximum number of iterations,
            :attr:`~culebra.DEFAULT_MAX_NUM_ITERS`, is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return (
            DEFAULT_MAX_NUM_ITERS
            if self._max_num_iters is None
            else self._max_num_iters
        )

    @max_num_iters.setter
    def max_num_iters(self, value: int | None) -> None:
        """Set the maximum number of iterations.

        :param value: The new maximum number of iterations. If set to
            :data:`None`, the default maximum number of iterations,
            :attr:`~culebra.DEFAULT_MAX_NUM_ITERS`, is chosen
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

        # Reset the algorithm
        self.reset()

    @property
    def current_iter(self) -> int:
        """Current iteration.

        :rtype: int
        """
        return self._current_iter

    @property
    def custom_termination_func(self) -> Callable[
            [Trainer],
            bool
    ]:
        """Custom termination criterion.

        The custom termination criterion must be a function which receives
        the trainer as its unique argument and returns a boolean value,
        :data:`True` if the search should terminate or :data:`False`
        otherwise.

        If more than one arguments are needed to define the termniation
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
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        return self._custom_termination_func

    @custom_termination_func.setter
    def custom_termination_func(
        self,
        func: Callable[
                [Trainer],
                bool
        ] | None
    ) -> None:
        """Set the custom termination criterion.

        The custom termination criterion must be a function which receives
        the trainer as its unique argument and returns a boolean value,
        :data:`True` if the search should terminate or :data:`False`
        otherwise.

        :param func: The new custom termination criterion. If set to
            :data:`None`, the default termination criterion is used.
            Defaults to :data:`None`
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not callable
        """
        # Check func
        self._custom_termination_func = (
            None if func is None else check_func(
                func, "custom termination criterion"
            )
        )

        # Reset the algorithm
        self.reset()

    @property
    def checkpoint_enable(self) -> bool:
        """Checkpointing enablement.

        :return: :data:`True` if checkpoinitng is enabled, or
            :data:`False` otherwise
        :rtype: bool
        :setter: Modify the checkpointing enablement
        :param value: New value for the checkpoint enablement. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not a boolean value
        """
        return (
            DEFAULT_CHECKPOINT_ENABLE if self._checkpoint_enable is None
            else self._checkpoint_enable
        )

    @checkpoint_enable.setter
    def checkpoint_enable(self, value: bool | None) -> None:
        """Modify the checkpointing enablement.

        :param value: New value for the checkpoint enablement. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not a boolean value
        """
        self._checkpoint_enable = (
            None if value is None else check_bool(
                value, "checkpoint enablement"
            )
        )

    @property
    def checkpoint_freq(self) -> int:
        """Checkpoint frequency.

        :rtype: int

        :setter: Modify the checkpoint frequency
        :param value: New value for the checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is not a positive number
        """
        return (
            DEFAULT_CHECKPOINT_FREQ if self._checkpoint_freq is None
            else self._checkpoint_freq
        )

    @checkpoint_freq.setter
    def checkpoint_freq(self, value: int | None) -> None:
        """Set a value for the checkpoint frequency.

        :param value: New value for the checkpoint frequency. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
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
    def checkpoint_filename(self) -> str:
        """Checkpoint file path.

        :rtype: str

        :setter: Modify the checkpoint file path
        :param value: New value for the checkpoint file path. If set to
            :data:`None`,
            :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` is chosen
        :type value: str
        :raises TypeError: If *value* is not a valid file name
        :raises ValueError: If the *value* extension is not
            :attr:`~culebra.SERIALIZED_FILE_EXTENSION`
        """
        return (
            DEFAULT_CHECKPOINT_FILENAME if self._checkpoint_filename is None
            else self._checkpoint_filename
        )

    @checkpoint_filename.setter
    def checkpoint_filename(self, value: str | None) -> None:
        """Set a value for the checkpoint file path.

        :param value: New value for the checkpoint file path. If set to
            :data:`None`,
            :attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` is chosen
        :type value: str
        :raises TypeError: If *value* is not a valid file name
        :raises ValueError: If the *value* extension is not
            :attr:`~culebra.SERIALIZED_FILE_EXTENSION`
        """
        # Check the value
        self._checkpoint_filename = (
            None if value is None else check_filename(
                value,
                name="checkpoint file name",
                ext=SERIALIZED_FILE_EXTENSION
            )
        )

    @property
    def verbose(self) -> bool:
        """Verbosity of this trainer.

        :rtype: bool

        :setter: Set a new value for the verbosity
        :param value: The verbosity. If set to :data:`None`, :data:`__debug__`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not boolean
        """
        return (
            DEFAULT_VERBOSITY if self._verbose is None
            else self._verbose
        )

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbosity of this trainer.

        :param value: The verbosity. If set to :data:`None`, :data:`__debug__`
            is chosen
        :type value: bool
        :raises TypeError: If *value* is not boolean
        """
        self._verbose = (
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

        # Reset the algorithm
        self.reset()

    @property
    def logbook(self) -> Logbook | None:
        """Trainer logbook.

        :return: A logbook with the statistics of the search or :data:`None`
            if the search has not been done yet
        :rtype: ~deap.tools.Logbook
        """
        return self._logbook

    @property
    def num_evals(self) -> int | None:
        """Number of evaluations performed while training.

        :return: The number of evaluations or :data:`None` if the search has
            not been done yet
        :rtype: int
        """
        return self._num_evals

    @property
    def runtime(self) -> float | None:
        """Training runtime.

        :return: The training runtime or :data:`None` if the search has not
            been done yet.

        :rtype: float
        """
        return self._runtime

    @property
    def index(self) -> int:
        """Trainer index.

        The trainer index is only used by distributed trainers. For the rest
        of trainers :attr:`~culebra.DEFAULT_INDEX` is used.

        :rtype: int
        :setter: Set a new value for trainer index.
        :param value: New value for the trainer index. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_INDEX` is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is a negative number
        """
        return (
            DEFAULT_INDEX if self._index is None
            else self._index
        )

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a value for trainer index.

        The trainer index is only used by distributed trainers. For the rest
        of trainers :attr:`~culebra.DEFAULT_INDEX` is used.

        :param value: New value for the trainer index. If set to
            :data:`None`, :attr:`~culebra.DEFAULT_INDEX` is chosen
        :type value: int
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is a negative number
        """
        # Check the value
        self._index = (
            None if value is None else check_int(value, "index", ge=0)
        )

    @property
    def container(self) -> Trainer | None:
        """Container of this trainer.

        The trainer container is only used by distributed trainers. For the
        rest of trainers defaults to :data:`None`.

        :rtype: ~culebra.abc.Trainer
        :setter: Set a new value for container of this trainer
        :param value: New value for the container or :data:`None`
        :type value: ~culebra.abc.Trainer
        :raises TypeError: If *value* is not a valid trainer
        """
        return self._container

    @container.setter
    def container(self, value: Trainer | None) -> None:
        """Set a container for this trainer.

        The trainer container is only used by distributed trainers. For the
        rest of trainers defaults to :data:`None`.

        :param value: New value for the container or :data:`None`
        :type value: ~culebra.abc.Trainer
        :raises TypeError: If *value* is not a valid trainer
        """
        # Check the value
        self._container = (
            None if value is None else check_instance(
                value, "container", cls=Trainer
            )
        )

    @property
    def representatives(self) -> Sequence[Sequence[Solution | None]] | None:
        """Representatives of the other species.

        Only used by cooperative trainers. If the trainer does not use
        representatives, :data:`None` is returned.

        :rtype:
            ~collections.abc.Sequence[~collections.abc.Sequence[~culebra.abc.Solution]]
        """
        return self._representatives

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Default state is a dictionary composed of the values of the
        :attr:`~culebra.abc.Trainer.logbook`,
        :attr:`~culebra.abc.Trainer.num_evals`,
        :attr:`~culebra.abc.Trainer.runtime`,
        :attr:`~culebra.abc.Trainer.current_iter`, and
        :attr:`~culebra.abc.Trainer.representatives`
        trainer properties, along with a private boolean attribute that informs
        if the search has finished and also the states of the :mod:`random`
        and :mod:`numpy.random` modules.

        If subclasses use any more properties to keep their state, the
        :meth:`~culebra.abc.Trainer._get_state` and
        :meth:`~culebra.abc.Trainer._set_state` methods must be
        overridden to take into account such properties.

        :rtype: dict
        """
        # Fill in the dictionary with the trainer state
        return dict(logbook=self._logbook,
                    num_evals=self._num_evals,
                    runtime=self._runtime,
                    current_iter=self._current_iter,
                    representatives=self._representatives,
                    search_finished=self._search_finished,
                    rnd_state=random.getstate(),
                    np_rnd_state=np.random.get_state())

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of this trainer.

        If subclasses use any more properties to keep their state, the
        :meth:`~culebra.abc.Trainer._get_state` and
        :meth:`~culebra.abc.Trainer._set_state` methods must be
        overridden to take into account such properties.

        :param state: The last loaded state
        :type state: dict
        """
        self._logbook = state["logbook"]
        self._num_evals = state["num_evals"]
        self._runtime = state["runtime"]
        self._current_iter = state["current_iter"]
        self._representatives = state["representatives"]
        self._search_finished = state["search_finished"]
        random.setstate(state["rnd_state"])
        np.random.set_state(state["np_rnd_state"])

    def reset(self) -> None:
        """Reset the trainer.

        Delete the state of the trainer (with
        :meth:`~culebra.abc.Trainer._reset_state`) and also all the internal
        data structures needed to perform the search
        (with :meth:`~culebra.abc.Trainer._reset_internals`).

        This method should be invoqued each time a hyper parameter is
        modified.
        """
        # Reset the trainer internals
        self._reset_internals()

        # Reset the trainer state
        self._reset_state()

    def evaluate(
        self,
        sol: Solution,
        fitness_func: Optional[FitnessFunction] = None,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Sequence[Solution | None]]] = None
    ) -> None:
        """Evaluate one solution.

        Its fitness will be modified according with the fitness function
        results. Besides, if called during training, the number of evaluations
        will be also updated.

        :param sol: The solution
        :type sol: ~culebra.abc.Solution
        :param fitness_func: The fitness function. If omitted, the default
            training fitness function
            (:attr:`~culebra.abc.Trainer.fitness_function`) is used
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param index: Index where *sol* should be inserted in the
            *representatives* sequence to form a complete solution for the
            problem. If omitted, :attr:`~culebra.abc.Trainer.index` is used
        :type index: int
        :param representatives: Sequence of representatives of other species
            or :data:`None` (if no representatives are needed to evaluate
            *sol*). If omitted, the current value of
            :attr:`~culebra.abc.Trainer.representatives` is used
        :type representatives:
            ~collections.abc.Sequence[~collections.abc.Sequence[~culebra.abc.Solution]]
        """
        # Select the representatives to be used
        context_seq = (
            self.representatives
            if representatives is None
            else representatives
        )

        # Select the fitness function to be used
        if fitness_func is not None:
            # Select the provided function
            func = fitness_func
            # If other fitness function is used, the solution's fitness
            # must be changed
            sol.fitness = func.fitness_cls()
        else:
            # Select the training fitness function
            func = self.fitness_function

        the_index = index if index is not None else self.index

        # If context is not None -> cooperation
        if context_seq is not None:
            # Different fitness trials values, one per solution in the context
            fitness_trials_values = []
            for context in context_seq:
                fitness_trials_values.append(
                    func.evaluate(
                        sol,
                        index=the_index,
                        representatives=context
                    ).values
                )

            self._set_cooperative_fitness(sol, fitness_trials_values)
        else:
            func.evaluate(sol, index=the_index)

        # Increase the number of evaluations performed
        if self._current_iter_evals is not None:
            self._current_iter_evals += (
                    len(context_seq)
                    if context_seq is not None
                    else 1
                )

    def _set_cooperative_fitness(
        self,
        sol: Solution,
        fitness_trials_values: [Sequence[Tuple[float]]]
    ) -> None:
        """Estimate a solution fitness from multiple evaluation trials.

        Applies an average of the fitness trials values. Trainers requiring
        another estimation should override this method.

        :param sol: The solution
        :type sol: ~culebra.abc.Solution
        :param fitness_trials_values: Sequence of fitness trials values. Each
            trial should be obtained with a different context in a cooperative
            trainer approach.
        :type fitness_trials_values: ~collections.abc.Sequence[tuple[float]]
        """
        sol.fitness.values = np.average(fitness_trials_values, axis=0)

    @abstractmethod
    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        This method must be overridden by subclasses to return a correct
        value.

        :return: One Hall of Fame for each species
        :rtype: ~collections.abc.Sequence[~deap.tools.HallOfFame]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The best_solutions method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def best_representatives(self) -> List[List[Solution]] | None:
        """Return a list of representatives from each species.

        Only used for cooperative trainers.

        :return: A list of representatives lists if the trainer is
            cooperative or :data:`None` in other cases.
        :rtype: list[list[~culebra.abc.Solution]]
        """
        return None

    def train(self, state_proxy: Optional[DictProxy] = None) -> None:
        """Perform the training process.

        :param state_proxy: Dictionary proxy to copy the output state of the
            trainer procedure. Only used if train is executed within a
            :class:`multiprocess.Process`. Defaults to :data:`None`
        :type state_proxy: ~multiprocess.managers.DictProxy
        """
        # Check state_proxy
        if state_proxy is not None:
            check_instance(state_proxy, "state proxy", cls=DictProxy)

        # Init the search process
        self._init_search()

        # Search the best solutions
        self._search()

        # Finish the search process
        self._finish_search()

        # Copy the ouput state (if needed)
        if state_proxy is not None:
            state = self._get_state()
            for key in state:
                # state_proxy[key] = deepcopy(state[key])
                state_proxy[key] = state[key]

    def test(
        self,
        best_found: Sequence[HallOfFame],
        fitness_func: Optional[FitnessFunction] = None,
        representatives: Optional[Sequence[Sequence[Solution]]] = None
    ) -> None:
        """Apply the test fitness function to the solutions found.

        Update the solutions in *best_found* with their test fitness.

        :param best_found: The best solutions found for each species.
            One :class:`~deap.tools.HallOfFame` for each species
        :type best_found: ~collections.abc.Sequence[~deap.tools.HallOfFame]
        :param fitness_func: Fitness function used to evaluate the final
            solutions. If ommited, the default training fitness function
            (:attr:`~culebra.abc.Trainer.fitness_function`) will be used
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param representatives: Sequence of representatives of other species
            or :data:`None` (if no representatives are needed). If omitted,
            the current value of
            :attr:`~culebra.abc.Trainer.representatives` is used
        :type representatives:
            ~collections.abc.Sequence[~collections.abc.Sequence[~culebra.abc.Solution]]
        :raises TypeError: If any parameter has a wrong type
        :raises ValueError: If any parameter has an invalid value.
        """
        # Check best_found
        check_sequence(
            best_found,
            "best found solutions",
            item_checker=partial(check_instance, cls=HallOfFame)
        )

        # Check fitness_function, if provided
        if fitness_func is not None:
            check_instance(
                fitness_func, "fitness function", cls=FitnessFunction
            )

        # Check representatives, if provided
        if representatives is not None:
            check_sequence(
                representatives,
                "representatives",
                item_checker=partial(check_instance, cls=Sequence)
            )
            for context in representatives:
                check_sequence(
                    context,
                    "representatives",
                    size=len(best_found),
                    item_checker=partial(check_instance, cls=Solution)
                )

        # For each hof
        for species_index, hof in enumerate(best_found):
            # For each solution found in this hof
            for sol in hof:
                self.evaluate(
                    sol, fitness_func, species_index, representatives
                )

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

    def _new_state(self) -> None:
        """Generate a new trainer state.

        If subclasses add any new  property to keep their state, this method
        should be overridden to initialize the full state of the trainer.
        """
        # Create a new logbook
        self._logbook = Logbook()

        # Init the logbook
        self._logbook.header = (
            list(
                self.stats_names
                if self.container is None
                else self.container.stats_names
            ) +
            self._stats.fields if self._stats else []
        )

        # Init the current iteration
        self._current_iter = 0

        # Init the number of evaluations
        self._num_evals = 0

        # Init the computing runtime
        self._runtime = 0

        # The trainer hasn't trained yet
        self._search_finished = False

        # Init the representatives
        self._init_representatives()

    def _init_state(self) -> None:
        """Init the trainer state.

        If there is any checkpoint file, the state is initialized from it with
        the :meth:`~culebra.abc.Trainer._load_state` method. Otherwise a new
        initial state is generated with the
        :meth:`~culebra.abc.Trainer._new_state` method.
        """
        # Init the trainer state
        create_new_state = True
        if self.checkpoint_enable:
            # Try to load the state of the last checkpoint
            try:
                self._load_state()
                create_new_state = False
            # If a checkpoint can't be loaded, make initialization
            except Exception:
                pass

        if create_new_state:
            self._new_state()

    def _reset_state(self) -> None:
        """Reset the trainer state.

        If subclasses overwrite the :meth:`~culebra.abc.Trainer._new_state`
        method to add any new property to keep their state, this method should
        also be overridden to reset the full state of the trainer.
        """
        self._logbook = None
        self._num_evals = None
        self._runtime = None
        self._representatives = None
        self._search_finished = None
        self._current_iter = None

    def _init_internals(self) -> None:
        """Set up the trainer internal data structures to start searching.

        Create all the internal objects, functions and data structures needed
        to run the search process. For the :class:`~culebra.abc.Trainer`
        class, only a :class:`~deap.tools.Statistics` object is created.
        Subclasses which need more objects or data structures should override
        this method.
        """
        # Initialize statistics object
        self._stats = Statistics(self._get_fitness_values)

        # Configure the stats
        for name, func in self.objective_stats.items():
            self._stats.register(name, func, axis=0)

        self._current_iter_evals = None
        self._current_iter_start_time = None

    def _reset_internals(self) -> None:
        """Reset the internal structures of the trainer.

        If subclasses overwrite the
        :meth:`~culebra.abc.Trainer._init_internals` method to add any new
        internal object, this method should also be overridden to reset all the
        internal objects of the trainer.
        """
        self._stats = None
        self._current_iter_evals = None
        self._current_iter_start_time = None

    def _init_search(self) -> None:
        """Init the search process.

        Initialize the state of the trainer (with
        :meth:`~culebra.abc.Trainer._init_state`) and all the internal data
        structures needed
        (with :meth:`~culebra.abc.Trainer._init_internals`) to perform the
        search.
        """
        # Init the trainer internals
        self._init_internals()

        # Init the state of the trainer
        self._init_state()

    def _search(self) -> None:
        """Apply the search algorithm.

        Execute the trainer until the termination condition is met. Each
        iteration is composed by the following steps:

        * :meth:`~culebra.abc.Trainer._start_iteration`
        * :meth:`~culebra.abc.Trainer._preprocess_iteration`
        * :meth:`~culebra.abc.Trainer._do_iteration`
        * :meth:`~culebra.abc.Trainer._postprocess_iteration`
        * :meth:`~culebra.abc.Trainer._finish_iteration`
        * :meth:`~culebra.abc.Trainer._do_iteration_stats`
        """
        # Run all the iterations
        while not self._termination_criterion():
            self._start_iteration()
            self._preprocess_iteration()
            self._do_iteration()
            self._postprocess_iteration()
            self._finish_iteration()
            self._do_iteration_stats()
            self._current_iter += 1

    def _finish_search(self) -> None:
        """Finish the search process.

        This method is called after the search has finished. It can be
        overridden to perform any treatment of the solutions found.
        """
        self._search_finished = True

        # Save the last state
        if self.checkpoint_enable:
            self._save_state()

    def _start_iteration(self) -> None:
        """Start an iteration.

        Prepare the iteration metrics (number of evaluations, execution time)
        before each iteration is run.
        """
        self._current_iter_evals = 0
        self._current_iter_start_time = perf_counter()

    def _preprocess_iteration(self) -> None:
        """Preprocess before doing the iteration.

        Subclasses should override this method to make any preprocessment
        before performing an iteration.
        """

    @abstractmethod
    def _do_iteration(self) -> None:
        """Implement an iteration of the search process.

        This abstract method should be implemented by subclasses in order to
        implement the desired behavior.
        """

    def _postprocess_iteration(self) -> None:
        """Postprocess after doing the iteration.

        Subclasses should override this method to make any postprocessment
        after performing an iteration.
        """

    def _finish_iteration(self) -> None:
        """Finish an iteration.

        Finish the iteration metrics (number of evaluations, execution time)
        after each iteration is run.
        """
        end_time = perf_counter()
        self._runtime += end_time - self._current_iter_start_time
        self._num_evals += self._current_iter_evals

        # Save the trainer state at each checkpoint
        if (self.checkpoint_enable and
                self._current_iter % self.checkpoint_freq == 0):
            self._save_state()

    def _do_iteration_stats(self) -> None:
        """Perform the iteration stats.

        This method should be implemented by subclasses in order to perform
        the adequate stats.
        """

    def _default_termination_func(self) -> bool:
        """Default termination criterion.

        :return: :data:`True` if :attr:`~culebra.abc.Trainer.max_num_iters`
            iterations have been run
        :rtype: bool
        """
        if self._current_iter < self.max_num_iters:
            return False

        return True

    def _termination_criterion(self) -> bool:
        """Control the search termination.

        :return: :data:`True` if either the default termination criterion or
            a custom termination criterion is met. The default termination
            criterion is implemented by the
            :meth:`~culebra.abc.Trainer._default_termination_func` method.
            Another custom termination criterion can be set with
            :attr:`~culebra.abc.Trainer.custom_termination_func` method.
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

    def _init_representatives(self) -> None:
        """Init the representatives of the other species.

        Only used for cooperative approaches, which need representatives of
        all the species to form a complete solution for the problem.
        Cooperative subclasses of the :class:`~culebra.abc.Trainer` class
        should override this method to get the representatives of the other
        species initialized.
        """
        self._representatives = None

    def __copy__(self) -> Trainer:
        """Shallow copy the trainer.

        :return: The copied triner
        :rtype: ~culebra.abc.Trainer
        """
        cls = self.__class__
        result = cls(self.fitness_function)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Trainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: dict
        :return: The copied triner
        :rtype: ~culebra.abc.Trainer
        """
        cls = self.__class__
        result = cls(self.fitness_function)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, (self.fitness_function,), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Trainer:
        """Return a trainer from a state.

        :param state: The state
        :type state: dict
        :return: The triner
        :rtype: ~culebra.abc.Trainer
        """
        obj = cls(state['_fitness_function'])
        obj.__setstate__(state)
        return obj


# Exported symbols for this module
__all__ = [
    'Base',
    'Fitness',
    'FitnessFunction',
    'Species',
    'Solution',
    'Trainer'
]
