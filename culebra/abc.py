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
from typing import Any
from collections.abc import Sequence, Callable
import gzip
from copy import deepcopy
from functools import partial

import numpy as np
import dill
from deap.tools import Logbook, HallOfFame

from culebra import (
    SERIALIZED_FILE_EXTENSION,
    DEFAULT_SIMILARITY_THRESHOLD
)
from culebra.checker import (
    check_int,
    check_float,
    check_instance,
    check_sequence,
    check_filename,
    check_func
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


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

    def __setstate__(self, state: dict[str, Any]) -> None:
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
        return deepcopy(obj)


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

    def __init__(self, values: Sequence[float, ...] | None = None) -> None:
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
    def values(self) -> tuple[float | None]:
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
    def wvalues(self) -> tuple[float]:
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

    def __init__(self) -> None:
        """Construct the fitness function."""
        # Init the superclasses
        super().__init__()

        # Set the default similarity thresholds
        self.obj_thresholds = None

    @property
    @abstractmethod
    def obj_weights(self) -> tuple[int, ...]:
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
    def obj_names(self) -> tuple[str, ...]:
        """Objective names.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: tuple[str]
        """
        raise NotImplementedError(
            "The obj_names property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @property
    def _default_similarity_threshold(self) -> float:
        """Default similarity threshold for fitnesses.

        :return: :attr:`~culebra.DEFAULT_SIMILARITY_THRESHOLD`
        :rtype: float
        """
        return DEFAULT_SIMILARITY_THRESHOLD

    @property
    def obj_thresholds(self) -> list[float]:
        """Objective similarity thresholds.

        :rtype: list[float]
        :setter: Set new thresholds.
        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.abc.FitnessFunction._default_similarity_threshold`
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if self._obj_thresholds is None:
            return [
                self._default_similarity_threshold
            ] * self.num_obj

        return self._obj_thresholds

    @obj_thresholds.setter
    def obj_thresholds(
        self, values: float | Sequence[float] | None
    ) -> None:
        """Set new objective similarity thresholds.

        :param values: The new values. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a :class:`~collections.abc.Sequence`.
            If set to :data:`None`, all the thresholds are set to
            :attr:`~culebra.abc.FitnessFunction._default_similarity_threshold`
        :type values: float | ~collections.abc.Sequence[float]
        :raises TypeError: If neither a real number nor a
            :class:`~collections.abc.Sequence` of real numbers is provided
        :raises ValueError: If any value is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if values is None:
            self._obj_thresholds = None
        elif isinstance(values, Sequence):
            self._obj_thresholds = check_sequence(
                values,
                "objective similarity thresholds",
                size=self.num_obj,
                item_checker=partial(check_float, ge=0)
            )
        else:
            self._obj_thresholds = [
                check_float(values, "objective similarity threshold", ge=0)
            ] * self.num_obj

    @property
    def num_obj(self) -> int:
        """Number of objectives.

        :rtype: int
        """
        return len(self.obj_weights)

    @property
    def fitness_cls(self) -> type[Fitness]:
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
        index: int | None = None,
        cooperators: Sequence[Solution | None] | None = None
    ) -> Fitness:
        """Evaluate a solution.

        Parameters *cooperators* and *index* are used only for cooperative
        evaluations

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param index: Index where *sol* should be inserted in the cooperators
            sequence to form a complete solution for the problem, optional
        :type index: int
        :param cooperators: Cooperators of each species  being optimized,
            optional
        :type cooperators:
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
        fitness_cls: type[Fitness]
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
        result = cls(
            deepcopy(self.species, memo),
            deepcopy(self.fitness.__class__)
        )
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo | {id(self.species): result.species}
            )
        )
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
        return deepcopy(obj)


class Trainer(Base):
    """Base class for all the training algorithms."""

    def __init__(self) -> None:
        """Construct a trainer."""
        # Init the superclass
        super().__init__()

        # Set the default cooperative fitness estimation function
        self.cooperative_fitness_estimation_func = None

    @property
    @abstractmethod
    def fitness_func(self) -> FitnessFunction:
        """Training fitness function.

        :rtype: ~culebra.abc.FitnessFunction

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: tuple[str]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The fitness_func property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def iteration_metric_names(self) -> tuple(str):
        """Names of the metrics recorded each iteration.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: tuple[str]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The iteration_metric_names property has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    @abstractmethod
    def iteration_obj_stats(self) -> dict(str, Callable):
        """Stats applied to each objective every iteration.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: dict
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The iteration_metric_names property has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    @abstractmethod
    def training_finished(self) -> bool:
        """Check if training has finished.
        
        This property must be overridden by subclasses to return a correct
        value.

        :return: :data`True` if training has finished
        :rtype: bool

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The training_finished property has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    @abstractmethod
    def logbook(self) -> Logbook | None:
        """Trainer logbook.

        This property must be overridden by subclasses to return a correct
        value.

        :return: A logbook with the statistics of the training or :data:`None`
            if the training has not been done yet
        :rtype: ~deap.tools.Logbook
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The logbook property has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    @abstractmethod
    def num_evals(self) -> int | None:
        """Number of evaluations performed while training.

        This property must be overridden by subclasses to return a correct
        value.

        :return: The number of evaluations or :data:`None` if the training has
            not been done yet
        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The num_evals property has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    @abstractmethod
    def num_iters(self) -> int | None:
        """Number of iterations performed while training.

        This property must be overridden by subclasses to return a correct
        value.

        :return: The number of iterations or :data:`None` if the training has
            not been done yet
        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The num_iters property has not been implemented in "
            f"the {self.__class__.__name__} class")

    @property
    @abstractmethod
    def runtime(self) -> float | None:
        """Training runtime.

        This property must be overridden by subclasses to return a correct
        value.

        :return: The training runtime or :data:`None` if the training has not
            been done yet.

        :rtype: float
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The runtime property has not been implemented in "
            f"the {self.__class__.__name__} class")

    def evaluate(
        self,
        sol: Solution,
        fitness_func: FitnessFunction = None,
        index: int | None = None,
        cooperators: Sequence[Sequence[Solution | None]] | None = None
    ) -> int:
        """Evaluate one solution.

        Its fitness will be modified according with the fitness function
        results.

        :param sol: The solution
        :type sol: ~culebra.abc.Solution
        :param fitness_func: The fitness function. If omitted, the training
            function is used
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param index: Index where *sol* should be inserted in the
            *cooperators* sequence to form a complete solution for the
            problem
        :type index: int
        :param cooperators: Sequence of cooperators of other species or
            :data:`None` (if no cooperators are needed to evaluate *sol*)
        :type cooperators:
            ~collections.abc.Sequence[~collections.abc.Sequence[~culebra.abc.Solution]]
        :return: The number of evaluations performed
        :rtype: int
        """
        if fitness_func is None:
            fitness_func = self.fitness_func
        elif fitness_func is not self.fitness_func:
            # Check the fitness function
            fitness_func = check_instance(
                fitness_func, "fitness function", FitnessFunction
            )
            sol.fitness = fitness_func.fitness_cls()

        # If cooperators is not None -> cooperation
        if cooperators is not None:
            # Different fitness trials values, one per solution in the context
            fitness_trials_values = []
            for context in cooperators:
                fitness_trials_values.append(
                    fitness_func.evaluate(
                        sol,
                        index=index,
                        cooperators=context
                    ).values
                )
            sol.fitness.values = self.cooperative_fitness_estimation_func(
                fitness_trials_values
            )
        else:
            fitness_func.evaluate(sol, index=index)

        # Return the number of evaluations performed
        return 1 if cooperators is None else len(cooperators)

    @property
    def _default_cooperative_fitness_estimation_func(
        self
    ) -> Callable[Sequence[Sequence[float]], Sequence[float]]:
        """Default cooperative fitness estimation function.
        
        Return the average of all fitness trials.
        """
        return lambda fitness_trials: np.average(fitness_trials, axis=0)

    @property
    def cooperative_fitness_estimation_func(
        self
    ) -> Callable[Sequence[Sequence[float]], Sequence[float]]:
        """Cooperative fitness estimation function.

        Funtion to estimate the cooperative fitness of a solution from
        several fitness trials with different cooperators. Only used by
        cooperative trainers.

        :rtype: ~collections.abc.Callable

        :setter: Set a new function
        :param func: The new function
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not a valid function
        """
        return (
            self._default_cooperative_fitness_estimation_func
            if self._cooperative_fitness_estimation_func is None
            else self._cooperative_fitness_estimation_func
        )

    @cooperative_fitness_estimation_func.setter
    def cooperative_fitness_estimation_func(
        self, func: Callable[Sequence[Sequence[float]], Sequence[float]]
    ) -> None:
        """Set a new cooperative fitness estimation function.

        Funtion to estimate the cooperative fitness of a solution from
        several fitness trials with different cooperators. Only used by
        cooperative trainers.

        :param func: The new function
        :type func: ~collections.abc.Callable
        :raises TypeError: If *func* is not a valid function
        """
        self._cooperative_fitness_estimation_func = (
            None if func is None else
            check_func(
                func, "cooperative fitness estimation function"
            )
        )

        # Reset the trainer
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset the trainer.

        This method must be overridden by subclasses to allow reseting the
        trainer.
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The reset method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _init_training(self) -> None:
        """Init the training process.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _init_training method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def _do_training(self) -> None:
        """Apply the training algorithm.

        This abstract method should be implemented by subclasses in order to
        implement the desired behavior.
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _do_training method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def _finish_training(self) -> None:
        """Finish the training process.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _finish_training method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )


    def train(self):
        """Perform the training process."""
        # Init the training process
        self._init_training()

        # Perform the training
        self._do_training()

        # Finish the training process
        self._finish_training()

    def test(
        self,
        best_found: Sequence[HallOfFame],
        fitness_func: FitnessFunction,
        cooperators: Sequence[Sequence[Solution]] | None = None
    ) -> None:
        """Apply the test fitness function to the solutions found.

        Update the solutions in *best_found* with their test fitness.

        :param best_found: The best solutions found for each species.
            One :class:`~deap.tools.HallOfFame` for each species
        :type best_found: ~collections.abc.Sequence[~deap.tools.HallOfFame]
        :param fitness_func: Fitness function used to evaluate the final
            solutions
        :type fitness_func: ~culebra.abc.FitnessFunction
        :param cooperators: Sequence of cooperators of other species or
            :data:`None` (if no cooperators are needed)
        :type cooperators: ~collections.abc.Sequence[
            ~collections.abc.Sequence[~culebra.abc.Solution]]
        :raises TypeError: If any parameter has a wrong type
        :raises ValueError: If any parameter has an invalid value.
        """
        # Check best_found
        check_sequence(
            best_found,
            "best found solutions",
            item_checker=partial(check_instance, cls=HallOfFame)
        )

        # Check cooperators, if provided
        if cooperators is not None:
            check_sequence(
                cooperators,
                "cooperators",
                item_checker=partial(check_instance, cls=Sequence)
            )
            for context in cooperators:
                check_sequence(
                    context,
                    "cooperators",
                    size=len(best_found),
                    item_checker=partial(check_instance, cls=Solution)
                )

        # For each hof
        for species_index, hof in enumerate(best_found):
            # For each solution found in this hof
            for sol in hof:
                self.evaluate(
                    sol, fitness_func, species_index, cooperators
                )

    @abstractmethod
    def best_solutions(self) -> tuple[HallOfFame]:
        """Get the best solutions found for each species.

        This method must be overridden by subclasses to return a correct
        value.

        :return: One Hall of Fame for each species
        :rtype: tuple[~deap.tools.HallOfFame]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The best_solutions method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def best_cooperators(self) -> list[list[Solution | None]] | None:
        """Return a list of cooperators from each species.

        Only used for cooperative trainers.

        :return: A list of cooperators lists if the trainer is cooperative or
            :data:`None` in other cases
        :rtype: list[list[~culebra.abc.Solution]]
        """
        return None


# Exported symbols for this module
__all__ = [
    'Base',
    'Fitness',
    'FitnessFunction',
    'Species',
    'Solution',
    'Trainer'
]
