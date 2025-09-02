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

Fundamental classes to solve optimization problems. The :py:mod:`~culebra.abc`
module defines:

  * A :py:class:`~culebra.abc.Base` class from which all the classes of culebra
    inherit
  * A :py:class:`~culebra.abc.Species` class to define the characteristics of
    solutions belonging to a given domain
  * A :py:class:`~culebra.abc.Solution` class, which will be used within the
    :py:class:`~culebra.abc.Trainer` class to seek the best solution(s) for a
    problem
  * A :py:class:`~culebra.abc.Fitness` class to store the fitness values for
    each :py:class:`~culebra.abc.Solution`
  * A :py:class:`~culebra.abc.FitnessFunction` class to evaluate a
    :py:class:`~culebra.abc.Solution` during the training process and assign
    it values for its :py:class:`~culebra.abc.Fitness`
  * A :py:class:`~culebra.abc.Trainer` class to find solutions for a problem
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
from deap.base import Fitness as DeapFitness
from deap.tools import Logbook, Statistics, HallOfFame

from culebra import (
    DEFAULT_SIMILARITY_THRESHOLD,
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
:py:class:`~culebra.abc.Trainer`.
"""

TRAINER_OBJECTIVE_STATS = {
    "Avg": np.mean,
    "Std": np.std,
    "Min": np.min,
    "Max": np.max,
}
"""Statistics calculated for each objective within a
:py:class:`~culebra.abc.Trainer`.
"""


class Base:
    """Base for all classes in culebra."""

    def dump(self, filename: str) -> None:
        """Serialize this object and save it to a file.

        :param filename: The file name.
        :type filename: :py:class:`~str`
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If the *filename* extension is not
            :py:attr:`~culebra.SERIALIZED_FILE_EXTENSION`
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
        :type filename: :py:class:`~str`
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If the *filename* extension is not
            :py:attr:`~culebra.SERIALIZED_FILE_EXTENSION`
        """
        filename = check_filename(
            filename,
            name="serialized file name",
            ext=SERIALIZED_FILE_EXTENSION
        )

        with gzip.open(filename, 'rb') as f:
            return cls.__fromstate__(dill.load(f).__dict__)

    def __copy__(self) -> Base:
        """Shallow copy the object."""
        cls = self.__class__
        result = cls()
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Base:
        """Deepcopy the object.

        :param memo: Object attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the object
        :rtype: The same than the original object
        """
        cls = self.__class__
        result = cls()
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __setstate__(self, state: dict) -> None:
        """Set the state of the object.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self) -> tuple:
        """Reduce the object.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (), self.__dict__)

    def __repr__(self) -> str:
        """Return the object representation."""
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

        :param state: The state.
        :type state: :py:class:`~dict`
        """
        obj = cls()
        obj.__setstate__(state)
        return obj


class Fitness(DeapFitness, Base):
    """Define the base class for the fitness of a solution."""

    weights = None
    """A :py:class:`tuple` containing an integer value for each objective
    being optimized. The weights are used in the fitness comparison. They are
    shared among all fitnesses of the same type. When subclassing
    :py:class:`~culebra.abc.Fitness`, the weights must be defined as a tuple
    where each element is associated to an objective. A negative weight element
    corresponds to the minimization of the associated objective and positive
    weight to the maximization. This attribute is inherited from
    :py:class:`deap.base.Fitness`.

    .. note::
        If weights is not defined during subclassing, the following error will
        occur at instantiation of a subclass fitness object:

        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """

    names = None
    """Names of the objectives.

    If not defined, generic names will be used.
    """

    thresholds = None
    """A list with the similarity thresholds defined for all the objectives.
    Fitness objects are compared lexicographically. The comparison applies a
    similarity threshold to assume that two fitness values are similar (if
    their difference is lower than or equal to the similarity threshold).
    If not defined, 0 will be used for each objective.
    """

    def __init__(self, values: Tuple[float, ...] = ()) -> None:
        """Construct a default fitness object.

        :param values: Initial values for the fitness, defaults to ()
        :type values: :py:class:`tuple`, optional
        """
        # Init the superclasses
        super().__init__(values)
        self._num_evaluations = [0] * self.num_obj

    @property
    def num_evaluations(self) -> List[float]:
        """Get the number of times each objective has been evaluated.

        Useful to implement Monte Carlo cross-validation

        :type: :py:class:`list` of :py:class:`float`
        """
        return self._num_evaluations

    @DeapFitness.values.deleter
    def values(self):
        """Update the values deletion to reset the number of evaluations."""
        super().delValues()
        self._num_evaluations = [0] * self.num_obj

    @property
    def num_obj(self) -> int:
        """Get the number of objectives.

        :type: :py:class:`int`
        """
        return len(self.weights)

    @property
    def pheromone_amount(self) -> Tuple[float, ...]:
        """Return the amount of pheromone to be deposited.

        This property is intended for ACO-based approaches. By default, the
        reciprocal of an objective fitness will be used for minimization
        objectives, while the objective's value will be used for maximization
        problems. Fitness classes pretending a different behavior should
        override this property.

        :return: The amount of pheromone to be deposited for each objective
        :rtype: :py:class:`tuple` of py:class:`float`
        """
        return tuple(
            1/self.values[i] if self.weights[i] < 0 else self.values[i]
            for i in range(len(self.wvalues))
        )

    def dominates(self, other: Fitness, which: slice = slice(None)) -> bool:
        """Check if this fitness dominates another one.

        :param other: The other fitness
        :type other: :py:class:`~culebra.abc.Fitness`
        :param which: Slice indicating on which objectives the domination is
            tested. The default value is :py:class:`slice` (:py:data:`None`),
            representing every objective
        :type which: :py:class:`slice`
        :return: :py:data:`True` if each objective of this fitness is not
            strictly worse than the corresponding objective of the *other* and
            at least one objective is strictly better
        :rtype: :py:class:`bool`
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
        """Return the hash number for this fitness."""
        return hash(self.wvalues)

    def __gt__(self, other: Fitness) -> bool:
        """Lexicographic greater than operator.

        :param other: The other fitness
        :type other: :py:class:`~culebra.abc.Fitness`
        """
        return not self.__le__(other)

    def __ge__(self, other: Fitness) -> bool:
        """Lexicographic greater than or equal to operator.

        :param other: The other fitness
        :type other: :py:class:`~culebra.abc.Fitness`
        """
        return not self.__lt__(other)

    def __le__(self, other: Fitness) -> bool:
        """Lexicographic less than or equal to operator.

        :param other: The other fitness
        :type other: :py:class:`~culebra.abc.Fitness`
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
        :type other: :py:class:`~culebra.abc.Fitness`
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
        :type other: :py:class:`~culebra.abc.Fitness`
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
        :type other: :py:class:`~culebra.abc.Fitness`
        """
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """Return the object representation."""
        return Base.__repr__(self)

    def __str__(self) -> str:
        """Return the object as a string."""
        return DeapFitness.__str__(self)


class FitnessFunction(Base):
    """Base fitness function."""

    def __init__(
        self
    ) -> None:
        """Create the fitness function."""
        super().__init__()
        self.obj_thresholds = None

    @property
    @abstractmethod
    def obj_weights(self) -> Tuple[int, ...]:
        """Get the objective weights.

        This property must be overridden by subclasses to return a correct
        value.

        :type: :py:class:`tuple` of :py:class:`int`
        :raises NotImplementedError: if has not been overridden
        """
        raise NotImplementedError(
            "The obj_weights property has not been implemented in the "
            f"{self.__class__.__name__} class")

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Get the objective names.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        suffix_len = len(str(self.num_obj-1))

        return tuple(
            f"obj_{i:0{suffix_len}d}" for i in range(self.num_obj)
        )

    @property
    def num_obj(self) -> int:
        """Get the number of objectives.

        :type: :py:class:`int`
        """
        return len(self.obj_weights)

    @property
    def obj_thresholds(self) -> List[float]:
        """Get and set new objective similarity thresholds.

        :getter: Return the current thresholds
        :setter: Set new thresholds. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a
            :py:class:`~collections.abc.Sequence`.
        :type thresholds: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If neither a real number nor a
            :py:class:`~collections.abc.Sequence` of real numbers id provided
        :raises ValueError: If any threshold is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        return self._obj_thresholds

    @obj_thresholds.setter
    def obj_thresholds(
        self, values: float | Sequence[float] | None
    ) -> None:
        """Get and set new objective similarity thresholds.

        :getter: Return the current thresholds
        :setter: Set new thresholds. If only a single value is provided, the
            same threshold will be used for all the objectives. Different
            thresholds can be provided in a
            :py:class:`~collections.abc.Sequence`.
        :type thresholds: :py:class:`float` or
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
        :raises TypeError: If neither a real number nor a
            :py:class:`~collections.abc.Sequence` of real numbers id provided
        :raises ValueError: If any threshold is negative
        :raises ValueError: If the length of the thresholds sequence does not
            match the number of objectives
        """
        if isinstance(values, Sequence):
            self._obj_thresholds = check_sequence(
                values,
                "objective similarity thresholds",
                size=self.num_obj,
                item_checker=partial(check_float, ge=0)
            )
        elif values is not None:
            self._obj_thresholds = [
                check_float(values, "objective similarity threshold", ge=0)
            ] * self.num_obj
        else:
            self._obj_thresholds = (
                [DEFAULT_SIMILARITY_THRESHOLD] * self.num_obj
            )

    @property
    def fitness_cls(self):
        """Return the fitness class for the fitness function.

        Subclasses must override this property to generate an adequate fitness
        class.
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

    @property
    def is_noisy(self) -> int:
        """Return :py:data:`True` if the fitness function is noisy."""
        return False

    @abstractmethod
    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        Parameters *representatives* and *index* are used only for cooperative
        evaluations

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: Solution to be evaluated.
        :type sol: :py:class:`~culebra.abc.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem
        :type index: :py:class:`int`, optional
        :param representatives: Representative solutions of each species
            being optimized
        :type representatives: A :py:class:`~collections.abc.Sequence`
            containing instances of :py:class:`~culebra.abc.Solution`,
            optional
        :raises NotImplementedError: if has not been overridden
        :return: The fitness values for *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        raise NotImplementedError(
            "The evaluate method has not been implemented in the "
            f"{self.__class__.__name__} class")


class Species(Base):
    """Base class for all the species in culebra.

    Each solution returned by a :py:class:`~culebra.abc.Trainer` must belong
    to a species which constraints its parameter values.
    """

    @abstractmethod
    def is_member(self, sol: Solution) -> bool:
        """Check if a solution meets the constraints imposed by the species.

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: The solution
        :type sol: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if the solution belongs to the species, or
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        raise NotImplementedError(
            "The check method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )


class Solution(Base):
    """Base class for all the solutions.

    All the solutions have the following attributes:

    * :py:attr:`~culebra.abc.Solution.species`: A species with the constraints
      that the solution must meet.
    * :py:attr:`~culebra.abc.Solution.fitness`: A
      :py:class:`~culebra.abc.Fitness` class for the solution.
    """

    species_cls = Species
    """Class for the species used by the :py:class:`~culebra.abc.Solution`
    class to constrain all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness]
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: :py:class:`~culebra.abc.Solution.species_cls`
        :param fitness: The solutions's fitness class
        :type fitness: Any subclass of :py:class:`~culebra.abc.Fitness`
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
        """Get the solution's species.

        :return: The species
        :rtype: :py:class:`~culebra.abc.Species`
        """
        return self._species

    @property
    def fitness(self) -> Fitness:
        """Get and set the solution's fitness.

        :getter: Return the current fitness
        :setter: Set a new Fitness
        :type: :py:class:`~culebra.abc.Fitness`
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value: Fitness) -> None:
        """Set a new fitness for the solution.

        :param value: The new fitness
        :type value: :py:class:`~culebra.abc.Fitness`
        """
        self._fitness = check_instance(value, "fitness class", cls=Fitness)

    def delete_fitness(self) -> None:
        """Delete the solution's fitness."""
        self._fitness = self.fitness.__class__()

    def __hash__(self) -> int:
        """Return the hash number for this solution.

        The hash number is used for equality comparisons. Currently is
        implemented as the hash of the solution's string representation.
        """
        return hash(str(self))

    def dominates(self, other: Solution) -> bool:
        """Dominate operator.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if each objective of the solution is not
            strictly worse than the corresponding objective of *other* and at
            least one objective is strictly better.
        :rtype: :py:class:`bool`
        """
        return self.fitness.dominates(other.fitness)

    def __eq__(self, other: Solution) -> bool:
        """Equality test.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if *other* codes the same solution, or
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        return hash(self) == hash(other)

    def __ne__(self, other: Solution) -> bool:
        """Not equality test.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`False` if *other* codes the same solutions, or
            :py:data:`True` otherwise
        :rtype: :py:class:`bool`
        """
        return not self.__eq__(other)

    def __lt__(self, other: Solution) -> bool:
        """Less than operator.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if the solution's fitness is less than the
            *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness < other.fitness

    def __gt__(self, other: Solution) -> bool:
        """Greater than operator.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if the solution's fitness is greater than
            the *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness > other.fitness

    def __le__(self, other: Solution) -> bool:
        """Less than or equal to operator.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if the solution's fitness is less than or
            equal to the *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness <= other.fitness

    def __ge__(self, other: Solution) -> bool:
        """Greater than or equal to operator.

        :param other: Other solution
        :type other: :py:class:`~culebra.abc.Solution`
        :return: :py:data:`True` if the solution's fitness is greater than
            or equal to the *other*'s fitness
        :rtype: :py:class:`bool`
        """
        return self.fitness >= other.fitness

    def __copy__(self) -> Solution:
        """Shallow copy the solution."""
        cls = self.__class__
        result = cls(self.species, self.fitness.__class__)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Solution:
        """Deepcopy the solution.

        :param memo: solution attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the solution
        :rtype: :py:class:`~culebra.abc.Solution`
        """
        cls = self.__class__
        result = cls(self.species, self.fitness.__class__)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the solution.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (
            self.__class__,
            (self.species, self.fitness.__class__),
            self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Solution:
        """Return a solution from a state.

        :param state: The state.
        :type state: :py:class:`~dict`
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
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_enable: :py:class:`bool`, optional
        :param checkpoint_freq: The checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            will be used. Defaults to :py:data:`None`
        :type checkpoint_freq: :py:class:`int`, optional
        :param checkpoint_filename: The checkpoint file path. If set to
            :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` will be used.
            Defaults to :py:data:`None`
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

        DEAP's :py:class:`~deap.tools.Statistics` class needs a function to
        obtain the fitness values of a solution.

        :param sol: The solution
        :type sol: Any subclass of :py:class:`~culebra.abc.Solution`
        :return: The fitness values of *sol*
        :rtype: :py:class:`tuple`
        """
        return sol.fitness.values

    @property
    def fitness_function(self) -> FitnessFunction:
        """Get and set the training fitness function.

        :getter: Return the fitness function
        :setter: Set a new fitness function
        :type: :py:class:`~culebra.abc.FitnessFunction`
        :raises TypeError: If set to a value which is not a fitness function
        """
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, func: FitnessFunction) -> None:
        """Set a new training fitness function.

        :param func: New training fitness function
        :type func: :py:class:`~culebra.abc.FitnessFunction`
        :raises TypeError: If set to a value which is not a fitness function
        """
        # Check the function
        self._fitness_function = check_instance(
            func, "fitness_function", FitnessFunction
        )

        # Reset the algorithm
        self.reset()

    @property
    def max_num_iters(self) -> int:
        """Get and set the maximum number of iterations.

        :getter: Return the current maximum number of iterations
        :setter: Set a new value for the maximum number of iterations. If set
            to :py:data:`None`, the default maximum number of iterations,
            :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS`, is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
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
            :py:data:`None`, the default maximum number of iterations,
            :py:attr:`~culebra.DEFAULT_MAX_NUM_ITERS`, is chosen
        :type value: An integer value or :py:data:`None`
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
        """Return the current iteration.

        :type: :py:class:`int`
        """
        return self._current_iter

    @property
    def custom_termination_func(self) -> Callable[
            [Trainer],
            bool
    ]:
        """Get and set the custom termination criterion.

        The custom termination criterion must be a function which receives
        the trainer as its unique argument and returns a boolean value,
        :py:data:`True` if the search should terminate or :py:data:`False`
        otherwise.

        If more than one arguments are needed to define the termniation
        condition, :py:func:`functools.partial` can be used:

        .. code-block:: python

            from functools import partial

            def my_crit(trainer, max_iters):
                if trainer.current_iter < max_iters:
                    return False
                return True

            trainer.custom_termination_func = partial(my_crit, max_iters=10)

        :getter: Return the current custom termination criterion
        :setter: Set a new custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type: :py:class:`~collections.abc.Callable`
        :raises TypeError: If set to a value which is not callable
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
        :py:data:`True` if the search should terminate or :py:data:`False`
        otherwise.

        :param func: The new custom termination criterion. If set to
            :py:data:`None`, the default termination criterion is used.
            Defaults to :py:data:`None`
        :type func: :py:class:`~collections.abc.Callable`
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
        """Enable or disable checkpointing.

        :getter: Return :py:data:`True` if checkpoinitng is enabled, or
            :py:data:`False` otherwise
        :setter: New value for the checkpoint enablement. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            is chosen
        :type: :py:class:`bool`
        :raises TypeError: If set to a value which is not boolean
        """
        return (
            DEFAULT_CHECKPOINT_ENABLE if self._checkpoint_enable is None
            else self._checkpoint_enable
        )

    @checkpoint_enable.setter
    def checkpoint_enable(self, value: bool | None) -> None:
        """Enable or disable checkpointing.

        :param value: New value for the checkpoint enablement. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_ENABLE`
            is chosen
        :type value: :py:class:`bool`
        :raises TypeError: If *value* is not a boolean value
        """
        self._checkpoint_enable = (
            None if value is None else check_bool(
                value, "checkpoint enablement"
            )
        )

    @property
    def checkpoint_freq(self) -> int:
        """Get and set the checkpoint frequency.

        :getter: Return the checkpoint frequency
        :setter: Set a value for the checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is not a positive number
        """
        return (
            DEFAULT_CHECKPOINT_FREQ if self._checkpoint_freq is None
            else self._checkpoint_freq
        )

    @checkpoint_freq.setter
    def checkpoint_freq(self, value: int | None) -> None:
        """Set a value for the checkpoint frequency.

        :param value: New value for the checkpoint frequency. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_CHECKPOINT_FREQ`
            is chosen
        :type value: :py:class:`int`
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
        """Get and set the checkpoint file path.

        :getter: Return the checkpoint file path
        :setter: Set a new value for the checkpoint file path. If set to
            :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` is chosen
        :type: :py:class:`str`
        :raises TypeError: If set to a value which is not a a valid file name
        :raises ValueError: If set to a value whose extension is not
            :py:attr:`~culebra.SERIALIZED_FILE_EXTENSION`
        """
        return (
            DEFAULT_CHECKPOINT_FILENAME if self._checkpoint_filename is None
            else self._checkpoint_filename
        )

    @checkpoint_filename.setter
    def checkpoint_filename(self, value: str | None) -> None:
        """Set a value for the checkpoint file path.

        :param value: New value for the checkpoint file path. If set to
            :py:data:`None`,
            :py:attr:`~culebra.DEFAULT_CHECKPOINT_FILENAME` is chosen
        :type value: :py:class:`str`
        :raises TypeError: If *value* is not a valid file name
        :raises ValueError: If the *value* extension is not
            :py:attr:`~culebra.SERIALIZED_FILE_EXTENSION`
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
        """Get and set the verbosity of this trainer.

        :getter: Return the verbosity
        :setter: Set a new value for the verbosity. If set to
            :py:data:`None`, :py:data:`__debug__` is chosen
        :type: :py:class:`bool`
        :raises TypeError: If set to a value which is not boolean
        """
        return (
            DEFAULT_VERBOSITY if self._verbose is None
            else self._verbose
        )

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbosity of this trainer.

        :param value: The verbosity. If set to :py:data:`None`,
            :py:data:`__debug__` is chosen
        :type value: :py:class:`bool`
        :raises TypeError: If *value* is not boolean
        """
        self._verbose = (
            None if value is None else check_bool(value, "verbosity")
        )

    @property
    def random_seed(self) -> int:
        """Get and set the initial random seed used by this trainer.

        :getter: Return the seed
        :setter: Set a new value for the random seed
        :type: :py:class:`int`
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        """Set the random seed for this trainer.

        :param value: Random seed for the random generator
        :type value: :py:class:`int`
        """
        self._random_seed = value
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)

        # Reset the algorithm
        self.reset()

    @property
    def logbook(self) -> Logbook | None:
        """Get the training logbook.

        Return a logbook with the statistics of the search or :py:data:`None`
        if the search has not been done yet.

        :type: :py:class:`~deap.tools.Logbook`
        """
        return self._logbook

    @property
    def num_evals(self) -> int | None:
        """Get the number of evaluations performed while training.

        Return the number of evaluations or :py:data:`None` if the search has
        not been done yet.

        :type: :py:class:`int`
        """
        return self._num_evals

    @property
    def runtime(self) -> float | None:
        """Get the training runtime.

        Return the training runtime or :py:data:`None` if the search has not
        been done yet.

        :type: :py:class:`float`
        """
        return self._runtime

    @property
    def index(self) -> int:
        """Get and set the trainer index.

        The trainer index is only used by distributed trainers. For the rest
        of trainers :py:attr:`~culebra.DEFAULT_INDEX` is used.

        :getter: Return the trainer index
        :setter: Set a new value for trainer index. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_INDEX` is chosen
        :type: :py:class:`int`
        :raises TypeError: If set to a value which is not an integer
        :raises ValueError: If set to a value which is a negative number
        """
        return (
            DEFAULT_INDEX if self._index is None
            else self._index
        )

    @index.setter
    def index(self, value: int | None) -> None:
        """Set a value for trainer index.

        The trainer index is only used by distributed trainers. For the rest
        of trainers :py:attr:`~culebra.DEFAULT_INDEX` is used.

        :param value: New value for the trainer index. If set to
            :py:data:`None`, :py:attr:`~culebra.DEFAULT_INDEX` is chosen
        :type value: :py:class:`int`
        :raises TypeError: If *value* is not an integer
        :raises ValueError: If *value* is a negative number
        """
        # Check the value
        self._index = (
            None if value is None else check_int(value, "index", ge=0)
        )

    @property
    def container(self) -> Trainer | None:
        """Get and set the container of this trainer.

        The trainer container is only used by distributed trainers. For the
        rest of trainers defaults to :py:data:`None`.

        :getter: Return the container
        :setter: Set a new value for container of this trainer
        :type: :py:class:`~culebra.abc.Trainer`
        :raises TypeError: If set to a value which is not a valid trainer
        """
        return self._container

    @container.setter
    def container(self, value: Trainer | None) -> None:
        """Set a container for this trainer.

        The trainer container is only used by distributed trainers. For the
        rest of trainers defaults to :py:data:`None`.

        :param value: New value for the container or :py:data:`None`
        :type value: :py:class:`~culebra.abc.Trainer`
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
        """Return the representatives of the other species.

        Only used by cooperative trainers. If the trainer does not use
        representatives, :py:data:`None` is returned.
        """
        return self._representatives

    def _get_state(self) -> Dict[str, Any]:
        """Return the state of this trainer.

        Default state is a dictionary composed of the values of the
        :py:attr:`~culebra.abc.Trainer.logbook`,
        :py:attr:`~culebra.abc.Trainer.num_evals`,
        :py:attr:`~culebra.abc.Trainer.runtime`,
        :py:attr:`~culebra.abc.Trainer.current_iter`, and
        :py:attr:`~culebra.abc.Trainer.representatives`
        trainer properties, along with a private boolean attribute that informs
        if the search has finished and also the states of the :py:mod:`random`
        and :py:mod:`numpy.random` modules.

        If subclasses use any more properties to keep their state, the
        :py:meth:`~culebra.abc.Trainer._get_state` and
        :py:meth:`~culebra.abc.Trainer._set_state` methods must be
        overridden to take into account such properties.

        :type: :py:class:`dict`
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
        :py:meth:`~culebra.abc.Trainer._get_state` and
        :py:meth:`~culebra.abc.Trainer._set_state` methods must be
        overridden to take into account such properties.

        :param state: The last loaded state
        :type state: :py:class:`dict`
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
        :py:meth:`~culebra.abc.Trainer._reset_state`) and also all the internal
        data structures needed to perform the search
        (with :py:meth:`~culebra.abc.Trainer._reset_internals`).

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
        :type sol: :py:class:`~culebra.abc.Solution`
        :param fitness_func: The fitness function. If omitted, the default
            training fitness function
            (:py:attr:`~culebra.abc.Trainer.fitness_function`) is used
        :type fitness_func: :py:class:`~culebra.abc.FitnessFunction`, optional
        :param index: Index where *sol* should be inserted in the
            *representatives* sequence to form a complete solution for the
            problem. If omitted, :py:attr:`~culebra.abc.Trainer.index` is used
        :type index: :py:class:`int`, optional
        :param representatives: Sequence of representatives of other species
            or :py:data:`None` (if no representatives are needed to evaluate
            *sol*). If omitted, the current value of
            :py:attr:`~culebra.abc.Trainer.representatives` is used
        :type representatives: :py:class:`~collections.abc.Sequence`
            of :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution` or :py:data:`None`, optional
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
            # Invalidate the last fitness
            sol.fitness.delValues()

        the_index = index if index is not None else self.index

        # If context is not None -> cooperation
        if context_seq is not None:
            for context in context_seq:
                # Calculate their fitness
                fit_values = func.evaluate(
                    sol,
                    index=the_index,
                    representatives=context
                )

                # If it is the first trial ...
                if sol.fitness.valid is False:
                    sol.fitness.values = fit_values
                else:
                    trial_fitness = func.fitness_cls(fit_values)
                    # If the trial fitness is better
                    # TODO Other criteria should be tested to choose
                    # the better fitness estimation
                    # Average???
                    if trial_fitness > sol.fitness:
                        sol.fitness.values = fit_values
        else:
            sol.fitness.values = func.evaluate(sol, index=the_index)

        # Increase the number of evaluations performed
        if self._current_iter_evals is not None:
            self._current_iter_evals += (
                    len(context_seq)
                    if context_seq is not None
                    else 1
                )

    @abstractmethod
    def best_solutions(self) -> Sequence[HallOfFame]:
        """Get the best solutions found for each species.

        This method must be overridden by subclasses to return a correct
        value.

        :return: A list containing :py:class:`~deap.tools.HallOfFame` of
            solutions. One hof for each species
        :rtype: :py:class:`list` of :py:class:`~deap.tools.HallOfFame`
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
            cooperative or :py:data:`None` in other cases.
        :rtype: :py:class:`list` of :py:class:`list` of
            :py:class:`~culebra.abc.Solution` or :py:data:`None`
        """
        return None

    def train(self, state_proxy: Optional[DictProxy] = None) -> None:
        """Perform the training process.

        :param state_proxy: Dictionary proxy to copy the output state of the
            trainer procedure. Only used if train is executed within a
            :py:class:`multiprocess.Process`. Defaults to :py:data:`None`
        :type state_proxy: :py:class:`~multiprocess.managers.DictProxy`,
            optional
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
            One :py:class:`~deap.tools.HallOfFame` for each species
        :type best_found: :py:class:`~collections.abc.Sequence` of
            :py:class:`~deap.tools.HallOfFame`
        :param fitness_func: Fitness function used to evaluate the final
            solutions. If ommited, the default training fitness function
            (:py:attr:`~culebra.abc.Trainer.fitness_function`) will be used
        :type fitness_func: :py:class:`~culebra.abc.FitnessFunction`, optional
        :param representatives: Sequence of representatives of other species
            or :py:data:`None` (if no representatives are needed). If omitted,
            the current value of
            :py:attr:`~culebra.abc.Trainer.representatives` is used
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`
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
        the :py:meth:`~culebra.abc.Trainer._load_state` method. Otherwise a new
        initial state is generated with the
        :py:meth:`~culebra.abc.Trainer._new_state` method.
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

        If subclasses overwrite the :py:meth:`~culebra.abc.Trainer._new_state`
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
        to run the search process. For the :py:class:`~culebra.abc.Trainer`
        class, only a :py:class:`~deap.tools.Statistics` object is created.
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
        :py:meth:`~culebra.abc.Trainer._init_internals` method to add any new
        internal object, this method should also be overridden to reset all the
        internal objects of the trainer.
        """
        self._stats = None
        self._current_iter_evals = None
        self._current_iter_start_time = None

    def _init_search(self) -> None:
        """Init the search process.

        Initialize the state of the trainer (with
        :py:meth:`~culebra.abc.Trainer._init_state`) and all the internal data
        structures needed
        (with :py:meth:`~culebra.abc.Trainer._init_internals`) to perform the
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

            * :py:meth:`~culebra.abc.Trainer._start_iteration`
            * :py:meth:`~culebra.abc.Trainer._preprocess_iteration`
            * :py:meth:`~culebra.abc.Trainer._do_iteration`
            * :py:meth:`~culebra.abc.Trainer._postprocess_iteration`
            * :py:meth:`~culebra.abc.Trainer._finish_iteration`
            * :py:meth:`~culebra.abc.Trainer._do_iteration_stats`
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
        """Set the default termination criterion.

        Return :py:data:`True` if :py:attr:`~culebra.abc.Trainer.max_num_iters`
        iterations have been run.
        """
        if self._current_iter < self.max_num_iters:
            return False

        return True

    def _termination_criterion(self) -> bool:
        """Return true if the search should terminate.

        Returns :py:data:`True` if either the default termination criterion or
        a custom termination criterion is met. The default termination
        criterion is implemented by the
        :py:meth:`~culebra.abc.Trainer._default_termination_func` method.
        Another custom termination criterion can be set with
        :py:attr:`~culebra.abc.Trainer.custom_termination_func` method.
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
        Cooperative subclasses of the :py:class:`~culebra.abc.Trainer` class
        should override this method to get the representatives of the other
        species initialized.
        """
        self._representatives = None

    def __copy__(self) -> Trainer:
        """Shallow copy the trainer."""
        cls = self.__class__
        result = cls(self.fitness_function)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Trainer:
        """Deepcopy the trainer.

        :param memo: Trainer attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the trainer
        :rtype: :py:class:`~culebra.abc.Trainer`
        """
        cls = self.__class__
        result = cls(self.fitness_function)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the trainer.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.fitness_function,), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Trainer:
        """Return a trainer from a state.

        :param state: The state.
        :type state: :py:class:`~dict`
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
