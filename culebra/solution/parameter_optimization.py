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

"""Species and solutions for parameter optimization problems.

This module supports parameter optimization problems. The possible solutions
to the problem are handled by:

* A :class:`~culebra.solution.parameter_optimization.Solution` class
  containing all the parameter values found for the problem.
* A :class:`~culebra.solution.parameter_optimization.Species` class to
  define the characteristics that the desired parameter values should meet.

In order to make possible the application of evolutionary approaches, the
:class:`~culebra.solution.parameter_optimization.Solution` class has been
extended by the :class:`~culebra.solution.parameter_optimization.Individual`
class, which inherits from both
:class:`~culebra.solution.parameter_optimization.Solution` and
:class:`~culebra.solution.abc.Individual` classes to provide the crossover
and mutation operators to the
:class:`~culebra.solution.parameter_optimization.Solution` class.

"""

from __future__ import annotations

from typing import Tuple, NamedTuple, Type, Optional
from collections.abc import Sequence
from collections import namedtuple
from string import ascii_letters, digits
from copy import deepcopy
from numbers import Integral, Real
from functools import partial
from random import random

from deap.tools import cxSimulatedBinaryBounded, mutPolynomialBounded

from culebra.abc import (
    Species as BaseSpecies,
    Solution as BaseSolution,
    Fitness
)
from culebra.checker import (
    check_str,
    check_subclass,
    check_sequence,
    check_int,
    check_float
)
from culebra.solution.abc import Individual as BaseIndividual


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_PARAMETER_NAME = "param"
"""Default parameter name."""

VALID_PARAMETER_NAME_CHARS = ascii_letters + digits + '_'
"""Valid characters for paramter names."""

DEFAULT_ETA = 5
r"""Default value for :math:`{\eta}` (eta), which controls the probability
distribution used in SBX and polynomial mutation.
"""


class Species(BaseSpecies):
    """Species for the parameter optimization solutions.

    Species instances have the following attributes:

        * :attr:`~culebra.solution.parameter_optimization.Species.lower_bounds`:
          Lower bound for each parameter
        * :attr:`~culebra.solution.parameter_optimization.Species.upper_bounds`:
          Upper bound for each parameter
        * :attr:`~culebra.solution.parameter_optimization.Species.types`:
          Type of each parameter
    """

    def __init__(
        self,
        lower_bounds: Sequence[int | float],
        upper_bounds: Sequence[int | float],
        types: Optional[Sequence[Type[int] | Type[float]]] = None,
        names: Optional[Sequence[str]] = None
    ) -> None:
        """Create a new species.

        :param lower_bounds: Lower bound for each parameter
        :type lower_bounds: ~collections.abc.Sequence[int | float]
        :param upper_bounds: Upper bound for each parameter
        :type upper_bounds: ~collections.abc.Sequence[int | float]
        :param types: Type of each parameter. If omited, all the parameters
            will be treated as :class:`float`. Defaults to :data:`None`
        :type types: ~collections.abc.Sequence[type]
        :param names: Name of each parameter. Defaults to :data:`None`
        :type names: ~collections.abc.Sequence[str]
        :raises TypeError: If any of the attributes is not a
            :class:`~collections.abc.Sequence`
        :raises ValueError: If the sequences have different lengths
        :raises ValueError: If the sequences are empty
        :raises ValueError: If the type of any bound does not match with its
            corresponding type in *types*
        :raises ValueError: If any lower bound is greater to or equal than its
            corresponding upper bound
        :raises ValueError: If any name is not an instance of :class:`str`
        :raises ValueError: If there is any repeated name
        """
        super().__init__()

        (
            self._lower_bounds,
            self._upper_bounds,
            self._types,
            self._names
        ) = self._check_attributes(lower_bounds, upper_bounds, types, names)

    @property
    def lower_bounds(self) -> Tuple[int | float, ...]:
        """Lower bound for each parameter.

        :rtype: tuple[int | float]
        """
        return self._lower_bounds

    @property
    def upper_bounds(self) -> Tuple[int | float, ...]:
        """Upper bound for each parameter.

        :rtype: tuple[int | float]
        """
        return self._upper_bounds

    @property
    def types(self) -> Tuple[Type[int] | Type[float], ...]:
        """Type of each parameter.

        :rtype: tuple[type]
        """
        return self._types

    @property
    def names(self) -> Tuple[str, ...]:
        """Name of each parameter.

        :rtype: tuple[str]
        """
        return self._names

    @property
    def num_params(self) -> int:
        """Number of parameters to be optimized.

        :rtype: int
        """
        return len(self.lower_bounds)

    def is_member(self, sol: Solution) -> bool:
        """Check if a solution meets the constraints imposed by the species.

        :param sol: The solution
        :type sol: ~culebra.solution.parameter_optimization.Solution
        :return: :data:`True` if the solution belongs to the species.
            :data:`False` otherwise
        :rtype: bool
        """
        sol_is_member = True

        # Check the number of parameters
        if len(sol.values) != self.num_params:
            sol_is_member = False
        else:
            # Check all the values
            for (val, lower, upper, name, t) in zip(
                sol.values,
                self.lower_bounds,
                self.upper_bounds,
                self.names,
                self.types
            ):
                # Check the value
                try:
                    if issubclass(t, Integral):
                        check_int(val, name, ge=lower, le=upper)
                    else:
                        check_float(val, name, ge=lower, le=upper)
                except (TypeError, ValueError):
                    sol_is_member = False
                    break

        return sol_is_member

    @staticmethod
    def _retype(
        values: Sequence[float],
        types: Sequence[Type[int] | Type[float]]
    ) -> Tuple[int | float, ...]:
        """Retype the input values to the types provided.

        :param values: The values
        :type values: ~collections.abc.Sequence[float]
        :param types: The types
        :type types: ~collections.abc.Sequence[type]
        :return: A tuple of retyped values according to *types*
        :rtype: tuple[int | floattype]
        """
        return tuple(
            int(val) if t is int else float(val)
            for val, t in zip(values, types)
        )

    @staticmethod
    def _check_attributes(
        lower_bounds: Sequence[int | float],
        upper_bounds: Sequence[int | float],
        types: Optional[Sequence[Type[int] | Type[float]]] = None,
        names: Optional[Sequence[str]] = None
    ) -> Tuple[
        Tuple[int | float, ...],
        Tuple[int | float, ...],
        Tuple[Type[int] | Type[float], ...],
        Tuple[str, ...]
    ]:
        """Check the species attributes.

        :param lower_bounds: Proposed lower bound for each parameter
        :type lower_bounds: ~collections.abc.Sequence[int | float]
        :param upper_bounds: Proposed upper bound for each parameter
        :type upper_bounds: ~collections.abc.Sequence[int | float]
        :param types: Proposed type of each parameter. If omitted, all the
            parameters will be treated as :class:`float`. Defaults to
            :data:`None`
        :type types: ~collections.abc.Sequence[type]
        :param names: Name of each parameter. Defaults to :data:`None`
        :type names: ~collections.abc.Sequence[str]
        :return: A tuple of valid values for the attributes
        :rtype:
            tuple[tuple[int|float], tuple[int|float], tuple[type], tuple[str]]
        :raises TypeError: If any of the attributes is not a
            :class:`~collections.abc.Sequence`
        :raises ValueError: If the sequences have different lengths
        :raises ValueError: If the sequences are empty
        :raises ValueError: If the type of any bound does not match with its
            corresponding type in *types*
        :raises ValueError: If any lower bound is greater or equal than its
            corresponding upper bound
        :raises ValueError: If name is not an instance of :class:`str`
        :raises ValueError: If name contains invalid characters
        :raises ValueError: If there is any repeated name
        """
        # Check that lower_bounds is a sequence
        lower_bounds = check_sequence(lower_bounds, "lower bounds")

        # Number of parameters
        num_params = len(lower_bounds)

        # Check that there is at least one parameter
        if num_params == 0:
            raise ValueError("Lower bounds must have at least one element")

        # Check that upper_bounds is a sequence of the same size
        upper_bounds = check_sequence(
            upper_bounds, "upper bounds", size=num_params
        )

        # If types are provided
        if types is not None:
            # Check that types is a sequence of the same size of subclasses
            # of float
            types = check_sequence(
                types,
                "types",
                size=num_params,
                item_checker=partial(check_subclass, cls=Real)
            )
            # Change types for their abstract classes
            types = [
                int if issubclass(t, Integral) else float for t in types
            ]
        else:
            types = [float] * num_params

        # Check the bounds
        for (lower, upper, t) in zip(lower_bounds, upper_bounds, types):
            # Check the value
            try:
                if issubclass(t, Integral):
                    check_int(lower, "lower bound")
                    check_int(upper, "upper bound", gt=lower)
                else:
                    check_float(lower, "lower bound")
                    check_float(upper, "upper bound", gt=lower)
            except TypeError as error:
                raise ValueError(str(error)) from error

        # If names are provided
        if names is not None:
            # Check that names is a sequence of the same size of valid names
            names = check_sequence(
                names,
                "parameter names",
                size=num_params,
                item_checker=partial(
                    check_str,
                    valid_chars=VALID_PARAMETER_NAME_CHARS
                )
            )
        else:
            # Maximum length of the index (in chars)
            index_len = len(str((num_params-1)))

            # Generate default parameter names
            names = list(
                f"{DEFAULT_PARAMETER_NAME}_"
                f"{i:0{index_len}d}" for i in range(num_params)
            )

        # Check if there is any repeated name
        for i in range(num_params):
            name = names[i]
            for other in names[i + 1:]:
                if name == other:
                    raise ValueError(f"Repeated name: {name}")

        return (
            Species._retype(lower_bounds, types),
            Species._retype(upper_bounds, types),
            tuple(types),
            tuple(names)
        )

    def __copy__(self) -> Species:
        """Shallow copy the species.

        :return: The copied object
        :rtype: ~culebra.solution.parameter_optimization.Species
        """
        cls = self.__class__
        result = cls(self.lower_bounds, self.upper_bounds)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Species:
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: dict
        :return: The copied object
        :rtype: ~culebra.solution.parameter_optimization.Species
        """
        cls = self.__class__
        result = cls(self.lower_bounds, self.upper_bounds)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the species.

        :return: The reduction
        :rtype: tuple
        """
        return (
            self.__class__,
            (self.lower_bounds, self.upper_bounds),
            self.__dict__
        )

    @classmethod
    def __fromstate__(cls, state: dict) -> Species:
        """Return a species from a state.

        :param state: The state
        :type state: dict
        :return: The species
        :rtype: ~culebra.solution.parameter_optimization.Species
        """
        obj = cls(
            state['_lower_bounds'],
            state['_upper_bounds']
        )
        obj.__setstate__(state)
        return obj


class Solution(BaseSolution):
    """Parameter optimization solution."""

    species_cls = Species
    """Class for the species used by the
    :class:`~culebra.solution.parameter_optimization.Solution` class to
    constrain all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness],
        values: Optional[Sequence[int | float]] = None
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: ~culebra.solution.parameter_optimization.Species
        :param fitness_cls: The solution's fitness class
        :type fitness_cls: type[~culebra.abc.Fitness]
        :param values: Initial values, optional
        :type values: ~collections.abc.Sequence[int | float]
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)

        if values is not None:
            self.values = values
        else:
            self._setup()

    @property
    def named_values_cls(self) -> Type[NamedTuple]:
        """Named tuple class to hold the parameter values.
        
        :rtype: type[~typing.NamedTuple]
        """
        return namedtuple("NamedValues", self.species.names)

    @property
    def values(self) -> NamedTuple[int | float, ...]:
        """Parameter values evolved by the solution.
        
        :rtype:
            :attr:`~culebra.solution.parameter_optimization.Solution.named_values_cls`

        :setter: Set new parameter values.
        :param values: The new parameters values
        :type values: ~collections.abc.Sequence[int | float]
        :raises ValueError: If the values do not meet the species constraints
        """
        return self.named_values_cls(
            *[
                int(val) if issubclass(t, Integral) else val
                for t, val in zip(self.species.types, self._values)
            ]
        )

    @values.setter
    def values(self, values: Sequence[int | float]) -> None:
        """Set new values for the parameter values.

        :param values: The new parameters values
        :type values: ~collections.abc.Sequence[int | float]
        :raises ValueError: If the values do not meet the species constraints
        """
        # Values are stored as float
        self._values = values

        if not self.species.is_member(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

        # Store all the values as float
        self._retype()

    def get(self, name: str) -> int | float:
        """Get the value of the parameter with the given name.

        :param name: Name of the parameter
        :type name: str
        :return: The value of the parameter
        :rtype: int | float
        """
        try:
            index = self.species.names.index(name)
            return self._values[index]
        except ValueError as exc:
            raise ValueError(f"'{name}' is not a valid value name") from exc

    def _setup(self) -> None:
        """Init the values of this solution randomly."""
        # Values are stored as float
        self._values = [
            lower + random() * (upper - lower)
            for (lower, upper) in zip(
                self.species.lower_bounds,
                self.species.upper_bounds
            )
        ]

        # Round the int values
        self._retype()

    def _retype(self) -> None:
        """Round values of int parameters."""
        self._values = [
            float(round(val)) if issubclass(t, Integral) else float(val)
            for t, val in zip(self.species.types, self._values)
        ]

    def __str__(self) -> str:
        """Solution as a string.

        :rtype: str
        """
        values = self.values

        msg = ''
        sep = ''
        for val, name, t in zip(
            values, self.species.names, self.species.types
        ):
            if issubclass(t, Integral):
                value_str = str(val)
            else:
                value_str = f"{val:.6f}"
            msg += sep + name + ": " + value_str
            sep = ", "

        return msg

    def __repr__(self) -> str:
        """Solution representation.

        :rtype: str
        """
        cls_name = self.__class__.__name__
        species_info = self.species.__str__()
        fitness_info = self.fitness.values

        return (f"{cls_name}(species={species_info}, fitness={fitness_info}, "
                f"values={self.__str__()})")


class Individual(Solution, BaseIndividual):
    """Parameter optimization individual."""

    eta = DEFAULT_ETA
    """Default value for eta, which controls the probability distribution used
    in SBX and polynomial mutation.
    """

    def crossover(self, other: Individual) -> Tuple[Individual, Individual]:
        """Cross this individual with another one.

        SBX is used.

        :param other: The other individual
        :type other: ~culebra.solution.parameter_optimization.Individual
        :return: The two offspring
        :rtype: tuple[~culebra.solution.parameter_optimization.Individual]
        """
        # Apply SBX
        self._values, other._values = cxSimulatedBinaryBounded(
            self._values,
            other._values,
            self.eta,
            self.species.lower_bounds,
            self.species.upper_bounds
        )

        # Round the int values
        self._retype()
        other._retype()

        # Return the new offspring
        del self.fitness.values
        del other.fitness.values

        return self, other

    def mutate(self, indpb: float) -> Tuple[Individual]:
        """Mutate the individual.

        Polynomial mutation is used.

        :param indpb: Independent probability for each parameter to be mutated.
        :type indpb: float
        :return: The mutant
        :rtype: tuple[~culebra.solution.parameter_optimization.Individual]
        """
        # Apply mutation
        (self._values,) = mutPolynomialBounded(
            self._values,
            self.eta,
            self.species.lower_bounds,
            self.species.upper_bounds,
            indpb
        )

        # Round the int values
        self._retype()

        # Return the new individual
        del self.fitness.values

        return (self,)


# Exported symbols for this module
__all__ = [
    'Species',
    'Solution',
    'Individual',
    'DEFAULT_PARAMETER_NAME',
    'DEFAULT_ETA'
]
