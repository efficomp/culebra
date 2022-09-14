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

"""Abstract classes for feature selection."""

from __future__ import annotations
from typing import Tuple, NamedTuple, Type, Optional
from collections.abc import Sequence
from collections import namedtuple
from string import ascii_letters, digits
from copy import deepcopy
from numbers import Integral, Real
from functools import partial
from deap.tools import cxSimulatedBinaryBounded, mutPolynomialBounded
from random import random
from culebra.base import (
    Species as BaseSpecies,
    Individual as BaseIndividual,
    Fitness,
    check_str,
    check_subclass,
    check_sequence,
    check_int,
    check_float
)

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_ETA = 5
"""Default value for :math:`{\eta}` (eta), which controls the probability
distribution used in SBX and polynomial mutation.
"""

DEFAULT_HYPERPARAMETER_NAME = "hyperparam"
"""Default hyperparameter name."""

VALID_HYPERPARAMETER_NAME_CHARS = ascii_letters + digits + '_'
"""Valid characters for hyperparamter names."""


class Species(BaseSpecies):
    """Species for the classifier hyperparameters individuals.

    Species instances have the following attributes:

        * :py:attr:`~genotype.classifier_optimization.Species.lower_bounds`:
          Lower bound for each hyperparameter
        * :py:attr:`~genotype.classifier_optimization.Species.upper_bounds`:
          Upper bound for each hyperparameter
        * :py:attr:`~genotype.classifier_optimization.Species.types`: Type of
          each hyperparameter
    """

    def __init__(
        self,
        lower_bounds: Sequence[int | float],
        upper_bounds: Sequence[int | float],
        types: Optional[Sequence[Type[int] | Type[float]]] = None,
        names: Optional[Sequence[str]] = None
    ) -> None:
        """Create a new species.

        :param lower_bounds: Lower bound for each hyperparameter
        :type lower_bounds: :py:class:`~collections.abc.Sequence` of
            :py:class:`int` or :py:class:`float` values
        :param upper_bounds: Upper bound for each hyperparameter
        :type upper_bounds: :py:class:`~collections.abc.Sequence` of
            :py:class:`int` or :py:class:`float` values
        :param types: Type of each hyperparameter. All the hyperparameters will
            be treated as :py:class:`float` if omitted. Defaults to
            :py:data:`None`
        :type types: :py:class:`~collections.abc.Sequence` of
            :py:class:`type` (:py:class:`int` or :py:class:`float`), optional
        :param names: Name of each hyperparameter. Defaults to :py:data:`None`
        :type names: :py:class:`~collections.abc.Sequence` of
            :py:class:`str`, optional
        :raises TypeError: If any of the attributes is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If the sequences have different lengths
        :raises ValueError: If the sequences are empty
        :raises ValueError: If the type of any bound does not match with its
            corresponding type in *types*
        :raises ValueError: If any lower bound is greater to or equal than its
            corresponding upper bound
        :raises ValueError: If any name is not an instance of :py:class:`str`
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
        """Get the lower bound for each hyperparameter.

        :type: :py:class:`tuple` of :py:class:`int` or :py:class:`float`
            values
        """
        return self._lower_bounds

    @property
    def upper_bounds(self) -> Tuple[int | float, ...]:
        """Get the upper bound for each hyperparameter.

        :type: :py:class:`tuple` of :py:class:`int` or :py:class:`float`
            values
        """
        return self._upper_bounds

    @property
    def types(self) -> Tuple[Type[int] | Type[float], ...]:
        """Get the type of each hyperparameter.

        :type: :py:class:`tuple` of :py:class:`int` or :py:class:`float`
        """
        return self._types

    @property
    def names(self) -> Tuple[str, ...]:
        """Get the name of each hyperparameter.

        :type: :py:class:`tuple` of :py:class:`str`
        """
        return self._names

    @property
    def num_hyperparams(self) -> int:
        """Get the number of hyperparameters to be optimized.

        :type: :py:class:`int`
        """
        return len(self.lower_bounds)

    def is_member(self, ind: Individual) -> bool:
        """Check if an individual meets the constraints imposed by the species.

        :param ind: The individual
        :type ind: :py:class:`~genotype.classifier_optimization.Individual`
        :return: :py:data:`True` if the individual belongs to the species.
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        ind_is_member = True

        # Check the number of hyperparameters
        if len(ind.values) != self.num_hyperparams:
            ind_is_member = False
        else:
            # Check all the values
            for (val, lower, upper, name, t) in zip(
                ind.values,
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
                except Exception:
                    ind_is_member = False
                    break

        return ind_is_member

    @staticmethod
    def _retype(
        values: Sequence[float],
        types: Sequence[Type[int] | Type[float]]
    ) -> Tuple[int | float, ...]:
        """Retype the input values to the types provided.

        :param values: The values
        :type values: :py:class:`~collections.abc.Sequence` of numbers
        :param types: The types
        :type types: :py:class:`~collections.abc.Sequence` of types
        :return: A tuple of retyped values according to *types*
        :rtype: :py:class:`tuple` of :py:class:`int` or :py:class:`float`
            values
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

        :param lower_bounds: Proposed lower bound for each hyperparameter
        :type lower_bounds: :py:class:`~collections.abc.Sequence` of
            :py:class:`int` or :py:class:`float`
        :param upper_bounds: Proposed upper bound for each hyperparameter
        :type upper_bounds: :py:class:`~collections.abc.Sequence` of
            :py:class:`int` or :py:class:`float`
        :param types: Proposed type of each hyperparameter. All the
            hyperparameters will be treated as :py:class:`float` if omitted.
            Defaults to :py:data:`None`
        :type types: :py:class:`~collections.abc.Sequence` of
            :py:class:`type` (:py:class:`int` or :py:class:`float`)
        :param names: Name of each hyperparameter. Defaults to :py:data:`None`
        :type names: :py:class:`~collections.abc.Sequence` of
        :return: A tuple of valid values for the attributes
        :rtype: :py:class:`tuple`
        :raises TypeError: If any of the attributes is not a
            :py:class:`~collections.abc.Sequence`
        :raises ValueError: If the sequences have different lengths
        :raises ValueError: If the sequences are empty
        :raises ValueError: If the type of any bound does not match with its
            corresponding type in *types*
        :raises ValueError: If any lower bound is greater or equal than its
            corresponding upper bound
        :raises ValueError: If name is not an instance of :py:class:`str`
        :raises ValueError: If name contains invalid characters
        :raises ValueError: If there is any repeated name
        """
        # Check that lower_bounds is a sequence
        lower_bounds = check_sequence(lower_bounds, "lower bounds")

        # Number of hyperparameters
        num_hyperparams = len(lower_bounds)

        # Check that there is at least one hyperparameter
        if num_hyperparams == 0:
            raise ValueError("Lower bounds must have at least one element")

        # Check that upper_bounds is a sequence of the same size
        upper_bounds = check_sequence(
            upper_bounds, "upper bounds", size=num_hyperparams
        )

        # If types are provided
        if types is not None:
            # Check that types is a sequence of the same size of subclasses
            # of float
            types = check_sequence(
                types,
                "types",
                size=num_hyperparams,
                item_checker=partial(check_subclass, cls=Real)
            )
            # Change types for their abstract classes
            types = [
                int if issubclass(t, Integral) else float for t in types
            ]
        else:
            types = [float] * num_hyperparams

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
                "hyperparameter names",
                size=num_hyperparams,
                item_checker=partial(
                    check_str,
                    valid_chars=VALID_HYPERPARAMETER_NAME_CHARS
                )
            )
        else:
            # Maximum length of the index (in chars)
            index_len = len((num_hyperparams-1).__str__())

            # Generate default hyperparameter names
            names = list(
                f"{DEFAULT_HYPERPARAMETER_NAME}_"
                f"{i:0{index_len}d}" for i in range(num_hyperparams)
            )

        # Check if there is any repeated name
        for i in range(num_hyperparams):
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
        """Shallow copy the species."""
        cls = self.__class__
        result = cls(self.lower_bounds, self.upper_bounds)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Species:
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the species
        :rtype: :py:class:`~feature_selector.Species`
        """
        cls = self.__class__
        result = cls(self.lower_bounds, self.upper_bounds)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the species.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (
            self.__class__,
            (self.lower_bounds, self.upper_bounds),
            self.__dict__
        )


class Individual(BaseIndividual):
    """Classifier hyperparameters individuals."""

    species_cls = Species
    """Class for the species used by the
    :py:class:`~genotype.classifier_optimization.Individual` class to
    constrain all its instances."""

    eta = DEFAULT_ETA
    """Default value for eta, which controls the probability distribution used
    in SBX and polynomial mutation.
    """

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness],
        values: Optional[Sequence[int | float]] = None
    ) -> None:
        """Construct a default individual.

        :param species: The species the individual will belong to
        :type species:
            :py:class:`~genotype.classifier_optimization.Individual.species_cls`
        :param fitness: The individual's fitness class
        :type fitness: :py:class:`~base.Fitness`
        :param values: Initial values
        :type values:
            :py:class:`~collections.abc.Sequence` of :py:class:`int` or
            :py:class:`float` values, optional
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)

        if values is not None:
            self.values = values
        else:
            self._random_init()

    def _retype(self):
        """Round values of int hyperparameters."""
        self._values = [
            float(round(val)) if issubclass(t, Integral) else float(val)
            for t, val in zip(self.species.types, self._values)
        ]

    def _random_init(self) -> None:
        """Init the values of this individual randomly."""
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

    @property
    def named_values_cls(self) -> Type[NamedTuple]:
        """Return the named tuple class to hold the hyperparameter values."""
        return namedtuple("NamedValues", self.species.names)

    @property
    def values(self) -> NamedTuple[int | float, ...]:
        """Get and set the hyperparameter values evolved by the individual.

        :getter: Return the hyperparameter values
        :setter: Set the new hyperparameter values.
        :type:
            :py:class:`~genotype.classifier_optimization.Individual.named_values_cls`
        :raises ValueError: If set to new hyperparameter values which do not
            meet the species constraints.
        """
        return self.named_values_cls(
            *[
                int(val) if issubclass(t, Integral) else val
                for t, val in zip(self.species.types, self._values)
            ]
        )

    @values.setter
    def values(self, values: Sequence[int | float]) -> None:
        """Set new values for the hyperparameter values.

        :param values: The new hyperparameters values
        :type values: :py:class:`~collections.abc.Sequence`
        :raises ValueError: If the values do not meet the species constraints.
        """
        # Values are stored as float
        self._values = values

        if not self.species.is_member(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

        # Store all the values as float
        self._retype()

    def get(self, name: str) -> int | float:
        """Get the value of the hyperparameter with the given name.

        :param name: Name of the hyperparameter
        :type name: :py:class:`str`
        """
        pass

    def crossover(self, other: Individual) -> Tuple[Individual, Individual]:
        """Cross this individual with another one.

        SBX is used.

        :param other: The other individual
        :type other: :py:class:`~classifier_optimization.Individual`
        :return: The two offspring
        :rtype: :py:class:`tuple`
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
        return self, other

    def mutate(self, indpb: float) -> Tuple[Individual]:
        """Mutate the individual.

        Polynimial mutation is used.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: :py:class:`float`
        :return: The mutant
        :rtype: :py:class:`tuple`
        """
        # Apply mutation
        # print("Mutation", end=": ")
        # print(self, end=" -> ")

        (self._values,) = mutPolynomialBounded(
            self._values,
            self.eta,
            self.species.lower_bounds,
            self.species.upper_bounds,
            indpb
        )

        # Round the int values
        self._retype()

        # print(self)
        # Return the new individual
        return (self,)

    def __str__(self) -> str:
        """Return the individual as a string."""
        values = self.values

        msg = ''
        sep = ''
        for val, name, t in zip(
            values, self.species.names, self.species.types
        ):
            if issubclass(t, Integral):
                value_str = str(val)
            else:
                value_str = "{:.6f}".format(round(val, 6))
            msg += sep + name + ": " + value_str
            sep = ", "

        return msg

    def __repr__(self) -> str:
        """Return the individual representation."""
        cls_name = self.__class__.__name__
        species_info = self.species.__str__()
        fitness_info = self.fitness.values

        return (f"{cls_name}(species={species_info}, fitness={fitness_info}, "
                f"values={self.__str__()})")


# Exported symbols for this module
__all__ = [
    'Species',
    'Individual',
    "DEFAULT_ETA",
    "DEFAULT_HYPERPARAMETER_NAME"
]
