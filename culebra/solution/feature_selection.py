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

"""Species and solutions for feature selection problems.

This module provides all the classes necessary to solve feature selection
problems with culebra. The possible solutions to the problem are handled by:

* A :class:`~culebra.solution.feature_selection.Solution` abstract class
  defining the interface for feature selectors.
* A :class:`~culebra.solution.feature_selection.Species` class to define
  the characteristics that the desired selected features should meet.

Two :class:`~numpy.ndarray`-based implementations of the
:class:`~culebra.solution.feature_selection.Solution` class are also
provided:

* :class:`~culebra.solution.feature_selection.BinarySolution`:
  Solutions are coded as boolean vectors of a fixed length, the number of
  selectable features. One boolean value is associated to each selectable
  feature. If the boolean value is :data:`True`, then the feature is
  selected, otherwise the feature is ignored.
* :class:`~culebra.solution.feature_selection.IntSolution`: Solutions
  are coded as vectors of feature indices. Only features whose indices are
  present in the solution are selected. Thus, solutions may have different
  lengths and can not contain repeated indices.

In order to make possible the application of evolutionary approaches to this
problem, the following :class:`~culebra.solution.abc.Individual`
implementations are provided:

* :class:`~culebra.solution.feature_selection.BitVector`: Inherits from
  :class:`~culebra.solution.feature_selection.BinarySolution` and
  :class:`~culebra.solution.abc.Individual` to provide crossover and
  mutation operators to binary solutions.
* :class:`~culebra.solution.feature_selection.IntVector`: Inherits from
  :class:`~culebra.solution.feature_selection.IntSolution` and
  :class:`~culebra.solution.abc.Individual` to provide crossover and
  mutation operators to feature index-based solutions.

Ant Colony Optimization is also supported by the
:class:`~culebra.solution.feature_selection.Ant` class, which inherits from
:class:`~culebra.solution.feature_selection.IntSolution` and
:class:`culebra.solution.abc.Ant` to add all the path handling stuff to
feature index-based solutions.

Finally, this module also provides the
:class:`~culebra.solution.feature_selection.Metrics` class, which supplies
several metrics about the selected features finally obtained.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import copy, deepcopy
from random import randint

import numpy as np
from pandas import Series

from culebra.abc import (
    Base,
    Species as BaseSpecies,
    Solution as BaseSolution,
    Fitness
)
from culebra.checker import (
    check_int,
    check_float
)
from culebra.solution.abc import (
    Individual as BaseIndividual,
    Ant as BaseAnt
)


__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


DEFAULT_PROP = 0.15
"""Default proportion for the generation of a species."""

MAX_PROP = 0.25
"""Maximum proportion for the generation of a species."""


class Species(BaseSpecies):
    """Species for the feature selector solutions."""

    def __init__(
        self,
        num_feats: int,
        min_feat: int | None = None,
        min_size: int | None = None,
        max_feat: int | None = None,
        max_size: int | None = None
    ) -> None:
        """Create a new species.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: int
        :param min_feat: Smallest feature index considered in this species.
            Must be in the interval [0, *num_feats*). If omitted, the minimum
            value (0) will be used. Defaults to :data:`None`
        :type min_feat: int
        :param min_size: Minimum size of solutions (minimum number of
            features selected by solutions in the species). Must be in the
            interval [0, *max_feat - min_feat + 1*]. If omitted, the minimum
            value (0) will be used. Defaults to :data:`None`
        :type min_size: int
        :param max_feat: Largest feature index considered in this species.
            Must be in the interval [*min_feat*, *num_feats*). If omitted, the
            maximum possible feature index (*num_feats* - 1) will be used.
            Defaults to :data:`None`
        :type max_feat: int
        :param max_size: Maximum size of solutions. Must be in the interval
            [*min_size*, *max_feat - min_feat + 1*]. If omitted, the maximum
            possible size is used. Defaults to :data:`None`
        :type max_size: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        super().__init__()

        (
            self._num_feats,
            self._min_feat,
            self._min_size,
            self._max_feat,
            self._max_size
        ) = self._check_attributes(
            num_feats,
            min_feat,
            min_size,
            max_feat,
            max_size
        )

    @property
    def num_feats(self) -> int:
        """Number of features for this species.

        :rtype: int
        """
        return self._num_feats

    @property
    def min_feat(self) -> int:
        """Minimum feature index for this species.

        :rtype: int
        """
        return self._min_feat

    @property
    def max_feat(self) -> int:
        """Maximum feature index for this species.

        :rtype: int
        """
        return self._max_feat

    @property
    def min_size(self) -> int:
        """Minimum subset size for this species.

        :rtype: int
        """
        return self._min_size

    @property
    def max_size(self) -> int:
        """Maximum subset size for this species.

        :rtype: int
        """
        return self._max_size

    @classmethod
    def from_proportion(
        cls,
        num_feats: int,
        prop: float | None = None
    ) -> Species:
        """Create a parametrized species for testing purposes.

        Fix *min_feat*, *max_feat*, *min_size* and *max_size* proportionally
        to the number of features, according to *prop*, in this way:

            - *min_feat* = *num_feats* * *prop*
            - *max_feat* = *num_feats* - *min_feat* - 1
            - *min_size* = *min_feat*
            - *max_size* = *max_feat* - (2 * *min_feat*) + 1

        Here are some examples for *num_feats* = 1000

        ======  ==========  ==========  ==========  ==========
        *prop*  *min_feat*  *max_feat*  *min_size*  *max_size*
        ======  ==========  ==========  ==========  ==========
          0.00           0         999           0        1000
          0.05          50         949          50         850
          0.10         100         899         100         700
          0.15         150         849         150         550
          0.20         200         799         200         400
          0.25         250         749         250         250
        ======  ==========  ==========  ==========  ==========

        The maximum value for *prop* is
        :attr:`~culebra.solution.feature_selection.MAX_PROP`.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: int
        :param prop: Proportion of the number of features used to fix the
            species parameters. If omitted, the default proportion
            (:attr:`~culebra.solution.feature_selection.DEFAULT_PROP`) is used
        :type prop: float
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        :return: A Species object
        :rtype: ~culebra.solution.feature_selection.Species
        """
        # Check num_feats
        (num_feats, _, _, _, _) = cls._check_attributes(num_feats)

        # Check prop
        if prop is None:
            prop = DEFAULT_PROP
        else:
            prop = check_float(prop, "proportion", ge=0, le=MAX_PROP)

        # Parametrize the species from num_feats and prop
        min_feat = int(num_feats * prop)
        max_feat = num_feats - min_feat - 1
        min_size = min_feat
        max_size = max_feat - (2 * min_feat) + 1

        return cls(num_feats, min_feat, min_size, max_feat, max_size)

    def is_member(self, sol: Solution) -> bool:
        """Check if a solution meets the constraints imposed by the species.

        :param sol: The solution
        :type sol: ~culebra.solution.feature_selection.Solution
        :return: :data:`True` if the solution belongs to the species.
            :data:`False` otherwise
        :rtype: bool
        """
        sol_is_member = True
        # Check the number of features
        selected_num_feats = sol.num_feats
        if (selected_num_feats < self.min_size or
                selected_num_feats > self.max_size):
            sol_is_member = False

        # Check the minimum and maximum index
        if selected_num_feats > 0 and (
                sol.min_feat < self.min_feat or sol.max_feat > self.max_feat):
            sol_is_member = False

        return sol_is_member

    @staticmethod
    def _check_attributes(
        num_feats: int,
        min_feat: int | None = None,
        min_size: int | None = None,
        max_feat: int | None = None,
        max_size: int | None = None
    ) -> tuple[int, ...]:
        """Check the species attributes.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: int
        :param min_feat: Smallest feature index considered in this species.
            Must be in the interval [0, *num_feats*). If omitted, the minimum
            value (0) will be used. Defaults to :data:`None`
        :type min_feat: int
        :param min_size: Minimum size of solutions (minimum number of
            features selected by solutions in the species). Must be in the
            interval [0, *max_feat - min_feat + 1*]. If omitted, the minimum
            value (0) will be used. Defaults to :data:`None`
        :type min_size: int
        :param max_feat: Largest feature index considered in this species.
            Must be in the interval [*min_feat*, *num_feats*). If omitted, the
            maximum possible feature index (*num_feats* - 1) will be used.
            Defaults to :data:`None`
        :type max_feat: int
        :param max_size: Maximum size of solutions. Must be in the interval
            [*min_size*, *max_feat - min_feat + 1*]. If omitted, the maximum
            possible size is used. Defaults to :data:`None`
        :type max_size: int
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        # Check num_feats
        num_feats = check_int(num_feats, "number of features", gt=0)

        # Check min_feat
        if min_feat is None:
            min_feat = 0
        min_feat = check_int(
            min_feat, "minimum feature index", ge=0, lt=num_feats
        )

        # Check max_feat
        if max_feat is None:
            max_feat = num_feats - 1
        max_feat = check_int(
            max_feat, "maximum feature index", ge=min_feat, lt=num_feats)

        # Size limit
        size_limit = max_feat - min_feat + 1

        # Check min_size
        if min_size is None:
            min_size = 0
        min_size = check_int(
            min_size, "minimum subset size", ge=0, le=size_limit
        )

        # Check max_size
        if max_size is None:
            max_size = size_limit
        max_size = check_int(
            max_size, "maximum subset size", ge=min_size, le=size_limit
        )

        return (num_feats, min_feat, min_size, max_feat, max_size)

    def __copy__(self) -> Species:
        """Shallow copy the species.

        :return: The copied species
        :rtype: ~culebra.solution.feature_selection.Species
        """
        cls = self.__class__
        result = cls(self.num_feats)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Species:
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: dict
        :return: The copied species
        :rtype: ~culebra.solution.feature_selection.Species
        """
        cls = self.__class__
        result = cls(self.num_feats)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the species.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, (self.num_feats,), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> Species:
        """Return a species from a state.

        :param state: The state
        :type state: dict
        :return: The object
        :rtype: ~culebra.solution.feature_selection.Species
        """
        obj = cls(
            state['_num_feats']
        )
        obj.__setstate__(state)
        return obj


class Solution(BaseSolution):
    """Abstract base class for all the feature selector solutions."""

    species_cls = Species
    """Class for the species used by the
    :class:`~culebra.solution.feature_selection.Solution` class to constrain
    all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: type[Fitness],
        features: Sequence[int] | None = None
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: ~culebra.solution.feature_selection.Species
        :param fitness_cls: The solution's fitness class
        :type fitness_cls: type[~culebra.abc.Fitness]
        :param features: Initial features
        :type features: ~collections.abc.Sequence[int]
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)

        if features is not None:
            self.features = features
        else:
            self._setup()

    @property
    @abstractmethod
    def features(self) -> np.ndarray[int]:
        """Features selected by the solution

        This property must be overridden by subclasses to return a correct
        value.

        :return: Return an ordered sequence with the indices of the selected
            features
        :rtype: ~numpy.ndarray[int]

        :setter: Set a new set of selected features
        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The features property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @features.setter
    @abstractmethod
    def features(self, values: Sequence[int]) -> None:
        """Set a new set of selected features.

        This property setter must be overridden by subclasses.

        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The features property setter has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def num_feats(self) -> int:
        """Number of features selected by the solution.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The num_feats property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def min_feat(self) -> int | None:
        """Minimum feature index selected by the solution.

        This property must be overridden by subclasses to return a correct
        value.

        :return: The minimum feature index or :data:`None` if no feature has
            been selected
        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The min_feat property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @property
    @abstractmethod
    def max_feat(self) -> int | None:
        """Maximum feature index selected by the solution.

        This property must be overridden by subclasses to return a correct
        value.

        :return: The maximum feature index or :data:`None` if no feature has
            been selected
        :rtype: int
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The max_feat property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @abstractmethod
    def _setup(self) -> None:
        """Init the features of this solution randomly.

        This method must be overridden by subclasses.

        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _setup method has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    def __str__(self) -> str:
        """Solution as a string.

        :rtype: str
        """
        return str(self.features)

    def __repr__(self) -> str:
        """Solution representation.

        :rtype: str
        """
        cls_name = self.__class__.__name__
        species_info = str(self.species)
        fitness_info = self.fitness.values

        return (f"{cls_name}(species={species_info}, fitness={fitness_info}, "
                f"features={str(self)})")


class BinarySolution(Solution):
    """Feature selector implementation based on boolean arrays."""

    @property
    def features(self) -> np.ndarray[int]:
        """Features selected by the solution

        :return: Return an ordered sequence with the indices of the selected
            features
        :rtype: ~numpy.ndarray[int]

        :setter: Set a new set of selected features
        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        return self._features.nonzero()[0] + self.species.min_feat

    @features.setter
    def features(self, values: Sequence[int]) -> None:
        """Set a new set of selected features.

        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        # Get the set of indices
        indices = np.unique(
            np.asarray(values, dtype=int)
        ) - self.species.min_feat

        num_feats = self.species.max_feat - self.species.min_feat + 1
        self._features = np.zeros(num_feats, dtype=bool)
        self._features[indices] = True

        if not self.species.is_member(self):
            raise ValueError("The values provided do not meet the species "
                             "constraints")

    @property
    def num_feats(self) -> int:
        """Number of features selected by the solution.

        :rtype: int
        """
        return np.count_nonzero(self._features)

    @property
    def min_feat(self) -> int | None:
        """Minimum feature index selected by the solution.

        :return: The minimum feature index or :data:`None` if no feature has
            been selected
        :rtype: int
        """
        index = None
        if self.num_feats > 0:
            index = self._features.nonzero()[0][0] + self.species.min_feat
        return index

    @property
    def max_feat(self) -> int | None:
        """Maximum feature index selected by the solution.

        :return: The maximum feature index or :data:`None` if no feature has
            been selected
        :rtype: int
        """
        index = None
        if self.num_feats > 0:
            index = self._features.nonzero()[0][-1] + self.species.min_feat
        return index

    def _setup(self) -> None:
        """Init the features of this solution randomly."""
        # Select the features randomly
        max_num_feats = self.species.max_feat - self.species.min_feat + 1
        self._features = np.random.random(max_num_feats) < 0.5

        # Number of selected features
        selected_num_feats = np.count_nonzero(self._features)

        # Repair too small BitVectors
        if selected_num_feats < self.species.min_size:
            # Indices of the unselected features
            unselected = np.where(~self._features)[0]

            # Random choice of the minimal set of features to be selected
            np.random.shuffle(unselected)
            self._features[
                unselected[:self.species.min_size - selected_num_feats]] = True

            # The number of selected features is now the minimim allowed size
            selected_num_feats = self.species.min_size

        # Repair too large BitVectors
        if selected_num_feats > self.species.max_size:
            # Indices of the selected features
            selected = np.where(self._features)[0]

            # Random choice of the minimal set of features to be deselected
            np.random.shuffle(selected)
            self._features[
                selected[:selected_num_feats - self.species.max_size]
            ] = False


class IntSolution(Solution):
    """Feature selector implementation based on arrays of indices."""

    @property
    def features(self) -> np.ndarray[int]:
        """Features selected by the solution

        :return: Return an ordered sequence with the indices of the selected
            features
        :rtype: ~numpy.ndarray[int]

        :setter: Set a new set of selected features
        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        # Sort the array
        self._features.sort()
        return self._features

    @features.setter
    def features(self, values: Sequence[int]) -> None:
        """Set a new set of selected features.

        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        # Get the set of indices
        self._features = np.unique(np.asarray(values, dtype=int))

        if not self.species.is_member(self):
            raise ValueError(
                "The values provided do not meet the species constraints"
            )

    @property
    def num_feats(self) -> int:
        """Number of features selected by the solution.

        :rtype: int
        """
        return self._features.size

    @property
    def min_feat(self) -> int | None:
        """Minimum feature index selected by the solution.

        :return: The minimum feature index or :data:`None` if no feature has
            been selected
        :rtype: int
        """
        return min(self._features) if self.num_feats > 0 else None

    @property
    def max_feat(self) -> int | None:
        """Maximum feature index selected by the solution.

        :return: The maximum feature index or :data:`None` if no feature has
            been selected
        :rtype: int
        """
        return max(self._features) if self.num_feats > 0 else None

    def _setup(self) -> None:
        """Init the features of this solution randomly."""
        # All possible indices for the species
        indices = np.arange(self.species.min_feat,
                            self.species.max_feat + 1)

        # Random size
        size = randint(self.species.min_size, self.species.max_size)

        # Select the features of the new solution
        np.random.shuffle(indices)
        self._features = indices[:size]


class BitVector(BinarySolution, BaseIndividual):
    """BitVector Individual."""

    def crossover1p(self, other: BitVector) -> tuple[BitVector, BitVector]:
        """Cross this individual with another one.

        This method implements the single-point crossover.

        :param other: The other individual
        :type other: ~culebra.solution.feature_selection.BitVector
        :return: The two offspring
        :rtype: tuple[~culebra.solution.feature_selection.BitVector]
        """
        while True:
            # Cross point. Use randint to include also max_feat
            min_feat = self.species.min_feat
            max_feat = self.species.max_feat
            cross_point = randint(0, max_feat - min_feat)

            # Try a crossing of the the individuals
            (self._features[:cross_point],
             other._features[:cross_point]) = \
                (other._features[:cross_point].copy(),
                 self._features[:cross_point].copy())

            # Check if the number of features meets the species constraints
            if (self.species.min_size <= self.num_feats <=
                    self.species.max_size) and \
               (self.species.min_size <= other.num_feats <=
                    self.species.max_size):
                break

        # Return the offspring
        del self.fitness.values
        del other.fitness.values
        return self, other

    def crossover2p(self, other: BitVector) -> tuple[BitVector, BitVector]:
        """Cross this individual with another one.

        This method implements the two-points crossover.

        :param other: The other individual
        :type other: ~culebra.solution.feature_selection.BitVector
        :return: The two offspring
        :rtype: tuple[~culebra.solution.feature_selection.BitVector]
        """
        min_feat = self.species.min_feat
        max_feat = self.species.max_feat

        while True:
            # Cross points. Use randint to include also max_feat
            cross_point_0 = randint(0, max_feat - min_feat)
            cross_point_1 = randint(0, max_feat - min_feat)
            if cross_point_0 > cross_point_1:
                cross_point_0, cross_point_1 = cross_point_1, cross_point_0

            # Cross the individuals
            (self._features[cross_point_0:cross_point_1],
             other._features[cross_point_0:cross_point_1]) = \
                (other._features[cross_point_0:cross_point_1].copy(),
                 self._features[cross_point_0:cross_point_1].copy())

            # Check if the number of features meets the species constraints
            if (self.species.min_size <= self.num_feats <=
                    self.species.max_size) and \
               (self.species.min_size <= other.num_feats <=
                    self.species.max_size):
                break

        # Return the offspring
        del self.fitness.values
        del other.fitness.values
        return self, other

    crossover = crossover1p
    """Default crossover operator.

    Implemented as the single-point crossover.

    :param other: The other individual
    :type other: ~culebra.solution.feature_selection.BitVector
    :return: The two offspring
    :rtype: tuple[~culebra.solution.feature_selection.BitVector]
    """

    def mutate(self, indpb: float) -> tuple[BitVector]:
        """Mutate the individual.

        Each feature is independently mutated according to the given
        probability.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: float
        :return: The mutant
        :rtype: tuple[~culebra.solution.feature_selection.BitVector]
        """
        while True:
            # Mask to select the features that will be flipped
            mask = np.random.random(self._features.size) < indpb

            self._features[mask] = ~self._features[mask]
            # Check if the number of features meets the species constraints
            if (self.species.min_size <= self.num_feats <=
                    self.species.max_size):
                break

        # Return the mutant
        del self.fitness.values
        return (self,)


class IntVector(IntSolution, BaseIndividual):
    """Individual implementation based on arrays of indices."""

    def crossover(self, other: IntVector) -> tuple[IntVector, IntVector]:
        """Cross this individual with another one.

        All the common features will remain common in the new offspring. The
        remaining features will be randomly distributed to generate two new
        individuals.

        :param other: The other individual
        :type other: ~culebra.solution.feature_selection.IntVector
        :return: The two offspring
        :rtype: tuple[~culebra.solution.feature_selection.IntVector]
        """
        # Common features to both individuals
        common = np.intersect1d(
            self._features, other._features, assume_unique=True
        )

        # Uncommon features
        uncommon = np.setxor1d(
            self._features, other._features, assume_unique=True
        )

        # Create the two offspring
        np.random.shuffle(uncommon)

        lower_limit = max(
            0,
            self.species.min_size - common.size,
            common.size + uncommon.size - self.species.max_size
        )

        upper_limit = uncommon.size - lower_limit

        cross_point = randint(lower_limit, upper_limit)

        # Return the new offspring
        self._features = np.concatenate((common, uncommon[:cross_point]))
        other._features = np.concatenate((common, uncommon[cross_point:]))

        del self.fitness.values
        del other.fitness.values

        return self, other

    def mutate(self, indpb: float) -> tuple[IntVector]:
        """Mutate the individual.

        Each feature is independently mutated according to the given
        probability.

        :param indpb: Independent probability for each feature to be mutated.
        :type indpb: float
        :return: The mutant
        :rtype: tuple[~culebra.solution.feature_selection.IntVector]
        """
        # All possible features for the species
        all_feats = np.arange(
            self.species.min_feat,
            self.species.max_feat + 1
        )

        # Features to be kept
        to_be_kept = self._features[
            np.random.random(self.num_feats) >= indpb
        ]

        # Features not selected by the original individual
        not_selected = np.setdiff1d(all_feats, self._features)

        # Features to be added to the mutated individual
        to_be_added = not_selected[np.random.random(not_selected.size) <
                                   indpb]

        # Generate the new mutated individual
        new_features = np.concatenate((to_be_kept, to_be_added))

        # Repair too small individuals
        if new_features.size < self.species.min_size:
            # Features not considered by the individual
            not_considered = np.setdiff1d(all_feats, new_features)

            # Number of needed features to achieve the minimum size for the
            # species
            needed = self.species.min_size - new_features.size

            # Obtain the patch for the individual
            patch = np.random.choice(not_considered, needed, replace=False)

            # Apply the patch
            new_features = np.concatenate((new_features, patch))

        # Repair too large individuals
        if new_features.size > self.species.max_size:
            # Select only some of the features to maintain the maximum size
            # for this species
            new_features = np.random.choice(
                new_features, self.species.max_size, replace=False
            )

        self._features = new_features
        del self.fitness.values

        # Return the new individual
        return (self,)


class Ant(IntSolution, BaseAnt):
    """Ant to apply ACO to Fs problems."""

    def __init__(
        self,
        species: Species,
        fitness_cls: type[Fitness],
        features: Sequence[int] | None = None
    ) -> None:
        """Construct a default solution.

        :param species: The species the solution will belong to
        :type species: ~culebra.solution.feature_selection.Species
        :param fitness_cls: The solution's fitness class
        :type fitness_cls: type[~culebra.abc.Fitness]
        :param features: Initial features
        :type features: ~collections.abc.Sequence[int]
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls, features)
        self._discarded = np.empty(shape=(0,), dtype=int)

    def _setup(self) -> None:
        """Set the default path for ants.

        An empty path is set.
        """
        self._features = np.empty(shape=(0,), dtype=int)

    @property
    def path(self) -> np.ndarray[int]:
        """Path traveled by the ant.

        :rtype: ~numpy.ndarray[int]
        """
        return self._features

    @property
    def discarded(self) -> np.ndarray[int]:
        """Nodes discarded by the ant.

        :rtype: ~numpy.ndarray[int]
        """
        return self._discarded

    def append(self, feature: int) -> None:
        """Append a new feature to the ant's path.

        :param feature: The feature index
        :type feature: int
        :raises TypeError: If *feature* is not an integer number
        :raises ValueError: If *feature* does not meet the species
            constraints.
        :raises ValueError: If *feature* is already in the path or has been
            previously discarded
        """
        # Check the feature type
        feature = check_int(feature, "feature index")

        if feature in self.path:
            raise ValueError(
                f"Feature {feature} is already in the path"
            )
        if feature in self.discarded:
            raise ValueError(
                f"Feature {feature} has been previously discarded"
            )
        self._features = np.append(self.path, (feature))
        del self.fitness.values

    def discard(self, feature: int) -> None:
        """Discard a feature.

        The discarded feature is not appended to the ant's path.

        :param feature: The feature index
        :type feature: int
        :raises TypeError: If *feature* is not an integer number
        :raises ValueError: If *feature* is already in the path or has been
            previously discarded
        """
        # Check the feature type
        feature = check_int(feature, "feature index")

        if feature in self.path:
            raise ValueError(
                f"Feature {feature} is already in the path"
            )
        if feature in self.discarded:
            raise ValueError(
                f"Feature {feature} has been previously discarded"
            )
        self._discarded = np.append(self.discarded, (feature))

    @property
    def features(self) -> np.ndarray[int]:
        """Features selected by the ant

        :return: Return an ordered sequence with the indices of the selected
            features. Use :attr:`~culebra.solution.feature_selection.Ant.path`
            to get the path in the order the ant traveled it
        :rtype: ~numpy.ndarray[int]

        :setter: Set a new set of selected features
        :param values: The indices of the new features
        :type values: ~collections.abc.Sequence[int]
        :raises ValueError: If set to new feature indices values which do not
            meet the species constraints.
        """
        # Sort the array
        the_features = copy(self.path)
        the_features.sort()
        return the_features

    @features.setter
    def features(self, path: Sequence[int]) -> None:
        """Set a new traveled features path for the ant.

        :param path: The new features path
        :type path: ~collections.abc.Sequence[int]
        :raises ValueError: If any feature in *path* does not meet the species
            constraints.
        :raises ValueError: If any feature in *path* is visited more than once.
        """
        # Get the path as an array
        path = np.asarray(path, dtype=int)

        if len(np.unique(path)) != len(path):
            raise ValueError(
                "The path provided contains repeated features"
            )

        self._features = path

        if not self.species.is_member(self):
            raise ValueError(
                "The path provided does not meet the species constraints"
            )

    def __repr__(self) -> str:
        """Return the ant representation.

        :rtype: str
        """
        return BaseAnt.__repr__(self)


class Metrics(Base):
    """Provide some metrics about the selected features finally obtained.

    Evaluate the set of solutions found by a :class:`~culebra.abc.Trainer`
    and calculate some metrics about the frequency of each selected feature.
    More information about such metrics can be found in [Gonzalez2019]_.
    """

    """.. [Gonzalez2019] J. González, J. Ortega, M. Damas, P. Martín-Smith,
       John Q. Gan. *A new multi-objective wrapper method for feature
       selection - Accuracy and stability analysis for BCI*.
       **Neurocomputing**, 333:407-418, 2019.
       https://doi.org/10.1016/j.neucom.2019.01.017.
    """

    @staticmethod
    def relevance(
        solutions: Sequence[Solution],
    ) -> Series:
        """Return the relevance of the features finally selected.

        The relevance is calculated according to the method proposed in
        [Gonzalez2019]_.

        :param solutions: Best solutions returned by the
            :class:`~culebra.abc.Trainer`
        :type solutions:
            ~collections.abc.Sequence[~culebra.solution.feature_selection.Solution]
        :return: The relevance of each feature appearing in the solutions
        :rtype: ~pandas.Series
        """
        # all relevances are initialized to 0
        relevances = dict(
            (feat, 0) for feat in np.arange(0, solutions[0].species.num_feats)
        )
        n_sol = 0
        for sol in solutions:
            if sol.num_feats > 0:
                n_sol += 1
                for feat in sol.features:
                    if feat in relevances:
                        relevances[feat] += 1
                    else:
                        relevances[feat] = 1

        if n_sol > 0:
            relevances = {
                feat: relevances[feat] / n_sol for feat in relevances
            }

        return Series(relevances).sort_index()

    @staticmethod
    def rank(
        solutions: Sequence[Solution],
    ) -> Series:
        """Return the rank of the features finally selected.

        The rank is calculated according to the method proposed in
        [Gonzalez2019]_.

        :param solutions: Best solutions returned by the
            :class:`~culebra.abc.Trainer`.
        :type solutions:
            ~collections.abc.Sequence[~culebra.solution.feature_selection.Solution]
        :return: The relevance of each feature appearing in the solutions.
        :rtype: ~pandas.Series
        """
        # Obtain the relevance of each feature. The series is sorted, starting
        # with the most relevant feature
        relevances = Metrics.relevance(solutions)

        # Obtain the different relevance values
        rel_values = np.sort(np.unique(relevances.values))[::-1]

        ranks = {}

        index = 0
        for val in rel_values:
            feats = [feat for feat, rel in relevances.items() if rel == val]
            n_feats = len(feats)
            the_rank = (2 * index + n_feats - 1) / 2
            index += n_feats
            for feat in feats:
                ranks[feat] = the_rank

        return Series(ranks).sort_index()


# Exported symbols for this module
__all__ = [
    'Species',
    'Solution',
    'BinarySolution',
    'IntSolution',
    'BitVector',
    'IntVector',
    'Ant',
    'Metrics',
    'DEFAULT_PROP',
    'MAX_PROP'
]
