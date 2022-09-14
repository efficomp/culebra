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
from abc import abstractmethod
from typing import Tuple, Type, Optional
from collections.abc import Sequence
from copy import deepcopy
from culebra.base import (
    Species as BaseSpecies,
    Individual as BaseIndividual,
    Fitness,
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

DEFAULT_PROP = 0.15
"""Default proportion for the generation of a species."""

MAX_PROP = 0.25
"""Maximum proportion for the generation of a species."""


class Species(BaseSpecies):
    """Species for the feature selector individuals."""

    def __init__(
        self,
        num_feats: int,
        min_feat: Optional[int] = None,
        min_size: Optional[int] = None,
        max_feat: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> None:
        """Create a new species.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param min_feat: Smallest feature index considered in this species.
            Must be in the interval [0, *num_feats*). If omitted, the minumum
            value (0) will be used. Defaults to :py:data:`None`
        :type min_feat: :py:class:`int`, optional
        :param min_size: Minimum size of individuals (minimum number of
            features selected by individuals in the species). Must be in the
            interval [0, *max_feat - min_feat + 1*]. If omitted, the minumum
            value (0) will be used. Defaults to :py:data:`None`
        :type min_size: :py:class:`int`, optional
        :param max_feat: Largest feature index considered in this species.
            Must be in the interval [*min_feat*, *num_feats*). If omitted, the
            maximum possible feature index (*num_feats* - 1) will be used.
            Defaults to :py:data:`None`
        :type max_feat: :py:class:`int`, optional
        :param max_size: Maximum size of individuals. Must be in the interval
            [*min_size*, *max_feat - min_feat + 1*]. If omitted, the maximum
            possible size is used. Defaults to :py:data:`None`
        :type max_size: :py:class:`int`, optional
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

    @classmethod
    def from_proportion(
        cls,
        num_feats: int,
        prop: float = DEFAULT_PROP
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
        :py:attr:`~genotype.feature_selection.MAX_PROP`.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param prop: Proportion of the number of features used to fix the
            species parameters. Defaults to
            :py:attr:`~genotype.feature_selection.DEFAULT_PROP`. The maximum
            allowed value is :py:attr:`~genotype.feature_selection.MAX_PROP`.
        :type prop: :py:class:`float`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        :return: A Species object
        :rtype: :py:class:`~genotype.feature_selection.Species`
        """
        # Check num_feats
        (num_feats, _, _, _, _) = cls._check_attributes(num_feats)

        # Check prop
        prop = check_float(prop, "proportion", ge=0, le=MAX_PROP)

        # Parametrize the species from num_feats and prop
        min_feat = int(num_feats * prop)
        max_feat = num_feats - min_feat - 1
        min_size = min_feat
        max_size = max_feat - (2 * min_feat) + 1

        return cls(num_feats, min_feat, min_size, max_feat, max_size)

    @property
    def num_feats(self) -> int:
        """Get the number of features for this species.

        :type: :py:class:`int`
        """
        return self._num_feats

    @property
    def min_feat(self) -> int:
        """Get the minimum feature index for this species..

        :type: :py:class:`int`
        """
        return self._min_feat

    @property
    def max_feat(self) -> int:
        """Get the maximum feature index for this species.

        :type: :py:class:`int`
        """
        return self._max_feat

    @property
    def min_size(self) -> int:
        """Get the minimum subset size for this species.

        :type: :py:class:`int`
        """
        return self._min_size

    @property
    def max_size(self) -> int:
        """Get the maximum subset size for this species.

        :type: :py:class:`int`
        """
        return self._max_size

    def is_member(self, ind: Individual) -> bool:
        """Check if an individual meets the constraints imposed by the species.

        :param ind: The individual
        :type ind: :py:class:`~genotype.feature_selection.Individual`
        :return: :py:data:`True` if the individual belongs to the species.
            :py:data:`False` otherwise
        :rtype: :py:class:`bool`
        """
        ind_is_member = True
        # Check the number of features
        selected_num_feats = ind.num_feats
        if (selected_num_feats < self.min_size or
                selected_num_feats > self.max_size):
            ind_is_member = False

        # Check the minimum and maximum index
        if selected_num_feats > 0 and (
                ind.min_feat < self.min_feat or ind.max_feat > self.max_feat):
            ind_is_member = False

        return ind_is_member

    @staticmethod
    def _check_attributes(
        num_feats: int,
        min_feat: Optional[int] = None,
        min_size: Optional[int] = None,
        max_feat: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> Tuple[int, ...]:
        """Check the species attributes.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param min_feat: Smallest feature index considered in this species.
            Must be in the interval [0, *num_feats*). If omitted, the minumum
            value (0) will be used. Defaults to :py:data:`None`
        :type min_feat: :py:class:`int`, optional
        :param min_size: Minimum size of individuals (minimum number of
            features selected by individuals in the species). Must be in the
            interval [0, *max_feat - min_feat + 1*]. If omitted, the minumum
            value (0) will be used. Defaults to :py:data:`None`
        :type min_size: :py:class:`int`, optional
        :param max_feat: Largest feature index considered in this species.
            Must be in the interval [*min_feat*, *num_feats*). If omitted, the
            maximum possible feature index (*num_feats* - 1) will be used.
            Defaults to :py:data:`None`
        :type max_feat: :py:class:`int`, optional
        :param max_size: Maximum size of individuals. Must be in the interval
            [*min_size*, *max_feat - min_feat + 1*]. If omitted, the maximum
            possible size is used. Defaults to :py:data:`None`
        :type max_size: :py:class:`int`, optional
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
        """Shallow copy the species."""
        cls = self.__class__
        result = cls(self.num_feats)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> Species:
        """Deepcopy the species.

        :param memo: Species attributes
        :type memo: :py:class:`dict`
        :return: A deep copy of the species
        :rtype: :py:class:`~genotype.feature_selection.Species`
        """
        cls = self.__class__
        result = cls(self.num_feats)
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result

    def __reduce__(self) -> tuple:
        """Reduce the species.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.num_feats,), self.__dict__)


class Individual(BaseIndividual):
    """Base class for all the feature selector individuals."""

    species_cls = Species
    """Class for the species used by the
    :py:class:`~genotype.feature_selection.Individual` class to
    constrain all its instances."""

    def __init__(
        self,
        species: Species,
        fitness_cls: Type[Fitness],
        features: Optional[Sequence[int]] = None
    ) -> None:
        """Construct a default individual.

        :param species: The species the individual will belong to
        :type species:
            :py:class:`~genotype.feature_selection.Individual.species_cls`
        :param fitness: The individual's fitness class
        :type fitness: :py:class:`~base.Fitness`
        :param features: Initial features
        :type features:
                :py:class:`~collections.abc.Sequence` of :py:class:`int`
        :raises TypeError: If *species* is not a valid species
        :raises TypeError: If *fitness_cls* is not a valid fitness class
        """
        # Init the superclass
        super().__init__(species, fitness_cls)

        if features is not None:
            self.features = features
        else:
            self._random_init()

    @abstractmethod
    def _random_init(self) -> None:
        """Init the features of this individual randomly.

        This method must be overriden by subclasses.

        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError("The _random_init method has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def features(self) -> Sequence[int]:
        """Get and set the indices of the features selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :getter: Return the indices of the selected features
        :setter: Set the new feature indices. An array-like object of
            feature indices is expected
        :type: :py:class:`~collections.abc.Sequence` of :py:class:`int`

        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError("The features property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @features.setter
    @abstractmethod
    def features(self, values: Sequence[int]) -> None:
        """Set the indices of the new features selected by the individual.

        This property setter must be overriden by subclasses.

        :param values: The new feature indices
        :type values: :py:class:`~collections.abc.Sequence`  of :py:class:`int`
        :raises NotImplementedError: if has not been overriden
        """
        raise NotImplementedError("The features property seter has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def num_feats(self) -> int:
        """Get the number of features selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int`
        """
        raise NotImplementedError("The num_feats property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def min_feat(self) -> int | None:
        """Minimum feature index selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int` or :py:data:`None` if no feature has been
            selected
        """
        raise NotImplementedError("The min_feat property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    @property
    @abstractmethod
    def max_feat(self) -> int | None:
        """Maximum feature index selected by the individual.

        This property must be overriden by subclasses to return a correct
        value.

        :raises NotImplementedError: if has not been overriden
        :type: :py:class:`int` or :py:data:`None` if no feature has been
            selected
        """
        raise NotImplementedError("The max_feat property has not been "
                                  "implemented in the "
                                  f"{self.__class__.__name__} class")

    def __str__(self) -> str:
        """Return the individual as a string."""
        return self.features.__str__()

    def __repr__(self) -> str:
        """Return the individual representation."""
        cls_name = self.__class__.__name__
        species_info = self.species.__str__()
        fitness_info = self.fitness.values

        return (f"{cls_name}(species={species_info}, fitness={fitness_info}, "
                f"features={self.__str__()})")


# Exported symbols for this module
__all__ = [
    'Species',
    'Individual',
    'DEFAULT_PROP',
    'MAX_PROP'
]
