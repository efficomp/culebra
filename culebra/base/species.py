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

"""This module implements the :py:class:`~base.species.Species` class.

All individuals evolved within the same subpopulation must belong to the same
species. Species control the common parameters for all the individuals in the
population, such as their number of input features, the minimum and maximum
sizes for individuals, and the range of features evolved by the species.
"""

import numbers

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


class Species:
    """Species defining the characteristics of individuals."""

    def __init__(self, num_feats, *, min_feat=0, max_feat=-1, min_size=0,
                 max_size=-1):
        """Create a new species.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param min_feat: Smallest feature index considered in this species.
            Must be in the interval [0, *num_feats*). Defaults to 0
        :type min_feat: :py:class:`int`, optional
        :param max_feat: Largest feature index considered in this species.
            Must be in the interval [*min_feat*, *num_feats*). Negative values
            are interpreted as the maximum possible feature index
            (*num_feats* - 1). Defaults to -1
        :type max_feat: :py:class:`int`, optional
        :param min_size: Minimum size of individuals (minimum number of
            features selected by individuals in the species). Must be in the
            interval [0, *max_feat - min_feat + 1*]. Defaults to 0
        :type min_size: :py:class:`int`, optional
        :param max_size: Maximum size of individuals. Must be in the interval
            [*min_size*, *max_feat - min_feat + 1*]. Negative values are
            interpreted as the maximum possible size. Defaults to -1
        :type max_size: :py:class:`int`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        """
        self.__num_feats = Species.__check_num_feats(self, num_feats)
        self.__min_feat = Species.__check_min_feat(self, min_feat)
        self.__max_feat = Species.__check_max_feat(self, max_feat)
        self.__min_size = Species.__check_min_size(self, min_size)
        self.__max_size = Species.__check_max_size(self, max_size)

    @classmethod
    def from_proportion(cls, num_feats, *, prop=DEFAULT_PROP):
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

        The maximum value for *prop* is :py:attr:`~base.species.MAX_PROP`.

        :param num_feats: Number of input features considered in the feature
            selection problem
        :type num_feats: :py:class:`int`
        :param prop: Proportion of the number of features used to fix the
            species parameters. Defaults to
            :py:attr:`~base.species.DEFAULT_PROP`. The maximum allowed value
            is :py:attr:`~base.species.MAX_PROP`.
        :type prop: :py:class:`float`, optional
        :raises TypeError: If any argument is not of the appropriate type
        :raises ValueError: If any argument has an incorrect value
        :return: A Species object
        :rtype: :py:class:`~base.species.Species`
        """
        nf = cls.__check_num_feats(cls, num_feats)

        # Checks prop
        p = cls.__check_prop(cls, prop)

        # Parametrize the species from num_feats and prop
        minf = int(nf * p)
        maxf = nf - minf - 1
        mins = minf
        maxs = maxf - (2 * minf) + 1

        return cls(nf, min_feat=minf, max_feat=maxf, min_size=mins,
                   max_size=maxs)

    @property
    def num_feats(self):
        """Number of features for this species.

        :type: :py:class:`int`
        """
        return self.__num_feats

    @property
    def min_feat(self):
        """Minimum feature index for this species..

        :type: :py:class:`int`
        """
        return self.__min_feat

    @property
    def max_feat(self):
        """Maximum feature index for this species.

        :type: :py:class:`int`
        """
        return self.__max_feat

    @property
    def min_size(self):
        """Minimum subset size for this species.

        :type: :py:class:`int`
        """
        return self.__min_size

    @property
    def max_size(self):
        """Maximum subset size for this species.

        :type: :py:class:`int`
        """
        return self.__max_size

    @property
    def __size_limit(self):
        """Limit for the individuals size for this species.

        :type: :py:class:`int`
        """
        return self.__max_feat - self.__min_feat + 1

    def __check_num_feats(self, num_feats):
        """Check if the type or value of num_feats is valid.

        :param num_feats: Proposed value for num_feats
        :type num_feats: :py:class:`int`
        :raise TypeError: If num_feats is not an integer
        :raise ValueError: If num_feats is lower than or equal to 0
        :return: A valid value for the number of features
        :rtype: :py:class:`int`
        """
        if not isinstance(num_feats, numbers.Integral):
            raise TypeError("The number of features should be an integer "
                            f"number: 'num_feats = {num_feats}'")
        if num_feats <= 0:
            raise ValueError("The number of features should be an integer "
                             f"greater than 0: 'num_feats = {num_feats}'")
        return num_feats

    def __check_min_feat(self, min_feat):
        """Check if the type or value of min_feat is valid.

        :param min_feat: Proposed value for min_feat
        :type min_feat: :py:class:`int`
        :raise TypeError: If min_feat is not an integer
        :raise ValueError: If min_feat is lower than 0 or if it is greater than
            or equal to the number of features
        :return: A valid value for the minimum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(min_feat, numbers.Integral):
            raise TypeError("The minimum feature index should be an integer "
                            f"number: 'min_feat = {min_feat}'")
        if min_feat < 0:
            raise ValueError("The minimum feature index should be greater "
                             f"than or equal to 0: 'min_feat = {min_feat}'")
        if min_feat >= self.__num_feats:
            raise ValueError("The minimum feature index should be lower than "
                             "the number of features: 'num_feats = "
                             f"{self.__num_feats}, min_feat = {min_feat}'")
        return min_feat

    def __check_max_feat(self, max_feat):
        """Check if the type or value of max_feat is valid.

        :param max_feat: Proposed value for max_feat
        :type max_feat: :py:class:`int`
        :raise TypeError: If max_feat is not an integer
        :raise ValueError: If max_feat is lower than checked_min_feat or if it
            is greater than or equal to the number of features.
        :return: A valid value for the maximum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(max_feat, numbers.Integral):
            raise TypeError("The maximum feature index should be an integer "
                            f"number: 'max_feat = {max_feat}'")

        if max_feat < 0:
            max_feat = self.__num_feats - 1

        if max_feat < self.__min_feat:
            raise ValueError("The maximum feature index should be greater "
                             "than or equal to min_feat: 'min_feat = "
                             f"{self.__min_feat}, max_feat = {max_feat}'")
        if max_feat >= self.__num_feats:
            raise ValueError("The maximum feature index should be lower than "
                             "the number of features: 'num_feats = "
                             f"{self.__num_feats}, max_feat = {max_feat}'")
        return max_feat

    def __check_min_size(self, min_size):
        """Check if the type or value of min_size is valid.

        :param min_size: Proposed value for min_size
        :type min_size: :py:class:`int`
        :raise TypeError: If min_size is not an integer
        :raise ValueError: If min_size is lower than 0 or if it is greater than
            checked_max_feat - checked_min_feat + 1.
        :return: A valid value for the minimum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(min_size, numbers.Integral):
            raise TypeError("The minimum subset size should be an integer "
                            f"number: 'min_size = {min_size}'")

        if min_size < 0:
            raise ValueError("The minimum subset size should be greater than "
                             f"or equal to 0: 'min_size = {min_size}'")

        if min_size > self.__size_limit:
            raise ValueError("The minimum subset size should be lower than or "
                             "equal to (max_feat - min_feat + 1): 'min_feat = "
                             f"{self.__min_feat}, max_feat = "
                             f"{self.__max_feat}, min_size = {min_size}'")

        return min_size

    def __check_max_size(self, max_size):
        """Check if the type or value of max_size is valid.

        :param max_size: Proposed value for max_size
        :type max_size: :py:class:`int`
        :raise TypeError: If max_size is not an integer
        :raise ValueError: If max_size is lower than checked_min_size or if it
            is greater than checked_max_feat - checked_min_feat + 1.
        :return: A valid value for the maximum feature index
        :rtype: :py:class:`int`
        """
        if not isinstance(max_size, numbers.Integral):
            raise TypeError("The maximum subset size should be an integer "
                            f"number: 'max_size = {max_size}'")

        if max_size < 0:
            max_size = self.__size_limit

        if max_size < self.__min_size:
            raise ValueError("The maximum subset size should be greater than "
                             "or equal to the minimum size: 'min_size = "
                             f"{self.__min_size}, max_size = {max_size}'")

        if max_size > self.__size_limit:
            raise ValueError("The maximum subset size should be lower than or "
                             "equal to (max_feat - min_feat + 1): 'min_feat = "
                             f"{self.__min_feat}, max_feat = "
                             f"{self.__max_feat}, max_size = {max_size}'")

        return max_size

    def __check_prop(self, prop):
        """Check if the type or value of prop is valid.

        :param prop: Proportion of the number of features used to fix the
            attributes of a species
        :type prop: :py:class:`float`
        :raise TypeError: If prop is not a valid real number
        :raise ValueError: If prop is not in the interval [0, *MAX_PROP*]
        :return: A vaild value for the proportion of the number of features
        :rtype: :py:class:`float`
        """
        if not isinstance(prop, numbers.Real):
            raise TypeError("The proportion must be a real value: 'prop = "
                            f"{prop}'")

        if not 0 <= prop <= MAX_PROP:
            raise ValueError("The proportion must be in the interval "
                             f"[0, {MAX_PROP}]: 'prop = {prop}'")

        return prop

    def __repr__(self):
        """Return the Species representation."""
        cls = self.__class__
        cls_name = cls.__name__
        properties = (
                p for p in dir(cls)
                if isinstance(getattr(cls, p), property) and
                not p.startswith('_')
                      )

        repr = cls_name
        sep = "("
        for p in properties:
            repr += sep + p + "=" + getattr(self, p).__str__()
            sep = ", "

        repr += ")"
        return repr

    def __setstate__(self, state):
        """Set the state of the individual.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce the individual.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (self.num_feats, ), self.__dict__)
