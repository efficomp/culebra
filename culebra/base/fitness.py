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

"""Provides the :py:class:`~base.fitness.Fitness` class."""

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import numbers
from deap.base import Fitness as DeapFitness

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_THRESHOLD = 0
"""Default threshold."""


class Fitness(DeapFitness):
    """Define the base class for the fitness evaluation of an individual."""

    names = ()
    """Names of the objectives."""

    def __init__(self, **params):
        """Create the Fitness object.

        Fitness objects are compared lexicographically. The comparison applies
        a similarity threshold to assume that two fitness values are similar
        (if their difference is lower then the similarity threshold).

        :param thresholds: Thresholds to assume if two fitness values are
            equivalent. If only a single value is provided, the same threshold
            will be used for all the objectives. A different threshold can be
            provided for each objective with a
            :py:class:`~collections.abc.Sequence`. Defaults to
            :py:attr:`~base.fitness.DEFAULT_THRESHOLD`
        :type thresholds: :py:class:`float` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float` numbers,
            optional
        :raises TypeError: If *thresholds* is not a :py:class:`float` value or
            a :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises ValueError: If a negative threshold is provided
        """
        super().__init__()

        # Get the fitness thresholds
        self.thresholds = params.pop('thresholds', DEFAULT_THRESHOLD)

    def eval(self, ind, dataset):
        """Evaluate an individual.

        This method must be overriden by subclasses to return a correct
        value.

        :param ind: The individual
        :type ind: Any subclass of :py:class:`~base.individual.Individual`
        :param dataset: A dataset
        :type dataset: :py:class:`~base.dataset.Dataset`
        :raises NotImplementedError: if has not been overriden
        :return: The fitness of *ind*
        :rtype: :py:class:`tuple`
        """
        raise NotImplementedError("The evaluation operator has not been "
                                  "implemented")

    @property
    def n_obj(self):
        """Number of objectives to be optimized.

        :type: :py:class:`int`
        """
        return len(self.names)

    @property
    def thresholds(self):
        """Similarity thresholds.

        Applied to assume if two fitness values are equivalent.

        :getter: Return the similarity threshold for each objective
        :setter: Set the similarity thresholds. If only a single value is
            provided, the same threshold will be used for all the objectives.
            A different threshold can be provided for each objective with a
            :py:class:`~collections.abc.Sequence`

        :type: :py:class:`tuple` of :py:class:`float` numbers
        :raises TypeError: If set with a value which is not a
            :py:class:`float` value or a :py:class:`~collections.abc.Sequence`
            of :py:class:`float` numbers
        :raises ValueError: If a negative threshold is provided
        """
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value):
        """Similarity thresholds.

        Applied to assume if two fitness values are equivalent.

        :param value: If only a single value is provided, the same threshold
            will be used for all the objectives. A different threshold can be
            provided for each objective with a
            :py:class:`~collections.abc.Sequence`
        :type value: :py:class:`float` or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises TypeError: If *value* is not a :py:class:`float` value or a
            :py:class:`~collections.abc.Sequence` of :py:class:`float`
            numbers
        :raises ValueError: If a negative threshold is provided
        """
        def check_value(val):
            # Check that it is a real number
            if not isinstance(val, numbers.Real):
                raise TypeError("The fitness threshold must be a real number "
                                "or a sequence of real numbers")
            # Check that it is not positive
            if val < 0:
                raise ValueError("The fitness threshold can not be negative")

        # If value is not a sequence ...
        if not isinstance(value, Sequence):
            # Check the value
            check_value(value)
            # Use it for all the objectives
            self._thresholds = (value,) * self.n_obj
        else:
            if len(value) != self.n_obj:
                raise ValueError("Incorrect number of thresholds. The number "
                                 f"of objectives is {self.n_obj}")

            # Check the values
            for v in value:
                check_value(v)

            # Set the thresholds
            self._thresholds = tuple(value)

    def __setstate__(self, state):
        """Set the state of the fitness.

        :param state: The state
        :type state: :py:class:`dict`
        """
        self.__dict__.update(state)

    def __reduce__(self):
        """Reduce the fitness.

        :return: The reduction
        :rtype: :py:class:`tuple`
        """
        return (self.__class__, (), self.__dict__)

    def __repr__(self):
        """Return the fitness representation."""
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

    def dominates(self, other, which=slice(None)):
        """Domination operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        :param which: Slice indicating on which objectives the domination is
            tested. The default value is `slice(None)`, representing every
            objectives
        :type which: :py:class:`slice`
        :return: `True` if each objective of *self* is not strictly worse than
            the corresponding objective of *other* and at least one objective
            is strictly better
        :rtype: :py:class:`bool`
        """
        not_equal = False
        for sw, ow, th in zip(self.wvalues[which], other.wvalues[which],
                              self.thresholds[which]):
            if sw > ow and sw-ow > th:
                not_equal = True
            elif sw < ow and ow-sw > th:
                return False
        return not_equal

    def __hash__(self):
        """Hash number for this individual."""
        return hash(self.wvalues)

    def __gt__(self, other):
        """Greater than operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        """
        return not self.__le__(other)

    def __ge__(self, other):
        """Greater than or equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        """
        return not self.__lt__(other)

    def __le__(self, other):
        """Less than or equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        """
        for sw, ow, th in zip(self.wvalues, other.wvalues, self.thresholds):
            if sw < ow and ow-sw > th:
                return True
            elif sw > ow and sw-ow > th:
                return False
        return True

    def __lt__(self, other):
        """Less than operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        """
        for sw, ow, th in zip(self.wvalues, other.wvalues, self.thresholds):
            if sw < ow and ow-sw > th:
                return True
            elif sw > ow and sw-ow > th:
                return False
        return False

    def __eq__(self, other):
        """Equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        """
        for sw, ow, th in zip(self.wvalues, other.wvalues, self.thresholds):
            if abs(sw-ow) > th:
                return False
        return True

    def __ne__(self, other):
        """Not equal to operator.

        :param other: The other fitness
        :type other: Any :py:class:`~base.fitness.Fitness` subclass
        """
        return not self.__eq__(other)
