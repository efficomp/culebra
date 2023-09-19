#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""Unit test for :py:class:`culebra.abc.Base`."""

import unittest
import pickle
from copy import copy, deepcopy

from culebra.abc import Base


class BaseSubclass(Base):
    """Base for all classes in culebra."""

    def __init__(self, my_list=[]):
        """Construct a base object."""
        self.my_list = my_list

    @property
    def my_list(self):
        """Return the list."""
        return self._my_list

    @my_list.setter
    def my_list(self, value):
        """Set my_list.

        :param value: New value for my_list
        :type value: :py:class:`list`
        """
        self._my_list = value


the_list = [[1, 2, 3], [4, 5, 6]]


class BaseTester(unittest.TestCase):
    """Test the :py:class:`~culebra.abc.Base` class."""

    def test_copy(self):
        """Test the :py:meth:`~culebra.abc.Base.__copy__` method."""
        base1 = BaseSubclass(the_list)
        base2 = copy(base1)

        # Copy only copies the first level (base1 != base2)
        self.assertNotEqual(id(base1), id(base2))

        # The objects attributes are shared
        self.assertEqual(id(base1._my_list), id(base2._my_list))

    def test_deepcopy(self):
        """Test the :py:meth:`~culebra.abc.Base.__deepcopy__` method."""
        base1 = BaseSubclass(the_list)
        base2 = deepcopy(base1)

        # Check the copy
        self._check_deepcopy(base1, base2)

    def test_serialization(self):
        """Serialization test.

        Test the :py:meth:`~culebra.abc.Base.__setstate__` and
        :py:meth:`~culebra.abc.Base.__reduce__` methods.
        """
        base1 = BaseSubclass(the_list)

        data = pickle.dumps(base1)
        base2 = pickle.loads(data)

        # Check the serialization
        self._check_deepcopy(base1, base2)

    def test_repr(self):
        """Test the: py: meth: `~culebra.abc.Base.__repr__` method."""
        lists = [[], the_list]

        for value in lists:
            base = BaseSubclass(my_list=value)
            self.assertEqual(repr(base),
                             base.__class__.__name__ +
                             f"(my_list: {base.my_list})")

    def _check_deepcopy(self, obj1, obj2):
        """Check if *obj1* is a deepcopy of *obj2*.

        :param obj1: The first object
        :type obj1: :py:class:`BaseSubclass`
        :param obj2: The second object
        :type obj2: :py:class:`BaseSubclass`
        """
        # Copies all the levels
        self.assertNotEqual(id(obj1), id(obj2))
        self.assertNotEqual(id(obj1._my_list), id(obj2._my_list))
        for list1, list2 in zip(obj1._my_list, obj2._my_list):
            self.assertNotEqual(id(list1), id(list2))
            for value1, value2 in zip(list1, list2):
                self.assertEqual(value1, value2)


if __name__ == '__main__':
    unittest.main()
