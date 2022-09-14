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

"""Base classes of culebra."""

from __future__ import annotations
from copy import deepcopy


__author__ = "Jesús González"
__copyright__ = "Copyright 2021, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.1.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


class Base:
    """Base for all classes in culebra."""

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
                if isinstance(value, type) else value.__str__()
            )
            msg += sep + prop + ": " + value_str
            sep = ", "

        if sep[0] == "(":
            msg += sep
        msg += ")"
        return msg


# Exported symbols for this module
__all__ = ['Base']
