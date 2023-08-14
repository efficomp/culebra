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

"""Checker functions for different data types."""

from __future__ import annotations

from typing import (
    Any,
    Optional,
    Callable,
    List,
    Dict,
    Sequence)
from numbers import Integral, Real
from os.path import normcase, normpath, splitext


__author__ = "Jesús González"
__copyright__ = "Copyright 2023, EFFICOMP"
__license__ = "GNU GPL-3.0-or-later"
__version__ = "0.2.1"
__maintainer__ = "Jesús González"
__email__ = "jesusgonzalez@ugr.es"
__status__ = "Development"


def check_bool(value: bool, name: str) -> bool:
    """Check if the given value is a valid boolean.

    :param value: The value
    :type value: :py:class:`bool`
    :param name: The value name
    :type name: :py:class:`str`
    :return: A valid boolean
    :rtype: :py:class:`bool`
    :raises TypeError: If *value* is not a boolean value
    """
    # Check that value is a bool
    if not isinstance(value, bool):
        raise TypeError(
            f"The {name} must be a boolean value: {value}"
        )

    return value


def check_str(
    value: str,
    name: str,
    valid_chars: Optional[str] = None,
    invalid_chars: Optional[str] = None
) -> str:
    """Check if the given value is a valid string.

    :param value: The value
    :type value: :py:class:`str`
    :param name: The value name
    :type name: :py:class:`str`
    :param valid_chars: If provided, contains the valid chars for the string
    :type valid_chars: :py:class:`str`, optional
    :param invalid_chars: If provided, contains the forbiden chars for the
        string
    :type invalid_chars: :py:class:`str`, optional
    :return: A valid string
    :rtype: :py:class:`str`
    :raises TypeError: If *value* is not a string
    """
    # Check that value is a string
    if not isinstance(value, str):
        raise TypeError(
            f"The {name} must be a string: {value}"
        )

    if valid_chars is not None:
        for ch in value:
            if ch not in valid_chars:
                raise ValueError(
                    f"The {name} contains an invalid char: {ch}")

    if invalid_chars is not None:
        for ch in value:
            if ch in invalid_chars:
                raise ValueError(
                    f"The {name} contains an invalid char: {ch}")

    return value


def check_limits(
    value: Real,
    name: str,
    gt: Optional[Real] = None,
    ge: Optional[Real] = None,
    lt: Optional[Real] = None,
    le: Optional[Real] = None
) -> Real:
    """Check if the given value meets the limits.

    :param value: The value
    :type value: :py:class:`~numbers.Real`
    :param name: The value name
    :type name: :py:class:`str`
    :param gt: Inferior limit. If provided, *value* must be greater than *gt*
    :type gt: :py:class:`~numbers.Real`, optional
    :param ge: Inferior limit. If provided, *value* must be greater than or
        equal to *ge*
    :type ge: :py:class:`~numbers.Real`, optional
    :param lt: Superior limit. If provided, *value* must be lower than *lt*
    :type lt: :py:class:`~numbers.Real`, optional
    :param le: Superior limit. If provided, *value* must be lower than or
        equal to *le*
    :type le: :py:class:`~numbers.Real`, optional
    :return: A valid integer
    :rtype: :py:class:`~numbers.Real`
    :raises ValueError: If *value* does not meet any imposed limit
    """
    # Check the limits
    if gt is not None and value <= gt:
        raise ValueError(
            f"The {name} must be greater than {gt}: {value}"
        )
    if ge is not None and value < ge:
        raise ValueError(
            f"The {name} must be greater than or equal to {ge}: {value}"
        )

    if lt is not None and value >= lt:
        raise ValueError(
            f"The {name} must be less than {lt}: {value}"
        )
    if le is not None and value > le:
        raise ValueError(
            f"The {name} must be less than or equal to {le}: {value}"
        )

    return value


def check_int(
    value: int,
    name: str,
    gt: Optional[int] = None,
    ge: Optional[int] = None,
    lt: Optional[int] = None,
    le: Optional[int] = None,
    ne: Optional[int] = None
) -> int:
    """Check if the given value is a valid integer.

    :param value: The value
    :type value: :py:class:`int`
    :param name: The value name
    :type name: :py:class:`str`
    :param gt: Inferior limit. If provided, *value* must be greater than *gt*
    :type gt: :py:class:`int`, optional
    :param ge: Inferior limit. If provided, *value* must be greater than or
        equal to *ge*
    :type ge: :py:class:`int`, optional
    :param lt: Superior limit. If provided, *value* must be lower than *lt*
    :type lt: :py:class:`int`, optional
    :param le: Superior limit. If provided, *value* must be lower than or
        equal to *le*
    :type le: :py:class:`int`, optional
    :param ne: Not equal. If provided, *value* can not be equal to *ne*
    :type ne: :py:class:`int`, optional
    :return: A valid integer
    :rtype: :py:class:`int`
    :raises TypeError: If *value* is not an integer number
    :raises ValueError: If *value* does not meet any imposed constraint
    """
    # Check that value is an integer number
    if not (
        isinstance(value, Integral) or
        (isinstance(value, Real) and value.is_integer())
    ):
        raise TypeError(
            f"The {name} must be an integer number: {value}"
        )

    # Check that value is not equal to ne
    if value == ne:
        raise ValueError(
            f"The {name} can not be equal to {ne}"
        )

    # Check the limits
    return check_limits(int(value), name, gt, ge, lt, le)


def check_float(
    value: float,
    name: str,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None
) -> float:
    """Check if the given value is a valid float.

    :param value: The value
    :type value: :py:class:`float`
    :param name: The value name
    :type name: :py:class:`str`
    :param gt: Inferior limit. If provided, *value* must be greater than *gt*
    :type gt: :py:class:`float`, optional
    :param ge: Inferior limit. If provided, *value* must be greater than or
        equal to *ge*
    :type ge: :py:class:`float`, optional
    :param lt: Superior limit. If provided, *value* must be lower than *lt*
    :type lt: :py:class:`float`, optional
    :param le: Superior limit. If provided, *value* must be lower than or
        equal to *le*
    :type le: :py:class:`float`, optional
    :return: A valid float
    :rtype: :py:class:`float`
    :raises TypeError: If *value* is not a floating point number
    """
    # Check that value is a float
    if not isinstance(value, Real):
        raise TypeError(
            f"The {name} must be a floating point number: {value}"
        )

    return check_limits(float(value), name, gt, ge, lt, le)


def check_instance(value: object, name: str, cls: type) -> object:
    """Check if the given value is an instance of *cls*.

    :param value: An object
    :type value: :py:class:`object`
    :param name: The value name
    :type name: :py:class:`str`
    :param cls: A class
    :type cls: :py:class:`type`
    :raises TypeError: If *value* is not instance of *cls*
    """
    if not isinstance(value, cls):
        raise TypeError(
            f"The {name} is not a valid {cls.__name__} instance: {value}"
        )

    return value


def check_subclass(value: type, name: str, cls: type) -> type:
    """Check if the given value is subclass of *cls*.

    :param value: A class
    :type value: :py:class:`type`
    :param name: The value name
    :type name: :py:class:`str`
    :param cls: Another class
    :type cls: :py:class:`type`
    :raises TypeError: If *value* is not subclass of *cls*
    """
    if not (isinstance(value, type) and issubclass(value, cls)):
        raise TypeError(
            f"The {name} is not subclass of {cls.__name__}: {value}"
        )

    return value


def check_func(
        value: Callable,
        name: str
) -> Callable:
    """Check if the given value is a valid function.

    :param value: A function
    :type value: :py:class:`~collections.abc.Callable`
    :param name: The value name
    :type name: :py:class:`str`
    :raises TypeError: If *value* is not callable
    """
    # Check that func is callable
    if not callable(value):
        raise TypeError(f"The {name} must be callable: {value}")

    return value


def check_func_params(value: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Check if the given value is a valid set of function parameters.

    :param value: A dictionary
    :type value: A :py:class:`dict`
    :param name: The value name
    :type name: :py:class:`str`
    :raises TypeError: If *value* is not a dictionary
    :raises ValueError: If the keys in *value* are not strings
    """
    # Check that func_params is a valid dict
    if not isinstance(value, dict):
        raise TypeError(f"The {name} must be in a dictionary: "
                        f"{value}")

    # Check that the keys are strings
    for key in value.keys():
        if not isinstance(key, str):
            raise ValueError(f"The {name} keys must be strings: {key}")

    return value


def check_sequence(
    seq: Sequence[Any],
    name: str,
    size: Optional[int] = None,
    item_checker: Optional[Callable[[Any, str], Any]] = None
) -> List[Any]:
    """Check a sequence of items.

    :param seq: The sequence.
    :type seq: :py:class:`~collections.abc.Sequence`
    :param name: The value name
    :type name: :py:class:`str`
    :param size: Length of the sequence. If omitted, sequences of any
        length are allowed
    :type size: :py:class:`int`, optional
    :param item_checker: Function to check the sequence items. If omitted,
        no restrictions are applied to the sequence items
    :type item_checker: :py:class:`~collections.abc.Callable`, optional
    :return: The checked sequence
    :rtype: A :py:class:`list`
    :raises TypeError: If *seq* is not a :py:class:`~collections.abc.Sequence`
    :raises ValueError: If the number of items in the sequence does not match
        *size*
    :raises ValueError: If any item fails the *item_checker* function
    """
    # Check that seq is a sequence
    if not isinstance(seq, Sequence):
        raise TypeError(f"The {name} must be in a sequence: {seq}")

    # Check the length
    if size is not None and len(seq) != size:
        raise ValueError(f"The length of {name} must be {size}: {seq}")

    # Transform the sequence into a list
    seq = list(seq)

    # Check all the items
    if item_checker is not None:
        for i, item in enumerate(seq):
            try:
                seq[i] = item_checker(item, name)
            except TypeError as error:
                raise ValueError(
                    f"Incorrect values for {name}: {seq}"
                ) from error

    return list(seq)


def check_filename(
    value: str,
    name: str,
    ext: Optional[str] = None
) -> str:
    """Check if the given value is a valid filename.

    :param value: The value
    :type value: :py:class:`str`, bytes or os.PathLike object
    :param name: The value name
    :type name: :py:class:`str`
    :param ext: Required file extension
    :type ext: :py:class:`str`, bytes or os.PathLike object, optional
    :return: A valid filename
    :rtype: :py:class:`str`
    :raises TypeError: If *value* is not a valid file name
    :raises ValueError: If *value* does not meet the constraints
    """
    # Check that value is a valid file path
    try:
        value = normcase(normpath(value))
    except TypeError as error:
        raise TypeError(f"Not valid {name}: {value}") from error

    # Check extension
    if ext is not None:
        try:
            ext = normcase(normpath(ext))
        except TypeError as error:
            raise TypeError(
                f"Not valid required extension for {name}: {ext}"
            ) from error

        if not ext.startswith("."):
            raise ValueError(
                f"Not valid required extension for {name}. Extensions must "
                f"begin with a dot: {ext}"
            )

        # Split the proposed file name
        (_, value_ext) = splitext(value)
        # Check the extension
        if not ext == value_ext:
            raise ValueError(
                f"The extension of {name} should be {ext}: {value}"
            )

    return value


__all__ = [
    'check_bool',
    'check_str',
    'check_limits',
    'check_int',
    'check_float',
    'check_instance',
    'check_subclass',
    'check_func',
    'check_func_params',
    'check_sequence',
    'check_filename'
]
