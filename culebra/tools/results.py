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

"""Results management."""

from __future__ import annotations
from typing import Optional
from collections import UserDict
from pandas import DataFrame, read_csv, ExcelWriter
from pandas.io.pickle import to_pickle, read_pickle
from culebra.base import (
    Base,
    DEFAULT_SEP,
    check_instance,
    check_str,
    check_filename
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_KEY = "Unlabeled"
"""Default key for results."""

DEFAULT_RESULTS_BACKUP_FILENAME = "results.gz"
"""Default file name for results backups."""

DEFAULT_RESULTS_EXCEL_FILENAME = "results.xlsx"
"""Default name of the datasheet to store the results."""


class Results(UserDict, Base):
    """Manages the results produced by the evaluation of a wrapper."""

    def __init__(
        self,
        file: Optional[str] = None,
        key: Optional[str] = None,
        sep: Optional[str] = None
    ) -> None:
        """Create a results manager.

        If a *file* is provided, the results are loaded from it. If a *key* is
        also provided, the results are labeled with it. Otherwise, the results
        manager keeps empty.

        :param file: File containing the data
        :type file: :py:class:`str`, path object or file-like objects, optional
        :param key: Key for the data
        :type key: :py:class:`str`, optional
        :param sep: Separator, defaults to :py:attr:`~base.DEFAULT_SEP`
        :type sep: :py:class:`str`, optional
        """
        # Init the superclasses
        super().__init__()

        if file is not None:
            self.read_csv(file, key, sep)

    def __setitem__(self, key: str, data: DataFrame) -> DataFrame:
        """Overriden to verify the key and value.

        Assure that the key is a :py:class:`str` and the value is a
        :py:class:`Dataframe`.

        :param key: Key for the results
        :type key: :py:class:`str`
        :param data: The data
        :type data: :py:class:`Dataframe`
        :return: The data inserted
        :rtype: :py:class:`Dataframe`
        """
        return super().__setitem__(
            check_instance(key, "key for the results", str),
            check_instance(data, "data", DataFrame)
        )

    def read_csv(
        self, file: str,
        key: Optional[str] = None,
        sep: Optional[str] = None
    ) -> None:
        """Load data from a file.

        :param file: File containing the data
        :type file: :py:class:`str`, path object or file-like objects
        :param key: Key for the data. If set to :py:data:`None`,
            :py:attr:`~tools.DEFAULT_KEY` is used. Defaults
            to :py:data:`None`
        :type key: :py:class:`str`, optional
        :param sep: Separator. If :py:data:`None` is provided,
            :py:attr:`~base.DEFAULT_SEP`is used. Defaults to :py:data:`None`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *sep* is not a string
        """
        # Check sep
        sep = check_str(sep, "separator") if sep is not None else DEFAULT_SEP

        # Read the data
        data = read_csv(file, sep=sep)

        # Get the key
        key = DEFAULT_KEY if key is None else key

        # Insert the data
        self.__setitem__(key, data)

    def save(self, filename: Optional[str] = None) -> None:
        """Save these results.

        :param filename: File path. If set to :py:data:`None`,
            :py:attr:`~tools.DEFAULT_RESULTS_BACKUP_FILENAME` is used. Defaults
            to :py:data:`None`
        :type filename: :py:class:`str`, optional.
        """
        filename = check_filename(
            (
                DEFAULT_RESULTS_BACKUP_FILENAME
                if filename is None
                else filename
            ),
            name="results backup file name",
            ext=".gz"
        )
        to_pickle(self, filename)

    @classmethod
    def load(cls, filename: Optional[str] = None) -> None:
        """Load the results from a backup file.

        :param filename: File path. If set to :py:data:`None`,
            :py:attr:`~tools.DEFAULT_RESULTS_BACKUP_FILENAME` is used. Defaults
            to :py:data:`None`
        :type filename: :py:class:`str`, optional.
        """
        filename = check_filename(
            (
                DEFAULT_RESULTS_BACKUP_FILENAME
                if filename is None
                else filename
            ),
            name="results backup file name",
            ext=".gz"
        )
        return read_pickle(filename)

    def to_excel(self, filename: Optional[str] = None) -> None:
        """Save the results to a Excel file.

        :param filename: File path. If set to :py:data:`None`,
            :py:attr:`~tools.DEFAULT_RESULTS_EXCEL_FILENAME` is used. Defaults
            to :py:data:`None`
        :type filename: :py:class:`str`, optional.
        """
        filename = check_filename(
            (
                DEFAULT_RESULTS_EXCEL_FILENAME
                if filename is None
                else filename
            ),
            name="results excel datasheet file name",
            ext=".xlsx"
        )

        with ExcelWriter(filename) as writer: \
                # pylint: disable=abstract-class-instantiated
            for key, data in self.items():
                data.to_excel(writer, sheet_name=key)

            writer.save()


# Exported symbols for this module
__all__ = [
    'DEFAULT_KEY',
    'DEFAULT_RESULTS_BACKUP_FILENAME',
    'DEFAULT_RESULTS_EXCEL_FILENAME',
    'Results'
]
