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

from typing import Optional, Sequence
from collections import UserDict
from os.path import basename, splitext

from pandas import DataFrame, read_csv, ExcelWriter
from pandas.io.pickle import to_pickle, read_pickle

from culebra.abc import Base
from culebra.checker import (
    check_instance,
    check_str,
    check_sequence,
    check_filename
)

from . import DEFAULT_SEP

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.2.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_RESULTS_BASE_FILENAME = "results"
"""Default base file name for results backups."""

BACKUP_EXTENSION = ".gz"
"""Extension for backup files."""

EXCEL_EXTENSION = ".xlsx"
"""Extension for excel files."""


class Results(UserDict, Base):
    """Manages the results produced by the evaluation of a trainer."""

    default_base_filename = DEFAULT_RESULTS_BASE_FILENAME
    """Default base file name for results backups."""

    def __init__(self, base_filename: Optional[str] = None) -> None:
        """Create an empty results manager.

        :param base_filename: The base filename to save the results. If set to
            :py:data:`None`,
            :py:attr:`~culebra.tools.Results.default_base_filename` is used.
            Defaults to :py:data:`None`.
        :type base_filename: :py:class:`~str`, optional
        :type: :py:class:`~base.FitnessFunction`
        :raises TypeError: If *base_filename* is not a valid file name
        """
        super().__init__()

        # Set the defeault base name for backups
        self.base_filename = (
            DEFAULT_RESULTS_BASE_FILENAME
            if base_filename is None
            else base_filename
        )

    @classmethod
    def load(cls, filename: Optional[str] = None) -> None:
        """Load the results from a backup file.

        :param filename: File path (must have the ".gz" extension). If set to
            :py:data:`None`, the default
            :py:attr:`~culebra.tools.Results.default_base_filename` is used.
            Defaults to :py:data:`None`
        :type filename: :py:class:`str`, optional
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If *filename* has an invalid extension
        """
        filename = check_filename(
            (
                Results.default_base_filename + BACKUP_EXTENSION
                if filename is None
                else filename
            ),
            name="results backup file name",
            ext=BACKUP_EXTENSION
        )
        return read_pickle(filename)

    @classmethod
    def from_csv_files(
        cls,
        files: Sequence[str],
        keys: Sequence[str] = None,
        sep: Optional[str] = None
    ) -> None:
        """Load some results from several csv files.

        :param files: Sequence of files containing the results
        :type files: :py:class:`~collections.abc.Sequence` of :py:class:`str`,
            path objects or file-like objects
        :param keys: Keys for the different results. One key for each csv file.
            If omitted, the basename of each file in *files* (without
            extension) is used. Defaults to :py:data:`None`
        :type keys: :py:class:`~collections.abc.Sequence` of :py:class:`str`
        :param sep: Separator. If :py:data:`None` is provided,
            :py:attr:`~culebra.tools.DEFAULT_SEP` is used. Defaults to
            :py:data:`None`
        :type sep: :py:class:`str`, optional
        :raises TypeError: If *sep* is not a string
        :raises ValueError: If *files* and *keys* have different lengths
        :raises ValueError: If any key in *keys* is not a string
        :raises FileNotFoundError: If any file in *files* is not found
        """
        # Check files
        files = check_sequence(files, "csv files")

        # If no keys are provided, take the base filenames, without extension
        if keys is None:
            keys = list(splitext(basename(file))[0] for file in files)

        # Check keys
        keys = check_sequence(
            keys, "csv keys", size=len(files), item_checker=check_str
        )

        # Check sep
        sep = check_str(sep, "separator") if sep is not None else DEFAULT_SEP

        # Create the results
        results = cls()

        # Read the data
        for file, key in zip(files, keys):
            results[key] = read_csv(file, sep=sep)

        # Return the results
        return results

    @property
    def base_filename(self) -> str:
        """Get and set the base filename used to save the results.

        :getter: Return the current base_filename
        :setter: Set a new base_filename
        :type: :py:class:`str`
        :raises TypeError: If the new name is not a valid file name
        """
        return self._base_filename

    @base_filename.setter
    def base_filename(self, filename: str) -> None:
        """Set a new base filename used to save the results.

        :param filename: The new base filename
        :type filename: :py:class:`~str`
        :raises TypeError: If *filename* is not a valid file name
        """
        self._base_filename = check_filename(
            filename,
            name="base filename to save the results"
        )

    @property
    def backup_filename(self) -> str:
        """Get tha backup filename used to save the results.

        :type: :py:class:`str`
        """
        return self.base_filename + BACKUP_EXTENSION

    @property
    def excel_filename(self) -> str:
        """Get tha filename used to save the results in Excel format.

        :type: :py:class:`str`
        """
        return self.base_filename + EXCEL_EXTENSION

    def save(self, filename: Optional[str] = None) -> None:
        """Save these results.

        :param filename: File path (must have the ".gz" extension). If setto
            :py:data:`None`, :py:attr:`~culebra.tools.Results.backup_filename`
            is used. Defaults to :py:data:`None`
        :type filename: :py:class:`str`, optional.
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If *filename* has an invalid extension
        """
        filename = check_filename(
            (
                self.backup_filename
                if filename is None
                else filename
            ),
            name="results backup file name",
            ext=BACKUP_EXTENSION
        )
        to_pickle(self, filename)

    def to_excel(self, filename: Optional[str] = None) -> None:
        """Save the results to a Excel file.

        :param filename: File path (must have the ".xlsx" extension). If set to
            :py:data:`None`, :py:attr:`~culebra.tools.Results.excel_filename`
            is used. Defaults to :py:data:`None`
        :type filename: :py:class:`str`, optional.
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If *filename* has an invalid extension
        """
        filename = check_filename(
            (
                self.excel_filename
                if filename is None
                else filename
            ),
            name="results excel datasheet file name",
            ext=EXCEL_EXTENSION
        )

        with ExcelWriter(filename) as writer: \
                # pylint: disable=abstract-class-instantiated
            for key, data in self.items():
                data.to_excel(writer, sheet_name=key)

    def __setitem__(self, key: str, data: DataFrame) -> DataFrame:
        """Overridden to verify the key and value.

        Assure that the key is a :py:class:`str` and the value is a
        :py:class:`~pandas.DataFrame`.

        :param key: Key for the results
        :type key: :py:class:`str`
        :param data: The data
        :type data: :py:class:`~pandas.DataFrame`
        :return: The data inserted
        :rtype: :py:class:`~pandas.DataFrame`
        """
        return super().__setitem__(
            check_instance(key, "key for the results", str),
            check_instance(data, "data", DataFrame)
        )


# Exported symbols for this module
__all__ = [
    'Results'
]
