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
from culebra.base import (
    Base,
    DEFAULT_SEP,
    check_instance,
    check_str,
    check_sequence,
    check_filename
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'

DEFAULT_RESULTS_BACKUP_FILENAME = "results.gz"
"""Default file name for results backups."""

DEFAULT_RESULTS_EXCEL_FILENAME = "results.xlsx"
"""Default name of the datasheet to store the results."""


class Results(UserDict, Base):
    """Manages the results produced by the evaluation of a wrapper."""

    def __init__(self) -> None:
        """Create an empty results manager."""
        return super().__init__()

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
            :py:attr:`~base.DEFAULT_SEP` is used. Defaults to :py:data:`None`
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


# Exported symbols for this module
__all__ = [
    'DEFAULT_RESULTS_BACKUP_FILENAME',
    'DEFAULT_RESULTS_EXCEL_FILENAME',
    'Results'
]
