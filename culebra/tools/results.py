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

"""Results management."""

from __future__ import annotations

from typing import Optional, Sequence
from collections import UserDict
from os.path import basename, splitext

from pandas import DataFrame, read_csv, ExcelWriter

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
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


EXCEL_FILE_EXTENSION = ".xlsx"
"""File extension for Excel datasheets."""


class Results(UserDict, Base):
    """Manages the results produced by the evaluation of a trainer."""

    @classmethod
    def from_csv_files(
        cls,
        files: Sequence[str],
        keys: Sequence[str] = None,
        sep: Optional[str] = None
    ) -> None:
        """Load some results from several csv files.

        :param files: Sequence of files containing the results
        :type files: ~collections.abc.Sequence[str]
        :param keys: Keys for the different results. One key for each csv file.
            If omitted, the basename of each file in *files* (without
            extension) is used. Defaults to :data:`None`
        :type keys: ~collections.abc.Sequence[str]
        :param sep: Separator. If :data:`None` is provided,
            :attr:`~culebra.tools.DEFAULT_SEP` is used. Defaults to
            :data:`None`
        :type sep: str
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

    def to_excel(self, filename: str) -> None:
        """Save the results to a Excel file.

        :param filename: File path
        :type filename: str
        :raises TypeError: If *filename* is not a valid file name
        :raises ValueError: If the *filename* extension is not
            :attr:`~culebra.tools.EXCEL_FILE_EXTENSION`
        """
        filename = check_filename(
            filename,
            name="results excel datasheet file name",
            ext=EXCEL_FILE_EXTENSION
        )

        with ExcelWriter(filename) as writer: \
                # pylint: disable=abstract-class-instantiated
            for key, data in self.items():
                data.to_excel(writer, sheet_name=key)

    def __setitem__(self, key: str, data: DataFrame) -> DataFrame:
        """Overridden to verify the key and value.

        Assure that the key is a :class:`str` and the value is a
        :class:`~pandas.DataFrame`.

        :param key: Key for the results
        :type key: str
        :param data: The data
        :type data: ~pandas.DataFrame
        :return: The data inserted
        :rtype: ~pandas.DataFrame
        """
        return super().__setitem__(
            check_instance(key, "key for the results", str),
            check_instance(data, "data", DataFrame)
        )


# Exported symbols for this module
__all__ = [
    'Results',
    'EXCEL_FILE_EXTENSION'
]
