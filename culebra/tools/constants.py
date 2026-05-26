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

"""Constants of the module."""

from scipy.stats import shapiro, bartlett

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


DEFAULT_SEP = '\\s+'
"""Default column separator used within dataset files."""

DEFAULT_OUTLIER_PROPORTION = 0.05
"""Expected outlier proportion por class."""

DEFAULT_SMOTE_NUM_NEIGHBORS = 5
"""Default number of neighbors for :class:`~imblearn.over_sampling.SMOTE`"""

DEFAULT_EXCEL_FILE_EXTENSION = ".xlsx"
"""File extension for Excel datasheets."""

DEFAULT_SCRIPT_FILE_EXTENSION = ".py"
"""Default file extension for python scripts."""

DEFAULT_RESULTS_BASE_FILENAME = "results"
"""Default base name for results files."""

DEFAULT_NUM_EXPERIMENTS = 1
"""Default number of experiments in the batch."""

DEFAULT_RUN_SCRIPT_BASENAME = "run"
"""Default base name for the script to run an evaluation."""

DEFAULT_RUN_SCRIPT_FILENAME = (
    DEFAULT_RUN_SCRIPT_BASENAME + DEFAULT_SCRIPT_FILE_EXTENSION
)
"""Default file name for the script to run an evaluation."""

DEFAULT_CONFIG_SCRIPT_BASENAME = "config"
"""Default base name for configuration scripts."""

DEFAULT_CONFIG_SCRIPT_FILENAME = (
    DEFAULT_CONFIG_SCRIPT_BASENAME + DEFAULT_SCRIPT_FILE_EXTENSION
)
"""Default file name for configuration scripts."""

DEFAULT_ALPHA = 0.05
"""Default significance level for statistical tests."""

DEFAULT_NORMALITY_TEST = shapiro
"""Default normality test."""

DEFAULT_HOMOSCEDASTICITY_TEST = bartlett
"""Default homoscedasticity test."""

DEFAULT_P_ADJUST = 'fdr_tsbky'
"""Default method for adjusting the p-values with the Dunn's test."""


__all__ = [
    'DEFAULT_SEP',
    'DEFAULT_SCRIPT_FILE_EXTENSION',
    'DEFAULT_OUTLIER_PROPORTION',
    'DEFAULT_SMOTE_NUM_NEIGHBORS',
    'DEFAULT_NUM_EXPERIMENTS',
    'DEFAULT_RUN_SCRIPT_FILENAME',
    'DEFAULT_CONFIG_SCRIPT_FILENAME',
    'DEFAULT_RESULTS_BASE_FILENAME',    
    'DEFAULT_ALPHA',
    'DEFAULT_NORMALITY_TEST',
    'DEFAULT_HOMOSCEDASTICITY_TEST',
    'DEFAULT_P_ADJUST',
    'DEFAULT_EXCEL_FILE_EXTENSION'
]
