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

"""Tools to automate the execution of experiments.

Since many interesting problems are based on data processing, this module
provides the :class:`~culebra.tools.Dataset` class to hold and manage the
data samples.

Besides, since automated experimentation is also a quite valuable
characteristic when a :class:`~culebra.abc.Trainer` method has to be run
many times, culebra provides this features by means of the following classes:

* The :class:`~culebra.tools.Batch` class, which allows to run a batch of
  experiments with the same configuration
* The :class:`~culebra.tools.EffectSize` class, to keep the outcome
  of an effect size estimation of several batches results
* The :class:`~culebra.tools.Evaluation` class, a base class for the
  evaluation of trainers
* The :class:`~culebra.tools.Experiment` class, designed to run a single
  experiment with a :class:`~culebra.abc.Trainer`
* The :class:`~culebra.tools.Results` class, to manage the results
  provided by the evaluation of any :class:`~culebra.abc.Trainer`
* The :class:`~culebra.tools.ResultsAnalyzer` class, to perform
  statistical analysis over the results of several experimtent batchs
* The :class:`~culebra.tools.ResultsComparison` class, to keep the outcome
  of a comparison of several batches results
* The :class:`~culebra.tools.TestOutcome` class, to keep the outcome of a
  statistical test
"""

from .dataset import (
    Dataset,
    DEFAULT_SEP,
    DEFAULT_OUTLIER_PROPORTION,
    DEFAULT_SMOTE_NUM_NEIGHBORS
)

from .results import Results, EXCEL_FILE_EXTENSION
from .results_analyzer import (
    TestOutcome,
    ResultsComparison,
    EffectSize,
    ResultsAnalyzer,
    DEFAULT_ALPHA,
    DEFAULT_NORMALITY_TEST,
    DEFAULT_HOMOSCEDASTICITY_TEST,
    DEFAULT_P_ADJUST
)
from .evaluation import (
    Evaluation,
    Experiment,
    Batch,
    DEFAULT_STATS_FUNCTIONS,
    DEFAULT_FEATURE_METRIC_FUNCTIONS,
    DEFAULT_BATCH_STATS_FUNCTIONS,
    DEFAULT_NUM_EXPERIMENTS,
    DEFAULT_RUN_SCRIPT_FILENAME,
    DEFAULT_CONFIG_SCRIPT_FILENAME,
    DEFAULT_RESULTS_BASE_FILENAME
)


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


__all__ = [
    'Dataset',
    'Results',
    'TestOutcome',
    'ResultsComparison',
    'EffectSize',
    'ResultsAnalyzer',
    'Evaluation',
    'Experiment',
    'Batch',
    'DEFAULT_SEP',
    'DEFAULT_OUTLIER_PROPORTION',
    'DEFAULT_SMOTE_NUM_NEIGHBORS',
    'EXCEL_FILE_EXTENSION',
    'DEFAULT_ALPHA',
    'DEFAULT_NORMALITY_TEST',
    'DEFAULT_HOMOSCEDASTICITY_TEST',
    'DEFAULT_P_ADJUST',
    'DEFAULT_STATS_FUNCTIONS',
    'DEFAULT_FEATURE_METRIC_FUNCTIONS',
    'DEFAULT_BATCH_STATS_FUNCTIONS',
    'DEFAULT_NUM_EXPERIMENTS',
    'DEFAULT_RUN_SCRIPT_FILENAME',
    'DEFAULT_CONFIG_SCRIPT_FILENAME',
    'DEFAULT_RESULTS_BASE_FILENAME'
]
