..
   This file is part of

   Culebra is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
    If not, see <http://www.gnu.org/licenses/>.

   This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
   de Ciencia, Innovación y Universidades"), and by the European Regional
   Development Fund (ERDF).

:py:mod:`culebbra.tools` module
===============================

.. automodule:: culebra.tools

Attributes
----------
.. attribute:: DEFAULT_SEP
    :annotation: = '\\s+'

    Default column separator used within dataset files.

.. attribute:: DEFAULT_ALPHA
    :annotation: = 0.05

    Default significance level for statistical tests.

.. attribute:: DEFAULT_NORMALITY_TEST
    :annotation: = <function shapiro>

    Default normality test.

.. attribute:: DEFAULT_HOMOSCEDASTICITY_TEST
    :annotation: = <function bartlett>

    Default homoscedasticity test.

.. attribute:: DEFAULT_P_ADJUST
    :annotation: = 'fdr_tsbky'

    Default method for adjusting the p-values with the Dunn's test.

.. attribute:: DEFAULT_STATS_FUNCTIONS
    :annotation: = {'Avg': <function mean>, 'Max': <function amax>, 'Min': <function amin>, 'Std': <function std>}

    Default statistics calculated for the results.

.. attribute:: DEFAULT_FEATURE_METRIC_FUNCTIONS
    :annotation: = {'Rank': <function Metrics.rank>, 'Relevance': <function Metrics.relevance>}

    Default metrics calculated for the features in the set of solutions.

.. attribute:: DEFAULT_BATCH_STATS_FUNCTIONS
    :annotation: = {'Avg': <function NDFrame._add_numeric_operations.<locals>.mean>, 'Max': <function NDFrame._add_numeric_operations.<locals>.max>, 'Min': <function NDFrame._add_numeric_operations.<locals>.min>, 'Std': <function NDFrame._add_numeric_operations.<locals>.std>}

    Default statistics calculated for the results gathered from all the
    experiments.

.. attribute:: DEFAULT_NUM_EXPERIMENTS
    :annotation: = 1

    Default number of experiments in the batch.

.. attribute:: DEFAULT_SCRIPT_FILENAME
    :annotation: = 'run.py'

    Default file name for the script to run an evaluation.

.. attribute:: DEFAULT_CONFIG_FILENAME
    :annotation: = 'config.py'

    Default name for the configuration file for the evaluation.


..
    .. autodata:: DEFAULT_SEP
    .. autodata:: DEFAULT_ALPHA
    .. autodata:: DEFAULT_NORMALITY_TEST
    .. autodata:: DEFAULT_HOMOSCEDASTICITY_TEST
    .. autodata:: DEFAULT_P_ADJUST
    .. autodata:: DEFAULT_STATS_FUNCTIONS
    .. autodata:: DEFAULT_FEATURE_METRIC_FUNCTIONS
    .. autodata:: DEFAULT_BATCH_STATS_FUNCTIONS
    .. autodata:: DEFAULT_NUM_EXPERIMENTS
    .. autodata:: DEFAULT_SCRIPT_FILENAME
    .. autodata:: DEFAULT_CONFIG_FILENAME


.. toctree::
    :hidden:

    Dataset <tools/dataset>
    Results <tools/results>
    TestOutcome <tools/test_outcome>
    ResultsComparison <tools/results_comparison>
    ResultsAnalyzer <tools/results_analyzer>
    Evaluation <tools/evaluation>
    Experiment <tools/experiment>
    Batch <tools/batch>
