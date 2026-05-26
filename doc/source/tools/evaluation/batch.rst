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
   Culebra. If not, see <http://www.gnu.org/licenses/>.

   This work is supported by projects PGC2018-098813-B-C31 and
   PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
   Innovación y Universidades" and by the European Regional Development Fund
   (ERDF).

:class:`culebra.tools.evaluation.Batch` class
=============================================

.. autoclass:: culebra.tools.evaluation.Batch

Class attributes
----------------
.. autoattribute:: culebra.tools.evaluation.Batch.feature_metric_funcs
.. autoattribute:: culebra.tools.evaluation.Batch.stats_funcs
.. autoattribute:: culebra.tools.evaluation.Batch.ResultsKeys
.. autoattribute:: culebra.tools.evaluation.Batch.ResultsLabels

Class methods
-------------
.. automethod:: culebra.tools.evaluation.Batch.from_config
.. automethod:: culebra.tools.evaluation.Batch.generate_run_script
.. automethod:: culebra.tools.evaluation.Batch.load

Properties
----------
.. autoproperty:: culebra.tools.evaluation.Batch.decision_manager
.. autoproperty:: culebra.tools.evaluation.Batch.excel_results_filename
.. autoproperty:: culebra.tools.evaluation.Batch.experiment_basename
.. autoproperty:: culebra.tools.evaluation.Batch.experiment_labels
.. autoproperty:: culebra.tools.evaluation.Batch.hyperparameters
.. autoproperty:: culebra.tools.evaluation.Batch.num_experiments
.. autoproperty:: culebra.tools.evaluation.Batch.results
.. autoproperty:: culebra.tools.evaluation.Batch.results_base_filename
.. autoproperty:: culebra.tools.evaluation.Batch.serialized_results_filename
.. autoproperty:: culebra.tools.evaluation.Batch.test_fitness_func
.. autoproperty:: culebra.tools.evaluation.Batch.trainer

Private properties
------------------
.. autoproperty:: culebra.tools.evaluation.Batch._default_num_experiments
.. autoproperty:: culebra.tools.evaluation.Batch._default_results_base_filename
.. autoproperty:: culebra.tools.evaluation.Batch._default_test_fitness_func

Methods
-------
.. automethod:: culebra.tools.evaluation.Batch.dump
.. automethod:: culebra.tools.evaluation.Batch.reset
.. automethod:: culebra.tools.evaluation.Batch.run
.. automethod:: culebra.tools.evaluation.Batch.setup

Private methods
---------------
.. automethod:: culebra.tools.evaluation.Batch._add_execution_metrics_stats
.. automethod:: culebra.tools.evaluation.Batch._add_feature_metrics_stats
.. automethod:: culebra.tools.evaluation.Batch._add_fitness_stats
.. automethod:: culebra.tools.evaluation.Batch._append_data
.. automethod:: culebra.tools.evaluation.Batch._execute
.. automethod:: culebra.tools.evaluation.Batch._get_repr_properties
