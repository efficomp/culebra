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

:class:`culebra.tools.Batch` class
==================================

.. autoclass:: culebra.tools.Batch

Class attributes
----------------
.. autoattribute:: culebra.tools.Batch.feature_metric_functions
.. autoattribute:: culebra.tools.Batch.stats_functions

Class methods
-------------
.. automethod:: culebra.tools.Batch.from_config
.. automethod:: culebra.tools.Batch.generate_run_script
.. automethod:: culebra.tools.Batch.load

Properties
----------
.. autoproperty:: culebra.tools.Batch.excel_results_filename
.. autoproperty:: culebra.tools.Batch.experiment_basename
.. autoproperty:: culebra.tools.Batch.experiment_labels
.. autoproperty:: culebra.tools.Batch.hyperparameters
.. autoproperty:: culebra.tools.Batch.num_experiments
.. autoproperty:: culebra.tools.Batch.results
.. autoproperty:: culebra.tools.Batch.results_base_filename
.. autoproperty:: culebra.tools.Batch.serialized_results_filename
.. autoproperty:: culebra.tools.Batch.test_fitness_function
.. autoproperty:: culebra.tools.Batch.trainer
.. autoproperty:: culebra.tools.Batch.untie_best_fitness_function

Methods
-------
.. automethod:: culebra.tools.Batch.dump
.. automethod:: culebra.tools.Batch.reset
.. automethod:: culebra.tools.Batch.run
.. automethod:: culebra.tools.Batch.setup

Private methods
---------------
.. automethod:: culebra.tools.Batch._add_execution_metrics_stats
.. automethod:: culebra.tools.Batch._add_feature_metrics_stats
.. automethod:: culebra.tools.Batch._add_fitness_stats
.. automethod:: culebra.tools.Batch._append_data
.. automethod:: culebra.tools.Batch._execute
.. automethod:: culebra.tools.Batch._is_reserved
