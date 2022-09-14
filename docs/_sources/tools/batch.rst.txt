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

:py:class:`tools.Batch` class
=============================

.. autoclass:: tools.Batch

Class attributes
----------------
.. autoattribute:: tools.Batch.feature_metric_functions
.. autoattribute:: tools.Batch.stats_functions

Private class attributes
------------------------
.. class:: tools.Batch._ResultKeys

    Handle the keys for the batch results.

.. autoattribute:: tools.Batch._script_code

Properties
----------
.. autoproperty:: tools.Batch.wrapper
.. autoproperty:: tools.Batch.test_fitness_function
.. autoproperty:: tools.Batch.results
.. autoproperty:: tools.Batch.num_experiments
.. autoproperty:: tools.Batch.exp_labels

Class methods
-------------
.. automethod:: tools.Batch.from_config
.. automethod:: tools.Batch.generate_script

Methods
-------
.. automethod:: tools.Batch.reset
.. automethod:: tools.Batch.run

Private methods
---------------
.. automethod:: tools.Batch._append_data
.. automethod:: tools.Batch._add_execution_metrics_stats
.. automethod:: tools.Batch._add_feature_metrics_stats
.. automethod:: tools.Batch._add_fitness_stats
.. automethod:: tools.Batch._execute
