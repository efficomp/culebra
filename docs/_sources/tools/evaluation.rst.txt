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

:class:`culebra.tools.Evaluation` class
=======================================

.. autoclass:: culebra.tools.Evaluation

Class attributes
----------------
.. autoattribute:: culebra.tools.Evaluation.feature_metric_functions
.. autoattribute:: culebra.tools.Evaluation.stats_functions

Class methods
-------------
.. automethod:: culebra.tools.Evaluation.from_config
.. automethod:: culebra.tools.Evaluation.generate_run_script
.. automethod:: culebra.tools.Evaluation.load

Properties
----------
.. autoproperty:: culebra.tools.Evaluation.excel_results_filename
.. autoproperty:: culebra.tools.Evaluation.hyperparameters
.. autoproperty:: culebra.tools.Evaluation.results
.. autoproperty:: culebra.tools.Evaluation.results_base_filename
.. autoproperty:: culebra.tools.Evaluation.serialized_results_filename
.. autoproperty:: culebra.tools.Evaluation.test_fitness_function
.. autoproperty:: culebra.tools.Evaluation.trainer
.. autoproperty:: culebra.tools.Evaluation.untie_best_fitness_function

Methods
-------
.. automethod:: culebra.tools.Evaluation.dump
.. automethod:: culebra.tools.Evaluation.reset
.. automethod:: culebra.tools.Evaluation.run

Private methods
---------------
.. automethod:: culebra.tools.Evaluation._execute
.. automethod:: culebra.tools.Evaluation._is_reserved
