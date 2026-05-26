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

:class:`culebra.tools.abc.Evaluation` class
===========================================

.. autoclass:: culebra.tools.abc.Evaluation

Class attributes
----------------
.. autoattribute:: culebra.tools.abc.Evaluation.feature_metric_funcs
.. autoattribute:: culebra.tools.abc.Evaluation.stats_funcs
.. autoattribute:: culebra.tools.abc.Evaluation.ResultsLabels

Class methods
-------------
.. automethod:: culebra.tools.abc.Evaluation.from_config
.. automethod:: culebra.tools.abc.Evaluation.generate_run_script
.. automethod:: culebra.tools.abc.Evaluation.load

Properties
----------
.. autoproperty:: culebra.tools.abc.Evaluation.decision_manager
.. autoproperty:: culebra.tools.abc.Evaluation.excel_results_filename
.. autoproperty:: culebra.tools.abc.Evaluation.hyperparameters
.. autoproperty:: culebra.tools.abc.Evaluation.results
.. autoproperty:: culebra.tools.abc.Evaluation.results_base_filename
.. autoproperty:: culebra.tools.abc.Evaluation.serialized_results_filename
.. autoproperty:: culebra.tools.abc.Evaluation.test_fitness_func
.. autoproperty:: culebra.tools.abc.Evaluation.trainer

Private properties
------------------
.. autoproperty:: culebra.tools.abc.Evaluation._default_results_base_filename
.. autoproperty:: culebra.tools.abc.Evaluation._default_test_fitness_func

Methods
-------
.. automethod:: culebra.tools.abc.Evaluation.dump
.. automethod:: culebra.tools.abc.Evaluation.reset
.. automethod:: culebra.tools.abc.Evaluation.run

Private methods
---------------
.. automethod:: culebra.tools.abc.Evaluation._execute
.. automethod:: culebra.tools.abc.Evaluation._get_repr_properties
