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

:class:`culebra.tools.evaluation.Experiment` class
==================================================

.. autoclass:: culebra.tools.evaluation.Experiment

Class attributes
----------------
.. autoattribute:: culebra.tools.evaluation.Experiment.feature_metric_funcs
.. autoattribute:: culebra.tools.evaluation.Experiment.stats_funcs
.. autoattribute:: culebra.tools.evaluation.Experiment.ResultsKeys
.. autoattribute:: culebra.tools.evaluation.Experiment.ResultsLabels

Class methods
-------------
.. automethod:: culebra.tools.evaluation.Experiment.from_config
.. automethod:: culebra.tools.evaluation.Experiment.generate_run_script
.. automethod:: culebra.tools.evaluation.Experiment.load

Properties
----------
.. autoproperty:: culebra.tools.evaluation.Experiment.best_cooperators
.. autoproperty:: culebra.tools.evaluation.Experiment.best_solutions
.. autoproperty:: culebra.tools.evaluation.Experiment.decision_manager
.. autoproperty:: culebra.tools.evaluation.Experiment.excel_results_filename
.. autoproperty:: culebra.tools.evaluation.Experiment.hyperparameters
.. autoproperty:: culebra.tools.evaluation.Experiment.results
.. autoproperty:: culebra.tools.evaluation.Experiment.results_base_filename
.. autoproperty:: culebra.tools.evaluation.Experiment.serialized_results_filename
.. autoproperty:: culebra.tools.evaluation.Experiment.test_fitness_func
.. autoproperty:: culebra.tools.evaluation.Experiment.trainer

Private properties
------------------
.. autoproperty:: culebra.tools.evaluation.Experiment._default_results_base_filename
.. autoproperty:: culebra.tools.evaluation.Experiment._default_test_fitness_func

Methods
-------
.. automethod:: culebra.tools.evaluation.Experiment.dump
.. automethod:: culebra.tools.evaluation.Experiment.reset
.. automethod:: culebra.tools.evaluation.Experiment.run

Private methods
---------------
.. automethod:: culebra.tools.evaluation.Experiment._add_best
.. automethod:: culebra.tools.evaluation.Experiment._add_execution_metric
.. automethod:: culebra.tools.evaluation.Experiment._add_feature_metrics
.. automethod:: culebra.tools.evaluation.Experiment._add_fitness
.. automethod:: culebra.tools.evaluation.Experiment._add_fitness_stats
.. automethod:: culebra.tools.evaluation.Experiment._add_training_stats
.. automethod:: culebra.tools.evaluation.Experiment._do_test
.. automethod:: culebra.tools.evaluation.Experiment._do_training
.. automethod:: culebra.tools.evaluation.Experiment._execute
.. automethod:: culebra.tools.evaluation.Experiment._get_repr_properties
