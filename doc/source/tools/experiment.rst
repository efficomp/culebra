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

:py:class:`culebra.tools.Experiment` class
==========================================

.. autoclass:: culebra.tools.Experiment

Class attributes
----------------
.. autoattribute:: culebra.tools.Experiment.feature_metric_functions
.. autoattribute:: culebra.tools.Experiment.stats_functions

Properties
----------
.. autoproperty:: culebra.tools.Experiment.trainer
.. autoproperty:: culebra.tools.Experiment.test_fitness_function
.. autoproperty:: culebra.tools.Experiment.results_base_filename
.. autoproperty:: culebra.tools.Experiment.results
.. autoproperty:: culebra.tools.Experiment.best_solutions
.. autoproperty:: tools.Experiment.best_representatives

Class methods
-------------
.. automethod:: tools.Experiment.from_config
.. automethod:: tools.Experiment.generate_script

Methods
-------
.. automethod:: tools.Experiment.reset
.. automethod:: tools.Experiment.run

Private methods
---------------
.. automethod:: tools.Experiment._do_training
.. automethod:: tools.Experiment._add_training_stats
.. automethod:: tools.Experiment._add_fitness
.. automethod:: tools.Experiment._add_fitness_stats
.. automethod:: tools.Experiment._add_execution_metric
.. automethod:: tools.Experiment._add_feature_metrics
.. automethod:: tools.Experiment._do_test
.. automethod:: tools.Experiment._execute
