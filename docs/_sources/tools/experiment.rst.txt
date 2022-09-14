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

:py:class:`tools.Experiment` class
==================================

.. autoclass:: tools.Experiment

Class attributes
----------------
.. autoattribute:: tools.Experiment.feature_metric_functions
.. autoattribute:: tools.Experiment.stats_functions

Private class attributes
------------------------
.. class:: tools.Experiment._ResultKeys

    Handle the keys for the experiment results.

.. autoattribute:: tools.Experiment._script_code

Properties
----------
.. autoproperty:: tools.Experiment.wrapper
.. autoproperty:: tools.Experiment.test_fitness_function
.. autoproperty:: tools.Experiment.results
.. autoproperty:: tools.Experiment.best_solutions
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
