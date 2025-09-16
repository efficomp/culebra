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

:py:class:`culebra.fitness_function.svc_optimization.Accuracy` class
====================================================================

.. autoclass:: culebra.fitness_function.svc_optimization.Accuracy

Class methods
-------------
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy.load

Properties
----------
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.num_obj
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.obj_weights
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.obj_names
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.obj_thresholds
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.fitness_cls
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.index
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.training_data
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.test_data
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy.classifier

Private properties
------------------
.. autoproperty:: culebra.fitness_function.svc_optimization.Accuracy._worst_score

Methods
-------
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy.dump
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy.evaluate
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy.is_evaluable

Private methods
---------------
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy._score
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy._final_training_test_data
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy._evaluate_train_test
.. automethod:: culebra.fitness_function.svc_optimization.Accuracy._evaluate_kfcv
