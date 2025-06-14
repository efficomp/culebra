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

:py:class:`culebra.fitness_function.svc_optimization.AccuracyC` class
=====================================================================

.. autoclass:: culebra.fitness_function.svc_optimization.AccuracyC

Class attributes
----------------
.. class:: culebra.fitness_function.svc_optimization.AccuracyC.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.svc_optimization.AccuracyC.evaluate`
    method within a
    :py:class:`~culebra.solution.parameter_optimization.Solution`.

    .. autoattribute:: culebra.fitness_function.svc_optimization.AccuracyC.Fitness.weights
    .. autoattribute:: culebra.fitness_function.svc_optimization.AccuracyC.Fitness.names
    .. autoattribute:: culebra.fitness_function.svc_optimization.AccuracyC.Fitness.thresholds

Class methods
-------------
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.load_pickle
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.set_fitness_thresholds
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.get_fitness_objective_threshold
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.set_fitness_objective_threshold

Properties
----------
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.is_noisy
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.num_obj
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.num_nodes
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.training_data
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.test_data
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.test_prop
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC.classifier

Private properties
------------------
.. autoproperty:: culebra.fitness_function.svc_optimization.AccuracyC._worst_score

Methods
-------
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.save_pickle
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.heuristic
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC.evaluate

Private methods
---------------
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC._score
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC._final_training_test_data
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC._evaluate_train_test
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC._evaluate_mccv
.. automethod:: culebra.fitness_function.svc_optimization.AccuracyC._evaluate_kfcv
