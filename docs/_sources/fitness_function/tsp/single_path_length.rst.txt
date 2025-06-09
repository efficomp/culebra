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

:py:class:`culebra.fitness_function.tsp.SinglePathLength` class
===============================================================

.. autoclass:: culebra.fitness_function.tsp.SinglePathLength

Class attributes
----------------
.. class:: culebra.fitness_function.tsp.SinglePathLength.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.tsp.SinglePathLength.evaluate` method
    within a :py:class:`~culebra.solution.tsp.Solution`.

    .. autoattribute:: culebra.fitness_function.tsp.SinglePathLength.Fitness.weights
    .. autoattribute:: culebra.fitness_function.tsp.SinglePathLength.Fitness.names
    .. autoattribute:: culebra.fitness_function.tsp.SinglePathLength.Fitness.thresholds

Class methods
-------------
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.load_pickle
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.set_fitness_thresholds
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.get_fitness_objective_threshold
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.set_fitness_objective_threshold
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.fromPath
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.fromTSPLib

Properties
----------
.. autoproperty:: culebra.fitness_function.tsp.SinglePathLength.is_noisy
.. autoproperty:: culebra.fitness_function.tsp.SinglePathLength.num_obj
.. autoproperty:: culebra.fitness_function.tsp.SinglePathLength.num_nodes
.. autoproperty:: culebra.fitness_function.tsp.SinglePathLength.distance

Methods
-------
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.save_pickle
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.heuristic
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.greedy_solution
.. automethod:: culebra.fitness_function.tsp.SinglePathLength.evaluate
