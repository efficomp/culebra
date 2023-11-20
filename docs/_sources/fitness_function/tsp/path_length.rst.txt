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

:py:class:`culebra.fitness_function.tsp.PathLength` class
=========================================================

.. autoclass:: culebra.fitness_function.tsp.PathLength

Class attributes
----------------
.. class:: culebra.fitness_function.tsp.PathLength.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.tsp.PathLength.evaluate` method
    within a :py:class:`~culebra.solution.tsp.Solution`.

    .. autoattribute:: culebra.fitness_function.tsp.PathLength.Fitness.weights
    .. autoattribute:: culebra.fitness_function.tsp.PathLength.Fitness.names
    .. autoattribute:: culebra.fitness_function.tsp.PathLength.Fitness.thresholds

Class methods
-------------
.. automethod:: culebra.fitness_function.tsp.PathLength.fromPath
.. automethod:: culebra.fitness_function.tsp.PathLength.set_fitness_thresholds

Properties
----------
.. autoproperty:: culebra.fitness_function.tsp.PathLength.num_obj
.. autoproperty:: culebra.fitness_function.tsp.PathLength.num_nodes
.. autoproperty:: culebra.fitness_function.tsp.PathLength.distances

Methods
-------
.. automethod:: culebra.fitness_function.tsp.PathLength.heuristic
.. automethod:: culebra.fitness_function.tsp.PathLength.greedy_solution
.. automethod:: culebra.fitness_function.tsp.PathLength.evaluate
