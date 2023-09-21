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

:py:class:`culebra.fitness_function.svc_optimization.KappaC` class
==================================================================

.. autoclass:: culebra.fitness_function.svc_optimization.KappaC

Class attributes
----------------
.. class:: culebra.fitness_function.svc_optimization.KappaC.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.svc_optimization.KappaC.evaluate`
    method within a
    :py:class:`~culebra.solution.parameter_optimization.Solution`.


    .. autoattribute:: culebra.fitness_function.svc_optimization.KappaC.Fitness.weights
    .. autoattribute:: culebra.fitness_function.svc_optimization.KappaC.Fitness.names
    .. autoattribute:: culebra.fitness_function.svc_optimization.KappaC.Fitness.thresholds


Class methods
-------------
.. automethod:: culebra.fitness_function.svc_optimization.KappaC.set_fitness_thresholds

Properties
----------
.. autoproperty:: culebra.fitness_function.svc_optimization.KappaC.num_obj
.. autoproperty:: culebra.fitness_function.svc_optimization.KappaC.training_data
.. autoproperty:: culebra.fitness_function.svc_optimization.KappaC.test_data
.. autoproperty:: culebra.fitness_function.svc_optimization.KappaC.test_prop
.. autoproperty:: culebra.fitness_function.svc_optimization.KappaC.classifier

Methods
-------
.. automethod:: culebra.fitness_function.svc_optimization.KappaC.distances_matrix
.. automethod:: culebra.fitness_function.svc_optimization.KappaC.evaluate
