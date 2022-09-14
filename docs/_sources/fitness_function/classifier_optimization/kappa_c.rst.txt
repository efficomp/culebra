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

:py:class:`fitness_function.classifier_optimization.KappaC` class
=================================================================

.. autoclass:: fitness_function.classifier_optimization.KappaC

Class attributes
----------------
.. class:: fitness_function.classifier_optimization.KappaC.Fitness

    Fitness class to handle the values returned by the
    :py:meth:`~fitness_function.classifier_optimization.KappaC.evaluate`
    method within an :py:class:`~genotype.classifier_optimization.Individual`.

    .. autoattribute:: fitness_function.classifier_optimization.KappaC.Fitness.weights
    .. autoattribute:: fitness_function.classifier_optimization.KappaC.Fitness.names
    .. autoattribute:: fitness_function.classifier_optimization.KappaC.Fitness.thresholds


Properties
----------
.. autoproperty:: fitness_function.classifier_optimization.KappaC.num_obj
.. autoproperty:: fitness_function.classifier_optimization.KappaC.training_data
.. autoproperty:: fitness_function.classifier_optimization.KappaC.test_data
.. autoproperty:: fitness_function.classifier_optimization.KappaC.test_prop
.. autoproperty:: fitness_function.classifier_optimization.KappaC.classifier

Methods
-------
.. automethod:: fitness_function.classifier_optimization.KappaC.evaluate
