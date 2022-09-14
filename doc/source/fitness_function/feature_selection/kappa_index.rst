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

:py:class:`fitness_function.feature_selection.KappaIndex` class
===============================================================

.. autoclass:: fitness_function.feature_selection.KappaIndex

Class attributes
----------------
.. class:: fitness_function.feature_selection.KappaIndex.Fitness

    Fitness class to handle the values returned by the
    :py:meth:`~fitness_function.feature_selection.KappaIndex.evaluate` method
    within an :py:class:`~genotype.feature_selection.Individual`.

    .. autoattribute:: fitness_function.feature_selection.KappaIndex.Fitness.weights
    .. autoattribute:: fitness_function.feature_selection.KappaIndex.Fitness.names
    .. autoattribute:: fitness_function.feature_selection.KappaIndex.Fitness.thresholds


Properties
----------
.. autoproperty:: fitness_function.feature_selection.KappaIndex.num_obj
.. autoproperty:: fitness_function.feature_selection.KappaIndex.training_data
.. autoproperty:: fitness_function.feature_selection.KappaIndex.test_data
.. autoproperty:: fitness_function.feature_selection.KappaIndex.test_prop
.. autoproperty:: fitness_function.feature_selection.KappaIndex.classifier

Methods
-------
.. automethod:: fitness_function.feature_selection.KappaIndex.evaluate
