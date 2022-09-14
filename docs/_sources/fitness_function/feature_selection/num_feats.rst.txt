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

:py:class:`fitness_function.feature_selection.NumFeats` class
=============================================================

.. autoclass:: fitness_function.feature_selection.NumFeats

Class attributes
----------------
.. class:: fitness_function.feature_selection.NumFeats.Fitness

    Fitness class to handle the values returned by the
    :py:meth:`~fitness_function.feature_selection.NumFeats.evaluate` method
    within an :py:class:`~genotype.feature_selection.Individual`.

    .. autoattribute:: fitness_function.feature_selection.NumFeats.Fitness.weights
    .. autoattribute:: fitness_function.feature_selection.NumFeats.Fitness.names
    .. autoattribute:: fitness_function.feature_selection.NumFeats.Fitness.thresholds


Properties
----------
.. autoproperty:: fitness_function.feature_selection.NumFeats.num_obj
.. autoproperty:: fitness_function.feature_selection.NumFeats.training_data
.. autoproperty:: fitness_function.feature_selection.NumFeats.test_data
.. autoproperty:: fitness_function.feature_selection.NumFeats.test_prop
.. autoproperty:: fitness_function.feature_selection.NumFeats.classifier

Methods
-------
.. automethod:: fitness_function.feature_selection.NumFeats.evaluate
