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

:py:class:`fitness_function.cooperative.KappaNumFeatsC` class
=============================================================

.. autoclass:: fitness_function.cooperative.KappaNumFeatsC

More information about this fitness function can be found in [Gonzalez2021]_.

.. [Gonzalez2021] J. González, J. Ortega, J. J. Escobar, M. Damas.
   *A lexicographic cooperative co-evolutionary approach for feature
   selection*. **Neurocomputing**, 463:59-76, 2021.
   https://doi.org/10.1016/j.neucom.2021.08.003.


Class attributes
----------------
.. class:: fitness_function.cooperative.KappaNumFeatsC.Fitness

    Fitness class to handle the values returned by the
    :py:meth:`~fitness_function.cooperative.KappaNumFeatsC.evaluate`
    method within an :py:class:`~base.Individual`.

    .. autoattribute:: fitness_function.cooperative.KappaNumFeatsC.Fitness.weights
    .. autoattribute:: fitness_function.cooperative.KappaNumFeatsC.Fitness.names
    .. autoattribute:: fitness_function.cooperative.KappaNumFeatsC.Fitness.thresholds


Properties
----------
.. autoproperty:: fitness_function.cooperative.KappaNumFeatsC.num_obj
.. autoproperty:: fitness_function.cooperative.KappaNumFeatsC.training_data
.. autoproperty:: fitness_function.cooperative.KappaNumFeatsC.test_data
.. autoproperty:: fitness_function.cooperative.KappaNumFeatsC.test_prop
.. autoproperty:: fitness_function.cooperative.KappaNumFeatsC.classifier

Methods
-------
.. automethod:: fitness_function.cooperative.KappaNumFeatsC.evaluate
