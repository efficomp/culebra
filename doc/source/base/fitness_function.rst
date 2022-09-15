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

:py:class:`base.FitnessFunction` class
======================================

.. autoclass:: base.FitnessFunction

Class attributes
----------------
.. class:: base.FitnessFunction.Fitness

    Fitness class to handle the values returned by the
    :py:meth:`~base.FitnessFunction.evaluate` method within an
    :py:class:`~base.Individual`.

    This class must be implemented within all the
    :py:class:`~base.FitnessFunction` subclasses, as a subclass of the
    :py:class:`~base.Fitness` class, to define its three class attributes
    (:py:attr:`~base.Fitness.weights`, :py:attr:`~base.Fitness.names`, and
    :py:attr:`~base.Fitness.thresholds`) according to the fitness function.

Class methods
-------------
.. automethod:: base.FitnessFunction.set_fitness_thresholds

Properties
----------
.. autoproperty:: base.FitnessFunction.num_obj
.. autoproperty:: base.FitnessFunction.training_data
.. autoproperty:: base.FitnessFunction.test_data
.. autoproperty:: base.FitnessFunction.test_prop
.. autoproperty:: base.FitnessFunction.classifier

Methods
-------
.. automethod:: base.FitnessFunction.evaluate
