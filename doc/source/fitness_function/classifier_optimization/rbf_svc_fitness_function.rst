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

:py:class:`fitness_function.classifier_optimization.RBFSVCFitnessFunction` class
================================================================================

.. autoclass:: fitness_function.classifier_optimization.RBFSVCFitnessFunction

Class attributes
----------------
.. class:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.Fitness

    Fitness class to handle the values returned by the
    :py:meth:`~fitness_function.classifier_optimization.RBFSVCFitnessFunction.evaluate` method
    within an :py:class:`~genotype.classifier_optimization.Individual`.

    This class must be implemented within all the
    :py:class:`~fitness_function.classifier_optimization.RBFSVCFitnessFunction`
    subclasses, as a subclass of the
    :py:class:`~base.Fitness` class, to define its three class attributes
    (:py:attr:`~base.Fitness.weights`, :py:attr:`~base.Fitness.names`, and
    :py:attr:`~base.Fitness.thresholds`) according to the fitness function.


Properties
----------
.. autoproperty:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.num_obj
.. autoproperty:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.training_data
.. autoproperty:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.test_data
.. autoproperty:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.test_prop
.. autoproperty:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.classifier

Methods
-------
.. automethod:: fitness_function.classifier_optimization.RBFSVCFitnessFunction.evaluate
