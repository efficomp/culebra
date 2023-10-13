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

:py:class:`culebra.fitness_function.abc.FeatureSelectionFitnessFunction` class
==============================================================================

.. autoclass:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction

Class attributes
----------------
.. class:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.abc.FeatureSelectionFitnessFunction.evaluate`
    method within a :py:class:`~culebra.abc.Solution`.

    This class must be implemented within all the
    :py:class:`~culebra.fitness_function.abc.FeatureSelectionFitnessFunction`
    subclasses, as a subclass of the :py:class:`~culebra.abc.Fitness` class,
    to define its three class attributes (
    :py:attr:`~culebra.abc.Fitness.weights`,
    :py:attr:`~culebra.abc.Fitness.names`, and
    :py:attr:`~culebra.abc.Fitness.thresholds`) according to the fitness
    function.

Class methods
-------------
.. automethod:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.set_fitness_thresholds

Properties
----------
.. autoproperty:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.num_obj
.. autoproperty:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.num_nodes
.. autoproperty:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.training_data
.. autoproperty:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.test_data
.. autoproperty:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.test_prop
.. autoproperty:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.classifier

Methods
-------
.. automethod:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.heuristics
.. automethod:: culebra.fitness_function.abc.FeatureSelectionFitnessFunction.evaluate
