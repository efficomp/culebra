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

:py:class:`culebra.fitness_function.abc.CooperativeFSFitnessFunction` class
===========================================================================

.. autoclass:: culebra.fitness_function.abc.CooperativeFSFitnessFunction

Class attributes
----------------
.. class:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.abc.CooperativeFSFitnessFunction.evaluate`
    method within a :py:class:`~culebra.abc.Solution`.

    This class must be implemented within all the
    :py:class:`~culebra.fitness_function.abc.CooperativeFSFitnessFunction`
    subclasses, as a subclass of the :py:class:`~culebra.abc.Fitness` class,
    to define its three class attributes (
    :py:attr:`~culebra.abc.Fitness.weights`,
    :py:attr:`~culebra.abc.Fitness.names`, and
    :py:attr:`~culebra.abc.Fitness.thresholds`) according to the fitness
    function.

Class methods
-------------
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.load_pickle
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.set_fitness_thresholds
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.get_fitness_objective_threshold
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.set_fitness_objective_threshold

Properties
----------
.. autoproperty:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.num_obj
.. autoproperty:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.num_nodes

Methods
-------
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.save_pickle
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.heuristic
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.evaluate
.. automethod:: culebra.fitness_function.abc.CooperativeFSFitnessFunction.construct_solutions
