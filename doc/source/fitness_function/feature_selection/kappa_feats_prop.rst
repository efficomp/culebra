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

:py:class:`culebra.fitness_function.feature_selection.KappaFeatsProp` class
===========================================================================

.. autoclass:: culebra.fitness_function.feature_selection.KappaFeatsProp

Class attributes
----------------
.. class:: culebra.fitness_function.feature_selection.KappaFeatsProp.Fitness

    Handles the values returned by the
    :py:meth:`~culebra.fitness_function.feature_selection.KappaFeatsProp.evaluate`
    method within a :py:class:`~culebra.solution.feature_selection.Solution`.

    .. autoattribute:: culebra.fitness_function.feature_selection.KappaFeatsProp.Fitness.weights
    .. autoattribute:: culebra.fitness_function.feature_selection.KappaFeatsProp.Fitness.names
    .. autoattribute:: culebra.fitness_function.feature_selection.KappaFeatsProp.Fitness.thresholds

Class methods
-------------
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.load_pickle
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.set_fitness_thresholds
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.get_fitness_objective_threshold
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.set_fitness_objective_threshold

Properties
----------
.. autoproperty:: culebra.fitness_function.feature_selection.KappaFeatsProp.num_obj
.. autoproperty:: culebra.fitness_function.feature_selection.KappaFeatsProp.num_nodes
.. autoproperty:: culebra.fitness_function.feature_selection.KappaFeatsProp.training_data
.. autoproperty:: culebra.fitness_function.feature_selection.KappaFeatsProp.test_data
.. autoproperty:: culebra.fitness_function.feature_selection.KappaFeatsProp.test_prop
.. autoproperty:: culebra.fitness_function.feature_selection.KappaFeatsProp.classifier

Methods
-------
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.save_pickle
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.heuristic
.. automethod:: culebra.fitness_function.feature_selection.KappaFeatsProp.evaluate
