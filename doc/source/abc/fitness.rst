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

:py:class:`culebra.abc.Fitness` class
=====================================

.. autoclass:: culebra.abc.Fitness

Class attributes
----------------
.. autoattribute:: culebra.abc.Fitness.weights
.. autoattribute:: culebra.abc.Fitness.names
.. autoattribute:: culebra.abc.Fitness.thresholds

Class methods
-------------
.. automethod:: culebra.abc.Fitness.load_pickle
.. automethod:: culebra.abc.Fitness.get_objective_threshold
.. automethod:: culebra.abc.Fitness.set_objective_threshold

Properties
----------
.. autoproperty:: culebra.abc.Fitness.num_obj
.. autoproperty:: culebra.abc.Fitness.pheromone_amount

Methods
-------
.. automethod:: culebra.abc.Fitness.save_pickle
.. automethod:: culebra.abc.Fitness.dominates

Dunder methods
--------------
Intended to compare (lexicographically) two individuals according to their
fitness.

.. automethod:: culebra.abc.Fitness.__hash__
.. automethod:: culebra.abc.Fitness.__eq__
.. automethod:: culebra.abc.Fitness.__ne__
.. automethod:: culebra.abc.Fitness.__lt__
.. automethod:: culebra.abc.Fitness.__gt__
.. automethod:: culebra.abc.Fitness.__le__
.. automethod:: culebra.abc.Fitness.__ge__
