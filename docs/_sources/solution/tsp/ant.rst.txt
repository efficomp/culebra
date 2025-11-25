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

:class:`culebra.solution.tsp.Ant` class
=======================================

.. autoclass:: culebra.solution.tsp.Ant

Class attributes
----------------
.. autoattribute:: culebra.solution.tsp.Ant.species_cls

Class methods
-------------
.. automethod:: culebra.solution.tsp.Ant.load

Properties
----------
.. autoproperty:: culebra.solution.tsp.Ant.current
.. autoproperty:: culebra.solution.tsp.Ant.discarded
.. autoproperty:: culebra.solution.tsp.Ant.fitness
.. autoproperty:: culebra.solution.tsp.Ant.path
.. autoproperty:: culebra.solution.tsp.Ant.species

Methods
-------
.. automethod:: culebra.solution.tsp.Ant.append
.. automethod:: culebra.solution.tsp.Ant.delete_fitness
.. automethod:: culebra.solution.tsp.Ant.discard
.. automethod:: culebra.solution.tsp.Ant.dominates
.. automethod:: culebra.solution.tsp.Ant.dump

Private methods
---------------
.. automethod:: culebra.solution.tsp.Ant._setup

Dunder methods
--------------
Intended to compare (lexicographically) two ants according to their fitness.

.. automethod:: culebra.solution.tsp.Ant.__eq__
.. automethod:: culebra.solution.tsp.Ant.__ge__
.. automethod:: culebra.solution.tsp.Ant.__gt__
.. automethod:: culebra.solution.tsp.Ant.__hash__
.. automethod:: culebra.solution.tsp.Ant.__le__
.. automethod:: culebra.solution.tsp.Ant.__lt__
.. automethod:: culebra.solution.tsp.Ant.__ne__
.. automethod:: culebra.solution.tsp.Ant.__str__
