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

:py:class:`culebra.solution.tsp.Solution` class
===============================================

.. autoclass:: culebra.solution.tsp.Solution

Class attributes
----------------
.. autoattribute:: culebra.solution.tsp.Solution.species_cls


Properties
----------
.. autoproperty:: culebra.solution.tsp.Solution.species
.. autoproperty:: culebra.solution.tsp.Solution.fitness
.. autoproperty:: culebra.solution.tsp.Solution.path


Methods
-------
.. automethod:: culebra.solution.tsp.Solution.dominates
.. automethod:: culebra.solution.tsp.Solution.delete_fitness

Private methods
---------------
.. automethod:: culebra.solution.tsp.Solution._setup

Dunder methods
--------------
.. automethod:: culebra.solution.tsp.Solution.__hash__
.. automethod:: culebra.solution.tsp.Solution.__eq__
.. automethod:: culebra.solution.tsp.Solution.__ne__
.. automethod:: culebra.solution.tsp.Solution.__lt__
.. automethod:: culebra.solution.tsp.Solution.__gt__
.. automethod:: culebra.solution.tsp.Solution.__le__
.. automethod:: culebra.solution.tsp.Solution.__ge__
.. automethod:: culebra.solution.tsp.Solution.__str__
