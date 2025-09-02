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

:py:class:`culebra.abc.Solution` class
======================================

.. autoclass:: culebra.abc.Solution

Class attributes
----------------
.. autoattribute:: culebra.abc.Solution.species_cls

Class methods
-------------
.. automethod:: culebra.abc.Solution.load

Properties
----------
.. autoproperty:: culebra.abc.Solution.species
.. autoproperty:: culebra.abc.Solution.fitness

Methods
-------
.. automethod:: culebra.abc.Solution.dump
.. automethod:: culebra.abc.Solution.dominates
.. automethod:: culebra.abc.Solution.delete_fitness

Dunder methods
--------------
Intended to compare (lexicographically) two solutions according to their
fitness.

.. automethod:: culebra.abc.Solution.__hash__
.. automethod:: culebra.abc.Solution.__eq__
.. automethod:: culebra.abc.Solution.__ne__
.. automethod:: culebra.abc.Solution.__lt__
.. automethod:: culebra.abc.Solution.__gt__
.. automethod:: culebra.abc.Solution.__le__
.. automethod:: culebra.abc.Solution.__ge__
