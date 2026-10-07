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

:class:`culebra.solution.parameter_optimization.Individual` class
=================================================================

.. autoclass:: culebra.solution.parameter_optimization.Individual

Class attributes
----------------
.. autoattribute:: culebra.solution.parameter_optimization.Individual.eta
.. autoattribute:: culebra.solution.parameter_optimization.Individual.species_cls

Properties
----------
.. autoproperty:: culebra.solution.parameter_optimization.Individual.fitness
.. autoproperty:: culebra.solution.parameter_optimization.Individual.named_values_cls
.. autoproperty:: culebra.solution.parameter_optimization.Individual.species
.. autoproperty:: culebra.solution.parameter_optimization.Individual.values

Static methods
--------------
.. automethod:: culebra.solution.parameter_optimization.Individual.load

Methods
-------
.. automethod:: culebra.solution.parameter_optimization.Individual.crossover
.. automethod:: culebra.solution.parameter_optimization.Individual.dump
.. automethod:: culebra.solution.parameter_optimization.Individual.get
.. automethod:: culebra.solution.parameter_optimization.Individual.mutate

Private methods
---------------
.. automethod:: culebra.solution.parameter_optimization.Individual._get_repr_properties
.. automethod:: culebra.solution.parameter_optimization.Individual._setup

Dunder methods
--------------
Intended to compare (lexicographically) two individuals according to their
fitness.

.. automethod:: culebra.solution.parameter_optimization.Individual.__eq__
.. automethod:: culebra.solution.parameter_optimization.Individual.__ge__
.. automethod:: culebra.solution.parameter_optimization.Individual.__gt__
.. automethod:: culebra.solution.parameter_optimization.Individual.__hash__
.. automethod:: culebra.solution.parameter_optimization.Individual.__le__
.. automethod:: culebra.solution.parameter_optimization.Individual.__lt__
.. automethod:: culebra.solution.parameter_optimization.Individual.__ne__
.. automethod:: culebra.solution.parameter_optimization.Individual.__str__
