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

:class:`culebra.solution.feature_selection.Solution` class
==========================================================

.. autoclass:: culebra.solution.feature_selection.Solution

Class attributes
----------------
.. autoattribute:: culebra.solution.feature_selection.Solution.species_cls

Class methods
-------------
.. automethod:: culebra.solution.feature_selection.Solution.load

Properties
----------
.. autoproperty:: culebra.solution.feature_selection.Solution.features
.. autoproperty:: culebra.solution.feature_selection.Solution.fitness
.. autoproperty:: culebra.solution.feature_selection.Solution.max_feat
.. autoproperty:: culebra.solution.feature_selection.Solution.min_feat
.. autoproperty:: culebra.solution.feature_selection.Solution.num_feats
.. autoproperty:: culebra.solution.feature_selection.Solution.species

Methods
-------
.. automethod:: culebra.solution.feature_selection.Solution.delete_fitness
.. automethod:: culebra.solution.feature_selection.Solution.dominates
.. automethod:: culebra.solution.feature_selection.Solution.dump

Private methods
---------------
.. automethod:: culebra.solution.feature_selection.Solution._setup

Dunder methods
--------------
Intended to compare (lexicographically) two solutions according to their
fitness.

.. automethod:: culebra.solution.feature_selection.Solution.__eq__
.. automethod:: culebra.solution.feature_selection.Solution.__ge__
.. automethod:: culebra.solution.feature_selection.Solution.__gt__
.. automethod:: culebra.solution.feature_selection.Solution.__hash__
.. automethod:: culebra.solution.feature_selection.Solution.__le__
.. automethod:: culebra.solution.feature_selection.Solution.__lt__
.. automethod:: culebra.solution.feature_selection.Solution.__ne__
.. automethod:: culebra.solution.feature_selection.Solution.__str__
