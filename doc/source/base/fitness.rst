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

:py:class:`base.Fitness` class
==============================

.. autoclass:: base.Fitness

Class attributes
----------------
.. autoattribute:: base.Fitness.weights
.. autoattribute:: base.Fitness.names
.. autoattribute:: base.Fitness.thresholds

Properties
----------
.. autoproperty:: base.Fitness.num_obj

Methods
-------
.. automethod:: base.Fitness.dominates

Special methods
---------------
Intended to compare (lexicographically) two individuals according to their
fitness.

.. automethod:: base.Fitness.__eq__
.. automethod:: base.Fitness.__ne__
.. automethod:: base.Fitness.__lt__
.. automethod:: base.Fitness.__gt__
.. automethod:: base.Fitness.__le__
.. automethod:: base.Fitness.__ge__
