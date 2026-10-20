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

:class:`culebra.tools.abc.scripts.BatchGenerator` class
=======================================================

.. autoclass:: culebra.tools.abc.scripts.BatchGenerator

Class attributes
----------------
.. autoattribute:: culebra.tools.abc.scripts.BatchGenerator.default_batch_obj
.. autoattribute:: culebra.tools.abc.scripts.BatchGenerator.default_dm_obj_threshold
.. autoattribute:: culebra.tools.abc.scripts.BatchGenerator.default_experiments_per_batch
.. autoattribute:: culebra.tools.abc.scripts.BatchGenerator.default_test_func_obj
.. autoattribute:: culebra.tools.abc.scripts.BatchGenerator.default_trainer_obj

Static methods
--------------
.. automethod:: culebra.tools.abc.scripts.BatchGenerator.load

Methods
-------
.. automethod:: culebra.tools.abc.scripts.BatchGenerator.dump
.. automethod:: culebra.tools.abc.scripts.BatchGenerator.generate
.. automethod:: culebra.tools.abc.scripts.BatchGenerator.parse_args
.. automethod:: culebra.tools.abc.scripts.BatchGenerator.process

Private methods
---------------
.. automethod:: culebra.tools.abc.scripts.BatchGenerator._get_repr_properties
