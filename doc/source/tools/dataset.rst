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

:py:class:`culebra.tools.Dataset` class
=======================================

.. autoclass:: culebra.tools.Dataset

Class methods
-------------
.. automethod:: culebra.tools.Dataset.load_pickle
.. automethod:: culebra.tools.Dataset.load_train_test

Properties
----------
.. autoproperty:: culebra.tools.Dataset.num_feats
.. autoproperty:: culebra.tools.Dataset.size
.. autoproperty:: culebra.tools.Dataset.inputs
.. autoproperty:: culebra.tools.Dataset.outputs

Methods
-------
.. automethod:: culebra.tools.Dataset.save_pickle
.. automethod:: culebra.tools.Dataset.normalize
.. automethod:: culebra.tools.Dataset.robust_scale
.. automethod:: culebra.tools.Dataset.remove_outliers
.. automethod:: culebra.tools.Dataset.append_random_features
.. automethod:: culebra.tools.Dataset.split

