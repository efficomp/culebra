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

:py:class:`base.Dataset` class
==============================

.. autoclass:: base.Dataset

Properties
----------
.. autoproperty:: base.Dataset.num_feats
.. autoproperty:: base.Dataset.size
.. autoproperty:: base.Dataset.inputs
.. autoproperty:: base.Dataset.outputs

Class methods
-------------
.. automethod:: base.Dataset.load_train_test

Methods
-------
.. automethod:: base.Dataset.normalize
.. automethod:: base.Dataset.robust_scale
.. automethod:: base.Dataset.remove_outliers
.. automethod:: base.Dataset.append_random_features
.. automethod:: base.Dataset.split

