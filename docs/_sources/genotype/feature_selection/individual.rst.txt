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

:py:class:`genotype.feature_selection.Individual` class
=======================================================

.. autoclass:: genotype.feature_selection.Individual

Class attributes
----------------
.. autoattribute:: genotype.feature_selection.Individual.species_cls


Properties
----------
.. autoproperty:: genotype.feature_selection.Individual.species
.. autoproperty:: genotype.feature_selection.Individual.fitness
.. autoproperty:: genotype.feature_selection.Individual.features
.. autoproperty:: genotype.feature_selection.Individual.num_feats
.. autoproperty:: genotype.feature_selection.Individual.min_feat
.. autoproperty:: genotype.feature_selection.Individual.max_feat


Methods
-------
.. automethod:: genotype.feature_selection.Individual.crossover
.. automethod:: genotype.feature_selection.Individual.mutate
.. automethod:: genotype.feature_selection.Individual.dominates
.. automethod:: genotype.feature_selection.Individual.delete_fitness

Private methods
---------------
.. automethod:: genotype.feature_selection.Individual._random_init

Dunder methods
--------------
.. automethod:: genotype.feature_selection.Individual.__eq__
.. automethod:: genotype.feature_selection.Individual.__ne__
.. automethod:: genotype.feature_selection.Individual.__lt__
.. automethod:: genotype.feature_selection.Individual.__gt__
.. automethod:: genotype.feature_selection.Individual.__le__
.. automethod:: genotype.feature_selection.Individual.__ge__
.. automethod:: genotype.feature_selection.Individual.__str__
