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

:py:class:`genotype.classifier_optimization.Individual` class
=============================================================

.. autoclass:: genotype.classifier_optimization.Individual

Class attributes
----------------
.. autoattribute:: genotype.classifier_optimization.Individual.species_cls
.. autoattribute:: genotype.classifier_optimization.Individual.eta


Properties
----------
.. autoproperty:: genotype.classifier_optimization.Individual.species
.. autoproperty:: genotype.classifier_optimization.Individual.fitness
.. autoproperty:: genotype.classifier_optimization.Individual.named_values_cls
.. autoproperty:: genotype.classifier_optimization.Individual.values


Methods
-------
.. automethod:: genotype.classifier_optimization.Individual.crossover
.. automethod:: genotype.classifier_optimization.Individual.mutate
.. automethod:: genotype.classifier_optimization.Individual.get

Private methods
---------------
.. automethod:: genotype.classifier_optimization.Individual._random_init

Special methods
---------------

.. automethod:: genotype.classifier_optimization.Individual.__eq__
.. automethod:: genotype.classifier_optimization.Individual.__ne__
.. automethod:: genotype.classifier_optimization.Individual.__lt__
.. automethod:: genotype.classifier_optimization.Individual.__gt__
.. automethod:: genotype.classifier_optimization.Individual.__le__
.. automethod:: genotype.classifier_optimization.Individual.__ge__
.. automethod:: genotype.classifier_optimization.Individual.__str__
