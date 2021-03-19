..
   This file is part of culebra.

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

   This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
   de Ciencia, Innovación y Universidades"), and by the European Regional
   Development Fund (ERDF).

Base subpackage
===============

Culebra is based in some base modules to define the fundamental pieces that are
necessary to solve a feature selection problem. The base subpackage defines:

* A :doc:`dataset <base/dataset>` with samples, where the most relevant input
  features need to be selected
* The :doc:`individuals <base/individual>`, which will be used within the
  wrapper method to search the best subset of features
* A :doc:`species <base/species>` to define the characteristics of the
  individuals
* The :doc:`fitness function <base/fitness>` to guide the search towards
  optimal solutions
* The :doc:`wrapper <base/wrapper>` method to perform the search 


.. toctree::
   :hidden:

   base/dataset
   base/individual
   base/species
   base/fitness
   base/wrapper