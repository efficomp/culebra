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

Wrapper subpackage
==================

This package provides an abstract base class for all the generarational
wrapper procedures:

* :py:mod:`wrapper.generational_wrapper`

And two generational wrapper procedures: a simple evolutionary algorithm and
another one, based on Non-dominated sorting, able to run NSGA-II and NSGA-III
algorithms:
 
* :py:mod:`wrapper.evolutionary_wrapper`
* :py:mod:`wrapper.nsga_wrapper`

.. toctree::
   :hidden:
   
   wrapper/generational_wrapper
   wrapper/evolutionary_wrapper
   wrapper/nsga_wrapper