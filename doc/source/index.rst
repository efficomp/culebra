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


Welcome to culebra's documentation!
===================================

Culebra is a `DEAP <https://deap.readthedocs.io/en/master/>`_-based
evolutionary computation library designed to solve feature selection problems.

It provides several individual representations, such as bitvectors and set
of feature indices, several fitness functions and several wrapper algorithms.

Experiments and experiment batchs are automatized by means of the 
:py:class:`~tools.experiment.Experiment` and :py:class:`~tools.batch.Batch`
classes, both in the :doc:`tools` subpackage.


Contents:
=========
.. toctree::
   :titlesonly:

   Base subpackage <base>
   Individual subpackage <individual>
   Fitness subpackage <fitness>
   Wrapper subpackage <wrapper>
   Tools subpackage <tools> 



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
