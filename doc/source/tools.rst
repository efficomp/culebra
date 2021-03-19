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

Tools subpackage
================

This package provides several tools:

* Since individuals are generated and modified randomly, it is necessary an
  :doc:`Individual tester <tools/individual_tester>` to test extensively
  the random generation, crossover and mutation of all the
  :py:class:`~base.individual.Individual` implementations.

* Some :doc:`Feature metrics <tools/feature_metrics>` are also necessary to
  estimate the relevance of the input features from the
  :py:class:`~base.wrapper.Wrapper` results

* Automated experimentation is also a quite valuable characteristic when a
  :py:class:`~base.wrapper.Wrapper` method has to be run many times. Culebra
  provides this features by means of a `TOML
  <https://github.com/toml-lang/toml>`_ :doc:`configuration file
  <tools/config_file>` and the :py:mod:`~tools.config_manager` module, which
  is able to generate :doc:`experiments <tools/experiment>` and even
  :doc:`batchs of experiments <tools/batch>` from a single
  :doc:`configuration file <tools/config_file>` 
  
.. toctree::
   :hidden:
   
   tools/individual_tester
   tools/feature_metrics
   tools/config_manager
   tools/experiment
   tools/batch
   tools/config_file
