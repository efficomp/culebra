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

:py:mod:`experiment` module
===========================

Automated experimentation is also a quite valuable characteristic when a
:py:class:`~base.Wrapper` method has to be run many times. Culebra provides
this features by means of a `TOML <https://github.com/toml-lang/toml>`_
:doc:`configuration file <config_file>` and the
:py:class:`~experiment.ConfigManager` class, which is able to generate
experiments and even batchs of experiments from a single configuration file.

.. automodule:: experiment
   :members:
   :inherited-members:
   :special-members:
   :private-members:
   :show-inheritance: