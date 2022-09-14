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

:py:class:`tools.Evaluation` class
==================================

.. autoclass:: tools.Evaluation

Class attributes
----------------
.. autoattribute:: tools.Evaluation.feature_metric_functions
.. autoattribute:: tools.Evaluation.stats_functions

Private class attributes
------------------------
.. class:: tools.Evaluation._ResultKeys

    Result keys for the evaluation.

    It is empty, since :py:class:`~tools.Evaluation` is an abstract
    class. Subclasses should override this class to fill it with the
    appropriate result keys.

.. autoattribute:: tools.Evaluation._script_code

Properties
----------
.. autoproperty:: tools.Evaluation.wrapper
.. autoproperty:: tools.Evaluation.test_fitness_function
.. autoproperty:: tools.Evaluation.results


Class methods
-------------
.. automethod:: tools.Evaluation.from_config
.. automethod:: tools.Evaluation.generate_script

Methods
-------
.. automethod:: tools.Evaluation.reset
.. automethod:: tools.Evaluation.run

Private methods
---------------
.. automethod:: tools.Evaluation._execute
