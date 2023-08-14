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

:py:mod:`culebra.fitness_function` module
=========================================

.. automodule:: culebra.fitness_function

Attributes
----------
.. attribute:: DEFAULT_CLASSIFIER
    :annotation: = <class 'sklearn.naive_bayes.GaussianNB'>

    Default classifier for fitness functions

.. attribute:: DEFAULT_THRESHOLD
    :annotation: = 0.01

    Default similarity threshold for fitnesses


..
    .. autodata:: DEFAULT_CLASSIFIER
    .. autodata:: DEFAULT_THRESHOLD


.. toctree::
    :hidden:

    abc <fitness_function/abc>
    feature_selection <fitness_function/feature_selection>
    svc_optimization <fitness_function/svc_optimization>
    cooperative <fitness_function/cooperative>
