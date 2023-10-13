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

:py:mod:`culebra.trainer` module
================================

.. automodule:: culebra.trainer


Attributes
----------

.. attribute:: DEFAULT_NUM_SUBPOPS
    :annotation: = 1

    Default number of subpopulations.

.. attribute:: DEFAULT_REPRESENTATION_SIZE
    :annotation: = 5

    Default value for the number of representatives sent to the other
    subpopulations.

.. attribute:: DEFAULT_REPRESENTATION_FREQ
    :annotation: = 10

    Default value for the number of generations between representatives
    sending.

.. attribute:: DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
    :annotation: = <function full_connected_destinations>

    Default topology function for representatives sending.

.. attribute:: DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    :annotation: = {}

    Default parameters to obtain the destinations with the topology function.

.. attribute:: DEFAULT_REPRESENTATION_SELECTION_FUNC
    :annotation: = <function selTournament>

    Default selection policy function to choose the representatives.

.. attribute:: DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
    :annotation: = {'tournsize': 3}

    Default parameters for the representatives selection policy function.

.. attribute:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
    :annotation: = <function ring_destinations>

    Default topology function for the islands model.

.. attribute:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    :annotation: = {}

    Parameters for the default topology function in the islands model.

.. attribute:: DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC
    :annotation: = <full_connected_destinationss>

    Default topology function for the cooperative model.

.. attribute:: DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    :annotation: = {}

    Parameters for the default topology function in the cooperative model.


..
    .. autodata:: DEFAULT_NUM_SUBPOPS
    .. autodata:: DEFAULT_REPRESENTATION_SIZE
    .. autodata:: DEFAULT_REPRESENTATION_FREQ
    .. autodata:: DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
    .. autodata:: DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    .. autodata:: DEFAULT_REPRESENTATION_SELECTION_FUNC
    .. autodata:: DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
    .. autodata:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
    .. autodata:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    .. autodata:: DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC
    .. autodata:: DEFAULT_COOPERATIVE_REPRESENTATION_TOPOLOGY_FUNC_PARAMS

.. toctree::
    :hidden:

    abc <trainer/abc>
    ea <trainer/ea>
    aco <trainer/aco>
    topology <trainer/topology>
