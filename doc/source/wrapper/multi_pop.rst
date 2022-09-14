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

:py:class:`wrapper.multi_pop` module
====================================

.. automodule:: wrapper.multi_pop

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

.. attribute:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
    :annotation: = <function ring_destinations>

    Default topology function for the islands model.

.. attribute:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    :annotation: = {}

    Parameters for the default topology function in the islands model.

.. attribute:: DEFAULT_REPRESENTATION_SELECTION_FUNC
    :annotation: = <function selTournament>

    Default selection policy function to choose the representatives.

.. attribute:: DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS
    :annotation: = {'tournsize': 3}

    Default parameters for the representatives selection policy function.

..
    .. autodata:: DEFAULT_NUM_SUBPOPS
    .. autodata:: DEFAULT_REPRESENTATION_SIZE
    .. autodata:: DEFAULT_REPRESENTATION_FREQ
    .. autodata:: DEFAULT_REPRESENTATION_TOPOLOGY_FUNC
    .. autodata:: DEFAULT_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    .. autodata:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC
    .. autodata:: DEFAULT_ISLANDS_REPRESENTATION_TOPOLOGY_FUNC_PARAMS
    .. autodata:: DEFAULT_REPRESENTATION_SELECTION_FUNC
    .. autodata:: DEFAULT_REPRESENTATION_SELECTION_FUNC_PARAMS

.. toctree::
    :hidden:

    multi_pop <multi_pop/multi_pop>
    multi_pop <multi_pop/sequential_multi_pop>
    multi_pop <multi_pop/parallel_multi_pop>

    multi_pop <multi_pop/islands>
    multi_pop <multi_pop/homogeneous_islands>
    multi_pop <multi_pop/heterogeneous_islands>

    multi_pop <multi_pop/homogeneous_sequential_islands>
    multi_pop <multi_pop/homogeneous_parallel_islands>
    multi_pop <multi_pop/heterogeneous_sequential_islands>
    multi_pop <multi_pop/heterogeneous_parallel_islands>

    multi_pop <multi_pop/cooperative>
    multi_pop <multi_pop/sequential_cooperative>
    multi_pop <multi_pop/parallel_cooperative>

    multi_pop <multi_pop/topology>
