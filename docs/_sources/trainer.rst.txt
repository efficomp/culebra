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
   Culebra. If not, see <http://www.gnu.org/licenses/>.

   This work is supported by projects PGC2018-098813-B-C31 and
   PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
   Innovación y Universidades" and by the European Regional Development Fund
   (ERDF).

:mod:`culebra.trainer` module
=============================

.. automodule:: culebra.trainer

Attributes
----------
.. attribute:: DEFAULT_CHECKPOINT_ACTIVATION
    :annotation: = True

    Default checkpointing activation for a :class:`~culebra.trainer.abc.CentralizedTrainer`.

.. attribute:: DEFAULT_CHECKPOINT_BASENAME
     :annotation: = 'checkpoint'

     Default basename for checkpointing files.

.. attribute:: DEFAULT_CHECKPOINT_FREQ
    :annotation: = 10

    Default checkpointing frequency for a :class:`~culebra.trainer.abc.CentralizedTrainer`.

.. attribute:: DEFAULT_COOPERATIVE_TOPOLOGY_FUNC
    :annotation: = <function :func:`culebra.trainer.topology.full_connected_destinations`>

    Default topology function for the cooperative model.

.. attribute:: DEFAULT_ISLANDS_TOPOLOGY_FUNC
    :annotation: = <function :func:`culebra.trainer.topology.ring_destinations`>

    Default topology function for the islands model.

.. attribute:: DEFAULT_MAX_NUM_ITERS
    :annotation: = 100

    Default maximum number of iterations.

.. attribute:: DEFAULT_NUM_REPRESENTATIVES
    :annotation: = 5

    Default value for the number of representatives selected.

.. attribute:: DEFAULT_REPRESENTATIVES_EXCHANGE_FREQ
    :annotation: = 10

    Default value for the number of iterations between representatives
    sending.

.. attribute:: DEFAULT_REPRESENTATIVES_SELECTION_FUNC
    :annotation: = :func:`functools.partial`(<function :func:`deap.tools.selection.selTournament`>, tournsize=3)

    Default selection policy function to choose the representatives.

.. attribute:: DEFAULT_VERBOSITY
    :annotation: = True

    Default verbosity for a :class:`~culebra.trainer.abc.CentralizedTrainer`.


.. toctree::
    :hidden:

    abc <trainer/abc>
    ea <trainer/ea>
    aco <trainer/aco>
    topology <trainer/topology>
