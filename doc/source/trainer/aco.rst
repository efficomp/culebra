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

:py:class:`culebra.trainer.aco` module
======================================

.. automodule:: culebra.trainer.aco

Attributes
----------

.. attribute:: DEFAULT_PHEROMONE_INFLUENCE
    :annotation: = 1.0

    Default pheromone influence (:math:`{\alpha}`).

.. attribute:: DEFAULT_HEURISTIC_INFLUENCE
    :annotation: = 2.0

    Default heuristic influence (:math:`{\beta}`).

.. attribute:: DEFAULT_PHEROMONE_EVAPORATION_RATE
    :annotation: = 0.5

    Default pheromone evaporation rate (:math:`{\rho}`).

.. attribute:: DEFAULT_MMAS_PHEROMONE_EVAPORATION_RATE
    :annotation: = 0.2

    Default pheromone evaporation rate for the
    :math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS (:math:`{\rho}`).

.. attribute:: DEFAULT_MMAS_ITER_BEST_USE_LIMIT
    :annotation: = 250

    Default limit for the number of iterations for the
    :math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS to give up using the
    iteration-best ant to deposit pheromones. Iterations above this limit will
    use only the global-best ant.

.. attribute:: DEFAULT_MMAS_CONVERGENCE_CHECK_FREQ
    :annotation: = 100

    Default frequency to check if the
    :math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS has converged.


..
    .. autodata:: DEFAULT_PHEROMONE_INFLUENCE
    .. autodata:: DEFAULT_HEURISTIC_INFLUENCE
    .. autodata:: DEFAULT_PHEROMONE_EVAPORATION_RATE
    .. autodata:: DEFAULT_MMAS_PHEROMONE_EVAPORATION_RATE
    .. autodata:: DEFAULT_MMAS_ITER_BEST_USE_LIMIT
    .. autodata:: DEFAULT_MMAS_CONVERGENCE_CHECK_FREQ


.. toctree::
    :hidden:

    abc <aco/abc>

    AntSystem <aco/ant_system>
    ElitistAntSystem <aco/elitist_ant_system>
    MMAS <aco/mmas>
