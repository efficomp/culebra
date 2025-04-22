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

.. attribute:: DEFAULT_CONVERGENCE_CHECK_FREQ
    :annotation: = 100

    Default frequency to check if an elitist ACO has converged.

.. attribute:: DEFAULT_PHEROMONE_EVAPORATION_RATE
    :annotation: = 0.1

    Default pheromone evaporation rate (:math:`{\rho}`).

.. attribute:: DEFAULT_ELITE_WEIGHT
    :annotation: = 0.3

    Default weight for the elite ant (best-so-far ant) respect to the
    iteration-best ant.

.. attribute:: DEFAULT_MMAS_ITER_BEST_USE_LIMIT
    :annotation: = 250

    Default limit for the number of iterations for the
    :math:`{\small \mathcal{MAX}{-}\mathcal{MIN}}` AS to give up using the
    iteration-best ant to deposit pheromone. Iterations above this limit will
    use only the global-best ant.


.. attribute:: DEFAULT_ACO_FS_INITIAL_PHEROMONE
    :annotation: = 1

    Default initial pheromone.

.. attribute:: DEFAULT_ACO_FS_DISCARD_PROB
    :annotation: = 0.5

    Default probability of discarding a node (feature).


..
    .. autodata:: DEFAULT_PHEROMONE_INFLUENCE
    .. autodata:: DEFAULT_HEURISTIC_INFLUENCE
    .. autodata:: DEFAULT_CONVERGENCE_CHECK_FREQ
    .. autodata:: DEFAULT_PHEROMONE_EVAPORATION_RATE
    .. autodata:: DEFAULT_ELITE_WEIGHT
    .. autodata:: DEFAULT_MMAS_ITER_BEST_USE_LIMIT
    .. autodata:: DEFAULT_ACO_FS_INITIAL_PHEROMONE
    .. autodata:: DEFAULT_ACO_FS_DISCARD_PROB


.. toctree::
    :hidden:

    abc <aco/abc>

    AntSystem <aco/ant_system>
    ElitistAntSystem <aco/elitist_ant_system>
    MMAS <aco/mmas>
    AgeBasedPACO <aco/age_based_paco>
    QualityBasedPACO <aco/quality_based_paco>
    PACO_MO <aco/paco_mo>
    CPACO <aco/cpaco>
    PACO_FS <aco/paco_fs>
    ElitistACO_FS <aco/elitist_aco_fs>
    ACO_FSConvergenceDetector <aco/aco_fs_convergence_detector>
