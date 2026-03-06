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

:mod:`culebra.trainer.ea` module
================================

.. automodule:: culebra.trainer.ea

Attributes
----------

.. attribute:: DEFAULT_CROSSOVER_PROB
    :annotation: = 0.8

    Default crossover probability.

.. attribute:: DEFAULT_ELITE_SIZE
    :annotation: = 5

    Default number of elite individuals.

.. attribute:: DEFAULT_GENE_IND_MUTATION_PROB
    :annotation: = 0.1

    Default gene independent mutation probability.

.. attribute:: DEFAULT_MUTATION_PROB
    :annotation: = 0.2

    Default mutation probability.

.. attribute:: DEFAULT_NSGA_SELECTION_FUNC
    :annotation: = <function :func:`deap.tools.selection.selNSGA2`>

    Default selection function for NSGA-based algorithms.


.. attribute:: DEFAULT_NSGA3_REFERENCE_POINTS_P
    :annotation: = 4

    Default number of divisions along each objective for the reference points
    of NSGA-III.

.. attribute:: DEFAULT_POP_SIZE
    :annotation: = 100

    Default population size.

.. attribute:: DEFAULT_SELECTION_FUNC
    :annotation: = :func:`functools.partial`(<function :func:`deap.tools.selection.selTournament`>, tournsize=2)

    Default selection function.


.. toctree::
    :hidden:


    EA <ea/ea>
    ElitistEA <ea/elitist_ea>
    NSGA <ea/nsga>
