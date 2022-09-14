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

:py:class:`wrapper.single_pop` module
=====================================

.. automodule:: wrapper.single_pop

Attributes
----------

.. attribute:: DEFAULT_POP_SIZE
    :annotation: = 100

    Default population size.

.. attribute:: DEFAULT_CROSSOVER_PROB
    :annotation: = 0.9

    Default crossover probability.

.. attribute:: DEFAULT_MUTATION_PROB
    :annotation: = 0.1

    Default mutation probability.

.. attribute:: DEFAULT_GENE_IND_MUTATION_PROB
    :annotation: = 0.1

    Default gene independent mutation probability.

.. attribute:: DEFAULT_SELECTION_FUNC
    :annotation: = <function selTournament>

    Default selection function.

.. attribute:: DEFAULT_NSGA_SELECTION_FUNC
    :annotation: = <function selNSGA2>

    Default selection function for NSGA-based algorithms.

.. attribute:: DEFAULT_SELECTION_FUNC_PARAMS
    :annotation: = {'tournsize': 2}

    Default selection function parameters.

.. attribute:: DEFAULT_NSGA_SELECTION_FUNC_PARAMS
    :annotation: = {}

    Default selection function parameters for NSGA-based algorithms.

.. attribute:: DEFAULT_ELITE_SIZE
    :annotation: = 5

    Default number of elite individuals.

.. attribute:: DEFAULT_NSGA3_REFERENCE_POINTS_P
    :annotation: = 4

    Default number of divisions along each objective for the reference points
    of NSGA-III.

..
    .. autodata:: DEFAULT_POP_SIZE
    .. autodata:: DEFAULT_CROSSOVER_PROB
    .. autodata:: DEFAULT_MUTATION_PROB
    .. autodata:: DEFAULT_GENE_IND_MUTATION_PROB
    .. autodata:: DEFAULT_SELECTION_FUNC
    .. autodata:: DEFAULT_NSGA_SELECTION_FUNC
    .. autodata:: DEFAULT_SELECTION_FUNC_PARAMS
    .. autodata:: DEFAULT_NSGA_SELECTION_FUNC_PARAMS
    .. autodata:: DEFAULT_ELITE_SIZE
    .. autodata:: DEFAULT_NSGA3_REFERENCE_POINTS_P


.. toctree::
    :hidden:

    num_feats <single_pop/single_pop>
    evolutionary <single_pop/evolutionary>
    elitist <single_pop/elitist>
    nsga <single_pop/nsga>
