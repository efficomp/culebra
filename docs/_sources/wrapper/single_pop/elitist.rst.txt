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

:py:class:`wrapper.single_pop.Elitist` class
=================================================

.. autoclass:: wrapper.single_pop.Elitist

Class attributes
----------------
.. autoattribute:: wrapper.single_pop.Elitist.stats_names
.. autoattribute:: wrapper.single_pop.Elitist.objective_stats

Properties
----------
.. autoproperty:: wrapper.single_pop.Elitist.individual_cls
.. autoproperty:: wrapper.single_pop.Elitist.species
.. autoproperty:: wrapper.single_pop.Elitist.fitness_function
.. autoproperty:: wrapper.single_pop.Elitist.num_gens
.. autoproperty:: wrapper.single_pop.Elitist.pop_size
.. autoproperty:: wrapper.single_pop.Elitist.crossover_func
.. autoproperty:: wrapper.single_pop.Elitist.mutation_func
.. autoproperty:: wrapper.single_pop.Elitist.selection_func
.. autoproperty:: wrapper.single_pop.Elitist.crossover_prob
.. autoproperty:: wrapper.single_pop.Elitist.mutation_prob
.. autoproperty:: wrapper.single_pop.Elitist.gene_ind_mutation_prob
.. autoproperty:: wrapper.single_pop.Elitist.selection_func_params
.. autoproperty:: wrapper.single_pop.Elitist.elite_size
.. autoproperty:: wrapper.single_pop.Elitist.checkpoint_enable
.. autoproperty:: wrapper.single_pop.Elitist.checkpoint_freq
.. autoproperty:: wrapper.single_pop.Elitist.checkpoint_filename
.. autoproperty:: wrapper.single_pop.Elitist.verbose
.. autoproperty:: wrapper.single_pop.Elitist.random_seed
.. autoproperty:: wrapper.single_pop.Elitist.pop
.. autoproperty:: wrapper.single_pop.Elitist.logbook
.. autoproperty:: wrapper.single_pop.Elitist.num_evals
.. autoproperty:: wrapper.single_pop.Elitist.runtime
.. autoproperty:: wrapper.single_pop.Elitist.index
.. autoproperty:: wrapper.single_pop.Elitist.container
.. autoproperty:: wrapper.single_pop.Elitist.representatives

Private Properties
------------------
.. autoproperty:: wrapper.single_pop.Elitist._state

Methods
-------
.. automethod:: wrapper.single_pop.Elitist.reset
.. automethod:: wrapper.single_pop.Elitist.evaluate
.. automethod:: wrapper.single_pop.Elitist.best_solutions
.. automethod:: wrapper.single_pop.Elitist.best_representatives
.. automethod:: wrapper.single_pop.Elitist.train
.. automethod:: wrapper.single_pop.Elitist.test

Private methods
---------------
.. automethod:: wrapper.single_pop.Elitist._save_state
.. automethod:: wrapper.single_pop.Elitist._load_state
.. automethod:: wrapper.single_pop.Elitist._generate_initial_pop
.. automethod:: wrapper.single_pop.Elitist._evaluate_pop
.. automethod:: wrapper.single_pop.Elitist._do_generation_stats
.. automethod:: wrapper.single_pop.Elitist._new_state
.. automethod:: wrapper.single_pop.Elitist._reset_state
.. automethod:: wrapper.single_pop.Elitist._init_state
.. automethod:: wrapper.single_pop.Elitist._init_internals
.. automethod:: wrapper.single_pop.Elitist._reset_internals
.. automethod:: wrapper.single_pop.Elitist._init_search
.. automethod:: wrapper.single_pop.Elitist._start_generation
.. automethod:: wrapper.single_pop.Elitist._preprocess_generation
.. automethod:: wrapper.single_pop.Elitist._do_generation
.. automethod:: wrapper.single_pop.Elitist._postprocess_generation
.. automethod:: wrapper.single_pop.Elitist._finish_generation
.. automethod:: wrapper.single_pop.Elitist._search
.. automethod:: wrapper.single_pop.Elitist._finish_search
.. automethod:: wrapper.single_pop.Elitist._init_representatives
