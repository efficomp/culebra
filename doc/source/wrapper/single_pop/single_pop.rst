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

:py:class:`wrapper.single_pop.SinglePop` class
==============================================

.. autoclass:: wrapper.single_pop.SinglePop

Class attributes
----------------
.. autoattribute:: wrapper.single_pop.SinglePop.stats_names
.. autoattribute:: wrapper.single_pop.SinglePop.objective_stats

Properties
----------
.. autoproperty:: wrapper.single_pop.SinglePop.individual_cls
.. autoproperty:: wrapper.single_pop.SinglePop.species
.. autoproperty:: wrapper.single_pop.SinglePop.fitness_function
.. autoproperty:: wrapper.single_pop.SinglePop.num_gens
.. autoproperty:: wrapper.single_pop.SinglePop.pop_size
.. autoproperty:: wrapper.single_pop.SinglePop.crossover_func
.. autoproperty:: wrapper.single_pop.SinglePop.mutation_func
.. autoproperty:: wrapper.single_pop.SinglePop.selection_func
.. autoproperty:: wrapper.single_pop.SinglePop.crossover_prob
.. autoproperty:: wrapper.single_pop.SinglePop.mutation_prob
.. autoproperty:: wrapper.single_pop.SinglePop.gene_ind_mutation_prob
.. autoproperty:: wrapper.single_pop.SinglePop.selection_func_params
.. autoproperty:: wrapper.single_pop.SinglePop.checkpoint_enable
.. autoproperty:: wrapper.single_pop.SinglePop.checkpoint_freq
.. autoproperty:: wrapper.single_pop.SinglePop.checkpoint_filename
.. autoproperty:: wrapper.single_pop.SinglePop.verbose
.. autoproperty:: wrapper.single_pop.SinglePop.random_seed
.. autoproperty:: wrapper.single_pop.SinglePop.pop
.. autoproperty:: wrapper.single_pop.SinglePop.logbook
.. autoproperty:: wrapper.single_pop.SinglePop.num_evals
.. autoproperty:: wrapper.single_pop.SinglePop.runtime
.. autoproperty:: wrapper.single_pop.SinglePop.index
.. autoproperty:: wrapper.single_pop.SinglePop.container
.. autoproperty:: wrapper.single_pop.SinglePop.representatives

Private Properties
------------------
.. autoproperty:: wrapper.single_pop.SinglePop._state

Methods
-------
.. automethod:: wrapper.single_pop.SinglePop.reset
.. automethod:: wrapper.single_pop.SinglePop.evaluate
.. automethod:: wrapper.single_pop.SinglePop.best_solutions
.. automethod:: wrapper.single_pop.SinglePop.best_representatives
.. automethod:: wrapper.single_pop.SinglePop.train
.. automethod:: wrapper.single_pop.SinglePop.test

Private methods
---------------
.. automethod:: wrapper.single_pop.SinglePop._save_state
.. automethod:: wrapper.single_pop.SinglePop._load_state
.. automethod:: wrapper.single_pop.SinglePop._generate_initial_pop
.. automethod:: wrapper.single_pop.SinglePop._evaluate_pop
.. automethod:: wrapper.single_pop.SinglePop._do_generation_stats
.. automethod:: wrapper.single_pop.SinglePop._new_state
.. automethod:: wrapper.single_pop.SinglePop._reset_state
.. automethod:: wrapper.single_pop.SinglePop._init_state
.. automethod:: wrapper.single_pop.SinglePop._init_internals
.. automethod:: wrapper.single_pop.SinglePop._reset_internals
.. automethod:: wrapper.single_pop.SinglePop._init_search
.. automethod:: wrapper.single_pop.SinglePop._start_generation
.. automethod:: wrapper.single_pop.SinglePop._preprocess_generation
.. automethod:: wrapper.single_pop.SinglePop._do_generation
.. automethod:: wrapper.single_pop.SinglePop._postprocess_generation
.. automethod:: wrapper.single_pop.SinglePop._finish_generation
.. automethod:: wrapper.single_pop.SinglePop._search
.. automethod:: wrapper.single_pop.SinglePop._finish_search
.. automethod:: wrapper.single_pop.SinglePop._init_representatives
