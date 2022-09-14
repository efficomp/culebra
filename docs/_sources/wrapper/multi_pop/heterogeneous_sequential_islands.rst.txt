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

:py:class:`wrapper.multi_pop.HeterogeneousSequentialIslands` class
==================================================================

.. autoclass:: wrapper.multi_pop.HeterogeneousSequentialIslands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.HeterogeneousSequentialIslands.stats_names
.. autoattribute:: wrapper.multi_pop.HeterogeneousSequentialIslands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.individual_cls
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.species
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.fitness_function
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.num_gens
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.pop_sizes
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.crossover_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.mutation_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.selection_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.crossover_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.mutation_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.gene_ind_mutation_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.selection_funcs_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.num_subpops
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representation_size
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representation_freq
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.verbose
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.random_seed
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.logbook
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.num_evals
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.runtime
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.index
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.container
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.HeterogeneousSequentialIslands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.receive_representatives
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.reset
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.evaluate
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.best_solutions
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.best_representatives
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.train
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._save_state
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._load_state
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._new_state
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._reset_state
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._init_state
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._init_internals
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._reset_internals
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._init_search
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._start_generation
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._preprocess_generation
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._do_generation
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._postprocess_generation
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._finish_generation
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._search
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._finish_search
.. automethod:: wrapper.multi_pop.HeterogeneousSequentialIslands._init_representatives
