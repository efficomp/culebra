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

:py:class:`wrapper.multi_pop.HeterogeneousParallelIslands` class
================================================================

.. autoclass:: wrapper.multi_pop.HeterogeneousParallelIslands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.HeterogeneousParallelIslands.stats_names
.. autoattribute:: wrapper.multi_pop.HeterogeneousParallelIslands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.individual_cls
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.species
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.fitness_function
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.num_gens
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.pop_sizes
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.crossover_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.mutation_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.selection_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.crossover_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.mutation_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.gene_ind_mutation_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.selection_funcs_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.num_subpops
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representation_size
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representation_freq
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.verbose
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.random_seed
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.logbook
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.num_evals
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.runtime
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.index
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.container
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.HeterogeneousParallelIslands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.receive_representatives
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.reset
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.evaluate
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.best_solutions
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.best_representatives
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.train
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._save_state
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._load_state
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._new_state
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._reset_state
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._init_state
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._init_internals
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._reset_internals
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._init_search
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._start_generation
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._preprocess_generation
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._do_generation
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._postprocess_generation
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._finish_generation
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._search
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._finish_search
.. automethod:: wrapper.multi_pop.HeterogeneousParallelIslands._init_representatives
