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

:py:class:`wrapper.multi_pop.HeterogeneousIslands` class
========================================================

.. autoclass:: wrapper.multi_pop.HeterogeneousIslands

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.HeterogeneousIslands.stats_names
.. autoattribute:: wrapper.multi_pop.HeterogeneousIslands.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.individual_cls
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.species
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.fitness_function
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.num_gens
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.pop_sizes
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.crossover_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.mutation_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.selection_funcs
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.crossover_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.mutation_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.gene_ind_mutation_probs
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.selection_funcs_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.num_subpops
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representation_size
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representation_freq
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representation_topology_func
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representation_selection_func
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.verbose
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.random_seed
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.logbook
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.num_evals
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.runtime
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.index
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.container
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.HeterogeneousIslands._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.receive_representatives
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.send_representatives


Methods
-------
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.reset
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.evaluate
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.best_solutions
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.best_representatives
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.train
.. automethod:: wrapper.multi_pop.HeterogeneousIslands.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._save_state
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._load_state
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._new_state
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._reset_state
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._init_state
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._init_internals
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._reset_internals
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._init_search
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._start_generation
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._preprocess_generation
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._do_generation
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._postprocess_generation
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._finish_generation
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._search
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._finish_search
.. automethod:: wrapper.multi_pop.HeterogeneousIslands._init_representatives
