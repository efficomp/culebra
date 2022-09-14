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

:py:class:`wrapper.multi_pop.ParallelCooperative` class
=======================================================

.. autoclass:: wrapper.multi_pop.ParallelCooperative

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.ParallelCooperative.stats_names
.. autoattribute:: wrapper.multi_pop.ParallelCooperative.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.individual_classes
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.species
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.fitness_function
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.num_gens
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.pop_sizes
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.crossover_funcs
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.mutation_funcs
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.selection_funcs
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.crossover_probs
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.mutation_probs
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.gene_ind_mutation_probs
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.selection_funcs_params
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.num_subpops
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representation_size
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representation_freq
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representation_topology_func
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representation_selection_func
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.verbose
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.random_seed
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.logbook
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.num_evals
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.runtime
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.index
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.container
.. autoproperty:: wrapper.multi_pop.ParallelCooperative.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.ParallelCooperative._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.ParallelCooperative.receive_representatives
.. automethod:: wrapper.multi_pop.ParallelCooperative.send_representatives

Private static methods
----------------------
.. automethod:: wrapper.multi_pop.ParallelCooperative._init_subpop_wrapper_representatives

Methods
-------
.. automethod:: wrapper.multi_pop.ParallelCooperative.reset
.. automethod:: wrapper.multi_pop.ParallelCooperative.evaluate
.. automethod:: wrapper.multi_pop.ParallelCooperative.best_solutions
.. automethod:: wrapper.multi_pop.ParallelCooperative.best_representatives
.. automethod:: wrapper.multi_pop.ParallelCooperative.train
.. automethod:: wrapper.multi_pop.ParallelCooperative.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.ParallelCooperative._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.ParallelCooperative._save_state
.. automethod:: wrapper.multi_pop.ParallelCooperative._load_state
.. automethod:: wrapper.multi_pop.ParallelCooperative._new_state
.. automethod:: wrapper.multi_pop.ParallelCooperative._reset_state
.. automethod:: wrapper.multi_pop.ParallelCooperative._init_state
.. automethod:: wrapper.multi_pop.ParallelCooperative._init_internals
.. automethod:: wrapper.multi_pop.ParallelCooperative._reset_internals
.. automethod:: wrapper.multi_pop.ParallelCooperative._init_search
.. automethod:: wrapper.multi_pop.ParallelCooperative._start_generation
.. automethod:: wrapper.multi_pop.ParallelCooperative._preprocess_generation
.. automethod:: wrapper.multi_pop.ParallelCooperative._do_generation
.. automethod:: wrapper.multi_pop.ParallelCooperative._postprocess_generation
.. automethod:: wrapper.multi_pop.ParallelCooperative._finish_generation
.. automethod:: wrapper.multi_pop.ParallelCooperative._search
.. automethod:: wrapper.multi_pop.ParallelCooperative._finish_search
.. automethod:: wrapper.multi_pop.ParallelCooperative._init_representatives
