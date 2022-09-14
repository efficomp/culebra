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

:py:class:`wrapper.multi_pop.SequentialCooperative` class
=========================================================

.. autoclass:: wrapper.multi_pop.SequentialCooperative

Class attributes
----------------
.. autoattribute:: wrapper.multi_pop.SequentialCooperative.stats_names
.. autoattribute:: wrapper.multi_pop.SequentialCooperative.objective_stats

Properties
----------
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.individual_classes
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.species
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.fitness_function
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.subpop_wrapper_cls
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.subpop_wrapper_params
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.num_gens
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.pop_sizes
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.crossover_funcs
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.mutation_funcs
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.selection_funcs
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.crossover_probs
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.mutation_probs
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.gene_ind_mutation_probs
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.selection_funcs_params
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.num_subpops
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representation_size
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representation_freq
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representation_topology_func
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representation_topology_func_params
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representation_selection_func
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representation_selection_func_params
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.checkpoint_enable
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.checkpoint_freq
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.checkpoint_filename
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.verbose
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.random_seed
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.subpop_wrappers
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.subpop_wrapper_checkpoint_filenames
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.logbook
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.num_evals
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.runtime
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.index
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.container
.. autoproperty:: wrapper.multi_pop.SequentialCooperative.representatives

Private Properties
------------------
.. autoproperty:: wrapper.multi_pop.SequentialCooperative._state

Static methods
--------------
.. automethod:: wrapper.multi_pop.SequentialCooperative.receive_representatives
.. automethod:: wrapper.multi_pop.SequentialCooperative.send_representatives

Private static methods
----------------------
.. automethod:: wrapper.multi_pop.SequentialCooperative._init_subpop_wrapper_representatives

Methods
-------
.. automethod:: wrapper.multi_pop.SequentialCooperative.reset
.. automethod:: wrapper.multi_pop.SequentialCooperative.evaluate
.. automethod:: wrapper.multi_pop.SequentialCooperative.best_solutions
.. automethod:: wrapper.multi_pop.SequentialCooperative.best_representatives
.. automethod:: wrapper.multi_pop.SequentialCooperative.train
.. automethod:: wrapper.multi_pop.SequentialCooperative.test


Private methods
---------------
.. automethod:: wrapper.multi_pop.SequentialCooperative._generate_subpop_wrappers
.. automethod:: wrapper.multi_pop.SequentialCooperative._save_state
.. automethod:: wrapper.multi_pop.SequentialCooperative._load_state
.. automethod:: wrapper.multi_pop.SequentialCooperative._new_state
.. automethod:: wrapper.multi_pop.SequentialCooperative._reset_state
.. automethod:: wrapper.multi_pop.SequentialCooperative._init_state
.. automethod:: wrapper.multi_pop.SequentialCooperative._init_internals
.. automethod:: wrapper.multi_pop.SequentialCooperative._reset_internals
.. automethod:: wrapper.multi_pop.SequentialCooperative._init_search
.. automethod:: wrapper.multi_pop.SequentialCooperative._start_generation
.. automethod:: wrapper.multi_pop.SequentialCooperative._preprocess_generation
.. automethod:: wrapper.multi_pop.SequentialCooperative._do_generation
.. automethod:: wrapper.multi_pop.SequentialCooperative._postprocess_generation
.. automethod:: wrapper.multi_pop.SequentialCooperative._finish_generation
.. automethod:: wrapper.multi_pop.SequentialCooperative._search
.. automethod:: wrapper.multi_pop.SequentialCooperative._finish_search
.. automethod:: wrapper.multi_pop.SequentialCooperative._init_representatives
